"""M7 - A tiny inference server with DYNAMIC BATCHING (the serving payoff module).

Why this exists: M6 showed that a bigger *batch* raises tokens/sec. But real
requests arrive one at a time. A production server therefore *collects* requests
that arrive close together and runs them as ONE batch - trading a little latency
for a lot of throughput. That is the single most important idea in LLM serving,
and this ~120-line server implements it from scratch so you can see it work.

    client --HTTP--> [queue] --collect up to MAX_BATCH within MAX_WAIT_MS--> model.generate(batch) --> reply

This is *request-level* dynamic batching. vLLM / TGI go further with *token-level*
continuous batching + a paged KV-cache, but the lever is the same one you tune here.

Run (GPU strongly recommended - Colab T4, a GPU droplet, or any CUDA box):
    pip install -r requirements-serve.txt
    python m7_server.py                 # serves on http://0.0.0.0:8000
Then, in another terminal:
    python m7_loadtest.py               # measures p50/p99 latency + tokens/s vs concurrency

No GPU? Run in SIMULATION mode - no model, no torch, no download, just the batching
logic with a fake "forward pass" whose time depends on tokens (not on batch size):
    SIM=1 pip-free: python m7_server.py    # only needs fastapi + uvicorn
The whole point (throughput rises with batch size) is visible in SIM mode too.

Knobs (env vars): SIM (0/1), MAX_BATCH (default 8), MAX_WAIT_MS (default 10), MODEL, PORT.
Set MAX_BATCH=1 to DISABLE batching and see throughput collapse under load.
"""

import asyncio
import os
import time

SIM = os.environ.get("SIM", "0") == "1"
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
MAX_BATCH = int(os.environ.get("MAX_BATCH", "8"))
MAX_WAIT_MS = float(os.environ.get("MAX_WAIT_MS", "10"))
PORT = int(os.environ.get("PORT", "8000"))
SIM_STEP_S = float(os.environ.get("SIM_STEP_S", "0.004"))  # fake seconds per decoded token

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

if not SIM:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
else:
    DEVICE = "sim"

app = FastAPI(title="M7 dynamic-batching LLM server")
queue: "asyncio.Queue" = None  # set on startup
tok = model = None
stats = {"requests": 0, "batches": 0, "batched_tokens": 0}

class Req(BaseModel):
    prompt: str
    max_new_tokens: int = 128

def _load():
    global tok, model
    if SIM:
        return
    tok = AutoTokenizer.from_pretrained(MODEL)
    tok.padding_side = "left"  # decoder-only models must left-pad for batched generate
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dtype).to(DEVICE).eval()

def _run_batch(prompts, max_new):
    """Run ONE forward/generate for the whole batch, return (completions, total_new_tokens).

    The key property (real or simulated): a batch of B sequences costs about the SAME
    wall-time as a single sequence, because they share each forward pass. That is why
    batching multiplies throughput.
    """
    if SIM:
        time.sleep(SIM_STEP_S * max_new)  # one batched pass; time depends on tokens, not batch size
        return ["(sim output)"] * len(prompts), max_new * len(prompts)
    return _run_batch_real(prompts, max_new)

def _run_batch_real(prompts, max_new):
    """* Tokenize a list of prompts (padded), generate once, return decoded completions. """
    texts = [tok.apply_chat_template([{"role": "user", "content": p}],
                                     tokenize=False, add_generation_prompt=True) for p in prompts]
    enc = tok(texts, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    gen = out[:, enc["input_ids"].shape[1]:]
    n_tokens = int((gen != tok.pad_token_id).sum().item())
    return tok.batch_decode(gen, skip_special_tokens=True), n_tokens

async def _scheduler():
    """The heart of the server: pull requests, form a batch, run it, fan out replies."""
    while True:
        prompt, max_new, fut = await queue.get()              # block for the first request
        batch = [(prompt, max_new, fut)]
        deadline = time.perf_counter() + MAX_WAIT_MS / 1000.0
        while len(batch) < MAX_BATCH:                         # then grab any others that arrive within the window
            timeout = deadline - time.perf_counter()
            if timeout <= 0:
                break
            try:
                batch.append(await asyncio.wait_for(queue.get(), timeout))
            except asyncio.TimeoutError:
                break
        prompts = [b[0] for b in batch]
        max_new = max(b[1] for b in batch)
        texts, n_tokens = await asyncio.to_thread(_run_batch, prompts, max_new)
        stats["batches"] += 1
        stats["batched_tokens"] += n_tokens
        for (_, _, f), text in zip(batch, texts):
            f.set_result(text)

@app.on_event("startup")
async def _startup():
    global queue
    _load()
    queue = asyncio.Queue()
    asyncio.create_task(_scheduler())
    print(f"[M7] model={MODEL} device={DEVICE} MAX_BATCH={MAX_BATCH} MAX_WAIT_MS={MAX_WAIT_MS}")

@app.post("/generate")
async def generate(r: Req):
    t0 = time.perf_counter()
    fut = asyncio.get_event_loop().create_future()
    await queue.put((r.prompt, r.max_new_tokens, fut))
    text = await fut
    stats["requests"] += 1
    return {"text": text, "latency_s": round(time.perf_counter() - t0, 4)}

@app.get("/stats")
async def get_stats():
    avg_batch = stats["batched_tokens"] and stats["batches"] and (
        stats["requests"] / stats["batches"])
    return {**stats, "device": DEVICE, "max_batch": MAX_BATCH,
            "avg_requests_per_batch": round(avg_batch, 2) if avg_batch else None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")
