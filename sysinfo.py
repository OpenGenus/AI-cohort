"""Shared system-info header. Printed with every measurement so numbers have context.
Usage:  python3 sysinfo.py         (prints the header)
        from sysinfo import header  # in other modules
"""

import os, platform, subprocess


def get_info():
    info = {}
    info["os"] = platform.platform()
    info["python"] = platform.python_version()
    model = platform.processor() or "unknown"
    try:
        for line in open("/proc/cpuinfo"):
            if "model name" in line:
                model = line.split(":", 1)[1].strip()
                break
    except Exception:
        pass
    info["cpu"] = model
    info["logical_cores"] = os.cpu_count()

    # physical cores (from lscpu: sockets * cores-per-socket)
    try:
        out = subprocess.check_output(["lscpu"], text=True)
        vals = {}
        for l in out.splitlines():
            if ":" in l:
                k, v = l.split(":", 1)
                vals[k.strip()] = v.strip()

        sockets = int(vals.get("Socket(s)", "1"))
        cps = int(vals.get("Core(s) per socket", "0")) or 0
        if cps:
            info["physical_cores"] = sockets * cps

        for c in ("L1d cache", "L1i cache", "L2 cache", "L3 cache"):
            if c in vals:
                info[c] = vals[c]
    except Exception:
        pass

    try:
        mem_kb = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1])
        info["ram_gb"] = round(mem_kb / 1024 / 1024, 1)
    except Exception:
        info["ram_gb"] = "n/a"

    # ISA flags that matter for AI perf (explains why bf16/int8 are fast or slow)
    try:
        flags = ""
        for line in open("/proc/cpuinfo"):
            if line.startswith("flags"):
                flags = line.split(":", 1)[1]
                break
                watch = ["avx2", "avx512f", "avx512_vnni", "avx512_bf16", "amx_bf16", "amx_int8", "f16c"]
                present = [f for f in watch if (" " + f + " ") in (" " + flags + " ")]
                info["isa_features"] = ", ".join(present) if present else "none of the watched set"
            except Exception:
                pass

            try:
                import numpy
                info["numpy"] = numpy.__version__
            except Exception:
                pass

            try:
                os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
                import torch
                info["torch"] = torch.__version__
                info["torch_threads"] = torch.get_num_threads()
            except Exception:
                pass

            return info


def header():
    info = get_info()
    print("=" * 62)
    print("SYSTEM INFO")
    print("=" * 62)
    for k, v in info.items():
        print(f"{k:16}: {v}")
    print("=" * 62)
    return info


if __name__ == "__main__":
    header()
