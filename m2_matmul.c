/*
 * M2 - Custom CPU matmul kernel: the optimization "ladder"
 *   naive  -> cache-blocked (tiled) -> tiled + OpenMP threads
 * Each variant is verified for correctness against the naive result, and its
 * GFLOP/s is reported. This shows how hand optimization climbs toward the
 * roofline. Portable C + OpenMP (compiles on x86 and ARM; no x86 intrinsics).
 *
 * Build & run: see m2_run.sh    (gcc -O3 -fopenmp -march=native)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#ifndef N
#define N 1024
#endif

#ifndef TILE
#define TILE 64
#endif

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void naive(const float *A, const float *B, float *C) {
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float s = 0.0f;
            for (int k = 0; k < N; k++)
                s += A[i * N + k] * B[k * N + j];
            C[i * N + j] = s;
        }
}

static void tiled(const float *A, const float *B, float *C) {
    for (int i = 0; i < N * N; i++)
        C[i] = 0.0f;

    for (int ii = 0; ii < N; ii += TILE)
        for (int kk = 0; kk < N; kk += TILE)
            for (int jj = 0; jj < N; jj += TILE)
                for (int i = ii; i < ii + TILE; i++)
                    for (int k = kk; k < kk + TILE; k++) {
                        float a = A[i * N + k];
                        for (int j = jj; j < jj + TILE; j++)
                            C[i * N + j] += a * B[k * N + j];
                    }
}

static void tiled_omp(const float *A, const float *B, float *C) {
    for (int i = 0; i < N * N; i++) C[i] = 0.0f;
    #pragma omp parallel for schedule(static)
    for (int ii = 0; ii < N; ii += TILE)
        for (int kk = 0; kk < N; kk += TILE)
            for (int jj = 0; jj < N; jj += TILE)
                for (int i = ii; i < ii + TILE; i++)
                    for (int k = kk; k < kk + TILE; k++) {
                        float a = A[i * N + k];
                        for (int j = jj; j < jj + TILE; j++)
                            C[i * N + j] += a * B[k * N + j];
                    }
}

static double maxdiff(const float *X, const float *Y) {
    double m = 0.0;
    for (int i = 0; i < N * N; i++) {
        double d = fabs((double)X[i] - (double)Y[i]);
        if (d > m) m = d;
    }
    return m;
}

static double gflops(double seconds) { return (2.0 * N * N * N) / seconds / 1e9; }

/* A memory-bound kernel: 2x2 max-pooling over a large image.
 * Reported in GB/s (not GFLOP/s) because it does almost no arithmetic -- it is
 * limited by how fast memory can be read/written (contrast with matmul). */
#define PN 4096
static void bench_maxpool(void) {
    int PO = PN / 2;
    float *X = malloc(sizeof(float) * PN * PN);
    float *Y = malloc(sizeof(float) * PO * PO);
    for (int i = 0; i < PN * PN; i++) X[i] = (float)rand() / RAND_MAX;
    double t = now();
    for (int i = 0; i < PO; i++)
        for (int j = 0; j < PO; j++) {
            float a = X[(2 * i) * PN + 2 * j],     b = X[(2 * i) * PN + 2 * j + 1];
            float c = X[(2 * i + 1) * PN + 2 * j], d = X[(2 * i + 1) * PN + 2 * j + 1];
            float m = a; if (b > m) m = b; if (c > m) m = c; if (d > m) m = d;
            Y[i * PO + j] = m;
        }
    t = now() - t;
    double bytes = (double)PN * PN * 4 + (double)PO * PO * 4;  /* read in + write out */
    printf("\nmaxpool 2x2 on %dx%d image: %.4f s, %.2f GB/s (memory-bound)\n",
           PN, PN, t, bytes / t / 1e9);
    free(X); free(Y);
}

/*
 * ---------------------------------------------------------------------------
 * STUDENT TODO: implement your own optimized matmul variant below.
 * Ideas: try a different TILE size, add '#pragma omp simd' on the inner loop,
 * or combine OpenMP + a better blocking. Then time it in main() like the others
 * and verify max_diff vs naive stays ~0. */
static void student_matmul(const float *A, const float *B, float *C) {
    (void)A; (void)B; (void)C;    /* your code here */
}
/* --------------------------------------------------------------------------- */

int main(void) {
    float *A = malloc(sizeof(float) * N * N);
    float *B = malloc(sizeof(float) * N * N);
    float *C = malloc(sizeof(float) * N * N);
    float *R = malloc(sizeof(float) * N * N);
    for (int i = 0; i < N * N; i++) { A[i] = (float)rand() / RAND_MAX; B[i] = (float)rand() / RAND_MAX; }

    printf("Matrix N=%d  TILE=%d  threads=%d\n", N, TILE, omp_get_max_threads());
    printf("%-18s%12s%12s%14s\n", "variant", "time(s)", "GFLOP/s", "max_diff");
    printf("--------------------------------------------------------------\n");

    double t = now(); naive(A, B, R); t = now() - t;
    printf("%-18s%12.4f%12.2f%14s\n", "naive", t, gflops(t), "(reference)");

    t = now(); tiled(A, B, C); t = now() - t;
    printf("%-18s%12.4f%12.2f%14.2e\n", "tiled", t, gflops(t), maxdiff(R, C));

    t = now(); tiled_omp(A, B, C); t = now() - t;
    printf("%-18s%12.4f%12.2f%14.2e\n", "tiled+openmp", t, gflops(t), maxdiff(R, C));

    printf("--------------------------------------------------------------\n");
    printf("max_diff vs naive should be ~1e-3 or smaller (float rounding) -> correct.\n");

    bench_maxpool();  /* a memory-bound kernel for contrast (reported in GB/s) */
    (void)student_matmul;  /* implement it above, then benchmark it here */

    free(A); free(B); free(C); free(R);
    return 0;
}
