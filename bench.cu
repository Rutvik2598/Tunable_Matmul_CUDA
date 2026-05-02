// Benchmark: naive vs tunable vs cuBLAS vs CUTLASS at M=N=K in {1024, 2048, 4096}.
// Verifies tunable against naive, times each method, prints a table.
//
// Our two kernel implementations live in:
//     naive_kernel.cu      -- launch_naive
//     tunable_kernel.cu    -- launch_tunable + tunable_config_name (per-size knobs)
// cuBLAS and CUTLASS are called directly below.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cutlass/gemm/device/gemm.h"

// ---- error helpers ----
#define CUDA_CHECK(x) do {                                                   \
    cudaError_t _e = (x);                                                    \
    if (_e != cudaSuccess) {                                                 \
        std::fprintf(stderr, "CUDA error %s at %s:%d\n",                     \
                     cudaGetErrorString(_e), __FILE__, __LINE__);            \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

#define CUBLAS_CHECK(x) do {                                                 \
    cublasStatus_t _s = (x);                                                 \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                       \
        std::fprintf(stderr, "cuBLAS error %d at %s:%d\n",                   \
                     (int)_s, __FILE__, __LINE__);                           \
        std::exit(1);                                                        \
    }                                                                        \
} while (0)

// ---- our kernels (defined in their own TUs) ----
extern void launch_naive  (const float* A, const float* B, float* C,
                           int M, int N, int K);
extern void launch_tunable(const float* A, const float* B, float* C,
                           int M, int N, int K);
extern const char* tunable_config_name(int N);

// ---- cuBLAS row-major SGEMM ----
// cuBLAS is column-major. To compute row-major C = A*B we use
//     C^T (col-major) = B^T (col-major) * A^T (col-major)
// which means we can pass row-major buffers directly with the operand order swapped.
static void launch_cublas(cublasHandle_t h,
                          const float* A, const float* B, float* C,
                          int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K,
                             &alpha,
                             B, N,
                             A, K,
                             &beta,
                             C, N));
}

// ---- CUTLASS row-major SGEMM (default device::Gemm config) ----
using CutlassGemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor>;

static void launch_cutlass(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    CutlassGemm gemm_op;
    CutlassGemm::Arguments args(
        {M, N, K},
        {A, K},
        {B, N},
        {C, N},
        {C, N},
        {1.0f, 0.0f});
    cutlass::Status st = gemm_op(args);
    if (st != cutlass::Status::kSuccess) {
        std::fprintf(stderr, "CUTLASS gemm failed: %d\n", (int)st);
        std::exit(1);
    }
}

// ---- timing ----
template <typename Fn>
static float time_ms(Fn&& fn, int warmup, int iters) {
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(s));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(e));
    CUDA_CHECK(cudaEventSynchronize(e));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, s, e));
    CUDA_CHECK(cudaEventDestroy(s));
    CUDA_CHECK(cudaEventDestroy(e));
    return ms / iters;
}

static double gflops(int M, int N, int K, float ms) {
    double ops = 2.0 * (double)M * (double)N * (double)K;
    return ops / (ms * 1.0e-3) / 1.0e9;
}

static bool verify(const std::vector<float>& got,
                   const std::vector<float>& ref,
                   int K,
                   double& max_abs, double& max_rel) {
    max_abs = 0.0;
    max_rel = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        double d = std::fabs((double)got[i] - (double)ref[i]);
        double r = d / (std::fabs((double)ref[i]) + 1e-12);
        if (d > max_abs) max_abs = d;
        if (r > max_rel) max_rel = r;
    }
    double atol = 1e-3 * (double)K;
    double rtol = 1e-3;
    return max_abs <= atol || max_rel <= rtol;
}

struct Row {
    int n;
    const char* tun_cfg;
    float t_naive, t_tun, t_cublas, t_cutlass;
    double g_naive, g_tun, g_cublas, g_cutlass;
    bool tun_ok;
    double tun_max_abs, tun_max_rel;
};

static Row run_size(cublasHandle_t handle, int N) {
    int M = N, K = N;
    size_t bytes = (size_t)M * N * sizeof(float);

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    for (auto& v : hA) v = (float)((std::rand() % 201 - 100) / 1000.0);
    for (auto& v : hB) v = (float)((std::rand() % 201 - 100) / 1000.0);

    float *dA, *dB, *dC_naive, *dC_tun, *dC_cublas, *dC_cutlass;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC_naive,   bytes));
    CUDA_CHECK(cudaMalloc(&dC_tun,     bytes));
    CUDA_CHECK(cudaMalloc(&dC_cublas,  bytes));
    CUDA_CHECK(cudaMalloc(&dC_cutlass, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    launch_naive  (dA, dB, dC_naive,   M, N, K);
    launch_tunable(dA, dB, dC_tun,     M, N, K);
    launch_cublas (handle, dA, dB, dC_cublas, M, N, K);
    launch_cutlass(dA, dB, dC_cutlass, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> hRef(bytes / sizeof(float));
    std::vector<float> hTun(hRef.size());
    CUDA_CHECK(cudaMemcpy(hRef.data(), dC_naive, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hTun.data(), dC_tun,   bytes, cudaMemcpyDeviceToHost));

    Row r{};
    r.n = N;
    r.tun_cfg = tunable_config_name(N);
    r.tun_ok = verify(hTun, hRef, K, r.tun_max_abs, r.tun_max_rel);

    int iters_naive = (N <= 1024) ? 5 : (N <= 2048) ? 3 : 2;
    int iters_fast  = 10;
    int warmup      = 2;

    r.t_naive   = time_ms([&]{ launch_naive  (dA, dB, dC_naive,   M, N, K); },
                          warmup, iters_naive);
    r.t_tun     = time_ms([&]{ launch_tunable(dA, dB, dC_tun,     M, N, K); },
                          warmup, iters_fast);
    r.t_cublas  = time_ms([&]{ launch_cublas (handle, dA, dB, dC_cublas, M, N, K); },
                          warmup, iters_fast);
    r.t_cutlass = time_ms([&]{ launch_cutlass(dA, dB, dC_cutlass, M, N, K); },
                          warmup, iters_fast);

    r.g_naive   = gflops(M, N, K, r.t_naive);
    r.g_tun     = gflops(M, N, K, r.t_tun);
    r.g_cublas  = gflops(M, N, K, r.t_cublas);
    r.g_cutlass = gflops(M, N, K, r.t_cutlass);

    cudaFree(dA); cudaFree(dB);
    cudaFree(dC_naive); cudaFree(dC_tun);
    cudaFree(dC_cublas); cudaFree(dC_cutlass);
    return r;
}

int main() {
    std::srand(0xC0FFEE);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s  (sm_%d%d)\n\n", prop.name, prop.major, prop.minor);
    std::printf("Tunable kernel: per-size config selected from sweep results.\n\n");

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::vector<int> sizes = {1024, 2048, 4096};
    std::vector<Row> rows;
    for (int n : sizes) {
        std::printf("Running M=N=K=%d  (tunable cfg: %s) ...\n",
                    n, tunable_config_name(n));
        rows.push_back(run_size(handle, n));
    }
    CUBLAS_CHECK(cublasDestroy(handle));

    std::printf("\nVerification (tunable vs naive):\n");
    for (auto& r : rows) {
        std::printf("  N=%-5d  cfg=%s  %s  (max_abs=%.3e, max_rel=%.3e)\n",
                    r.n, r.tun_cfg, r.tun_ok ? "OK" : "FAIL",
                    r.tun_max_abs, r.tun_max_rel);
    }

    std::printf("\nTime per call (ms) and effective throughput (GFLOP/s):\n");
    std::printf("+-------+------------+------------+------------+------------+------------+------------+------------+------------+\n");
    std::printf("| N=M=K | Naive ms   | Naive GF/s | Tunable ms | Tun.  GF/s | cuBLAS ms  | cuBLAS GF/s| CUTLASS ms | CUT.  GF/s |\n");
    std::printf("+-------+------------+------------+------------+------------+------------+------------+------------+------------+\n");
    for (auto& r : rows) {
        std::printf("| %5d | %10.3f | %10.1f | %10.3f | %10.1f | %10.3f | %10.1f | %10.3f | %10.1f |\n",
                    r.n,
                    r.t_naive,   r.g_naive,
                    r.t_tun,     r.g_tun,
                    r.t_cublas,  r.g_cublas,
                    r.t_cutlass, r.g_cutlass);
    }
    std::printf("+-------+------------+------------+------------+------------+------------+------------+------------+------------+\n");

    std::printf("\nSpeedup vs naive (higher = better):\n");
    std::printf("+-------+----------+----------+----------+\n");
    std::printf("| N=M=K | Tunable  | cuBLAS   | CUTLASS  |\n");
    std::printf("+-------+----------+----------+----------+\n");
    for (auto& r : rows) {
        std::printf("| %5d | %7.2fx | %7.2fx | %7.2fx |\n",
                    r.n,
                    r.t_naive / r.t_tun,
                    r.t_naive / r.t_cublas,
                    r.t_naive / r.t_cutlass);
    }
    std::printf("+-------+----------+----------+----------+\n");
    return 0;
}
