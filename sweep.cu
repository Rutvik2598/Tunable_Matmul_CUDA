// Tunable-kernel sweep experiment: instantiates the tunable GEMM kernel for
// every (BDX, BDY, TX, TY, TK) tuple emitted by gen_configs.py, verifies each
// one against the naive reference, times it, and reports the best.
// cuBLAS and CUTLASS are also timed for reference. Each size's full results
// are dumped to sweep_<N>.csv.
//
// This is a SEPARATE binary from bench. The day-to-day comparison should use
// build/bench. This binary is just for retuning.

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <map>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cutlass/gemm/device/gemm.h"

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

// ---- naive reference (defined in naive_kernel.cu) ----
extern void launch_naive(const float* A, const float* B, float* C,
                         int M, int N, int K);

// ---- tunable kernel template (file-local copy) ----
// Same body as in tunable_kernel.cu; duplicated here so this experiment can
// instantiate it for hundreds of configs without touching the bench TU.
template <int BDX_, int BDY_, int TX_, int TY_, int TK_>
__global__ static void sweep_tunable_gemm(const float* __restrict__ A,
                                          const float* __restrict__ B,
                                          float* __restrict__ C,
                                          int M, int N, int K) {
    constexpr int BLOCK_M = BDY_ * TY_;
    constexpr int BLOCK_N = BDX_ * TX_;

    __shared__ float As[BLOCK_M][TK_];
    __shared__ float Bs[TK_][BLOCK_N];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tid = ty * BDX_ + tx;
    constexpr int NTHREADS = BDX_ * BDY_;

    int block_row = blockIdx.y * BLOCK_M;
    int block_col = blockIdx.x * BLOCK_N;

    float acc[TY_][TX_];
    for (int i = 0; i < TY_; ++i)
        for (int j = 0; j < TX_; ++j)
            acc[i][j] = 0.0f;

    constexpr int A_TILE = BLOCK_M * TK_;
    constexpr int B_TILE = TK_ * BLOCK_N;

    for (int k_tile = 0; k_tile < K; k_tile += TK_) {
        for (int i = tid; i < A_TILE; i += NTHREADS) {
            int r = i / TK_;
            int c = i % TK_;
            int gr = block_row + r;
            int gc = k_tile + c;
            As[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (int i = tid; i < B_TILE; i += NTHREADS) {
            int r = i / BLOCK_N;
            int c = i % BLOCK_N;
            int gr = k_tile + r;
            int gc = block_col + c;
            Bs[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();

        for (int kk = 0; kk < TK_; ++kk) {
            float a_reg[TY_];
            float b_reg[TX_];
            for (int i = 0; i < TY_; ++i) a_reg[i] = As[i * BDY_ + ty][kk];
            for (int j = 0; j < TX_; ++j) b_reg[j] = Bs[kk][j * BDX_ + tx];
            for (int i = 0; i < TY_; ++i)
                for (int j = 0; j < TX_; ++j)
                    acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads();
    }

    for (int i = 0; i < TY_; ++i) {
        int gr = block_row + i * BDY_ + ty;
        if (gr >= M) continue;
        for (int j = 0; j < TX_; ++j) {
            int gc = block_col + j * BDX_ + tx;
            if (gc < N) C[gr * N + gc] = acc[i][j];
        }
    }
}

template <int Bx, int By, int Tx, int Ty, int Tk>
static void launch_sweep_config(const float* A, const float* B, float* C,
                                int M, int N, int K) {
    constexpr int BLOCK_M = By * Ty;
    constexpr int BLOCK_N = Bx * Tx;
    dim3 block(Bx, By);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    sweep_tunable_gemm<Bx, By, Tx, Ty, Tk><<<grid, block>>>(A, B, C, M, N, K);
}

// ---- config registry (generated from tunable_configs.inc) ----
struct TunableConfig {
    const char* name;
    int bdx, bdy, tx, ty, tk;
    int block_m, block_n, threads, smem_bytes;
    void (*launch)(const float*, const float*, float*, int, int, int);
};

#define CFG(NAME, BX, BY, TX, TY, TK)                              \
    { NAME, BX, BY, TX, TY, TK,                                    \
      (BY) * (TY), (BX) * (TX), (BX) * (BY),                       \
      (int)(((BY)*(TY)*(TK) + (TK)*(BX)*(TX)) * sizeof(float)),    \
      &launch_sweep_config<BX, BY, TX, TY, TK> }

static const TunableConfig kConfigs[] = {
#include "tunable_configs.inc"
};

#undef CFG

static const int kNumConfigs = sizeof(kConfigs) / sizeof(kConfigs[0]);

// ---- cuBLAS / CUTLASS reference launchers ----
static void launch_cublas(cublasHandle_t h, const float* A, const float* B,
                          float* C, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N,
                             N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

using CutlassGemm = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor,
    float, cutlass::layout::RowMajor>;

static void launch_cutlass(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    CutlassGemm op;
    CutlassGemm::Arguments args({M,N,K},{A,K},{B,N},{C,N},{C,N},{1.f, 0.f});
    if (op(args) != cutlass::Status::kSuccess) std::abort();
}

// ---- timing / verification ----
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
                   const std::vector<float>& ref, int K,
                   double& max_abs, double& max_rel) {
    max_abs = 0.0; max_rel = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        double d = std::fabs((double)got[i] - (double)ref[i]);
        double r = d / (std::fabs((double)ref[i]) + 1e-12);
        if (d > max_abs) max_abs = d;
        if (r > max_rel) max_rel = r;
    }
    return max_abs <= 1e-3 * (double)K || max_rel <= 1e-3;
}

struct TunResult {
    int cfg_idx;
    bool launched, ok;
    double max_abs, max_rel;
    float t_ms;
    double gflops;
};

struct SizeReport {
    int n;
    std::vector<TunResult> tun;
    float t_naive, t_cublas, t_cutlass;
    double g_naive, g_cublas, g_cutlass;
};

static SizeReport run_size(cublasHandle_t handle, int N) {
    int M = N, K = N;
    size_t bytes = (size_t)M * N * sizeof(float);

    std::vector<float> hA((size_t)M * K), hB((size_t)K * N);
    for (auto& v : hA) v = (float)((std::rand() % 201 - 100) / 1000.0);
    for (auto& v : hB) v = (float)((std::rand() % 201 - 100) / 1000.0);

    float *dA, *dB, *dC_ref, *dC_tun, *dC_cublas, *dC_cutlass;
    CUDA_CHECK(cudaMalloc(&dA, (size_t)M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, (size_t)K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC_ref,     bytes));
    CUDA_CHECK(cudaMalloc(&dC_tun,     bytes));
    CUDA_CHECK(cudaMalloc(&dC_cublas,  bytes));
    CUDA_CHECK(cudaMalloc(&dC_cutlass, bytes));
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    launch_naive(dA, dB, dC_ref, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hRef(bytes / sizeof(float));
    CUDA_CHECK(cudaMemcpy(hRef.data(), dC_ref, bytes, cudaMemcpyDeviceToHost));

    SizeReport rep;
    rep.n = N;
    int iters  = (N <= 1024) ? 10 : (N <= 2048) ? 6 : 3;
    int warmup = 1;

    std::vector<float> hTun(hRef.size());
    rep.tun.resize(kNumConfigs);

    int n_launch_fail = 0;
    for (int i = 0; i < kNumConfigs; ++i) {
        const auto& cfg = kConfigs[i];
        TunResult& tr = rep.tun[i];
        tr.cfg_idx = i;

        (void)cudaGetLastError();
        CUDA_CHECK(cudaMemset(dC_tun, 0xFF, bytes));   // poison so failed launch can't pass verify

        cfg.launch(dA, dB, dC_tun, M, N, K);
        cudaError_t err = cudaGetLastError();
        if (err == cudaSuccess) err = cudaDeviceSynchronize();

        if (err != cudaSuccess) {
            tr.launched = false; tr.ok = false;
            tr.max_abs = tr.max_rel = NAN;
            tr.t_ms = NAN; tr.gflops = 0.0;
            (void)cudaGetLastError();
            ++n_launch_fail;
            continue;
        }
        tr.launched = true;
        CUDA_CHECK(cudaMemcpy(hTun.data(), dC_tun, bytes, cudaMemcpyDeviceToHost));
        tr.ok = verify(hTun, hRef, K, tr.max_abs, tr.max_rel);

        tr.t_ms   = time_ms([&]{ cfg.launch(dA, dB, dC_tun, M, N, K); }, warmup, iters);
        tr.gflops = gflops(M, N, K, tr.t_ms);
    }
    (void)cudaGetLastError();
    std::printf("  (sweep done; %d/%d configs failed to launch)\n", n_launch_fail, kNumConfigs);

    int iters_naive = (N <= 1024) ? 5 : (N <= 2048) ? 3 : 2;
    rep.t_naive   = time_ms([&]{ launch_naive(dA,dB,dC_ref,M,N,K); }, warmup, iters_naive);
    rep.g_naive   = gflops(M, N, K, rep.t_naive);
    rep.t_cublas  = time_ms([&]{ launch_cublas(handle,dA,dB,dC_cublas,M,N,K); }, warmup, iters);
    rep.g_cublas  = gflops(M, N, K, rep.t_cublas);
    rep.t_cutlass = time_ms([&]{ launch_cutlass(dA,dB,dC_cutlass,M,N,K); }, warmup, iters);
    rep.g_cutlass = gflops(M, N, K, rep.t_cutlass);

    cudaFree(dA); cudaFree(dB);
    cudaFree(dC_ref); cudaFree(dC_tun); cudaFree(dC_cublas); cudaFree(dC_cutlass);
    return rep;
}

// ---- output helpers ----
static int best_cfg_idx(const SizeReport& r) {
    int best = -1; double best_gf = -1.0;
    for (auto& t : r.tun) {
        if (!t.launched || !t.ok) continue;
        if (t.gflops > best_gf) { best_gf = t.gflops; best = t.cfg_idx; }
    }
    return best;
}

static void print_topN(const SizeReport& r, int N_top) {
    std::vector<int> order(r.tun.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        const auto& ra = r.tun[a]; const auto& rb = r.tun[b];
        if (!ra.launched || !ra.ok) return false;
        if (!rb.launched || !rb.ok) return true;
        return ra.gflops > rb.gflops;
    });

    std::printf("\n=== Top %d tunable configs at M=N=K=%d ===\n", N_top, r.n);
    std::printf("+----+---------------------+-----+--------+----------+------------+------------+--------+\n");
    std::printf("| #  | Config              | TPB | smem   | Block MxN| Time (ms)  |  GFLOP/s   | Verify |\n");
    std::printf("+----+---------------------+-----+--------+----------+------------+------------+--------+\n");
    int rank = 0;
    for (int idx : order) {
        const auto& t = r.tun[idx];
        const auto& c = kConfigs[idx];
        if (++rank > N_top) break;
        char tile[16];
        std::snprintf(tile, sizeof(tile), "%dx%d", c.block_m, c.block_n);
        if (!t.launched) {
            std::printf("| %2d | %-19s | %3d | %5dB | %-8s | %10s | %10s | LAUNCH |\n",
                        rank, c.name, c.threads, c.smem_bytes, tile, "n/a", "n/a");
        } else {
            std::printf("| %2d | %-19s | %3d | %5dB | %-8s | %10.3f | %10.1f | %-6s |\n",
                        rank, c.name, c.threads, c.smem_bytes, tile,
                        t.t_ms, t.gflops, t.ok ? "OK" : "FAIL");
        }
    }
    std::printf("+----+---------------------+-----+--------+----------+------------+------------+--------+\n");
    std::printf("Reference: naive=%.3f ms (%.1f GF/s)  cuBLAS=%.3f ms (%.1f GF/s)  CUTLASS=%.3f ms (%.1f GF/s)\n",
                r.t_naive, r.g_naive, r.t_cublas, r.g_cublas, r.t_cutlass, r.g_cutlass);
}

static void print_per_tpb_best(const SizeReport& r) {
    std::map<int, int> best_at_tpb;
    for (auto& t : r.tun) {
        if (!t.launched || !t.ok) continue;
        const auto& c = kConfigs[t.cfg_idx];
        auto it = best_at_tpb.find(c.threads);
        if (it == best_at_tpb.end()) best_at_tpb[c.threads] = t.cfg_idx;
        else if (t.gflops > r.tun[it->second].gflops) it->second = t.cfg_idx;
    }
    std::printf("\n=== Best tunable per threads-per-block bucket (M=N=K=%d) ===\n", r.n);
    std::printf("+------+---------------------+----------+------------+------------+\n");
    std::printf("| TPB  | Best config         | Block    | Time (ms)  |  GFLOP/s   |\n");
    std::printf("+------+---------------------+----------+------------+------------+\n");
    for (auto& kv : best_at_tpb) {
        const auto& t = r.tun[kv.second];
        const auto& c = kConfigs[kv.second];
        char tile[16];
        std::snprintf(tile, sizeof(tile), "%dx%d", c.block_m, c.block_n);
        std::printf("| %4d | %-19s | %-8s | %10.3f | %10.1f |\n",
                    kv.first, c.name, tile, t.t_ms, t.gflops);
    }
    std::printf("+------+---------------------+----------+------------+------------+\n");
}

static void print_cross_size_summary(const std::vector<SizeReport>& reps) {
    std::printf("\n=== Best tunable per size vs cuBLAS / CUTLASS ===\n");
    std::printf("+-------+---------------------+------------+------------+------------+------------+------------+------------+\n");
    std::printf("| N=M=K | Best tunable cfg    | Best ms    | Best GF/s  | cuBLAS ms  | cuBLAS GF/s| CUTLASS ms | CUT.  GF/s |\n");
    std::printf("+-------+---------------------+------------+------------+------------+------------+------------+------------+\n");
    for (auto& r : reps) {
        int bi = best_cfg_idx(r);
        const char* name = "(none)";
        float bms = NAN; double bgf = 0;
        if (bi >= 0) {
            const auto& t = r.tun[bi];
            name = kConfigs[bi].name;
            bms = t.t_ms; bgf = t.gflops;
        }
        std::printf("| %5d | %-19s | %10.3f | %10.1f | %10.3f | %10.1f | %10.3f | %10.1f |\n",
                    r.n, name, bms, bgf,
                    r.t_cublas, r.g_cublas, r.t_cutlass, r.g_cutlass);
    }
    std::printf("+-------+---------------------+------------+------------+------------+------------+------------+------------+\n");
}

static void dump_csv(const SizeReport& r) {
    char path[64];
    std::snprintf(path, sizeof(path), "sweep_%d.csv", r.n);
    FILE* f = std::fopen(path, "w");
    if (!f) { std::perror(path); return; }
    std::fprintf(f, "config,bdx,bdy,tx,ty,tk,block_m,block_n,threads,smem_bytes,launched,verify_ok,time_ms,gflops\n");
    for (auto& t : r.tun) {
        const auto& c = kConfigs[t.cfg_idx];
        std::fprintf(f, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%g,%g\n",
            c.name, c.bdx, c.bdy, c.tx, c.ty, c.tk,
            c.block_m, c.block_n, c.threads, c.smem_bytes,
            t.launched ? 1 : 0, t.ok ? 1 : 0,
            t.launched ? (double)t.t_ms : 0.0,
            t.launched ? t.gflops : 0.0);
    }
    std::fclose(f);
    std::printf("Wrote %s (%d rows)\n", path, (int)r.tun.size());
}

int main() {
    std::srand(0xC0FFEE);
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("Device: %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    std::printf("Sweeping %d tunable configurations.\n", kNumConfigs);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    std::vector<int> sizes = {1024, 2048, 4096};
    std::vector<SizeReport> reps;
    reps.reserve(sizes.size());
    for (int n : sizes) {
        std::printf("\nRunning M=N=K=%d ...\n", n);
        reps.push_back(run_size(handle, n));
        print_topN(reps.back(), 15);
        print_per_tpb_best(reps.back());
        dump_csv(reps.back());
    }
    CUBLAS_CHECK(cublasDestroy(handle));

    print_cross_size_summary(reps);
    return 0;
}
