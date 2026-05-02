// Tunable GEMM: C[M,N] = A[M,K] * B[K,N], all row-major, alpha=1 beta=0.
// Knobs (compile-time template params):
//   BDX, BDY  : threads per block in x / y
//   TX,  TY   : output elements per thread (register tile)
//   TK        : K-dimension tile loaded into shared memory per step
// Block tile of C is (BDY*TY) rows by (BDX*TX) cols.
//
// Exposed as:
//   void launch_tunable(const float*, const float*, float*, int M, int N, int K);
//   const char* tunable_config_name(int N);
//
// The (BDX, BDY, TX, TY, TK) tuple chosen for each problem size was picked
// by running build/sweep on this device; see the dispatcher below.

template <int BDX_, int BDY_, int TX_, int TY_, int TK_>
__global__ static void tunable_gemm(const float* __restrict__ A,
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
static void launch_config(const float* A, const float* B, float* C,
                          int M, int N, int K) {
    constexpr int BLOCK_M = By * Ty;
    constexpr int BLOCK_N = Bx * Tx;
    dim3 block(Bx, By);
    dim3 grid((N + BLOCK_N - 1) / BLOCK_N, (M + BLOCK_M - 1) / BLOCK_M);
    tunable_gemm<Bx, By, Tx, Ty, Tk><<<grid, block>>>(A, B, C, M, N, K);
}

// Per-size best knobs (top result of build/sweep on RTX 5080 / sm_120):
//   N <= 1024:  BDX=32 BDY=32 TX=4 TY=4 TK=8     (block tile 128x128, 1024 TPB)
//   N <= 2048:  BDX=16 BDY=8  TX=8 TY=8 TK=32    (block tile 64x128,  128 TPB)
//   N >  2048:  BDX=8  BDY=16 TX=8 TY=8 TK=16    (block tile 128x64,  128 TPB)
void launch_tunable(const float* A, const float* B, float* C,
                    int M, int N, int K) {
    if (N <= 1024)      launch_config<32, 32, 4, 4,  8>(A, B, C, M, N, K);
    else if (N <= 2048) launch_config<16,  8, 8, 8, 32>(A, B, C, M, N, K);
    else                launch_config< 8, 16, 8, 8, 16>(A, B, C, M, N, K);
}

const char* tunable_config_name(int N) {
    if (N <= 1024) return "B32x32_T4x4_K08";
    if (N <= 2048) return "B16x08_T8x8_K32";
    return "B08x16_T8x8_K16";
}
