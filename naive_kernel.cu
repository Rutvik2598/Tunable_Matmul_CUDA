// Naive GEMM: C[M,N] = A[M,K] * B[K,N], all row-major, alpha=1 beta=0.
// One thread per output element. No tiling, no shared memory.
//
// Exposed as:  void launch_naive(const float*, const float*, float*, int, int, int);

__global__ static void naive_gemm(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}

void launch_naive(const float* A, const float* B, float* C,
                  int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    naive_gemm<<<grid, block>>>(A, B, C, M, N, K);
}
