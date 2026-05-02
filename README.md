# Tunable_MatMul_MatVec

A small CUDA SGEMM benchmark for `C[M,N] = A[M,K] * B[K,N]` (row-major, FP32).
Compares four implementations:

1. **Naive** — one thread per output element. No tiling, no shared memory.
   Verification reference for the tunable kernel.
2. **Tunable** — shared-memory + register-tiled kernel templated on
   `BDX, BDY, TX, TY, TK`. The `(BDX, BDY, TX, TY, TK)` tuple used at run
   time is selected per matrix size from the sweep results below.
3. **cuBLAS** — `cublasSgemm` (row-major via the standard col-major-swap trick).
4. **CUTLASS** — `cutlass::gemm::device::Gemm<float, RowMajor, ...>`,
   default device-level config.

## Layout

```
Tunable_MatMul_MatVec/
├── naive_kernel.cu        our naive kernel
├── tunable_kernel.cu      our tunable kernel + per-size dispatcher
├── bench.cu               main benchmark; cuBLAS / CUTLASS calls inline
├── sweep.cu               separate retuning experiment (own copy of the kernel)
├── gen_configs.py         emits tunable_configs.inc for the sweep
├── tunable_configs.inc    generated; only built when needed by sweep
├── Makefile               builds build/bench (default) and build/sweep
└── build/                 generated artifacts (binaries, .o files)
```

## Tunable kernel knobs

- `BDX, BDY` — threads per block in x / y
- `TX,  TY`  — output elements per thread (register tile)
- `TK`        — K-dimension tile loaded into shared memory per step

Block tile of C is `(BDY*TY) × (BDX*TX)`, and each thread accumulates a
`TY × TX` register tile. The thread→output mapping is interleaved
(`row = i*BDY + ty`, `col = j*BDX + tx`) so that consecutive threads write
consecutive columns of C — coalesced global writes, conflict-free shared
reads of B.

## Build / run

Built with `nvcc 12.8` for `sm_120` (Blackwell). cuBLAS comes from CUDA 12.8;
CUTLASS headers from `/home/rithik/cutlass`. Override on the command line
with `make CUDA_HOME=... CUTLASS_HOME=... ARCH=sm_XX`.

```
make            # build/bench
./build/bench   # naive vs tunable vs cuBLAS vs CUTLASS

make sweep      # build/sweep (compile-heavy: ~400 template instantiations)
./build/sweep   # full tunable sweep + sweep_<N>.csv per size

make clean      # rm -rf build, tunable_configs.inc, sweep_*.csv
```

## Results — NVIDIA RTX 5080 FE (sm_120)

`./build/bench`, FP32, M = N = K, time in ms / throughput in GFLOP/s.
Tunable kernel verifies bit-identical to naive at all three sizes.

| N=M=K | Naive ms | Naive GF/s | Tunable ms | Tun. GF/s | cuBLAS ms | cuBLAS GF/s | CUTLASS ms | CUT. GF/s |
|------:|---------:|-----------:|-----------:|----------:|----------:|------------:|-----------:|----------:|
| 1024  |    0.666 |     3225.1 |      0.181 |   11862.6 |     0.086 |     24860.7 |      0.078 |   27565.8 |
| 2048  |    5.176 |     3319.3 |      1.019 |   16859.2 |     0.499 |     34443.1 |      0.595 |   28889.0 |
| 4096  |   41.425 |     3317.7 |      7.005 |   19619.4 |     3.690 |     37245.1 |      3.744 |   36706.4 |

GFLOP/s only:

| N=M=K | Naive | Tunable | cuBLAS | CUTLASS |
|------:|------:|--------:|-------:|--------:|
| 1024  |  3225 |  11,863 | 24,861 |  27,566 |
| 2048  |  3319 |  16,859 | 34,443 |  28,889 |
| 4096  |  3318 |  19,619 | 37,245 |  36,706 |

Speedup vs naive:

| N=M=K | Tunable | cuBLAS | CUTLASS |
|------:|--------:|-------:|--------:|
| 1024  |   3.68× |  7.71× |   8.55× |
| 2048  |   5.08× | 10.38× |   8.70× |
| 4096  |   5.91× | 11.23× |  11.06× |

cuBLAS and CUTLASS reach ~25–37 TFLOP/s, well above pure FP32 SIMT peak —
they're hitting the Blackwell tensor-core path (default SGEMM on `sm_120`
uses TF32 accumulation). Our tunable kernel is plain FP32 SIMT and tops out
around 19.6 TFLOP/s at N=4096, ~53 % of cuBLAS.

## Per-size tuned configs (baked into `tunable_kernel.cu`)

Picked from the top of `./build/sweep` on this device:

| Size       | Config name        | BDX | BDY | TX | TY | TK | Block tile | TPB  |
|-----------:|:-------------------|----:|----:|---:|---:|---:|:-----------|-----:|
| N ≤ 1024   | `B32x32_T4x4_K08`  |  32 |  32 |  4 |  4 |  8 | 128 × 128  | 1024 |
| N ≤ 2048   | `B16x08_T8x8_K32`  |  16 |   8 |  8 |  8 | 32 |  64 × 128  |  128 |
| N >  2048  | `B08x16_T8x8_K16`  |   8 |  16 |  8 |  8 | 16 | 128 ×  64  |  128 |

## Sweep experiment

`./build/sweep` instantiates the tunable kernel for every `(BDX, BDY, TX, TY, TK)`
in the cartesian product emitted by `gen_configs.py`:

- `BDX, BDY ∈ {8, 16, 32}`
- `TX,  TY  ∈ {1, 2, 4, 8}`
- `TK ∈ {8, 16, 32}`

filtered by `32 ≤ TPB ≤ 1024` and `smem ≤ 48 KB` → **431 unique kernels**.
Each is verified against the naive reference, timed, and dumped to
`sweep_<N>.csv` for offline analysis. Top-15 per size and best-per-TPB-bucket
are also printed to stdout.

### Best-per-TPB at N = 4096 on the RTX 5080 FE

| TPB  | Best config        | Block    | GFLOP/s |
|-----:|:-------------------|:---------|--------:|
|   64 | `B08x08_T8x8_K16`  | 64 × 64  |  18,336 |
|  128 | `B08x16_T8x8_K16`  | 128 × 64 | **19,652** |
|  256 | `B16x16_T8x8_K32`  | 128 ×128 |  19,316 |
|  512 | `B32x16_T8x8_K32`  | 128 ×256 |  15,942 |
| 1024 | `B32x32_T4x8_K32`  | 256 ×128 |  12,862 |

### Patterns from the sweep

- The `8×8` register tile (`TX*TY = 64` outputs/thread) dominates the top
  of every per-size leaderboard — register tiling is the single biggest knob.
- Best block tile is around `128×64` / `64×128` / `128×128` — small enough
  that ~2 blocks fit per SM with the registers each thread is using.
- **128 threads/block** is the sweet spot at N ≥ 2048 (more independent
  blocks → better latency hiding).
- 1024 TPB loses badly: with `TX = TY = 8` the kernel asks for ~85
  registers/thread × 1024 threads ≈ 85 K registers per block, exceeding
  the SM's 64 K register file. Two configs (`B32x32_T8x8_K{8,16}`) fail
  to launch at N ≥ 2048; the sweep handles this by checking
  `cudaGetLastError()` immediately after each launch and poisoning the
  output buffer with `cudaMemset(0xFF)` so a failed launch can't pass
  verification against a previous config's stale result.
- `TK = 16–32` wins at larger sizes (better arithmetic intensity per
  shared-memory load); `TK = 8` wins at N = 1024 where the SM is starved
  for blocks.

### Re-tuning on a different GPU

```
make sweep && ./build/sweep
```

Read off the best config per size from the printed summary (or
`sweep_*.csv`) and update the dispatcher in `tunable_kernel.cu:99-104`.

## Notes on numerics

The tunable kernel verifies bit-identical to the naive reference
(`max_abs = 0`, `max_rel = 0` at every tested size). Both kernels happen
to perform the K-dimension accumulation in the same order
(`k = 0, 1, …, K-1`), so although fp32 addition is not associative the
exact sequence of rounded operations is identical between the two
implementations. cuBLAS and CUTLASS use split-K reductions internally and
will not match bit-for-bit; we don't compare them against naive directly.
