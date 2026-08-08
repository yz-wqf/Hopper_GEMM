# `leet_gpu_half_mma` — warp-specialized FP16 GEMM for H100

    C = alpha * (A[M,K] @ B[K,N]) + beta * C     all row-major, half in / half out, fp32 accumulate

Single-file kernel (`mma_half.cu`) targeting sm_90a: TMA + `wgmma` + warp specialization +
2-CTA clusters + persistent grid + TMA-store epilogue, with a per-shape tile dispatcher.

**17 of 31 benchmarked shapes at ≥1.00× cuBLAS 12.9; 27 at ≥0.96×; 80–87% of the 989.4
TFLOPS hardware peak on large shapes.** Full table below.

| document | |
|---|---|
| **[`OPTIMIZATION_REPORT.md`](OPTIMIZATION_REPORT.md)** | how it was built: 11 steps, each with the code, a diagram, and the measurement — plus what was tried and rejected |
| [`SKILL.md`](../.claude/skills/gemm-hopper-optimization/SKILL.md) | the transferable method, as a reusable skill |

---

## Performance

**Hardware:** H100 SXM5 80GB @ 1980 MHz · **Toolkit:** CUDA 12.9 · **Baseline:** cuBLAS 12.9
(`cublasGemmEx`, `CUBLAS_COMPUTE_32F`) · **Peak:** 989.4 TFLOPS FP16 dense tensor core

**Method:** `alpha=1, beta=0`; median of 3 repeats × 20 iterations after 5 warm-ups; ours and
cuBLAS timed back-to-back in the same process, **serially** — concurrent runs contend for the
GPU and produce meaningless numbers.

**Columns:** `tile` = BM×BN chosen by the dispatcher · `GM` = GROUP_M chosen by the policy ·
`tiles` = resulting output tiles against 132 SMs · `ratio` = ours / cuBLAS.

Regenerate with `make perf` (`perf_all` produces this table).

| M × N × K | tile | GM | tiles | ours TF | cuBLAS TF | ratio | % peak |
|---|---|---|---|---|---|---|---|
| 4k×4095×4k | 128×256 | 8 | 512 | **508** | 148 | **3.43×** | 51% |
| 4095×4095×4095 | 128×256 | 8 | 512 | **440** | 157 | **2.81×** | 44% |
| 2k×2047×2k | 128×256 | 8 | 128 | **311** | 150 | **2.07×** | 31% |
| 1k×1023×1k | 128×64 | 2 | 128 | **136** | 71 | **1.90×** | 14% |
| 8k×8k×128 | 128×256 | 4 | 2048 | **304** | 287 | **1.06×** | 31% |
| 1k×1k×1k | 128×64 | 2 | 128 | **319** | 304 | **1.05×** | 32% |
| 8k×8k×4k | 128×256 | 8 | 2048 | **826** | 795 | **1.04×** | 84% |
| 4k×8k×128 | 128×256 | 4 | 1024 | **269** | 261 | **1.03×** | 27% |
| 16k×8k×128 | 128×256 | 4 | 4096 | **311** | 301 | **1.03×** | 31% |
| 8k×8k×8k | 128×256 | 8 | 2048 | **804** | 780 | **1.03×** | 81% |
| 16k×4k×8k | 128×256 | 8 | 2048 | **816** | 793 | **1.03×** | 82% |
| 16k×16k×16k | 128×256 | 8 | 8192 | **836** | 815 | **1.03×** | 84% |
| 16k×8k×4k | 128×256 | 8 | 4096 | **803** | 785 | **1.02×** | 81% |
| 8k×8k×16k | 128×256 | 8 | 2048 | **819** | 803 | **1.02×** | 83% |
| 8k×4k×8k | 128×256 | 8 | 1024 | **844** | 839 | **1.01×** | 85% |
| 8k×8k×2k | 128×256 | 8 | 2048 | **797** | 795 | **1.00×** | 81% |
| 32k×8k×2k | 128×256 | 8 | 8192 | **790** | 789 | **1.00×** | 80% |
| 4k×8k×8k | 128×256 | 8 | 1024 | 830 | 832 | 1.00× | 84% |
| 4k×4k×4k | 128×256 | 8 | 512 | 863 | 870 | 0.99× | 87% |
| 4k×4k×8k | 128×256 | 8 | 512 | 813 | 824 | 0.99× | 82% |
| 8k×1k×8k | 128×256 | 8 | 256 | 829 | 841 | 0.99× | 84% |
| 4k×8k×256 | 128×256 | 4 | 1024 | 426 | 432 | 0.99× | 43% |
| 384×2k×2k | 128×64 | 2 | 96 | 341 | 347 | 0.98× | 34% |
| 4k×8k×512 | 128×256 | 4 | 1024 | 564 | 575 | 0.98× | 57% |
| 4k×4k×1k | 128×256 | 8 | 512 | 630 | 646 | 0.98× | 64% |
| 4k×8k×1k | 128×256 | 8 | 1024 | 676 | 695 | 0.97× | 68% |
| 8k×8k×1k | 128×256 | 8 | 2048 | 670 | 690 | 0.97× | 68% |
| 3000×1000×2000 | 128×256 | 8 | 96 | 463 | 486 | 0.95× | 47% |
| 2k×2k×2k | 128×256 | 8 | 128 | 671 | 723 | 0.93× | 68% |
| 2k×2k×512 | 128×256 | 4 | 128 | 362 | 430 | 0.84× | 37% |
| 4k×512×4k | 128×128 | 16 | 128 | 543 | 707 | 0.77× | 55% |

**31 shapes — 17 at ≥1.00×, 27 at ≥0.96×.**

Large shapes run **80–87% of the 989.4 TFLOPS** hardware peak.

**Reading the extremes.** The top four rows are cuBLAS *losing*, not us winning: for
`N % 8 != 0` it abandons its Hopper kernel for an sm_75 CUTLASS `align1` fallback (confirmed
by nsys). We stay on the Hopper path by restriding B into scratch — but at 508 TF we are still
40% below our own aligned throughput. The bottom rows are grid starvation: every shape below
0.95× has ≤128 output tiles for 132 SMs.

**Reading the middle.** cuBLAS drifts ~12% run-to-run here even with medians, so the ≥1.00×
count moves ±2 between runs. The stable claim is *parity or better on large shapes*, not a
precise score.

---

## Quick start

```sh
make            # build everything
make check      # correctness  (~2 min)
make perf       # performance  (~5 min)
make san        # compute-sanitizer, all four tools (~20 min)
make probe      # re-derive the wgmma descriptor encodings
```

Requires CUDA 12.x and an H100 (or any sm_90a part). **`sm_90a`, not `sm_90`** — the plain
target rejects `wgmma`.

## The entry point

```cpp
extern "C" void solve(const half *A, const half *B, half *C,
                      int M, int N, int K, float alpha, float beta);
```

`h100_hgemm::launch(...)` is the same thing without the implicit `cudaDeviceSynchronize()`,
and takes a stream.

---

## Files

13 sources in four groups. **`mma_half.cu` is the only one that ships** — everything else
exists to prove it works, measure it, or derive a constant inside it.

### Kernel
| file | |
|---|---|
| `mma_half.cu` | the whole kernel — device code, host dispatch, `solve()` |

### Correctness
| file | what it checks |
|---|---|
| `check_all.cu` | **57 shapes** vs cuBLAS — every shape investigated during development: ragged tiles, degenerate (`1×1×8`, `4096×4096×8`), odd leading dimensions, both `beta` modes. The regression net. |
| `check_fp64.cu` | 15 shapes vs an **fp64 host reference**. The independent anchor — cuBLAS shares an accuracy class with us, so agreeing with it does not prove much on its own. Takes an iteration count in argv. |
| `check_large.cu` | The shapes the performance table actually reports (8192³, 16384³, 32768×8192×2048) plus odd-N. Exists because the suite once topped out at 4096×512×4096 while 16384³ numbers were being published. |
| `san_sweep.cu` | sanitizer target: 26 shapes × both `beta` modes. No timing, no cuBLAS — it just exercises code paths under `compute-sanitizer`. |
| `san_large.cu` | sanitizer target for large shapes, where coordinate arithmetic could overflow. |

Expected: `rel_l2` ≈ 2.07e-4 against fp64 — that is exactly the fp16 **output**-rounding
floor (2⁻¹¹ half-ulp), not error we introduce. Accumulation is fp32 and there is precisely
one narrowing, at the store. Against cuBLAS most shapes come out **bit-exact**.

### Performance
| file | |
|---|---|
| `perf_all.cu` + `perf_shapes.h` | the main sweep: 31 shapes vs cuBLAS, flags anything below parity. Source of the table above. |
| `perf_table.cu` | condensed headline table, more repeats per shape |
| `perf_cmp.cu` | one shape from argv: `./perf_cmp M N K [iters]` — the workhorse for spot checks |
| `perf_prof.cu` | one shape, ours only, no cuBLAS — for profiling under nsys without a second kernel in the trace |

### Introspection
| file | |
|---|---|
| `smem_budget.cu` | prints the 232,448 B dynamic smem limit and what the pipeline and epilogue staging consume |

This is how "STAGES=4 leaves 34.9 KB spare" was established, which decided that the TMA-store
epilogue got **chunked** rather than costing a pipeline stage — dropping 4→3 stages measured
−6.6%, more than the epilogue optimization it would have funded.

### Probes
The only three that **do not include the kernel**: standalone, link `-lcuda`, and sweep
candidate descriptor encodings against a CPU reference until one matches exactly. They exist
because a wrong `wgmma` descriptor produces *plausible but scrambled* output rather than a
crash — you cannot eyeball it.

| file | derives |
|---|---|
| `probe_kmajor.cu` | K-major operand: `LBO=1, SBO=64`, 128B swizzle, +32B per k-step |
| `probe_mnmajor.cu` | MN-major operand — whether one `wgmma` can read N-major B across multiple swizzle blocks: `LBO=512, SBO=64` |
| `probe_mnmajor_kadv.cu` | MN-major k-advance: `+2048 B` per k16 step |

`probe_mnmajor*` are the ones that deleted the B-transpose pass — worth 9–16% — after the
source had already been annotated claiming it was impossible.

`make probe` re-derives all three in seconds, which doubles as a live check that the
constants baked into `mma_half.cu` are still correct.

---

## Tuning knobs

Compile-time, via `-D`:

| macro | default | |
|---|---|---|
| `CFG_STAGES` | 4 | pipeline depth for the BN=256 tile (BN=128 uses 6) |
| `CFG_GROUP_M` | 8 | fallback L2 rasterization group height. The runtime policy overrides it for narrow tiles, `K≤512`, and `N≤512` — worth 2–8% on 6 shapes |
| `CFG_CLUSTER_M` | 2 | CTAs per cluster sharing a B tile; 1 disables multicast (−7.8%) |
| `CFG_CONSUMER_REGS` | 232 | consumer warpgroup register budget |
| `CFG_PRODUCER_REGS` | 32 | producer warpgroup register budget |
| `CFG_FORCE_BM`, `CFG_FORCE_BN` | *unset* | pin the tile shape, bypassing the dispatcher |
| `CFG_STAGES_OVERRIDE` | *unset* | pin the pipeline depth for the derived configs |

`CFG_FORCE_BM`/`CFG_FORCE_BN` exist for measuring the dispatcher: build each rung and compare.
The dispatcher picks the widest tile that still fills the machine —
`128×256 → 128×128 → 128×64` — with the threshold read off a measured sweep.

---

## Gotchas worth knowing before you touch this

1. **Run benchmarks serially.** Two on one GPU contend and produce garbage — the tell is
   cuBLAS reading ~265 TFLOPS at 8192³ instead of ~790.
2. **`sm_90a`, not `sm_90`.** `-arch=sm_90a` does not always propagate the `a`; use
   `-gencode arch=compute_90a,code=sm_90a`.
3. **cuBLAS drifts ~12% run-to-run** on this machine even with medians. Treat 0.96–1.04× as
   parity.
4. **`N % 8 != 0` costs ~40%** (508 vs 871 TF): B gets restrided into scratch and the
   epilogue drops off the TMA-store path. We are still 2–3.4× faster than cuBLAS there, but
   only because cuBLAS abandons its Hopper kernel for an sm_75 CUTLASS `align1` fallback —
   that is cuBLAS degrading, not us accelerating.
5. **The unit of parallelism is consumer *warpgroups*, not CTAs.** `BM=64` halves the
   warpgroups per CTA while doubling the CTAs, so it buys nothing; narrowing `BN` at `BM=128`
   is what adds capacity. Measured, and it is why the ladder holds `BM=128` throughout.
6. **Split-K does not help here.** Measured: the reduction over `S×M×N` costs more than it
   saves unless K is large relative to M·N. At 1024³ it lands at 0.80×.
7. **`4096×512×4096` (0.75×) and `2048×2048×512` (0.86×) are still open.** All five tile
   configurations were measured on both; the dispatcher already picks the best available, so
   these are not tile problems.
