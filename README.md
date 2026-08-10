# `leet_gpu_half_mma` — warp-specialized FP16 GEMM for H100

    C = alpha * (A[M,K] @ B[K,N]) + beta * C     all row-major, half in / half out, fp32 accumulate

Single-file kernel (`mma_half.cu`) targeting sm_90a: TMA + `wgmma` + warp specialization +
2-CTA clusters + persistent grid + TMA-store epilogue, with a per-shape tile dispatcher.

**Parity with cuBLAS 12.9 and a per-shape-tuned CUTLASS 4.7 across most of the range.**
16 of 31 shapes at ≥1.00× cuBLAS, 16 of 27 at ≥ CUTLASS, of which **14 are statistical
ties** the table marks as such. Exactly one shape is clearly ahead — `1024³` at 1.08× — and
thin-K is a clear weakness (`2k×2k×512` 0.86×, `4k×8k×512` 0.89×). Measured with **random
operand data, L2 flushed before every timed launch, interleaved timing, one shape per
process**; see Methodology, because those four choices each moved the numbers more than any
optimization in this repo. Full table below.

| document | |
|---|---|
| **[`OPTIMIZATION_REPORT.md`](OPTIMIZATION_REPORT.md)** | how it was built: 12 optimization steps, each with the code, a diagram, and the measurement — plus the CUTLASS cross-check (§12), the root-cause analysis of the shapes we lose (§14), and what was tried and rejected |
| **[`SKILL.md`](SKILL.md)** | the transferable method, as a reusable skill — installable at `.claude/skills/gemm-hopper-optimization/` |

---

## Performance

**Hardware:** H100 SXM5 80GB @ 1980 MHz · **Toolkit:** CUDA 12.9 · **Baseline:** cuBLAS 12.9
(`cublasGemmEx`, `CUBLAS_COMPUTE_32F`) · **Peak:** 989.4 TFLOPS FP16 dense tensor core

**Baselines:** cuBLAS 12.9 (`cublasGemmEx`, `CUBLAS_COMPUTE_32F`) and **CUTLASS 4.7**
(`mma_half_cutlass.cu` — same contract, built from `CollectiveBuilder`, tile/cluster/schedule
chosen by a 21-configuration sweep).

**Method:** `alpha=1, beta=0`; median of 3 repeats × 20 iterations after 5 warm-ups, then the
median of 3 whole runs. All three implementations timed back-to-back in the same process,
**serially** — concurrent runs contend for the GPU and produce meaningless numbers.

CUTLASS shows `n/a` where `N % 8 != 0` or `K % 8 != 0`: it validates TMA alignment against the
problem *extent*, so ragged shapes would need a zero-padded extent and a copy back. Our kernel
runs them in place because TMA zero-fills out-of-range elements natively.

Regenerate with `make perf` (`perf_all` for the cuBLAS table, `three_way` for this one).

| M × N × K | ours | CUTLASS | cuBLAS | ours/CUTLASS | ours/cuBLAS |
|---|---|---|---|---|---|
| 4k×4095×4k | **469** | n/a | 144 | — | **3.26×** |
| 4095×4095×4095 | **410** | n/a | 151 | — | **2.72×** |
| 2k×2047×2k | **270** | n/a | 135 | — | **2.00×** |
| 1k×1023×1k | **109** | n/a | 65 | — | **1.68×** |
| 8k×8k×128 | **292** | 301 | 282 | 0.97× | **1.04×** |
| 16k×8k×4k | **653** | 640 | 636 | **1.02×** | **1.03×** |
| 8k×8k×16k | **713** | 710 | 697 | tie | **1.02×** |
| 8k×8k×8k | **666** | 661 | 651 | tie | **1.02×** |
| 16k×4k×8k | **664** | 658 | 650 | tie | **1.02×** |
| 8k×8k×4k | **700** | 689 | 686 | tie | **1.02×** |
| 8k×8k×2k | **696** | 679 | 685 | **1.02×** | **1.02×** |
| 16k×8k×128 | **303** | 319 | 299 | 0.95× | **1.01×** |
| 4k×8k×128 | **263** | 260 | 260 | tie | **1.01×** |
| 16k×16k×16k | **666** | 668 | 659 | tie | **1.01×** |
| 32k×8k×2k | **702** | 676 | 698 | **1.04×** | **1.01×** |
| 1k×1k×1k | **221** | 205 | 221 | **1.08×** | **1.00×** |
| 4k×8k×8k | 715 | 710 | 718 | tie | 1.00× |
| 4k×4k×8k | 742 | 737 | 745 | tie | 1.00× |
| 8k×4k×8k | 683 | 678 | 686 | tie | 1.00× |
| 4k×4k×4k | 767 | 766 | 779 | tie | 0.98× |
| 8k×1k×8k | 779 | 787 | 795 | tie | 0.98× |
| 4k×512×4k | 505 | 536 | 522 | 0.94× | 0.97× |
| 4k×8k×512 | 559 | 625 | 578 | 0.89× | 0.97× |
| 4k×4k×1k | 624 | 621 | 646 | tie | 0.97× |
| 4k×8k×256 | 426 | 444 | 442 | 0.96× | 0.96× |
| 4k×8k×1k | 670 | 660 | 698 | tie | 0.96× |
| 8k×8k×1k | 696 | 673 | 725 | **1.03×** | 0.96× |
| 2k×2k×2k | 564 | 591 | 606 | 0.96× | 0.93× |
| 3000×1000×2000 | 402 | 414 | 459 | 0.97× | 0.88× |
| 384×2k×2k | 237 | 240 | 274 | tie | 0.87× |
| 2k×2k×512 | 281 | 327 | 328 | 0.86× | 0.86× |

**31 shapes — ours ≥ cuBLAS on 16; ours ≥ CUTLASS on 16 of 27 supported.**

### Methodology (read before the numbers)

**Operand data changes throughput by up to 19%.** Same kernel, same shape
(8192×4096×8192, CUTLASS 128×256×64 c2×1):

| operand data | TFLOPS |
|---|---|
| all zero (`memset 0`) | **906** |
| `0x11` constant | 809 |
| uniform, 200 levels | 833 |
| **uniform, full 16-bit entropy** | **761** |
| **normal(0, 0.05), NN-weight-like** | **759** |

Zero operands draw far less dynamic power in the tensor cores, so the GPU sustains higher
clocks. Benchmarking on `memset` patterns overstates everything. This harness fills both
operands with full-entropy uniform random data, which lands within 0.3% of NN-weight-like
normal data — so the absolute figures here are ~6% below a `0x11` benchmark and ~19% below
an all-zero one, and are the ones you would actually see.

**Timing is interleaved, not sequential.** On large shapes both kernels swing ±16% run to
run *together*; timing A then B lets whichever ran at high clocks win. At 8192×4096×8192
sequential timing reported 1.10× for a pair that measures **1.01×** interleaved. Every
large-shape "win" this repo previously claimed was that artifact.

**One shape per process.** Measuring 31 shapes back-to-back leaves the GPU throttled by the
later ones, and kernels degrade differently under throttle. `8192×8192×1024` read 1.08–1.15×
in a full single-process run — *three repeats all agreed* — and 1.01–1.05× in a fresh
process. That is systematic, not noise, so more repeats do not fix it; isolation does.
`make cutlass` drives one shape per process.

**L2 is flushed before every timed launch.** H100's L2 is 50 MB. Any shape whose working set
(A+B+C) fits inside it is otherwise measured entirely from cache, which flatters small shapes
and only small shapes:

| shape | working set | hot L2 | cold L2 |
|---|---|---|---|
| 1024³ | 6 MB | 1.13× | **1.08×** |
| 2048×2048×512 | 12 MB | 0.91× | **0.86×** |
| 384×2k×2k | 11 MB | 1.04× | **0.99×** |
| 4096³ | 96 MB | 1.02× | 1.01× |
| 8192³ | 384 MB | 1.00× | 1.00× |

Shapes above ~50 MB are unaffected either way. Cold is the conservative choice and models a
GEMM called once inside a larger pipeline rather than a tight loop over resident operands;
`nvbench` and the CUTLASS profiler both offer it. Absolute figures drop accordingly — 1024³
reads 221 TFLOPS cold against 323 hot.

**Ties are labelled.** Where the ours/CUTLASS ratio range across runs spans 1.00, or the
margin is under 2%, the table says `tie` rather than printing a number that implies
precision the measurement does not have. 14 of the 27 comparable rows are ties.

Rows within ~1% of each other are inside the run-to-run band on this machine: large shapes
drift ±50–100 TFLOPS between runs depending on clocks. Treat `0.99×`–`1.01×` as a tie and
only the ≥1.05× and ≤0.95× entries as decided. The medians above are over 3 serial runs.

Large shapes run **65–78% of the 989.4 TFLOPS** hardware peak on random data with L2
flushed — best single figure is 4096³ at 767 TFLOPS (77.5%). On `0x11` with a hot cache the
same code reads 864 (87.3%); that number is not wrong, it is measuring easier conditions.

**On the CUTLASS column.** CUTLASS's device API fixes the tile per instantiation — there is no
runtime tile heuristic (that is what its offline profiler is for) — so it gets the same two-rung
dispatch ours has, with configurations chosen by sweeping tile × cluster × kernel schedule.

**It gets the same tuning freedom our dispatcher has.** Our kernel picks from a three-rung
tile ladder (BN 256/128/64) with a per-shape GROUP_M policy. Giving CUTLASS two fixed tiles
was not a like-for-like test. It now gets seven configs — 128×256/128×128/128×64, clustered
and not, cooperative and pingpong — swept per shape together with the swizzle, best kept
(the `c<n>/sw<n>` column in `make cutlass`). That is what its offline profiler does.

Adding the middle rung and the pingpong schedule flipped **6 of the 10 shapes** we previously
claimed by >5%; `128×128×64` pingpong is the best CUTLASS config on nearly every thin-K
shape. Four >5% wins survive: `4k×8k×8k`, `1024³`, `8k×4k×8k`, `8k×8k×4k`.

**Its tile scheduler must be told to rasterize.** `max_swizzle_size` defaults to **1** —
no threadblock swizzle at all — which is not a fair baseline against a kernel that rasterizes
for L2. At 16384³ the default costs CUTLASS **23%** (691 → 849 TFLOPS): it alone accounted
for what previously read as a 1.22× win for us, which is really a **1.00× tie**. A fixed cap
overfits the other way (8 everywhere costs −38% on `384×2k×2k`), so the harness sweeps
`{1,2,4,8}` per shape in a cheap discovery pass and measures with the winner — the `sw=` column
in `make cutlass` output. This is what CUTLASS's own offline profiler does.

**It must be built with `-DNDEBUG`** (the Makefile does). Without it, CUTLASS's device-side
asserts block inlining, ptxas cannot keep the wgmma group open across the resulting call
boundary (warning **C7510**), and the SASS fences and *fully drains* around every single
`wgmma` instead of keeping one group in flight. That costs ~10% at 4096³ and ~2× at 1024³ —
large enough to invert several rows of this table. CUTLASS is **bit-exact against cuBLAS** on
every shape it runs.

**Reading the extremes.** The top four rows are cuBLAS *losing*, not us winning: for
`N % 8 != 0` it abandons its Hopper kernel for an sm_75 CUTLASS `align1` fallback (confirmed
by nsys). We stay on the Hopper path by restriding B into scratch — but at 500 TF we are still
42% below our own aligned throughput. The bottom rows are **not** simply "grid starvation", though every shape
below 0.95× does have ≤128 output tiles for 132 SMs. Section 14 of the report pins it down:
we carry a ~1.38 µs fixed per-CTA prologue against a ~0.59 µs per-tile advantage, so the
break-even is ~2.3 tiles per CTA. Below that the persistent design has nothing to amortise
its prologue against. That one constant predicts every shape CUTLASS wins.

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
make cutlass    # three-way vs CUTLASS and cuBLAS (needs CUTLASS=/path/to/cutlass)
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

15 sources in four groups. **`mma_half.cu` is the only one that ships** — everything else
exists to prove it works, measure it, or derive a constant inside it.

### Kernel
| file | |
|---|---|
| `mma_half.cu` | the whole kernel — device code, host dispatch, `solve()` |
| `mma_half_cutlass.cu` | the same contract built from CUTLASS 4.7 `CollectiveBuilder` primitives, as an independent cross-check. Aligned shapes only (see the table note). Tile/cluster/schedule chosen by a 21-configuration sweep |

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
| `three_way.cu` | ours vs CUTLASS vs cuBLAS across all 31 shapes, with a per-shape CUTLASS correctness check. `make cutlass` |

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
| `CFG_HOIST_DESC` | 1 | build the wgmma descriptor once per k-tile and reach each k16 issue with an immediate add, instead of rebuilding the 64-bit field per issue. Cuts the BN=256 wgmma region from 214 to 159 instructions; +1.6% at 1024³, +1.7% at 4k×8k×256, ~0 on large shapes. `0` restores the per-issue rebuild |
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
