# Optimization Report — `mma_half.cu`

FP16 GEMM for NVIDIA H100 (sm_90a):

    C = alpha * (A[M,K] @ B[K,N]) + beta * C     row-major, half in / half out, fp32 accumulate

All numbers measured on one machine: H100 SXM5 80GB @ 1980 MHz, CUDA 12.9, vs cuBLAS 12.9.
Hardware peak: **989.4 TFLOPS** FP16 dense tensor core.

> **Reading the numbers.** Measurements were taken at different points in the evolution, so a
> "before" from step 3 is not comparable to an "after" from step 8 — the baseline moved
> underneath. Within each step, before/after were measured back-to-back on the same build.
> cuBLAS drifts ~12% run-to-run here, so any ratio within a few percent of 1.0 is parity.
>
> **Benchmarks must run serially.** Two on one GPU contend and produce meaningless numbers;
> the tell is cuBLAS reading ~265 TFLOPS at 8192³ instead of ~790.

---

## Summary of the evolution

| # | Optimization | Headline effect |
|---|---|---|
| 0 | Base design: warp specialization + TMA + wgmma | 744 TF starting point |
| 1 | Empirical descriptor validation (probe) | correctness, first try |
| 2 | L2 rasterization (`GROUP_M`) | live knob; see step 11 |
| 3 | Cluster TMA multicast | **+5.6% … +7.8%** |
| 4 | Pipeline depth `STAGES=4` | **+71%** vs 2 stages |
| 5 | Eliminate the B transpose (N-major operand) | **+8.7% … +16%** |
| 6 | Stride pad-copy | **86×** on odd N (6 → 517 TF) |
| 7 | Persistent kernel | **+14%** at small K |
| 8 | Epilogue v2: shuffle-combined stores | **+21%** at small K |
| 9 | Epilogue v3: TMA store | **+6.7%** at small K |
| 10 | Tile dispatch (template on `BM`,`BN`) | **1024³: 0.69× → 1.04×** |
| 11 | Per-shape `GROUP_M` policy | **+2…8%** on 6 shapes |
| 12 | Cross-check vs tuned CUTLASS 4.7 | **1.04–1.33×** ahead on all 27 supported shapes |
| — | **Final** | **17/31 shapes ≥1.00× cuBLAS**, 27 at ≥0.96×, 80–87% of peak |

---

## 0. Base design — warp specialization + TMA + wgmma

Three warpgroups per CTA: WG0 is a pure **producer** issuing TMA bulk-tensor copies, WG1/WG2
are **consumers** running `wgmma`. The register file is repartitioned at runtime so consumers
get what they need and the producer gives up what it doesn't.

```cpp
if (wg == 0) {                       // producer
  setmaxnreg_dec<32>();
  if (lane_in_wg == 0) { /* issue TMA for each k-tile */ }
} else {                             // consumers
  setmaxnreg_inc<232>();
  /* wgmma.m64n256k16.f32.f16.f16 */
}
```

```
                   CTA = 384 threads = 3 warpgroups
   ┌────────────────┬─────────────────┬─────────────────┐
   │ WG0  PRODUCER  │ WG1  CONSUMER   │ WG2  CONSUMER   │
   │ 32 regs        │ 232 regs        │ 232 regs        │
   │ issues TMA     │ runs wgmma      │ runs wgmma      │
   │ (no math)      │ rows 0..63      │ rows 64..127    │
   └───────┬────────┴────────┬────────┴────────┬────────┘
           │ writes          │ reads           │ reads
           ▼                 ▼                 ▼
   ┌──────────────────────────────────────────────────────┐
   │  shared memory: STAGES-deep ring buffer              │
   │   ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐             │
   │   │  s0  │  │  s1  │  │  s2  │  │  s3  │  48 KB each │
   │   └──────┘  └──────┘  └──────┘  └──────┘             │
   │   A tile 128×64 half + B tile 256×64 half            │
   └──────────────────────────────────────────────────────┘
      mbarriers: full[s] (producer→consumers, TMA byte count)
                 empty[s] (consumers→producer, stage reusable)
```

Tile 128×256×64. `BK=64` halves is 128 bytes — exactly one 128B-swizzle atom row — which is
what makes the descriptor arithmetic in step 1 come out clean.

**Result:** 8192³ = **744 TF** end-to-end.

---

## 1. Empirical descriptor validation

Not a speed optimization, but the reason everything after it worked. The `wgmma` shared-memory
descriptor packs address / leading-byte-offset / stride-byte-offset / swizzle into 64 bits, and
getting it wrong produces *plausible but scrambled* results, not a crash. Rather than trust a
reading of the docs, a standalone probe swept candidate encodings against a CPU reference:

```
MATCH  lbo_enc=1     sbo_enc=64
```

i.e. LBO = 16 B (one core matrix along K), SBO = 1024 B (one swizzle atom along M/N), 128 B
swizzle, and +32 B of descriptor advance per k-step -- the layout is keyed on the 128-byte
swizzle row, not on the element type.

This methodology paid for itself again at step 5, where it overturned a conclusion I had
already written into the source as fact.

---

## 2. L2 rasterization (`GROUP_M`)

Also called threadblock swizzle, tile rasterization, or Triton's `GROUP_SIZE_M`. The point is
to choose *which* output tile each CTA takes, so that the ~132 tiles running concurrently
share operand tiles in L2 instead of sweeping across all of B.

```
   tiles_n = 6, GROUP_M = 4.  Numbers = order CTAs are launched (the work index w).

   ROW-MAJOR order                    GROUPED order (what we do)
   tile_n →                           tile_n →
   ┌───┬───┬───┬───┬───┬───┐          ┌───┬───┬───┬───┬───┬───┐
 t │ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │        t │ 0 │ 4 │ 8 │12 │16 │20 │
 i ├───┼───┼───┼───┼───┼───┤        i ├───┼───┼───┼───┼───┼───┤
 l │ 6 │ 7 │ 8 │ 9 │10 │11 │        l │ 1 │ 5 │ 9 │13 │17 │21 │
 e ├───┼───┼───┼───┼───┼───┤        e ├───┼───┼───┼───┼───┼───┤
 _ │12 │13 │14 │15 │16 │17 │        _ │ 2 │ 6 │10 │14 │18 │22 │
 m ├───┼───┼───┼───┼───┼───┤        m ├───┼───┼───┼───┼───┼───┤
   │18 │19 │20 │21 │22 │23 │          │ 3 │ 7 │11 │15 │19 │23 │
   └───┴───┴───┴───┴───┴───┘          └───┴───┴───┴───┴───┴───┘
   ▲                                  ▲
   w walks →  then wraps down         w walks ↓ within a GROUP_M-tall column,
                                      then jumps right

   With 12 CTAs resident:
     row-major : 2 tile-rows × 6 tile-cols →  2 A-blocks + 6 B-blocks = 8 operand tiles
     grouped   : 4 tile-rows × 3 tile-cols →  4 A-blocks + 3 B-blocks = 7 operand tiles
   The effect grows with the resident count: at 132 CTAs and tiles_n=32, row-major touches
   5 A + 32 B = 37 tiles, GROUP_M=8 touches 8 A + 17 B = 25.
```

Decoding a work index `w` into a tile, with the group possibly truncated at the bottom edge:

```cpp
const int group_sz = CGROUP_M * tiles_n;   // work items in one full group
const int gid      = w / group_sz;         // which group
const int in_grp   = w - gid * group_sz;   // position inside it
const int first_cm = gid * CGROUP_M;       // group's first tile-row (cluster units)
const int gcm      = min(ctiles_m - first_cm, CGROUP_M);   // may be short at the edge
tile_m = (first_cm + in_grp % gcm) * CLUSTER_M_ + rank;    // ↓ fastest
tile_n = in_grp / gcm;                                     // → slowest
```

`in_grp % gcm` is what makes `w` walk *down* the column; `in_grp / gcm` advances right only
after the column is exhausted. `gcm` (rather than `CGROUP_M`) handles a final group that ran
out of tile-rows. Everything is in *cluster* units — `CGROUP_M = GROUP_M / CLUSTER_M` — and
`rank` picks the CTA's row within its cluster.

**Measured across all 31 shapes** (spread = best-to-worst over `GROUP_M ∈ {2,4,8,16,32,64}`):

| shape | GM=2 | GM=4 | GM=8 | GM=16 | GM=32 | GM=64 | spread |
|---|---|---|---|---|---|---|---|
| 16384³ | 679 | 833 | 835 | 835 | 837 | **838** | **18.9%** |
| 16384×8192×128 | **314** | 313 | 310 | 293 | 298 | 286 | **9.1%** |
| 8192×8192×2048 | 775 | 762 | 799 | **826** | 797 | 765 | **7.7%** |
| 4096×512×4096 | 512 | 517 | 540 | 542 | **554** | 534 | **7.6%** |
| 4096×8192×512 | 549 | **575** | 533 | 556 | 536 | 539 | **7.2%** |
| 8192×8192×1024 | 673 | **721** | 672 | 675 | 679 | 693 | **6.8%** |
| 8192³ | 806 | 799 | **810** | 805 | 806 | 784 | 3.2% |
| 4096³ | 863 | 869 | **869** | 868 | 865 | 866 | 0.7% |

**9 of 31 shapes move by more than 5%.** The optimum is shape-dependent and spans the whole
range: 16384³ wants `GM=64`, 16384×8192×128 wants `GM=2`.

**Correction.** An earlier version of this report said GROUP_M was "not a lever for fp16",
based on the 8192³ sweep alone, where the spread is 3.2% — genuinely noise. Generalizing
from one shape was wrong: the largest shape in the suite swings 18.9%.

What *is* true is that the default `GROUP_M = 8` is a good compromise — within ~3% of the
per-shape optimum everywhere, and the 18.9% spread at 16384³ comes from `GM=2` being bad, not
from `GM=8`. A per-shape oracle would buy ~1.5% on average, ~3.4% at best. Not enough to
justify a dispatcher, but worth knowing the knob is live.

---

## 3. Cluster TMA multicast

Two CTAs form a cluster covering adjacent tile-rows at the same `tile_n`, so both consume the
*identical* B tile. Each CTA TMA-multicasts a slice of it to the whole cluster, halving B's
L2→SM traffic.

```cpp
tma_2d_multicast(bdst + blk * B_BLOCK_ELEMS, &tmap_b, ncol, kt * BK,
                 &bar_full[s], MC_MASK);   // .multicast::cluster
```

Consumers must then release a stage to *every* CTA in the cluster, via distributed shared memory:

```cpp
__device__ void bar_arrive_remote(uint64_t *bar, uint32_t rank) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
               :: "r"(map_to_cta(smem_u32(bar), rank)));
}
```

**Measured (8192³ mainloop):**

| | CLUSTER_M=1 | CLUSTER_M=2 | gain |
|---|---|---|---|
| at transpose stage | 764.8 TF | 807.4 TF | **+5.6%** |
| on final kernel | 722.6 TF | 779.1 TF | **+7.8%** |

---

## 4. Pipeline depth (`STAGES`)

A *stage* is one slot in the smem circular buffer holding one k-tile of operands
(`BM×BK` of A + `BN×BK` of B = 48 KB). The producer runs ahead filling stages; consumers drain
them. `STAGES` bounds how far ahead the producer may run — i.e. how much TMA latency
(~0.5–1 µs) can be covered by queued math.

```
 STAGES = 4                    TMA latency ~0.5-1 us must be covered by queued math
 producer ─┬ fill s0 ─┬ fill s1 ─┬ fill s2 ─┬ fill s3 ─┬ refill s0 ─┬ …
           │          │          │          │          │ (waits empty[0])
 consumer  └─ wait s0 ┴ wgmma s0 ┴ wgmma s1 ┴ wgmma s2 ┴ wgmma s3 ──┴ …
            └─ fill ─┘
              paid once per CTA

 too few stages → consumer catches up and stalls on bar_wait(full[s])
 too many      → smem is gone; it must come out of the tile or the epilogue buffer
```

```cpp
const int s = it % STAGES;
if (it >= STAGES) bar_wait(&bar_empty[s], ((it / STAGES) - 1) & 1);
```

**Measured:**

| STAGES | smem/CTA | K=2048 | K=8192 |
|---|---|---|---|
| 2 | 96 KB | 433 TF | **500 TF** |
| 3 | 144 KB | 634 | 803 |
| 4 | 192 KB | **650** | **858** |
| 5 | 240 KB | — | ✗ exceeds 227 KB |

**2 → 4 stages is +71%.** Five is impossible: H100 allows 232,448 B of dynamic smem.

**Side-effect:** stages trade directly against tile size and against epilogue staging space.
Re-measured at step 9, dropping to STAGES=3 to free 48 KB cost **−6.6%** at K=2048 and 4096³ —
which is why the TMA-store epilogue was chunked to fit the leftover 34.9 KB instead.

---

## 5. Eliminate the B transpose — consume B N-major

**The single largest structural win.**

`wgmma`'s default B operand is K-major. `A` (row-major M×K) already is; `B` (row-major K×N) is
**N-major** — `B[k][n]` sits at `k*N+n`, so n varies fastest. The naive port therefore
materialized `B^T` with a separate transpose kernel, costing ~7% and a workspace.

I first concluded this was unavoidable, reasoning that under 128 B swizzle TMA's innermost box
caps at 128 B = 64 halves, so a BN=256 N-major tile arrives as 4 separate blocks whose core
matrices are not a linear stride apart — "inexpressible in one descriptor". **That was wrong.**
An nsys trace showed cuBLAS running a single kernel with no transpose, which contradicted it.

```
 B is row-major K×N ⇒ n varies fastest ⇒ B is N-major.  wgmma's default B operand is
 K-major, which is why a naive port transposes B.

 Under 128B swizzle, TMA's innermost box caps at 128 B = 64 halves, so a BN=256
 N-major tile cannot arrive in one piece. It comes as 4 blocks:

    global B[k][n],  n →
         ┌──────────┬──────────┬──────────┬──────────┐
     k ↓ │ n  0..63 │ n 64..127│ n128..191│ n192..255│  BK rows
         └──────────┴──────────┴──────────┴──────────┘
             blk0        blk1       blk2       blk3

 Place them CONTIGUOUSLY in smem and the stride along N becomes uniform:

    smem  [   blk0   ][   blk1   ][   blk2   ][   blk3   ]
          └ BK*128 B ┘└ BK*128 B ┘└ BK*128 B ┘
                     ↑ one constant stride ⇒ exactly what the descriptor's LBO encodes

 That is the whole trick. I assumed the blocks were not a linear stride apart and
 concluded it was impossible. They are — if you lay them out this way.
```

Two probes settled it. If the blocks are placed **contiguously**, the stride along N *is*
uniform, and the MN-major descriptor expresses exactly that:

```
probe 1:  MATCH  lbo_enc=512  sbo_enc=64        (LBO = BK*128 B = the block stride)
probe 2:  kadv=2048 bytes  maxerr=0  <== MATCH  (a k16 step crosses 16 rows of 128 B)
```

```cpp
// Descriptor for an MN-major operand. The roles of the two offsets swap relative to the
// K-major case: LBO is the N-direction block stride, SBO the K-direction atom stride.
__device__ __forceinline__ uint64_t make_desc_mn(uint32_t addr) {
  uint64_t d = (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)(BK * 128 / 16) << 16;   // LBO = BK rows * 128B, per 64-N block
  d |= (uint64_t)64ull << 32;             // SBO = 1024B
  d |= (uint64_t)1ull << 62;              // swizzle = 128B
  return d;
}
```

```cpp
// producer: B arrives as B_BLOCKS contiguous [BK][64] blocks, in its native layout
for (int j = 0; j < BLOCKS_PER_CTA; ++j) {
  const int blk = rank * BLOCKS_PER_CTA + j;
  tma_2d_multicast(bdst + blk * B_BLOCK_ELEMS, &tmap_b,
                   tile_n * BN + blk * 64, kt * BK, &bar_full[s], MC_MASK);
}

// consumer: A advances +32B per k16 step; B crosses 16 rows of a 128B block = +2048B
wgmma_m64n256k16<1>(d, make_desc(a_base + ks * 32),
                       make_desc_mn(b_base + ks * MN_K_STRIDE));
```

and `imm-trans-b` flips from 0 to 1:

```
wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 {...}, %128, %129, %130, 1, 1, 0, 1;
                                                                                    ^ trans_b
```

The whole transpose kernel and its workspace were deleted.

**Measured:**

| shape | with transpose | N-major | gain |
|---|---|---|---|
| 4096³ | 617 TF | **710** | **+15%** |
| 8192³ | 745 | **810** | **+8.7%** |
| 8192×8192×4096 | 599 | **633** | +5.7% |
| 4096×4096×8192 | 637 | **738** | **+16%** |
| 16384×8192×4096 | 690 | **716** | +3.8% |

---

## 6. Stride pad-copy — fixing the cliff step 5 created

Removing the transpose moved TMA's 16-byte row-stride requirement from `K%8` onto `N%8`
(B's row stride is now N halves). Anything with odd N fell to the scalar fallback:

```
4096 x 4095 x 4096     6.0 TF     <- vs 708 TF aligned:  118x cliff
```

Fix: restride the offending operand into scratch with one streaming copy, instead of dropping
to the scalar kernel. Only the live columns are copied — TMA clamps reads to `globalDim`, so
the pad is never read.

```cpp
__global__ void pad_copy_kernel(const half *src, half *dst, int rows, int cols, int ld) {
  const int c = blockIdx.x * 256 + threadIdx.x;
  if (c >= cols) return;
  for (int r = blockIdx.y; r < rows; r += gridDim.y)
    dst[(long long)r * ld + c] = src[(long long)r * cols + c];
}
```

```cpp
const int lda = (K + 7) & ~7, ldb = (N + 7) & ~7;
if (padA) { pad_copy_kernel<<<...>>>(A, w,      M, K, lda); Ause = w; }
if (padB) { pad_copy_kernel<<<...>>>(B, w + na, K, N, ldb); Buse = w + na; }
```

**Measured:** N=4095 → **6.0 → 517 TF (86×)**. Caught only because I measured the swap rather
than assuming it was free.

---

## 7. Persistent kernel

Previously one CTA per output tile: 8192³ launched **2048 CTAs over 132 SMs** — 15.5 waves, so
the final wave ran half-empty and every CTA re-paid its prologue. nsys showed cuBLAS launching
exactly 132.

The change is a grid sized to residency plus a strided walk over the tile list. The subtlety
that makes it worth doing is a **monotonic pipeline counter across all tiles** — stage/phase
bookkeeping does *not* reset per tile, so the producer streams the next tile's k-tiles while
consumers are still in the current one's epilogue.

```cpp
int it = 0;                                   // monotonic across the CTA's whole work
for (int w = my_cluster; w < total_ctiles; w += num_clusters) {
  int tile_m, tile_n;
  decode(w, tile_m, tile_n);
  for (int kt = 0; kt < num_k_tiles; ++kt, ++it) {
    const int s = it % STAGES;                // <- not kt % STAGES
    ...
  }
}
```

```cpp
const int grid = min(total_ctiles * CLUSTER_M, (sm_count / CLUSTER_M) * CLUSTER_M);
```

**Measured:**

| shape | per-tile grid | persistent | gain |
|---|---|---|---|
| 8192×8192×**2048** | 575 TF | **657** | **+14%** |
| 4096³ | 708 | **749** | +6% |
| 16384×8192×4096 | 714 | **747** | +4.6% |
| 8192³ | 825 | **843** | +2% |

---

## 8. Epilogue v2 — shuffle-combined stores

After step 7, a decomposition (a build with the epilogue suppressed at runtime) showed the
epilogue was **the entire remaining gap**: our mainloop alone was 1.7–9.6% *faster* than
cuBLAS's whole kernel, at 92–93% of peak.

The cause is the `wgmma` accumulator layout. For a fixed `(g, j/2)`, a warp's 32 lanes cover
**8 rows** (`lane/4`) contributing only **16 contiguous bytes** each (`2*(lane%4)` plus the
`j%2` pair) — eight fragments, `N*2` bytes apart, against a 32-byte sector granularity. Every
fragment half-fills a sector:

```
 ONE store instruction, 32 lanes.   row = …+ lane/4      (changes every 4 lanes)
                                    col = …+ 2*(lane%4)  (4 lanes span 8 halves = 16 B)

   row 0  [16B]·······································   lanes  0- 3   bytes     0..15
   row 1  ·······[16B]································   lanes  4- 7   bytes 16384..16399
   row 2  ··············[16B]·························   lanes  8-11   bytes 32768..32783
   row 3  ·····················[16B]··················   lanes 12-15
    ⋮                                                     (8 fragments, N*2 = 16 KB apart)
   row 7  ····································[16B]···   lanes 28-31

   128 useful bytes scattered over 8 sectors = 256 B of sector traffic ⇒ 50% wasted
```

The missing half of each sector is the *next* g iteration (cols +8..+15) — the data exists, but
in a different instruction, because adjacent columns of one row live in different lanes. So the
fix is a **lane exchange**, not a wider store. Lanes `b` and `b^1` swap one of their two
column-pairs across `g` and `g+1`:

```
 One __shfl_xor between lanes b and b^1, swapping one column-pair across g and g+1:

  before   lane0 [0,1][8,9]   lane1 [2,3][10,11]   lane2 [4,5][12,13]   lane3 [6,7][14,15]
                  └─g─┘└g+1┘
  after    lane0 [0 1 2 3]    lane2 [4 5 6 7]      lane1 [8 9 10 11]    lane3 [12..15]
                  └── 8 B ──┘
           └──────────── 16 halves = 32 B = ONE FULL SECTOR ────────────────────────┘
```

Each lane now holds 4 contiguous halves (8 B) and the quad tiles **one full 32-byte sector**.

```cpp
const float a0 = alpha * d[i0], a1 = alpha * d[i0 + 1];   // my g   pair
const float e0 = alpha * d[i1], e1 = alpha * d[i1 + 1];   // my g+1 pair

// Odd lanes hand over their g pair, even lanes their g+1 pair.
const float r0 = __shfl_xor_sync(0xffffffffu, odd ? a0 : e0, 1);
const float r1 = __shfl_xor_sync(0xffffffffu, odd ? a1 : e1, 1);

float v0, v1, v2, v3;
if (odd) { v0 = r0; v1 = r1; v2 = e0; v3 = e1; }
else     { v0 = a0; v1 = a1; v2 = r0; v3 = r1; }

union { uint2 u; half2 h2[2]; } out;
out.h2[0] = __floats2half2_rn(v0, v1);
out.h2[1] = __floats2half2_rn(v2, v3);
*reinterpret_cast<uint2 *>(dst) = out.u;
```

**Critical implementation detail:** a first attempt used `olo[p]` with a runtime index
`p = b ^ i`. Local arrays indexed by a non-constant cannot live in registers, so ptxas spilled
them (64-byte stack frame) and the change was a net loss. Every `d[]` and result index must stay
compile-time constant; lane-dependent choices are ternaries, not array indices. The final version
reports **0 bytes stack frame, 0 spills**.

I used a 2-way exchange rather than 4-way deliberately: half the shuffles, and it already
reaches a full sector, so the extra work buys nothing.

**Measured:**

| shape | before | after | epilogue share |
|---|---|---|---|
| 8192×8192×**2048** | 657 TF | **796** (+21%) | 27.5% → **12.2%** |
| 4096³ | 749 | **836** (+12%) | 16.8% → **7.1%** |
| 16384×8192×4096 | 730 | **791** (+8%) | 12.8% → **5.5%** |
| 8192³ | 752 | **813** (+8%) | 10.0% → **2.7%** |
| 4096×4096×8192 | 835 | **886** (+6%) | 9.3% → **3.8%** |

---

## 9. Epilogue v3 — TMA store

After v2 the epilogue's implied write bandwidth was **~2.9 TB/s against a 3.35 TB/s HBM3 peak**,
and two shapes measured *above* peak — only possible if part of the write was already
overlapping. Conclusion: the store was no longer *inefficient*, it was **serialized**.

That reframes the problem. Store-widening (4-way exchange, wider vectors) would buy nothing.
The remaining lever is asynchrony — hand the drain to the TMA engine so the warps return to the
mainloop immediately:

```cpp
__device__ __forceinline__ void tma_store_2d(const CUtensorMap *map, int c0, int c1,
                                             const void *src) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
               :: "l"(map), "r"(c0), "r"(c1), "r"(smem_u32(src)) : "memory");
}
```

Each consumer WG stages a 64×64 chunk into smem in the 128B-swizzled layout the engine expects,
double-buffered so the register→smem fill of one chunk overlaps the store of the previous:

```cpp
for (int j = 0; j < EPI_CHUNKS; ++j) {
  half *buf = sEpi + (cwg * 2 + (j & 1)) * EPI_CHUNK_ELEMS;

  // Reclaim the buffer: with <=1 group outstanding, the store from two chunks ago is done.
  if (lane_in_wg == 0) tma_store_wait<1>();
  wg_barrier(bar_id);

  for (int h = 0; h < 2; ++h) {
    const int r = r_loc + 8 * h;
    const int sw = r & 7;                    // 128B swizzle: XOR chunk index with row%8
    for (int a = 0; a < 8; ++a) {
      const int i   = 4 * (8 * j + a) + 2 * h;
      const int off = r * EPI_COLS + ((a ^ sw) * 8) + 2 * b;
      *reinterpret_cast<half2 *>(buf + off) =
          __floats2half2_rn(alpha * d[i], alpha * d[i + 1]);
    }
  }
  asm volatile("fence.proxy.async.shared::cta;");   // generic proxy -> async proxy
  wg_barrier(bar_id);
  if (lane_in_wg == 0) {
    tma_store_2d(tmap_c, tile_n * BN + EPI_COLS * j, tile_m * BM + cwg * WG_M, buf);
    tma_store_commit();
  }
}
```

The swizzled smem write is bank-conflict-free by construction: bank =
`((a ^ (lane/4))*4 + lane%4) % 32`, and since XOR is a bijection the 32 lanes cover all 32 banks
exactly once.

Sizing: 2 WGs × 2 phases × 8 KB = **32 KB**, which fits the 34.9 KB left by a 4-stage pipeline —
so no stage had to be surrendered (measured at −6.6% if it had been).

**Measured:**

| shape | shuffle | TMA store | gain | epilogue share |
|---|---|---|---|---|
| 8192×8192×4096 | 848 TF | **899** | **+6.1%** | 8.4% → **2.8%** |
| 16384×8192×4096 | 779 | **846** | **+8.6%** | 6.0% → **~0%** |
| 8192×8192×2048 | 790 | **843** | **+6.7%** | 13.1% → **7.2%** |
| 4096³ | 842 | **864** | +2.6% | 7.2% → **4.8%** |
| 4096×4096×8192 | 886 | **909** | +2.6% | 3.8% → **1.3%** |
| 8192³ | 806 | 796 | −1.2% (noise) | 4.6% → 5.8% |

**Limitations:** a bulk store cannot read-modify-write, so `beta != 0` keeps the shuffle path;
and TMA needs a 16B-aligned row stride on C (`N%8`), which cannot be pad-fixed because C is the
caller's output buffer. Both fall back to v2, which is itself fast.

---

## 10. Tile-size dispatch — templating on `BM` and `BN`

Steps 0-9 all used one tile shape, 128x256 -- carried over from the initial design and never
re-examined. A per-shape sweep against cuBLAS showed it failing badly on
small problems, and not because of math:

| shape | ratio | tiles at 128x256 | CTAs | GPU used |
|---|---|---|---|---|
| 1024^3 | **0.47x** | 8 x 4 = 32 | 32 | **24%** |
| 384x2048x2048 | **0.41x** | 3 x 8 = 24 | 24 | **18%** |

Pure starvation. Templating on tile shape gives a ladder of configurations, chosen per call:

```cpp
template <int BM_, int BN_>
struct Cfg {
  static constexpr int CONSUMERS = BM_ / 64;               // one warpgroup per 64 rows
  static constexpr int THREADS   = (CONSUMERS + 1) * WGS;  // + the producer
  static constexpr int NREG      = 64 * BN_ / 128;         // accumulators per thread
  static constexpr int EPI       = CONSUMERS * 2 * EPI_CHUNK_ELEMS;
  static constexpr int STAGES    = ...;                    // fill the leftover smem budget
};
template <int CLUSTER_M_, int BM_, int BN_> __global__ void gemm_kernel(...);
```

### The thing I got wrong: CTAs are not the unit that matters

My first instinct was `BM=64` -- one consumer warpgroup instead of two, doubling the CTA
count. Built and measured, it disappointed: 1024^3 went 210 -> 236 TFLOPS, not the ~385 the
CTA-scaling scan projected.

The reason is that `BM=64` **adds no math capacity**. 128 CTAs x 1 consumer warpgroup is 128
warpgroups; the old 64 CTAs x 2 was also 128. It spreads the same warpgroups over more SMs,
leaving each SM with nothing to hide its own stalls behind.

What actually adds capacity is keeping two consumers and narrowing `BN` instead:

| config at 1024^3 | CTAs | **warpgroups** | TFLOPS |
|---|---|---|---|
| BM=128 BN=256 | 32 | 64 | 140 |
| BM=128 BN=128 | 64 | 128 | 205 |
| BM=64  BN=128 | 128 | 128 | 231 |
| **BM=128 BN=64** | 128 | **256** | **314** |

```
 The unit that matters is CONSUMER WARPGROUPS, not CTAs.

   BM=128 → 2 consumer WGs/CTA          BM=64 → 1 consumer WG/CTA
   ┌───────────────────┐                ┌─────────────────┐
   │ WG0  producer     │                │ WG0  producer   │
   │ WG1  rows   0..63 │ math           │ WG1  rows 0..63 │ math
   │ WG2  rows  64..127│ math           └─────────────────┘
   └───────────────────┘                  1/2 of the CTA does no math
     1/3 of the CTA does no math

  1024³:  BM=64 ,BN=128 → 128 CTAs × 1 WG = 128 WGs → 231 TF  (same WG count as before)
          BM=128,BN=64  → 128 CTAs × 2 WG = 256 WGs → 314 TF  ← the actual lever
```

Throughput tracks the warpgroup column, not the CTA column. `BM=64` was measured on every
shape below and **never won**, so the ladder holds `BM=128` throughout and only varies `BN`.

| shape | 128x256 | 128x128 | 128x64 | 64x128 | 64x64 | cuBLAS |
|---|---|---|---|---|---|---|
| 1024^3 | 140 | 205 | **314** | 231 | 257 | 306 |
| 384x2048x2048 | 141 | 244 | **338** | 272 | 241 | 354 |
| 4096x512x4096 | 389 | **534** | 417 | 411 | 344 | 688 |
| 2048x2048x512 | 359 | **361** | 343 | 315 | 278 | 430 |
| 2048^3 | **671** | 621 | 485 | 464 | 383 | 724 |
| 3000x1000x2000 | **456** | 378 | 257 | 218 | 193 | 477 |
| 4096^3 | **863** | 688 | 432 | 486 | 362 | 861 |
| 8192^3 | **839** | 672 | 371 | 478 | 369 | 754 |

### Dispatch

```cpp
auto tiles = [&](int bm, int bn) { return ((M+bm-1)/bm) * ((N+bn-1)/bn); };
const int need = (sm_count * 2 + 2) / 3;         // ~88 on a 132-SM part
if (tiles(128, 256) >= need) return launch_tile<128, 256>(...);
if (tiles(128, 128) >= need) return launch_tile<128, 128>(...);
return launch_tile<128, 64>(...);
```

The threshold is read off the table above, not assumed. An earlier attempt used "fewer tiles
than SMs", which **regressed** 2048^3 (0.94x -> 0.87x) and 3000x1000x2000 (0.92x -> 0.75x),
because a narrower tile is ~1.2x slower whenever the wider one has enough work.

**Measured:**

| shape | before | after | ratio |
|---|---|---|---|
| 1024^3 | 210 TF | **317** | 0.69x -> **1.04x** |
| 384x2048x2048 | 143 | **342** | 0.41x -> **0.97x** |
| 1024x1023x1024 | 75 | **133** | 1.05x -> **1.88x** |
| 4096x512x4096 | 398 | **529** | 0.58x -> 0.75x |

Overall **19 of 31 shapes at >=1.00x cuBLAS** (from 16), 26 at >=0.96x.

### A deadlock this introduced

`BN=64` yields exactly one 64-wide B block, and the cluster splits B by handing each CTA
`BLOCKS / CLUSTER_M` of them -- which is **zero**. B never arrives and the consumers hang on
`bar_full` forever. It showed up as a benchmark that never returned.

The fix decides clustering at compile time so the impossible instantiation is never formed,
plus a `static_assert` so any future tile that violates it fails to build instead of hanging:

```cpp
constexpr int CM = (C_::BLOCKS >= CLUSTER_M && C_::BLOCKS % CLUSTER_M == 0) ? CLUSTER_M : 1;
...
static_assert(C_::BLOCKS >= CLUSTER_M_ && C_::BLOCKS % CLUSTER_M_ == 0,
              "B tile must split evenly across the cluster, else CTAs load nothing and hang");
```

> **Measurement note.** One round of this sweep was discarded: I collected tile widths with
> `paste <(./a) <(./b) <(./a)`, which runs three benchmarks *concurrently* on one GPU. The
> tell was cuBLAS reading 265 TFLOPS at 8192^3 when it really does ~790. Run them serially.

---

## 11. Per-shape `GROUP_M` policy

Step 2 established that `GROUP_M` is a live knob but left it at a single global default of 8.
The obvious follow-up is to pick it per shape. The interesting part is what stopped that from
becoming a lookup table.

### First: is the winner even reproducible?

Searching each shape's best `GROUP_M` **three independent times**:

```
 shape              trial winners      gain over GM=8        stable?
 16384³             16, 16, 16         1.000 1.000 1.000     YES  (but no gain)
 4096x512x4096      16, 16, 16         1.042 1.045 1.048     YES  <- real
 16384x8192x128      4,  4,  4         1.037 1.027 1.033     YES  <- real
 4096x8192x512       4,  4,  4         1.029 1.031 1.028     YES  <- real
 1024x1023x1024      2,  2,  2         1.028 1.024 1.026     YES  <- real
 8192x8192x8192      2, 16,  8         1.008 1.000 1.000     no   <- noise
 4096x4096x4096      4,  8, 64         1.022 1.000 1.006     no   <- noise
 8192x4096x8192      2,  8,  4         1.027 1.000 1.000     no   <- noise
```

**Only 16 of 31 shapes picked the same winner three times.** For the other 15, "best
`GROUP_M`" is a noise artefact — a lookup table would have hard-coded it as fact.

This also **revised a number from step 2**. That sweep, run as one process per `GROUP_M`,
reported up to **7.2%** available. Measured in-process the same shape yields **2.6%**:
`4096x8192x512` goes 533→575 across processes but 580→595 within one. The GM=8 reading was a
low outlier. Cross-process comparison inflates spread; step 2's headline was overstated for
exactly the reason its original claim was wrong — uncontrolled measurement conditions.

### The policy

Every gain that was *both* stable across all three trials *and* worth >2% falls into three
buckets, so the rule is structural rather than a table:

```cpp
// group_m == 0 means "auto"; an explicit value (autotuner, benchmark) overrides.
if (group_m == 0) {
  if (bn == 64)      group_m = 2;   // narrow tile / tiny problems
  else if (K <= 512) group_m = 4;   // short k-loop, many tiles
  else if (N <= 512) group_m = 16;  // narrow N, long k-loop
  else               group_m = GROUP_M;
}
```

Shapes inside the noise band are deliberately left at the default rather than fitted.

Making this possible meant turning `GROUP_M` from a `constexpr` into a **runtime kernel
argument** (`cgroup_m`), so per-shape selection costs no extra instantiations. It is used once
per tile in the rasterization decode, never in the k-loop, so the runtime divide is free.

**Measured** — policy vs a pinned `GROUP_M=8`, same tile ladder, same process:

| shape | GM=8 | policy | GM | policy/GM=8 | policy/cuBLAS |
|---|---|---|---|---|---|
| 4096×512×4096 | 531 | **575** | 16 | **1.082×** | 0.82× |
| 4096×8192×128 | 265 | **271** | 4 | **1.024×** | 1.04× |
| 1024×1023×1024 | 133 | **136** | 2 | **1.023×** | 1.93× |
| 1024³ | 320 | **327** | 2 | **1.022×** | 1.07× |
| 8192×8192×128 | 297 | **303** | 4 | **1.021×** | 1.05× |
| 4096×8192×512 | 534 | **544** | 4 | **1.018×** | 0.98× |

Six shapes gain 2–8%; the remaining 25 are unchanged.

> **A measurement that was worthless, and how it announced itself.** The first run of this
> comparison showed ~1.00× on all 31 rows. Cause: the baseline passed `group_m = 8`, and the
> policy triggered on `group_m == GROUP_M`, which *is* 8 — both lambdas ran identical code.
> A row of 1.000× across every shape is not a result, it is a bug report. Fixed by making
> `0` mean "auto" so an explicit 8 is distinguishable.

### Autotuner

`autotune()` / `launch_tuned()` search (tile × `GROUP_M`) on the real shape and cache per
`(M,N,K)`. Kept **out of the default path**: the search costs ~18 configurations, which on a
16384³ shape is ~1.5 s before the first result — fine for a serving loop, fatal for a harness
that times the first call. `solve()` uses the heuristics; `launch_tuned()` is opt-in.

---

## 12. Cross-check against CUTLASS

The hand-written kernel builds warp specialization, the TMA pipeline, cluster multicast and
the TMA-store epilogue by hand. `mma_half_cutlass.cu` builds the same thing from CUTLASS 4.7
`CollectiveBuilder` primitives, with an identical `solve()` contract, as an independent check
that the hand-written machinery is worth its complexity.

### Tuning CUTLASS before comparing

CUTLASS's device API fixes the tile per instantiation -- there is **no runtime tile
heuristic** (that is what its offline profiler is for). Comparing our per-shape dispatcher
against a single fixed CUTLASS tile would measure our dispatcher, not the kernels. So the
tile, cluster and kernel schedule were swept, 21 configurations over two shapes:

| config | 1024³ | 4096³ |
|---|---|---|
| 128×256×64 c2×1 coop *(baseline)* | 131 | **776** |
| 128×256×64 c1×2 coop | 133 | 786 |
| 256×128×64 c2×1 coop | 129 | 744 |
| 128×128×64 c1×1 coop | 191 | 641 |
| 128×128×128 c1×1 coop | 202 | 541 |
| 128×64×64 c1×1 coop | 227 | 374 |
| **128×64×128 c1×1 coop** | **248** | 376 |
| 64×128×64 c1×1 ping | 204 | 385 |
| 64×64×64 c1×1 ping | 162 | 245 |
| 128×32×64 c1×1 coop | 160 | 200 |
| 128×256×128 c2×1 coop | 98 | 428 |
| 256×256×64 c2×1 coop | **7** | **33** |

Three things fall out:

- **1024³ improved 1.83x** (131 -> 240 in the final build) from `128×64×128 c1×1`.
- **4096³ could not be improved.** Nothing beat the baseline; the one nominal winner
  (c1×2 at 786 vs 776) is 1.3%, inside the noise band, so it was not taken.
- Small and large want *opposite* tiles, sharply: `128×64×128` is 1.83x better at 1024³ and
  2.1x worse at 4096³. CUTLASS hits the same starvation wall step 10 describes.

`256×256×64` collapsing to 7 / 33 TFLOPS is worth flagging: `can_implement` accepts it, so it
builds and runs while being ~100x slow -- almost certainly a shared-memory overrun forcing a
1-stage pipeline. A configuration sweep needs a sanity floor, not just a success check.

### Two traps in the CUTLASS build

**`EpilogueScheduleAuto` selects `DefaultEpilogue`, not the TMA one.** Measured 35% slower
(572 vs 774 TFLOPS at 4096³). Pairing the cooperative mainloop with an explicit
`TmaWarpSpecializedCooperative` epilogue -- what CUTLASS's own examples do -- is what the
comparison uses. `-DCUTLASS_EPI_AUTO` reproduces the slow variant.

**CUTLASS validates alignment against the problem *extent*, not the leading dimension.** A
first attempt padded the *stride* of ragged operands, as `mma_half.cu` does; `can_implement`
rejected all 19 such shapes and, because the return value was ignored, C was silently left
untouched -- 19 failures all reporting `rel_l2 ≈ 1.0`. Supporting them properly means running
a zero-padded *extent* and copying back, i.e. more work than the kernel under test performs,
so those shapes are skipped instead. `mma_half.cu` has no such restriction: TMA zero-fills
out-of-range elements, so it runs ragged shapes in place.

### Result

| M × N × K | ours | CUTLASS | cuBLAS | ours/CUTLASS | ours/cuBLAS |
|---|---|---|---|---|---|
| 4k×4095×4k | **503** | n/a | 145 | — | **3.47×** |
| 4095×4095×4095 | **437** | n/a | 154 | — | **2.83×** |
| 2k×2047×2k | **309** | n/a | 147 | — | **2.10×** |
| 1k×1023×1k | **137** | n/a | 71 | — | **1.93×** |
| 4k×4k×8k | **901** | 806 | 824 | **1.12×** | **1.09×** |
| 8k×8k×16k | **858** | 783 | 788 | **1.10×** | **1.09×** |
| 8k×4k×8k | **912** | 707 | 837 | **1.29×** | **1.09×** |
| 8k×8k×4k | **890** | 709 | 825 | **1.26×** | **1.08×** |
| 8k×8k×2k | **840** | 688 | 780 | **1.22×** | **1.08×** |
| 4k×8k×8k | **911** | 708 | 851 | **1.29×** | **1.07×** |
| 16k×16k×16k | **845** | 675 | 806 | **1.25×** | **1.05×** |
| 16k×8k×4k | **823** | 726 | 787 | **1.13×** | **1.05×** |
| 4k×8k×128 | **279** | 235 | 268 | **1.19×** | **1.04×** |
| 1k×1k×1k | **317** | 240 | 306 | **1.32×** | **1.04×** |
| 16k×8k×128 | **312** | 263 | 302 | **1.19×** | **1.03×** |
| 32k×8k×2k | **813** | 720 | 791 | **1.13×** | **1.03×** |
| 8k×8k×8k | **812** | 760 | 795 | **1.07×** | **1.02×** |
| 4k×8k×512 | 609 | 514 | 611 | **1.18×** | 1.00× |
| 4k×4k×4k | 865 | 774 | 874 | **1.12×** | 0.99× |
| 4k×8k×256 | 463 | 367 | 469 | **1.26×** | 0.99× |
| 8k×1k×8k | 878 | 795 | 892 | **1.10×** | 0.98× |
| 8k×8k×128 | 283 | 247 | 289 | **1.14×** | 0.98× |
| 4k×4k×1k | 693 | 614 | 708 | **1.13×** | 0.98× |
| 4k×8k×1k | 733 | 642 | 753 | **1.14×** | 0.97× |
| 8k×8k×1k | 758 | 652 | 780 | **1.16×** | 0.97× |
| 16k×4k×8k | 825 | 758 | 851 | **1.09×** | 0.97× |
| 384×2k×2k | 338 | 254 | 350 | **1.33×** | 0.97× |
| 3000×1000×2000 | 460 | 432 | 493 | **1.06×** | 0.93× |
| 2k×2k×2k | 672 | 622 | 726 | **1.08×** | 0.93× |
| 2k×2k×512 | 362 | 348 | 428 | **1.04×** | 0.85× |
| 4k×512×4k | 573 | 438 | 696 | **1.31×** | 0.82× |

**Ours is ahead on all 27 shapes CUTLASS supports (1.04–1.33x)**, and CUTLASS is bit-exact
against cuBLAS on every one of them. After tuning, the margin is a fairly uniform 1.05–1.30x
on large shapes -- a mainloop/epilogue difference rather than a tiling artifact, which is the
comparison worth having.

---

## Final performance

The measured table lives in **[`README.md`](README.md)** and in step 12 above — deliberately
not repeated a third time here, because two copies of these numbers have already drifted apart
once during this work.

Summary: **17 of 31 shapes at ≥1.00× cuBLAS 12.9**, 27 at ≥0.96×, and **ahead of a tuned
CUTLASS 4.7 on all 27 shapes CUTLASS supports (1.04–1.33×)**. Large shapes run 80–87% of the
989.4 TFLOPS hardware peak; the best single figure is 4096³ at 87% and 8192×4096×8192 at
912 TFLOPS.

Regenerate with `make perf` (vs cuBLAS) and `make cutlass` (three-way).

Every shape below 0.95× has ≤128 output tiles against 132 SMs — the bottom of the table is
grid starvation, not math.

---

## Rejected optimizations

Recorded because the reasoning is worth as much as the wins.

**Overlapping the B transpose with the mainloop (two streams, N-banded).** Measured **1–11%
slower**. My model — "the mainloop only uses ~1 of 3.3 TB/s, so the transpose hides under it" —
was wrong: the transpose's CTAs occupy SMs and contend with the mainloop, and banded launches
also give up wave quantization and L2 reuse. (Measured while the transpose still existed; it
was later deleted outright by step 5.)

**Matching cuBLAS's 320×128 tile.** Analysis put it at ~2% (7% less operand traffic on a
non-bottleneck). Then the epilogue decomposition showed our 128×256 mainloop was *already
faster* than their 320×128 one. Dead end — and I should have run the decomposition before the
tile analysis.

**A 4th warpgroup dedicated to the epilogue.** Register budget allows it (176/32/80 → 59,392 of
65,536), but **registers cannot cross warpgroups**, so the handoff needs the same smem staging;
once you have that, the TMA engine drains it without spending 128 threads. cuBLAS's 384-thread
block confirms they reached the same conclusion.

**STAGES=3 to free smem for epilogue staging.** −6.6% at K=2048 and 4096³ — more than the
epilogue win it would have funded. Chunked staging into the existing 34.9 KB instead.

---

## Correctness

- **15 shapes** vs an fp64 host reference — ragged tiles, `alpha`/`beta`, degenerate
  (`1×1×8`, `1×4096×4096`, `4096×4096×8`). `rel_l2 = 2.07e-4` uniformly, which is exactly the
  fp16 output-rounding floor (2⁻¹¹ half-ulp): accumulation is fp32, and there is precisely one
  narrowing, at the store.
- **8 shapes** vs cuBLAS including every large perf-table entry — 8192³, 16384³,
  32768×8192×2048 — **bit-exact** (max_abs = 0.000e+00 over up to 268M elements).
- Explicit coverage for **many-tiles-per-CTA × many-k-tiles**, which exercises the cross-tile
  pipeline bookkeeping introduced in step 7.
- **memcheck / racecheck / synccheck / initcheck: 0 errors** across 30 shapes × both beta modes,
  including large shapes where coordinate arithmetic could overflow.
- 168 registers, **0 spills**.

---

## Known limitations

1. **Small problems are grid-starved.** Anything producing <=64 tiles cannot fill 132 SMs.
   Step 10's dispatcher already drops to the narrowest available tile (BN=128) for these,
   worth 1.31x-1.75x, but 1024^3 is *still* only 64 tiles. Closing the rest needs BM=64 (one
   consumer warpgroup) or split-K -- a third kernel shape, not a tuning knob.
   Worst: 1024^3 at 0.69x, 384x2048x2048 at 0.71x, 4096x512x4096 at 0.77x.
2. **Short K is pipeline-fill-bound.** `2048x2048x512` (0.87x) has near-full occupancy at 128
   tiles, but K=512 is only 8 k-tiles against a 4-stage pipeline, so ~50% of the time is
   fill/drain. The BN=128 path's 6 stages make this *worse*, not better, which is why the
   dispatcher correctly selects the wide tile there.
3. **`N % 8 != 0` runs at ~60% of aligned speed** (508 TF vs 871). Takes the shuffle epilogue
   *and* a B restride. Still 2-3.4x faster than cuBLAS, which abandons its Hopper kernel for an
   sm_75 CUTLASS `align1` fallback -- but that is cuBLAS degrading, not us accelerating.
4. **`beta != 0` uses the shuffle epilogue**, not TMA store -- a bulk store cannot
   read-modify-write.
5. Timing harness uses `memset(0x11)` data. Tensor-core throughput is data-independent and
   0x1111 is a normal fp16 (~6.2e-4, no denormal penalty); numerics are verified separately
   (57 shapes, 0 failures, 38 bit-exact vs cuBLAS).

---
