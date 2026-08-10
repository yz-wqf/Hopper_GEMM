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
| 12 | Cross-check vs per-shape-tuned CUTLASS 4.7 | **parity**: 16/27 ≥, 14 ties; only `1024³` (1.08×) clearly ahead |
| 13 | Hoist the wgmma descriptor out of the k16 loop | **+1.6%** at 1024³, ~0 large |
| 14 | Root-cause: why the remaining losses look the way they do | *(analysis, no code change)* |
| 15 | L2 residency hints on TMA | **+1.5%** median on qualifying shapes |
| 16 | Decouple the cluster decision from GROUP_M | 18/27 ≥ CUTLASS (was 16); `g* = √(SM·BN/BM)` derived |
| — | **Final** | **parity**: 16/31 ≥ cuBLAS, 18/27 ≥ tuned CUTLASS (16 ties), 65–79% of peak, random data + cold L2 |

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

### The wgmma accumulator layout

To understand why the naive epilogue wastes bandwidth, first understand how wgmma distributes
output elements across threads. The instruction `wgmma.m64nBNk16.f32.f16.f16` computes a
64×BN output tile and spreads it across the 128 threads of one consumer warpgroup (4 warps ×
32 lanes). Each thread therefore holds `64×BN / 128 = BN/2` fp32 accumulator registers, named
`d[0]` through `d[BN/2 - 1]`.

The mapping from register index `i` and thread identity `(warp w, lane l)` to output
`(row, col)` is:

```
  row = 16*w + (l/4) + 8*h       where h = (i%4)/2  ∈ {0, 1}
  col = 8*g  + 2*b  + (i%2)      where g = i/4,  b = l%4
```

Two indices do all the structural work:

- **`g = i/4`** — the *column group*: which 8-column slab of the tile this register belongs
  to.  Registers `d[4g], d[4g+1], d[4g+2], d[4g+3]` all land in column group `g`, covering
  cols `8g .. 8g+7`.  `g` runs from 0 to BN/8−1.

- **`b = l%4`** — the *column offset within the group*: which pair of columns inside the
  8-column slab this lane contributes.  `b` ∈ {0,1,2,3} → columns `8g+2b` and `8g+2b+1`.

And for rows:
- `l/4` ∈ {0..7}: within one warp, 8 groups of 4 lanes each own a distinct row of the tile.
- `h` ∈ {0,1}: each thread covers 2 rows, 8 rows apart (the register pair d[4g+0]/d[4g+1]
  belongs to `h=0`, the pair d[4g+2]/d[4g+3] to `h=1`).

Laid out as a table for one warp (w=0), one column group (g=0), h=0 — showing which lane owns
which output element:

```
 Tile columns:    0   1   2   3   4   5   6   7        (group g=0, cols 8g .. 8g+7)
                ┌───┬───┬───┬───┬───┬───┬───┬───┐
 tile row 0     │L0 │L0 │L1 │L1 │L2 │L2 │L3 │L3 │   L0  = lane 0 (b=0)
                └───┴───┴───┴───┴───┴───┴───┴───┘   L1  = lane 1 (b=1)
 tile row 1     │L4 │L4 │L5 │L5 │L6 │L6 │L7 │L7 │   L2  = lane 2 (b=2)
                └───┴───┴───┴───┴───┴───┴───┴───┘   L3  = lane 3 (b=3)
 tile row 2     │L8 │L8 │L9 │L9 │L10│L10│L11│L11│   (lanes 4-7 own row 1, etc.)
                └───┴───┴───┴───┴───┴───┴───┴───┘
     ⋮
 tile row 7     │L28│L28│L29│L29│L30│L30│L31│L31│
                └───┴───┴───┴───┴───┴───┴───┴───┘
```

Every lane owns a 2-wide strip across 2 rows (h=0 and h=1) of one column group. The same
pattern repeats for every column group g=1,2,…,BN/8−1, with each lane's strip shifting right
by 8 columns.

### Why a straight store wastes 50%

Global memory is accessed in **32-byte sectors**. For BN=256 and the element's column
`col = 8g + 2b`, the 32 bytes of one sector span 16 consecutive half-precision elements
(32 B / 2 B per half = 16 elements = cols `16t .. 16t+15` for some tile offset `t`).

When we store group g without any reordering, a single warp issues 32 stores simultaneously —
one `half2` (4 B) per lane. In one row (say row 0), only 4 lanes (b=0..3, i.e. lanes 0–3)
contribute:

```
 Straight store of group g, row 0 of warp 0 only (4 lanes, 4 bytes each):

  col:  8g+0 8g+1 8g+2 8g+3 8g+4 8g+5 8g+6 8g+7 ║ 8g+8 8g+9 … 8g+15
        ┌────┬────┬────┬────┬────┬────┬────┬────╫────────────────────┐
        │ b=0│ b=0│ b=1│ b=1│ b=2│ b=2│ b=3│ b=3║  (empty — belongs ││
        └────┴────┴────┴────┴────┴────┴────┴────╫──  to group g+1)  ┘
        └──────── 16 B written ─────────────────╨─ 16 B untouched ──┘
        └────────────────── 32 B = one sector ──────────────────────┘
                               50% fill
```

The other 32 lanes (rows 1–7) are doing the same thing, each touching a different sector N×2
bytes away in global memory:

```
 ONE store instruction, 32 lanes.   row = 16w + l/4   (changes every 4 lanes)
                                    col = 8g + 2b      (b = l%4, 4 lanes span 8 halves = 16 B)

   row 0  [16B]·······································   lanes  0- 3   bytes     0..15
   row 1  ·······[16B]································   lanes  4- 7   bytes 16384..16399
   row 2  ··············[16B]·························   lanes  8-11   bytes 32768..32783
   row 3  ·····················[16B]··················   lanes 12-15
    ⋮                                                     (8 sectors, N×2 = 16 KB apart)
   row 7  ····································[16B]···   lanes 28-31

   128 useful bytes scattered over 8 sectors = 256 B of sector traffic ⇒ 50% wasted
```

The missing 16 bytes of each sector are the elements from **column group g+1** (cols
`8g+8..8g+15`). Those elements exist right now in the same warp's registers — but under the
accumulator layout they belong to the *next* loop iteration (group g+1), in a different
`d[i]`. The data is present; it just lives in the wrong register of the wrong lane.

### Why a lane exchange (not a wider store) is the right fix

A 4-wide store (`uint2`, 8 B per lane) over the same layout would still write only half a
sector: each lane would store cols `8g+2b` and `8g+2b+1` plus `8g+2b+2` and `8g+2b+3` — but
`8g+2b+2` is `b+1`'s territory, not data this lane holds. The problem is not the vector width;
it is that **adjacent columns of one row belong to adjacent lanes**, not adjacent registers of
the same lane.

The fix is a **lane exchange**: before storing, lanes `b` and `b^1` (i.e. 0↔1 and 2↔3) swap
column-pairs across groups g and g+1, so each lane ends up with 4 contiguous columns — one
full `uint2` worth of a single row, covering its 8-byte share of the 32-byte sector.

```
 One __shfl_xor between lanes b and b^1, swapping one column-pair across g and g+1:

  before   lane0 [0,1][8,9]   lane1 [2,3][10,11]   lane2 [4,5][12,13]   lane3 [6,7][14,15]
                  └─g─┘└g+1┘
  after    lane0 [0 1 2 3]    lane2 [4 5 6 7]      lane1 [8 9 10 11]    lane3 [12..15]
                  └── 8 B ──┘
           └──────────── 16 halves = 32 B = ONE FULL SECTOR ────────────────────────┘
```

Each lane now holds 4 contiguous halves (8 B) and the quad tiles **one full 32-byte sector**.

```
 Thread index → global memory layout after exchange
 (one row, one (g, g+1) pair; b = lane%4; c# = column offset from tile base):

 BEFORE exchange — each b holds two non-contiguous half-pairs, 8 columns apart:

   b=0  regs: ┌ a0=c0   a1=c1  ┐   ┌ e0=c8   e1=c9  ┐
   b=1  regs: ┌ a0=c2   a1=c3  ┐   ┌ e0=c10  e1=c11 ┐
   b=2  regs: ┌ a0=c4   a1=c5  ┐   ┌ e0=c12  e1=c13 ┐
   b=3  regs: ┌ a0=c6   a1=c7  ┐   ┌ e0=c14  e1=c15 ┐
               └─── group g ───┘   └─── group g+1 ───┘

 __shfl_xor(mask=1): b ↔ b^1  (0↔1, 2↔3)
   even b sends its e-pair, receives partner's a-pair:
   b=0 ──sends(c8, c9 )──► b=1     b=0 ◄──receives(c2, c3 )── b=1
   b=2 ──sends(c12,c13)──► b=3     b=2 ◄──receives(c6, c7 )── b=3

 AFTER exchange — each b holds 4 contiguous halves → one uint2 store (8 B):

   b=0  regs: [ c0  c1  c2  c3  ]  ──► uint2 store  ──► gmem cols  0 .. 3
   b=2  regs: [ c4  c5  c6  c7  ]  ──► uint2 store  ──► gmem cols  4 .. 7
   b=1  regs: [ c8  c9  c10 c11 ]  ──► uint2 store  ──► gmem cols  8 ..11
   b=3  regs: [ c12 c13 c14 c15 ]  ──► uint2 store  ──► gmem cols 12 ..15

 Global memory row (32 bytes = one L2 sector, one transaction):

  col:   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15
        [════════ b=0 ════════][════════ b=2 ════════][════════ b=1 ════════][════════ b=3 ════════]
         └──────────────────────────── 32 B, one full sector ──────────────────────────────────────┘
```

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
| 8192×8192×**2048** | 657 TF | **796** (+21%) | 27.5% → **13.1%** |
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

Overall **19 of 31 shapes at >=1.00x cuBLAS** (from 16), 26 at >=0.96x -- the state *after
this step*; sections 13-14 come later and the shipped figures are under Final performance.

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

### First, what the four levels actually are

Easy to conflate, and the rest of this section is unreadable without them:

| level | what it is |
|---|---|
| **CTA** | one `BM x BN` output tile of C. 384 threads = 1 producer + 2 consumer warpgroups. The unit the hardware puts on an SM. |
| **cluster** | `CLUSTER_M` CTAs stacked along M at the **same** `tile_n`. They consume the identical B tile, so one TMA multicast fills all of them -- the entire reason clusters exist here. **This is the unit of work assignment**: the scheduler hands out clusters, not CTAs. |
| **`w`** | linear index into the flat list of cluster-tiles, `total_ctiles = (tiles_m / CLUSTER_M) * tiles_n`. `decode(w)` yields `(tile_m, tile_n)`. |
| **group** | a band of the C tile grid, `cgroup_m` cluster-rows tall by **all** `tiles_n` columns, with `cgroup_m = GROUP_M / CLUSTER_M`. Purely a numbering device; no hardware knows it exists. It fixes the order of `w`. |

```
  group  >  cluster  >  CTA
  one group = cgroup_m * tiles_n clusters = cgroup_m * tiles_n * CLUSTER_M CTAs
```

The persistent loop is

```cpp
for (int w = my_cluster; w < total_ctiles; w += num_clusters)   // num_clusters = 132/2 = 66
```

so cluster 0 takes `w = 0, 66, 132...`, cluster 1 takes `1, 67, 133...`. **The consequence that
matters: at any instant the 66 resident clusters hold 66 *consecutive* values of `w`.** So the
order of `w` decides which tiles are co-resident, hence what hits in L2 -- and that ordering is
the only thing `GROUP_M` controls.

Concretely, for `GROUP_M = g` the concurrently-executing tiles form a `g x (SM/g)` rectangle of
the C grid. Those CTAs together need `g` A-tiles (one per M row) and `SM/g` B-tiles (one per N
column):

```
        g = 1                        g = 4
   ┌──┬──┬──┬──┬──┐            ┌──┬──┬──┐
   │  │  │  │  │  │  ...       │  │  │  │        1 A-tile, many B-tiles   (g=1)
   └──┴──┴──┴──┴──┘            ├──┼──┼──┤        vs
    one M row, many N          │  │  │  │        4 A-tiles, fewer B-tiles (g=4)
                               ├──┼──┼──┤
                               │  │  │  │
                               ├──┼──┼──┤
                               │  │  │  │
                               └──┴──┴──┘
```

Section 16 derives the optimum of that trade (`g* = sqrt(SM * BN / BM)`) and the condition
under which it stops applying.



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

`mma_half_cutlass.cu` builds the same contract from CUTLASS 4.7 `CollectiveBuilder`
primitives, as an independent check on whether the hand-written machinery earns its
complexity.

### The build flag that invalidated the first comparison

**CUTLASS must be compiled with `-DNDEBUG`.** Without it, its device-side asserts block
inlining; ptxas then cannot keep the wgmma group open across the resulting call boundary and
emits warning **C7510**. The SASS difference is stark -- at an identical 128x64x64 tile:

```
  OURS                                CUTLASS built WITHOUT -DNDEBUG
  WARPGROUP.ARRIVE                    WARPGROUP.ARRIVE
  HGMMA.64x64x16.F32  x4              HGMMA.64x64x16.F32  x1
  WARPGROUP.DEPBAR.LE gsb0, 0x1       WARPGROUP.DEPBAR.LE gsb0, 0x0
                      ^ 1 group in                        ^ FULLY DRAINED
                        flight                              after every wgmma
```

Four wgmma per commit group with one group in flight, versus one wgmma per group with a full
drain between each -- no overlap at all. Cost:

| | without `-DNDEBUG` | with |
|---|---|---|
| 4096³ | 776 | **869** |
| 1024³ | 131 | **289** |

I built it wrong, measured a 1.3x mainloop advantage that was mostly my own flag error, and
propagated it through a tile sweep, a K-scaling analysis and a written conclusion -- **while
having already quoted the C7510 warning that explained it**. A compiler warning naming the
exact mechanism is not background noise.

### Tuning

CUTLASS's device API fixes the tile per instantiation; there is no runtime tile heuristic.
Comparing our per-shape dispatcher against one fixed CUTLASS tile would measure our
dispatcher, so tile x cluster x schedule was swept (all figures with `-DNDEBUG`):

| config | 1024³ | 4096³ |
|---|---|---|
| 128x256x64 c2x1 | 143 | **857** |
| 128x256x64 c1x2 | 146 | 869 |
| **128x64x64 c1x1** | **289** | 374 |
| 128x64x128 c1x1 | 285 | 382 |
| 128x128x128 c1x1 | 225 | 579 |
| 64x64x64 c1x1 ping | 213 | 300 |
| 256x256x64 c2x1 | **7** | **34** |

A second overfit, caught late: `c1x2` looked 1.4% better than `c2x1` at 4096³ and shipped on
that basis -- then measured **1.6x worse** at 16384³ (429 vs 693). Same failure mode as the
first GROUP_M threshold: tuned on one shape, generalised. Reverted to `c2x1`.

`256x256x64` collapsing to 7 / 34 TFLOPS is worth flagging separately: `can_implement`
accepts it, so it builds and runs while being ~100x slow. A config sweep needs a sanity floor,
not just a success check.

### Alignment

CUTLASS validates TMA alignment against the problem **extent**, not the leading dimension, so
ragged shapes cannot be fixed by widening a stride -- they would need a zero-padded extent and
a copy back, i.e. more work than the kernel under test performs. Those four shapes are skipped
(`n/a`). `mma_half.cu` has no such restriction: TMA zero-fills out-of-range elements, so it
runs them in place.

### Result

| M × N × K | ours | CUTLASS | cuBLAS | ours/CUTLASS | ours/cuBLAS |
|---|---|---|---|---|---|
| 4k×4095×4k | **461** | n/a | 143 | — | **3.21×** |
| 4095×4095×4095 | **404** | n/a | 151 | — | **2.68×** |
| 2k×2047×2k | **271** | n/a | 135 | — | **2.01×** |
| 1k×1023×1k | **108** | n/a | 65 | — | **1.66×** |
| 16k×8k×128 | **320** | 319 | 300 | tie | **1.07×** |
| 8k×8k×128 | **301** | 297 | 283 | tie | **1.06×** |
| 16k×8k×4k | **733** | 718 | 704 | **1.02×** | **1.04×** |
| 8k×8k×16k | **721** | 711 | 697 | tie | **1.02×** |
| 4k×8k×128 | **266** | 258 | 260 | **1.03×** | **1.02×** |
| 8k×8k×8k | **687** | 681 | 672 | tie | **1.02×** |
| 16k×4k×8k | **686** | 681 | 672 | tie | **1.02×** |
| 8k×8k×4k | **732** | 721 | 720 | tie | **1.02×** |
| 16k×16k×16k | **656** | 654 | 659 | tie | **1.02×** |
| 8k×8k×2k | **746** | 735 | 740 | tie | **1.01×** |
| 32k×8k×2k | **720** | 693 | 716 | **1.04×** | **1.01×** |
| 4k×8k×512 | **578** | 628 | 577 | 0.92× | **1.00×** |
| 4k×8k×256 | 436 | 441 | 438 | tie | 1.00× |
| 4k×4k×1k | 644 | 622 | 647 | **1.03×** | 0.99× |
| 4k×8k×8k | 690 | 686 | 694 | tie | 0.99× |
| 8k×4k×8k | 712 | 709 | 716 | tie | 0.99× |
| 4k×4k×8k | 718 | 717 | 723 | tie | 0.99× |
| 4k×8k×1k | 690 | 659 | 697 | **1.05×** | 0.99× |
| 4k×4k×4k | 767 | 768 | 781 | tie | 0.98× |
| 8k×1k×8k | 781 | 789 | 797 | tie | 0.98× |
| 8k×8k×1k | 712 | 674 | 727 | **1.06×** | 0.98× |
| 2k×2k×2k | 589 | 593 | 607 | tie | 0.97× |
| 1k×1k×1k | 214 | 206 | 220 | **1.04×** | 0.97× |
| 4k×512×4k | 514 | 533 | 530 | 0.97× | 0.97× |
| 2k×2k×512 | 298 | 319 | 326 | 0.94× | 0.91× |
| 3000×1000×2000 | 391 | 412 | 458 | 0.95× | 0.85× |
| 384×2k×2k | 235 | 237 | 279 | tie | 0.84× |

**31 shapes — ours ≥ cuBLAS on 16; ours ≥ CUTLASS on 18 of 27 supported.**

The honest reading is **parity**: 16 of the 27 comparable rows are statistical ties and are
labelled as such. `1024^3` (1.06x) and a handful of large shapes lead, and
CUTLASS takes the thin-K end (`2k x2k x512` 0.86x, `4k x8k x512` 0.89x, `4k x512x4k` 0.94x)
-- section 14 explains why. It is bit-exact against cuBLAS on every shape it runs.

### Two measurement choices that mattered more than any optimization here

**1. Operand data is worth up to 19%.** Same kernel, same shape (8192x4096x8192):

| operand data | TFLOPS |
|---|---|
| all zero (`memset 0`) | **906** |
| `0x11` constant | 809 |
| uniform, 200 levels | 833 |
| uniform, full 16-bit entropy | **761** |
| normal(0, 0.05), NN-weight-like | **759** |

Zero operands draw far less dynamic power in the tensor cores, so the GPU holds higher
clocks. Every `memset`-based benchmark in this project was measuring easier data than a real
workload. Full-entropy uniform lands within 0.3% of NN-weight-like normal data, so that is
what the harness now uses; absolute figures dropped ~6% from the `0x11` numbers this report
previously carried, and 4096^3 went 864 -> 730.

This also invalidated a side experiment: a per-shape config probe that `memset` its inputs to
zero read CUTLASS at 936 TFLOPS on a shape where the harness measured 802. The probe was
measuring compression and clock headroom, not the kernel.

**2. Sequential A-then-B timing does not cancel clock drift.** On large shapes both kernels
swing +/-16% run to run *together*. At 8192x4096x8192, sequential timing reported **1.10x**
for a pair that measures **1.01x** interleaved -- and the same correction applied to
`4096x8192x8192` (1.13x -> 1.01x) and `8192x8192x4096` (1.09x -> 1.01x). Essentially every
large-shape margin this repo claimed was that artifact. The harness now round-robins the
three kernels within each repetition.

Note the correction runs both ways: interleaving *raised* `8192x8192x1024` from 1.04x to
1.11x. Sequential timing was not biased in our favour by design, it was simply noise being
read as signal.

**2b. Form the ratio per run, then reduce.** `median(ours) / median(theirs)` is not the same
as `median(ours/theirs)`, and only the second respects the pairing that interleaved timing
exists to create. Taking column medians independently lets an outlier in one column meet a
different run's value in the other:

```
  16384x4096x8192   per-run ratios 1.042, 1.007, 1.006
                    median(o)/median(c) = 1.042   <- one run's outlier, promoted to a "win"
                    median(o/c)         = 1.007   <- tie
```

Two rows of the table had their verdict changed by this, including one that had been reported
as a 1.04x win and is a tie. Cheap to get wrong, because both expressions look like "the
median ratio".

**3. One shape per process.** Interleaving was still not enough. Measuring all 31 shapes in
one process leaves the GPU throttled by the time it reaches the later ones, and kernels
degrade differently under throttle. `8192x8192x1024` read **1.08-1.15x** in a full run --
and three independent repeats *all agreed* -- against **1.01-1.05x** in a fresh process.
Agreement across repeats is not evidence when the error is systematic; only changing the
measurement setup exposed it. The harness now takes a shape index and `make cutlass` drives
one process per shape.

That correction cost the last large-shape claim: with isolation, `1024^3` was the only row
clearly exceeding 5%.

**4. Flush L2 before every timed launch.** H100's L2 is 50 MB, so any shape whose working set
(A+B+C) fits inside it was being measured entirely from cache -- which flatters small shapes,
and only small shapes:

| shape | working set | hot L2 | cold L2 |
|---|---|---|---|
| 1024^3 | 6 MB | 1.13x | **1.08x** |
| 2048x2048x512 | 12 MB | 0.91x | **0.86x** |
| 384x2048x2048 | 11 MB | 1.04x | **0.99x** |
| 4096x4096x4096 | 96 MB | 1.02x | 1.01x |
| 8192x8192x8192 | 384 MB | 1.00x | 1.00x |

The cutoff tracks the 50 MB L2 exactly. Cold is the conservative choice -- it models a GEMM
called once inside a larger pipeline rather than a tight loop over resident operands -- and is
what `nvbench` and the CUTLASS profiler offer. The harness now evicts L2 with a 150 MB
memset before each timed launch and times a single launch per sample (batching launches
inside one event window would leave all but the first hot, defeating the point), 25 samples
per kernel.

Absolutes drop accordingly: 1024^3 reads 221 TFLOPS cold against 323 hot. Our one surviving
win shrank from 1.13x to 1.08x, and `2048x2048x512` got worse rather than better -- the
correction is not uniformly against us, it is against *small* shapes in both directions.

**Four corrections, one direction.** Operand data (19%), sequential timing (10%),
single-process sweeps (10%), hot L2 (5% at small shapes). Each was individually larger than
any kernel optimization in this report, and every one had been inflating our numbers. The
kernel work in sections 1-11 is real; the margins claimed for it were mostly measurement.

### Giving CUTLASS the same tuning freedom

Our kernel picks from a three-rung tile ladder (BN 256/128/64) plus a per-shape GROUP_M
policy. For most of this report CUTLASS had **two** fixed tiles and no swizzle, which is not
a like-for-like test. It now gets seven configs -- 128x256 / 128x128 / 128x64, clustered and
not, cooperative and pingpong -- swept per shape alongside the swizzle, best kept.

The effect on the shapes we claimed by >5%:

| shape | two-tile + no swizzle | full ladder | CUTLASS best config |
|---|---|---|---|
| 4096x8192x512 | 1.06x | **0.90x** | 128x128x64 c1x1 **pingpong** sw2 |
| 4096x8192x256 | 1.15x | **0.99x** | 128x128x64 c1x1 **pingpong** sw8 |
| 8192x8192x128 | 1.07x | **1.00x** | 128x128x64 c1x1 **pingpong** sw2 |
| 16384x8192x128 | 1.02x | **0.96x** | 128x128x64 c1x1 **pingpong** sw1 |
| 8192x8192x2048 | 1.11x | 1.02x | 128x256x64 c2x1 sw8 |
| 4096x4096x8192 | 1.08x | 1.04x | 128x256x64 c2x1 sw8 |

Six of ten flipped. `128x128x64` **pingpong** is the best CUTLASS config on essentially every
thin-K shape -- a schedule dismissed earlier in this report on the strength of a single
measurement at 1024^3, where it happens to be 5.5% worse. That is the third time tuning on
one shape produced a wrong general conclusion here (see also the c1x2 cluster and the first
GROUP_M threshold).

Score across the three harness generations: **20/27 -> 17/27 -> 14/27**. Every correction
moved the same direction, which is what a systematically favourable setup looks like.

Two harness bugs found while building the sweep, both of which would have manufactured wins:

1. The correctness check inherited the *previous* shape's winning config. When that config
   could not implement the new shape, `launch()` returned false, `C` was left as memset, and
   the shape reported a spurious FAIL.
2. A config rejected by `can_implement` costs ~0 to "run", so it would win the sweep with a
   fabricated throughput. The sweep now checks the return value before timing.

Discovery is two-stage (config at swizzle 1, then swizzle for the winner): 11 candidates
rather than 28. The full cross-product is ~224 launches, enough to heat the GPU and depress
what follows -- 4096^3 read 825 for us against its 863 baseline. Every harness change is
validated by confirming ours and cuBLAS still reproduce their previous figures while only
the CUTLASS column moves.

### The tile scheduler, and a comparison that was not fair

CUTLASS's persistent tile scheduler takes `max_swizzle_size`, and it **defaults to 1** -- no
threadblock swizzle at all. Comparing that against a kernel with a GROUP_M rasterization
policy is not a like-for-like test, and the cost is not small:

| 128x256x64 c2x1 @ 16384^3 | TFLOPS |
|---|---|
| default (`max_swizzle_size = 1`) | 690.7 |
| swizzle 2 | 846.9 |
| swizzle 4 | 849.4 |
| AlongM / AlongN, swizzle 4-8 | 849.0-849.1 |

Interleaved medians, tight spreads. **That one setting was the whole of what previously read
as a 1.22x win for us at 16384^3; the fair figure is 1.00x.** It also inflated
`8192x8192x4096` (1.19x -> 1.06x), `8192x4096x8192` (1.17x -> 1.07x) and
`4096x8192x8192` (1.18x -> 1.09x). Every large-shape margin was mostly this artifact. The
small and short-K wins survive unchanged, which is consistent -- swizzle does nothing there,
and section 12's 1024^3 analysis was about constant-load volume, not L2.

Two failed attempts before the fix that stuck, both worth recording:

1. **A fixed cap is an overfit in the other direction.** `max_swizzle_size = 8` everywhere
   costs -22% on `4096x512x4096` and -38% on `384x2048x2048`, which have 2 and 3 tiles in
   one dimension -- swizzling scrambles a grid that was already fine.
2. **A hand-written policy for someone else's kernel is guesswork.** Gating on
   `tiles_m >= 8 && tiles_n >= 8` still cost `4096x512x4096` 27%, because that shape takes
   the *Small* 128x64 tile and so passes the gate on a grid that does not want swizzling.

So the harness sweeps `{1,2,4,8}` per shape and measures with the winner -- what CUTLASS's
offline profiler does, and what "tuned" should have meant from the start. The chosen value
appears in the `sw=` column of `make cutlass`.

Getting the sweep to not corrupt the measurement took two more iterations, both instructive:
a full-weight discovery pass left the GPU hot enough to cost **cuBLAS 15%** at 4096^3
(877 -> 763), and "fixing" that with a 400 ms settle was worse -- at 1024^3 the measurement
is microseconds, clocks never re-boost after an idle gap, and cuBLAS read **191 instead of
308**. The working recipe is a *cheap* discovery pass (~28 launches vs ~260) and no sleep.
Validation is that ours and cuBLAS reproduce their previous figures to within noise while
only the CUTLASS column moves.

---

## 13. Hoisting the wgmma descriptor out of the k16 loop

Profiling the mainloop showed 13.4 SASS instructions between consecutive `wgmma` issues
against CUTLASS's 7.9. The gap was the descriptor: the inner k16 loop called `make_desc()`
and `make_desc_mn()` per issue, rebuilding the whole 64-bit field each time.

Only bits `[13:0]` (`addr>>4`) vary with the k16 step, and the step is a compile-time
constant, so the rebuild is unnecessary:

```
  make_desc(a_base + ks*32)       ==  make_desc(a_base)    + ks*2
  make_desc_mn(b_base + ks*2048)  ==  make_desc_mn(b_base) + ks*(MN_K_STRIDE/16)
```

One base descriptor per operand per k-tile, plus an immediate add per issue. SASS confirms
it -- the base lives in a uniform register pair and each issue is a single 64-bit add:

```
  UIADD3.64 UR26, UR22, 0x80,  URZ
  HGMMA.64x256x16.F32 R24, gdesc[UR24].tnspB, R24
  UIADD3.64 UR26, UR22, 0x100, URZ
  HGMMA ...
```

| BN=256 wgmma region | per-issue rebuild | hoisted |
|---|---|---|
| `ULOP3` (field construction) | 26 | **8** |
| `USHF` (shifts) | 17 | **7** |
| **`ULDC`** | **0** | **0** |
| region size (instructions) | 214 | **159** |

**Note what this deliberately does *not* do.** CUTLASS solves the same problem by keeping
parameters in its 1280-byte `Params` block and re-reading them through `ULDC`, costing it
2.25x the uniform constant loads and 3.3x the `imc_miss` stalls. Hoisting into registers is
the good half of that idea without the constant traffic: whole-kernel `ULDC` is unchanged at
142, and zero in the wgmma region either way. A descriptor contains a runtime shared-memory
address, so it could not live in constant memory even if we wanted it to.

Measured with interleaved A/B (5 pairs, medians), because the first non-interleaved run
produced a **+6.2% phantom** on `16384x8192x4096` that became -0.7% once drift was cancelled:

| shape | rebuild | hoisted | delta |
|---|---|---|---|
| 1024x1024x1024 | 320 | 325 | **+1.6%** |
| 4096x8192x256 | 466 | 474 | **+1.7%** |
| 2048x2048x512 | 361 | 363 | +0.6% |
| 4096x4096x4096 | 862 | 863 | +0.1% |
| large shapes (8k-16k) | -- | -- | inside +/-50-100 TF noise |

The win concentrates where the mainloop is a large fraction of a short kernel and vanishes
where the tensor pipe is already saturated. Correctness unchanged: 57 shapes, 0 failures,
knob on and off. Default on; `-DCFG_HOIST_DESC=0` restores the rebuild.

---

## 14. Root cause of the shapes we still lose

`2048x2048x512` is our worst result against CUTLASS. Both dispatchers select the *same*
128x256x64 tile with a 2-CTA M-cluster, so this is not a tiling artifact.

**The K-sweep separates fixed cost from per-iteration cost.** At fixed M=N=2048:

| K | k-tiles | ours | CUTLASS | ratio | ours - CUTLASS |
|---|---|---|---|---|---|
| 128 | 2 | 125.3 | 138.9 | 0.90x | 0.84 us |
| 512 | 8 | 360.6 | 385.6 | 0.94x | 0.77 us |
| 2048 | 32 | 681.7 | 700.4 | 0.97x | 0.67 us |
| 8192 | 128 | 807.2 | 807.4 | 1.00x | 0.02 us |

The *ratio* converges but the *absolute* gap is flat at ~0.78 us across two decades of K --
a fixed per-tile cost, the exact inverse of the 1024^3 case where the ratio is flat and the
cost is per-iteration.

**Scaling tiles-per-CTA localises it.** At fixed K=512:

| M=N | tiles | tiles/CTA | ours | CUTLASS | delta |
|---|---|---|---|---|---|
| 2048 | 128 | 1.0 | 11.95 us | 11.16 us | **+0.79** |
| 4096 | 512 | 3.9 | 31.80 us | 32.73 us | **-0.93** |
| 8192 | 2048 | 15.5 | 105.62 us | 113.52 us | **-7.90** |

The deficit does not scale with tiles -- it *reverses*. Fitting:

```
  delta  =  1.38 us  -  0.59 us x (tiles per CTA)          break-even ~2.3 tiles/CTA
  predicted at 15.5 tiles/CTA: -7.81 us     measured: -7.90 us
```

**We carry a ~1.38 us fixed per-CTA startup cost against a ~0.59 us per-tile advantage.**
That is the persistent design's trade: front-load work into the prologue, amortise it over
many tiles. At exactly one wave there is nothing to amortise against.

One constant predicts every shape CUTLASS beats us on: `384x2k x2k` (0.18 tiles/CTA),
`2k x2k x512` (0.97), `2k x2k x2k` (0.97), `8k x1k x8k` (1.9) sit below 2.3; `4k^3` (3.9)
and `8k^3` (15.5) sit above, and we win.

Ruled out by intervention: pipeline fill (stages 3 vs 4 is a dead heat at short K) and TMA
descriptor re-encoding per launch (0.140 us, 10% of the gap). **Not decomposed:** the
remaining ~1.2 us -- candidates are first-stage TMA fill latency and per-CTA setup.

**The actionable item:** the dispatcher already computes tile counts, so it could detect
`tiles/CTA < ~2.3` and take a lighter path. Worth ~6% on the affected shapes, but blocked on
decomposing that 1.2 us first -- fixing the wrong half would be another null result. The
methodology is written up in the `gpu-perf-root-cause` skill.

---

## 15. L2 residency control on the TMA descriptors

Which operand deserves to stay in L2 is a property of the shape, not the kernel. PTX exposes
this directly: `createpolicy` builds a cache policy and `cp.async.bulk.tensor` accepts it via
`.L2::cache_hint`, so each TMA can state its intent. In SASS the descriptor shows up as an
extra operand:

```
  without:  UTMALDG.2D [UR12], [UR10]
  with:     UTMALDG.2D [UR12], [UR10], desc[UR6]
```

The shipped policy pins the smaller operand as `evict_last` when it is `<= 8 MB` **and**
`>= 4x` smaller than the other, and marks C's stores `evict_first` so the 500 MB write stream
cannot displace it. Everything else emits the plain instruction.

**Every intuitive version of this was wrong, and measurably so.**

| design | result |
|---|---|
| pin reused operand, mark the other `evict_first` | **-16%** on `4096x8192x512` |
| as above but non-pinned gets explicit `evict_normal` | **-27%** on `1024^3` |
| descriptor **only** on the pinned operand, plain instructions elsewhere | **+1.5%** median |

Two lessons in that table. Marking an operand `evict_first` destroys reuse that was happening
for free -- shapes where *no* operand qualified for pinning still lost 6-8%, and those tag
nothing as resident. And an explicit `evict_normal` policy is **not** equivalent to no policy:
carrying the descriptor at all costs something, enough to lose 27% at `1024^3`.

Validation across five shapes that satisfy the gate, plus a control whose code path is
provably identical in both builds:

| shape | A/B | ratio | delta | ranges |
|---|---|---|---|---|
| 4096x512x4096 | 32/4 MB | 8x | **+2.1%** | non-overlapping |
| 16384x256x8192 | 256/4 MB | 64x | **+1.6%** | non-overlapping |
| 2048x8192x256 | 1/4 MB | 4x | +1.5% | overlap |
| 4096x256x4096 | 32/2 MB | 16x | +1.2% | overlap |
| 8192x512x8192 | 128/8 MB | 16x | -0.1% | overlap |
| 4096x4096x1024 (control) | 8/8 MB | 1x | -0.3% | overlap |

Median +1.5%, control -0.3%. The null case is `8192x512x8192`, whose 8 MB operand sits exactly
at the cap -- the cap is probably still too loose, but moving it on one observation is how the
other three tuning mistakes in this report happened.

**It does nothing on the shape that motivated it.** `32768x8192x2048` is -0.2%: B is 32 MB,
already L2-resident there because the N-major rasterization sweeps it once per M-row, so there
was never headroom. The wins came from shapes nobody was looking at.

### The mechanism is not established

The obvious explanation -- the small reused operand is evicted by the large streaming operand
between reuses, and `evict_last` prevents that -- is **refuted**. Holding the pinned operand at
4 MB and sweeping the streaming one:

| streaming operand | A+B vs 50 MB L2 | delta |
|---|---|---|
| 8 MB | fits | -2.4% *(control: ratio 2x, gate does not fire)* |
| 16 MB | fits | -0.3% |
| **32 MB** | **fits, 36/50** | **+1.9%** <- peak |
| 64 MB | 1.4x over | +0.6% |
| 128 MB | 2.6x over | +0.5% |
| 256 MB | 5.2x over | +0.5% |

If eviction pressure were the cause the gain would grow with the streaming operand. It peaks
where the whole working set still *fits* in L2 and shrinks as pressure rises. No story worth
asserting fits that curve, and the control row puts this run's noise floor at 2.4 pp, against
which most of these deltas are marginal.

So: the effect is reproducible in low-noise runs (`4096x512x4096`, non-overlapping ranges over
15 alternations, +2.1%), it is small, and **"L2 residency" is a description of the policy, not
a demonstrated explanation of the gain**. Settling it needs `lts__t_sector_hit_rate` and
`dram__bytes_read.sum` per build: if DRAM reads do not drop with the hints on, the win is
coming from somewhere other than cache residency and the optimization is misnamed.

---

## 16. The cluster decision, and a correction to what GROUP_M was doing

**This section previously claimed that `GROUP_M=1` wins on short-K and small-grid shapes
because rasterizing straight down N is better there. That was wrong.**

`GROUP_M=1` was not selecting a rasterization. It was silently switching off the 2-CTA
cluster. The old condition read

```cpp
use_cluster = (CM > 1) && (tiles_m % CM == 0) && (group_m % CM == 0);
```

so any **odd** GROUP_M failed `group_m % CM == 0` and took the non-clustered launch, giving up
the TMA multicast of B along with it. The "+19% from GROUP_M=1" was the cluster being
disabled, and the two knobs were never independent.

Isolating the cluster properly -- `GROUP_M=1` (no cluster) against `GROUP_M=2` (cluster, same
`cgroup_m=1` grouping), then confirmed with a build-level `CFG_CLUSTER_M=1` at matched
M-grouping:

| shape | K | tiles_m | tiles_n | cluster |
|---|---|---|---|---|
| 2048x2048x512 | 512 | 16 | 8 | **-16.0%** |
| 2048x2048x2048 | 2048 | 16 | 8 | **-8.4%** |
| 4096x8192x128 | 128 | 32 | 32 | -7.0% |
| 8192x8192x128 | 128 | 64 | 32 | -7.0% |
| 4096x8192x256 | 256 | 32 | 32 | -6.7% |
| 4096x4096x4096 | 4096 | 32 | 16 | -0.6% |
| 16384x16384x16384 | 16384 | 128 | 64 | 0.0% |
| 8192x8192x4096 | 4096 | 64 | 32 | **+3.6%** |
| 32768x8192x2048 | 2048 | 256 | 32 | **+7.6%** |
| 8192x1024x8192 | 8192 | 64 | 4 | **+27.8%** |

The cluster's cost is **per tile** -- cluster launch, `cluster_sync`, mbarriers that must
collect `CONSUMERS * CLUSTER_M` arrivals. Its benefit -- halving B's L2->SM traffic by
multicasting one B tile to both CTAs -- is **per k-tile**, and needs enough M-tiles for that
tile to be reused. Hence the shipped rule:

```
  cluster  iff  (K >= 2048 and tiles_m >= 32)  or  tiles_n <= 4
```

K alone does not separate: at K=2048 the cluster is +7.6% on `32768x8192x2048` (tiles_m=256)
and -8.4% on `2048x2048x2048` (tiles_m=16). The narrow-N override exists because with few
N-tiles B is re-read once per M-row, so halving that traffic pays regardless of K --
`3000x1000x2000` (tiles_n=4) loses **14.6pp** against CUTLASS without it.

### Conclusion: CGA is a bandwidth optimisation -- but the threshold does not predict

**Read the held-out test above first.** The arithmetic in this subsection is exact; the
predictive claim built on it is not.

For this kernel -- 128x256x64 work quantum, `CLUSTER_M=2` -- enabling CGA is **fundamentally a
bandwidth optimisation, not a compute one**. Its only benefit is that two CTAs share one B tile
through TMA multicast, which cuts B-side L2->SM traffic by **50%** but total L2 traffic by only
**1/3**, because each CTA still loads its own A:

```
  per CTA per k-tile:   A = BM x BK = 16 KB      B = BK x BN = 32 KB      total 48 KB
  multicast halves B:   16 KB saved of 48  ->  33% of a CTA's L2 loads
```

The cost is **unconditional**. Clusters are GPC-co-resident, launched and retired as a unit,
and synchronised for multicast -- `release_stage` issues `CLUSTER_M` remote mbarrier arrivals
*per k-tile* and `bar_empty` collects `CONSUMERS * CLUSTER_M` of them.

So **CGA only pays when the unclustered kernel is already L2-read-bandwidth-bound.** On H100
that crossover is ~7 TB/s of L2 read demand. Because L2 traffic per FLOP is a property of the
tile alone -- `(BM+BN)/(BM*BN)`, with M, N and K cancelling -- demand is proportional to
throughput, and the crossover has a per-tile throughput form:

| tile | B/FLOP | FLOP/B | crossover | cluster legal? |
|---|---|---|---|---|
| 128x256x64 | **0.01171875** | 85 | **~597 TFLOPS** | yes (BLOCKS=4) |
| 128x128x64 | 0.015625 | 64 | ~448 TFLOPS | yes (BLOCKS=2) |
| 128x64x64 | 0.0234375 | 43 | ~299 TFLOPS | **no** -- BLOCKS=1, a cluster would starve |

Below the crossover CGA usually *hurts*, because the scheduling and synchronisation overhead is
not offset by any meaningful bandwidth saving.

### Recommendation

```
  predicted L2 read demand  ~=  unclustered FLOP/s  x  (BM + BN) / (BM * BN)

  demand < ~7 TB/s   ->  do NOT enable CGA
  demand > ~7 TB/s   ->  consider CGA, especially if profiling shows L2->SM as the
                         dominant bottleneck
```

For the 128x256x64 tile that is simply `TFLOPS x 0.01171875`, so in practice: **below ~600
TFLOPS unclustered, CGA is usually a bad idea; above it, CGA is more likely to help.**

Note the threshold is tile-independent in TB/s and tile-*dependent* in TFLOPS -- 597 at
BN=256 but 448 at BN=128 -- so state it in bandwidth, not throughput, when moving between
tiles.

In short: **CGA is bad when the kernel is not yet L2-bandwidth-bound, because you pay the
cluster scheduling and synchronisation cost without getting enough multicast benefit back.**

The constants are derived per tile in `Cfg::L2_BYTES_PER_FLOP` and `Cfg::CGA_CROSSOVER_FLOPS`
rather than hard-coded, so they follow the tile if it changes.

### Why the cluster pays only on some shapes: the L2 bandwidth wall

The rule above is three empirical clauses. There is one explanation underneath all of them.

The cluster's **cost is unconditional**: CTAs give up scheduling freedom. A cluster must be
co-resident within one GPC and is launched and retired as a unit, so the machine holds 66
independent clusters instead of 132 independent CTAs. That is paid on every shape.

Its **benefit is conditional**: multicast halves B's L2->SM traffic, which only helps if L2
bandwidth is the binding constraint. Tabulating the demand -- every CTA reads its own A and B
tile every k-tile, so `L2 bytes = CTAs * k_tiles * (BM + BN) * BK * 2` -- against the measured
cluster effect:

| L2->SM demand | shape | cluster |
|---|---|---|
| 3.17 TB/s | 4096x8192x128 | -7.0% |
| 3.58 TB/s | 8192x8192x128 | -7.0% |
| 3.69 TB/s | 2048x2048x512 | **-16.0%** |
| 5.24 TB/s | 4096x8192x256 | -6.7% |
| 5.94 TB/s | 8192x8192x8192 | -0.2% |
| 6.23 TB/s | 4096x4096x4096 | -1.2% |
| 6.46 TB/s | 4096x8192x512 | -0.4% |
| 6.98 TB/s | 2048x2048x2048 | -8.4% |
| **~7 TB/s -- H100 sustained L2 read bandwidth** | | |
| 7.54 TB/s | 8192x1024x8192 | +0.9% |
| 7.64 TB/s | 8192x8192x4096 | +2.6% |
| 7.77 TB/s | 32768x8192x2048 | **+10.8%** |
| 8.06 TB/s | 16384x16384x16384 | +1.1% |

**Twelve of twelve on the shapes it was built from -- and five of eleven on held-out shapes.**

Tested afterwards against the nine benchmark shapes whose cluster effect had never been
measured, plus two knowns as controls, it fails:

| shape | L2 demand | predicted | measured | |
|---|---|---|---|---|
| 3000x1000x2000 | 3.49 T | no cluster | **+11.1%** | MISS |
| 4096x512x4096 | 7.50 T | cluster | -4.1% | MISS |
| 4096x4096x4096 | 7.43 T | cluster | -1.0% | MISS |
| 32768x8192x2048 | 7.96 T | cluster | **-2.9%** | MISS |
| 8192x8192x2048 | 8.88 T | cluster | -0.4% | MISS |
| 4096x4096x8192 | 9.29 T | cluster | -0.3% | MISS |
| 8192x8192x16384 | 8.28 T | cluster | +3.2% | ok |
| 8192x4096x8192 | 8.93 T | cluster | +0.8% | ok |
| 16384x8192x4096 | 8.72 T | cluster | +0.4% | ok |
| 4096x8192x8192 | 8.97 T | cluster | +0.4% | ok |
| 16384x4096x8192 | 8.76 T | cluster | +0.2% | ok |

So the clean 12/12 separation above was **fitted**, not predictive, and this section previously
presented it as a mechanism. It is not one.

**And the measurements underneath it do not reproduce.** `32768x8192x2048` -- the strongest
single data point for clustering -- measured **+10.8%** in one session and **-2.9%** in
another, same shape, same method (`CFG_CLUSTER_M=1` vs `2`, process-alternated, 5 rounds,
GROUP_M=16). `CM=2` read 734.3 then and 659.9 now: an 11% swing in one configuration, which is
larger than the effect being attributed. `CLUSTER_M` is compile-time, so cluster on/off cannot
be interleaved in-process, and process-level alternation is evidently not enough.

What survives: the shipped cluster rule's *net* benefit was demonstrated in-harness with paired
per-run ratios against CUTLASS (16/31 and 18/27, up from 14/31 and 16/27), which is a more
reliable measurement than the cross-binary cluster deltas. The per-shape attribution -- which
shapes the cluster helps and why -- does not currently rest on reproducible ground, and the
bandwidth story below should be read as a plausible account rather than an established one.

The original observation still holds in the weak form: the threshold coincides with H100's
sustained L2 read bandwidth rather than being tuned to fit. Below it there is headroom, so halving B's traffic buys nothing and the GPC
constraint is paid for free. Above it, L2 is the wall and multicast is the only thing that
moves it.

There is a tidy consequence. For a fixed tile the L2 traffic per FLOP is **shape-independent**:

```
  L2 bytes = M*N*K * 2 * (BM + BN) / (BM * BN)        FLOPs = 2*M*N*K
  bytes/FLOP = (BM + BN) / (BM * BN) = 384 / 32768 = 0.0117    (85 FLOP/byte)
```

so L2 bandwidth demand is simply proportional to achieved throughput. "Above 7 TB/s" is
exactly "above ~600 TFLOPS", and the data agrees: every shape the cluster helps runs at
643-688 TFLOPS unclustered, every shape it hurts runs at 270-595.

Which means **the cluster pays only on shapes already fast enough to saturate L2** -- circular
for a runtime dispatcher, since throughput is not known before running. The shipped
`(K >= 2048 and tiles_m >= 32) or tiles_n <= 4` rule is best read as a *proxy* for "will this
shape reach the L2 wall". It also retrospectively justifies the `tiles_m >= 32` clause, which
had no mechanism attached: small grids never reach that throughput.

The same explanation covers `CLUSTER_M = 4` without needing the synchronisation arithmetic at
all. Halving the independent clusters (66 -> 33) costs scheduling freedom on *every* shape,
while the extra quarter of B traffic saved only helps at the wall -- and 33 clusters cannot
keep the machine fed even there.

`use_cluster` no longer looks at `group_m` at all, and `cgroup_m` rounds down rather than
rejecting odd values, so the two knobs are finally independent. The GROUP_M default also moves
8 -> 16, matching the derivation below.

Effect on the table: 16/31 at or above cuBLAS (was 14) and 18/27 at or above CUTLASS (was 16),
with `4096x4096x1024` +4.6pp, `4096x8192x1024` +4.4pp, `8192x8192x1024` +3.5pp,
`4096x8192x256` +2.9pp and `4096x8192x512` +2.8pp.

**What this says about the previous section.** Every number in the old section 16 was measured
correctly; the *attribution* was wrong, and it was wrong in a way no amount of re-measuring
would have caught -- only reading the launch path did. A knob that silently changes two things
is worse than two knobs.

### Why grouping helps at all, and why 16

Write out the DRAM traffic for a tiled GEMM under GROUP_M=g. With SM concurrent CTAs the
resident window is `g` M-tiles by `SM/g` N-tiles, so each A tile is re-fetched `tiles_n/(SM/g)`
times and each B tile `tiles_m/g` times. C is written once and its resident footprint is
`SM * BM * BN * 2` -- **both independent of g**, so C never moves the optimum, even though it
is up to 98% of the total bytes on short-K shapes (which is itself why grouping cannot buy much
there).

```
  traffic(g) = A·tiles_n·g/SM  +  B·tiles_m/g  +  C

  dt/dg = 0  ->  g* = sqrt(SM · B · tiles_m / (A · tiles_n))
               = sqrt(SM · (K·N)(M/BM) / ((M·K)(N/BN)))
               = sqrt(SM · BN / BM)          <- M, N and K all cancel
```

**The optimum is shape-independent.** For BM=128: `sqrt(132 * 256/128) = 16.2` at BN=256, and
`sqrt(132 * 64/128) = 8.1` at BN=64. That is exactly what the sweep measures wherever the
model's premise holds -- i.e. wherever re-fetching actually happens:

| shape | BN | g* | measured best |
|---|---|---|---|
| 8192x8192x1024 | 256 | 16.2 | **16** |
| 4096x4096x4096 | 256 | 16.2 | **16** |
| 8192x8192x4096 | 256 | 16.2 | **16** |
| 32768x8192x2048 | 256 | 16.2 | **16** |
| 384x2048x2048 | **64** | **8.1** | **8** |

Five for five, including the one shape on a different tile.

### The residency criterion: A+B, not A+B+C

Where the model falls silent is sharp and worth stating as a rule. **Every shape whose optimum
is `g=1` has `A + B <= 16 MB` against a 50 MB L2 -- five for five, no exceptions:**

| shape | A+B | A+B+C | best g |
|---|---|---|---|
| 2048x2048x512 | **4 MB** | 12 MB | 1 |
| 4096x8192x128 | **3 MB** | 67 MB | 1 |
| 8192x8192x128 | **4 MB** | 132 MB | 1 |
| 16384x8192x128 | **6 MB** | 262 MB | 1 |
| 2048x2048x2048 | **16 MB** | 24 MB | 1 |
| 4096x4096x4096 | 64 MB | 96 MB | 16 |
| 8192x8192x4096 | 128 MB | 256 MB | 16 |
| 32768x8192x2048 | 160 MB | 672 MB | 16 |
| 16384x16384x16384 | 1024 MB | 1536 MB | 4 |

Both operands are co-resident, so nothing is ever re-fetched, so there is no re-fetch traffic
for rasterization to reduce. That is the root cause, stated in one line.

**C must be excluded from the criterion.** C is write-once -- it never needs to be resident, it
just streams out. Three of the five `g=1` shapes carry 64-256 MB of C and obviously do not fit;
testing `A+B+C` would wrongly predict grouping for all three.

**It is necessary, not sufficient.** Six shapes also have `A+B <= 32 MB` and still prefer
grouping (`4096x8192x256` at 6 MB, `4096x8192x512` at 12 MB, `3000x1000x2000` at 15 MB,
`8192x8192x1024` at 32 MB). Scored as a predictor it gets 9/15, against 10/15 for the trivial
"always group" baseline. So `A+B <= L2` explains **why grouping cannot help** on the shapes it
covers; it does not predict where grouping still helps anyway. The likely reason is that
*fitting* is not *staying*: C pushes 64-160 MB of writes through the same L2 and evicts them,
so real residency depends on reuse distance, not just operand size.

Where it fails is where the premise fails: if A+B fits in L2 there is no re-fetching to
optimise, traffic stops depending on g, and the model is silent. All five GM=1 shapes have
A+B <= 16 MB against a 50 MB L2. That is *not* a sufficient rule -- `4096x8192x256` (6 MB) and
`4096x8192x512` (12 MB) also fit and still want GM=4 -- so the honest scope is: **the model
predicts the optimum wherever re-fetching occurs, and says nothing where it does not.** The
two shipped arms cover part of the region where it says nothing, empirically.

### On the magnitude

The `+19%` figure is setup-dependent and should be read as **+17..20%**:

```
  2-way rotation {1,4}        GM=1 317   GM=4 265   +19.9% / +19.6%
  blocked, no interleaving    GM=1 341   GM=4 291   +17.0% / +17.9%
  5-way {1,2,4,8,16}          GM=1 307   GM=4 297   +3.4%  / +5.9%
```

Two independent setups agree; the 5-way sweep is the outlier. GM=4 measures 265 when it
follows GM=1 in the rotation and 290-297 when it follows GM=2, so rotating many configurations
leaves state the 150 MB L2 flush does not clear. **Sweeping N configurations and sweeping 2 are
not the same experiment** -- a lesson that applies to every config sweep in this report.

**The uncomfortable part.** The `K<=512`, `N<=512` and default arms are still the ones fitted
under the distorted setup. They have not been re-derived, so they are suspect in exactly the
way `K<=512 -> 4` turned out to be. Re-tuning the whole policy under the corrected regime is
open work; this section fixes only the two cases where the error was large enough to find by
accident. The autotuner exists because a four-line rule cannot cover this space.

Effect on the table: `2k x2k x512` 0.89x -> 0.94x, and `2k x2k x2k`, `16k x8k x128`,
`8k x8k x128` all move from losses to ties.

---

## Final performance

Summary: **parity with cuBLAS 12.9 and a per-shape-tuned CUTLASS 4.7 across most of the
range** — 16 of 31 at ≥1.00× cuBLAS, 18 of 27 at ≥ CUTLASS, with 16 of those 27 statistical
ties. Those counts move ±3 between measurement sessions from identical code, so read the ties
and the direction rather than the score. `1024³` (1.06×) is the one clear win; thin-K is the clear weakness. Measured on random
operand data, L2 flushed per launch, interleaved, one shape per process (§12); large shapes
run 65–79% of the 989.4 TFLOPS peak, best 8192×1024×8192 at 780 (78.8%). The same code reads 864 on
`0x11` with a hot cache — that is the measurement conditions, not the kernel.

**On reading these numbers.** Rows within ~1% are inside the run-to-run band; only ≥1.05×
and ≤0.95× entries are decided. The band is wider than it looks on large shapes: the same
kernel and shape has read 912 and 838 TFLOPS in different sessions (8192×4096×8192), so
treat any single large-shape figure as ±5% and prefer the ratios, which are measured
back-to-back.

Regenerate with `make perf` (vs cuBLAS) and `make cutlass` (three-way; prints the chosen
CUTLASS swizzle per shape in the `sw=` column).

Every shape below 0.95× has ≤128 output tiles against 132 SMs, but "grid starvation" is only
the symptom — §14 measures the actual cause as a fixed ~1.38 µs per-CTA prologue against a
~0.59 µs per-tile advantage, break-even ~2.3 tiles/CTA.

---

### Rejected: `CLUSTER_M = 4`

If multicast is the cluster's only benefit, a 4-CTA cluster should save **3/4** of B's L2->SM
traffic instead of 1/2, and `BLOCKS = BN/64 = 4` makes it legal at BN=256. It is far worse:

| shape | CM=2 | CM=4 | |
|---|---|---|---|
| 32768x8192x2048 | 734.3 | 426.6 | **-41.9%** |
| 16384x16384x16384 | 695.3 | 436.2 | -37.3% |
| 8192x1024x8192 | 649.1 | 421.3 | -35.1% |
| 8192x8192x4096 | 668.5 | 457.2 | -31.6% |
| 8192x8192x8192 | 505.5 | 453.1 | -10.4% |
| 4096x4096x4096 | 525.6 | 477.7 | -9.1% |

(5 process-level alternations, medians; `CLUSTER_M` is compile-time so the builds cannot be
interleaved in-process.)

The sign was predictable from where the synchronisation lives. `release_stage` sits **inside
the k-tile loop** and issues `CLUSTER_M` remote `mbarrier` arrivals through
`mapa.shared::cluster`, and `bar_empty` must collect `CONSUMERS * CLUSTER_M` of them. So both
sides scale per k-tile, and they scale differently:

```
  benefit / k-tile  ~  (1 - 1/CM) * B_bytes      CM=2: 1/2   CM=4: 3/4    (x1.5)
  sync cost / k-tile ~  CM remote arrivals       CM=2: 2     CM=4: 4      (x2.0)
```

Cost grows faster than benefit, so CM=4 should lose. **But it loses far more than that
arithmetic predicts** (-42%, not -10..20%), so something else is also paying: most likely
scheduling, since a 4-CTA cluster must be co-resident within one GPC and halves the number of
independent clusters from 66 to 33, shrinking the pool of work available to hide latency.

Worth stating plainly because it is the general shape of the trade: **the cluster buys L2->SM
bandwidth and pays in synchronisation and scheduling freedom.** Two CTAs is where that came
out ahead here; it is not a knob to turn up.

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

1. **Small problems are grid-starved.** Anything producing ≤64 tiles cannot fill 132 SMs.
   Step 10's dispatcher drops to the narrowest available tile (BN=64, not BN=128) for these,
   adding pipeline depth and doubling tile count; that was worth 1.31–1.83× and moved the
   previously worst shapes off the list:

   | shape | before step 10 | after step 10 (BN=64) |
   |---|---|---|
   | 1024³ | 0.69× | **1.04×** |
   | 384×2048×2048 | 0.41× | **0.97×** |
   | 1024×1023×1024 | 1.05× | **1.88×** |

   Remaining open cases are smaller problems (≤32 tiles) or narrow-N shapes where the tile
   dispatcher cannot help further. The step that would close those is BM=64 (one consumer
   warpgroup, halving arithmetic intensity to double tile count) or split-K — neither has been
   implemented. Worst remaining: 4096×512×4096 at 0.82×, 2048×2048×512 at 0.85×.
2. **Short K is pipeline-fill-bound.** `2048x2048x512` (0.87x) has near-full occupancy at 128
   tiles, but K=512 is only 8 k-tiles against a 4-stage pipeline, so ~50% of the time is
   fill/drain. The BN=128 path's 6 stages make this *worse*, not better, which is why the
   dispatcher correctly selects the wide tile there.
3. **`N % 8 != 0` runs at ~58% of aligned speed** (500 TF vs 863). Takes the shuffle epilogue
   *and* a B restride. Still 2-3.4x faster than cuBLAS, which abandons its Hopper kernel for an
   sm_75 CUTLASS `align1` fallback -- but that is cuBLAS degrading, not us accelerating.
4. **`beta != 0` uses the shuffle epilogue**, not TMA store -- a bulk store cannot
   read-modify-write.
5. Timing harness uses `memset(0x11)` data. Tensor-core throughput is data-independent and
   0x1111 is a normal fp16 (~6.2e-4, no denormal penalty); numerics are verified separately
   (57 shapes, 0 failures, 38 bit-exact vs cuBLAS).

---
