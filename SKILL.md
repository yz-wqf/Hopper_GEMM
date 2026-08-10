---
name: gemm-hopper-optimization
description: Optimize a dense GEMM (or GEMM-like tensor-core kernel) on NVIDIA Hopper/sm_90a — TMA, wgmma, warp specialization, clusters, persistent grids, epilogue coalescing, tile dispatch. Use when writing a GEMM kernel from scratch, when an existing tensor-core kernel is below cuBLAS, when diagnosing which of {L2, smem, occupancy, epilogue, pipeline depth} is the bottleneck, or when reasoning about wgmma shared-memory descriptors and TMA swizzle layouts. Triggers on: wgmma, cp.async.bulk.tensor, TMA, sm_90a, warp specialization, matrix descriptor, GEMM tiling, split-K, threadblock swizzle.
---

# Hopper GEMM Optimization — from first principles to ahead of cuBLAS

Derived from taking an FP16 GEMM to 730 TFLOPS on random data / 918 on memset, H100 SXM5 80 GB (20/31 shapes
at parity with cuBLAS 12.9 and a per-shape-tuned CUTLASS 4.7 across most of the range (18/27 ≥, 16 ties), up to
92% of the 989.4 TFLOPS hardware peak). Numbers measured at H100 @ 1980 MHz, CUDA 12.9.
Treat them as *orders of magnitude and orderings*, not universal constants.

Source: `mma_half.cu` (single-file, ~1 400 lines) + `OPTIMIZATION_REPORT.md`.

---

## The Prime Directive

**Measure before and after every single change, and believe the measurement over the model.**

Every item below was reasoned convincingly — and was wrong:

| belief | reality | cost of belief |
|---|---|---|
| "the B-transpose is unavoidable — the descriptor can't express N-major" | it can; removing it was worth **9–16%** | wasted a transpose kernel + workspace |
| "the mainloop uses only 1/3 DRAM, so a transpose overlaps for free" | its CTAs occupy SMs; **1–11% slower** | 2-stream scheme abandoned |
| "matching cuBLAS's 320×128 tile is worth ~2%" | our mainloop was already faster; **0%** | wasted analysis |
| "BM=64 doubles CTAs → ~1.9× more capacity" | CTAs aren't the unit; warpgroups are. **+1%** | bad tile tested |
| "GROUP_M isn't a lever for fp16" | true at 8192³; **18.9% at 16384³** | knob left at default too long |
| "split-K will fix the starved shapes" | reduction cost > saving; **0.80×** | rejected |
| "GROUP_M winner is worth up to 7.2% per shape" | cross-process sweep inflated it; in-process **2.6%** | stale number in report |

**Write "unverified" in comments until a measurement exists.** Write a probe before writing a
kernel. A wrong wgmma descriptor produces *plausible but scrambled* output — you cannot
eyeball it.

---

## Step 0: Empirical descriptor validation (probe first)

A wrong descriptor produces plausible but scrambled output, not a crash. Never derive it from
docs alone. Write a standalone probe that sweeps encodings, compares to a CPU reference, and
prints exact matches. The probe takes an afternoon; a bug here poisons everything downstream.

```
descriptor layout (64-bit):
  [13:0]  addr >> 4
  [29:16] LBO  >> 4   (leading-byte-offset — stride across the "fast" matrix dimension)
  [45:32] SBO  >> 4   (stride-byte-offset  — stride across the "slow" matrix dimension)
  [51:49] base offset (usually 0)
  [63:62] swizzle mode (1 = 128B — use this; it is the only mode bank-conflict-free at BK=64)
```

Measured constants for fp16, 128B swizzle, BK=64:

| operand | trans imm | LBO enc | SBO enc | k16 descriptor advance |
|---|---|---|---|---|
| K-major (A: row-major M×K) | `trans_a=0` | 1 (= 16 B) | 64 (= 1024 B) | +32 B |
| MN-major (B: row-major K×N consumed in place) | `trans_b=1` | `BK*8` (= `BK*128B/16`) | 64 (= 1024 B) | +2048 B (= 16 rows × 128 B) |

```cpp
// K-major descriptor (A). Verified by probe_kmajor.cu.
__device__ __forceinline__ uint64_t make_desc(uint32_t addr) {
  uint64_t d = (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)1ull   << 16;  // LBO = 16 B
  d |= (uint64_t)64ull  << 32;  // SBO = 1024 B
  d |= (uint64_t)1ull   << 62;  // swizzle = 128B
  return d;
}

// MN-major descriptor (B in native row-major K×N layout). Verified by probe_mnmajor*.cu.
__device__ __forceinline__ uint64_t make_desc_mn(uint32_t addr) {
  uint64_t d = (uint64_t)((addr >> 4) & 0x3FFF);
  d |= (uint64_t)(BK * 128 / 16) << 16;  // LBO = BK rows × 128 B per 64-col block
  d |= (uint64_t)64ull             << 32;  // SBO = 1024 B
  d |= (uint64_t)1ull              << 62;  // swizzle = 128B
  return d;
}
```

Re-run probes (`make probe`) whenever tile geometry changes — it takes seconds and catches
stale constants that would otherwise cause silent corruption.

---

## Step 1: Warp specialization + TMA + wgmma skeleton

Three warpgroups/CTA: WG0 is a pure **producer** (issues `cp.async.bulk.tensor`, 32 regs);
WG1/WG2 are **consumers** (run `wgmma`, 232 regs). Repartition at runtime:

```cpp
if (wg == 0) {                    // producer
  setmaxnreg_dec<32>();
  if (lane_in_wg == 0) {
    for (int s = 0; s < STAGES; ++s) bar_init(&bar_full[s],  1);
    for (int s = 0; s < STAGES; ++s) bar_init(&bar_empty[s], CONSUMERS * CLUSTER_M);
  }
  // ... issue TMA
} else {                          // consumers
  setmaxnreg_inc<232>();
  // ... run wgmma
}
```

**Critical build flag:** `-gencode arch=compute_90a,code=sm_90a` — NOT `-arch=sm_90a` alone
(may not propagate the `a`) and NOT `sm_90` (rejects wgmma). Verify with
`cuobjdump --dump-sass <binary> | grep -E 'HGMMA|UTMALDG'`.

**Baseline measured:** 8192³ = **744 TFLOPS**.

---

## Step 2: Pipeline depth (STAGES) — usually the biggest single lever

One stage = one k-tile of both A and B in smem. The producer runs `STAGES` k-tiles ahead,
covering TMA latency (~0.5–1 µs). H100 dynamic smem cap: **232,448 B**.

```
STAGES = 4, ring buffer:
producer ─┬ fill s0 ─┬ fill s1 ─┬ fill s2 ─┬ fill s3 ─┬ refill s0 ─┬ …
          │          │          │          │          │ (waits empty[0])
consumer  └─ wait s0 ┴ wgmma s0 ┴ wgmma s1 ┴ wgmma s2 ┴ wgmma s3 ──┴ …
```

| STAGES | smem/CTA (BM=128, BN=256, BK=64) | K=2048 | K=8192 |
|---|---|---|---|
| 2 | 96 KB | 433 TF | 500 TF |
| 3 | 144 KB | 634 | 803 |
| **4** | **192 KB** | **650** | **858** |
| 5 | 240 KB | — | ✗ exceeds limit |

**2 → 4 stages: +71%.** Five is impossible on H100.

Stages trade directly against epilogue staging space. Price the trade before spending it:
dropping 4→3 to free 48 KB cost **−6.6%** — more than the epilogue optimization it would have
funded. Budget it: `smem_budget - epilogue_staging - alignment_pad` → available for stages.

For narrow tiles, fewer bytes per stage allows deeper pipelines at the same smem budget:
```
BN=256, STAGES=4: 4 × (128×64 + 256×64) × 2B = 192 KB  +  32 KB epi = 224 KB  ✓
BN=128, STAGES=6: 6 × (128×64 + 128×64) × 2B = 192 KB  +  32 KB epi = 224 KB  ✓
BN=64,  STAGES=9: fits ~191 KB, capped at 6 in practice  (K=1024 → only 16 k-tiles; >6 stages
                  spends more time on fill/drain than they hide)
```

---

## Step 3: Cluster TMA multicast

2-CTA clusters cover adjacent tile-rows at the same `tile_n` — both need the identical B tile.
Each CTA multicasts its share of B to the whole cluster, halving B's L2→SM traffic.

```cpp
// producer: each CTA in the cluster issues TMA for its BLOCKS_PER_CTA slices of B
tma_2d_multicast(bdst + blk * B_BLOCK_ELEMS, &tmap_b, ncol, kt * BK,
                 &bar_full[s], MC_MASK);   // .multicast::cluster

// consumer must release each stage to EVERY CTA in the cluster:
__device__ void bar_arrive_remote(uint64_t *bar, uint32_t rank) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
               :: "r"(map_to_cta(smem_u32(bar), rank)));
}
template <int CLUSTER_M_>
__device__ void release_stage(uint64_t *bar_empty, int stage) {
  if constexpr (CLUSTER_M_ == 1) { bar_arrive(&bar_empty[stage]); }
  else { for (int r = 0; r < CLUSTER_M_; ++r) bar_arrive_remote(&bar_empty[stage], r); }
}
```

**Measured (8192³):**

| | CLUSTER_M=1 | CLUSTER_M=2 | gain |
|---|---|---|---|
| mainloop | 764.8 TF | 807.4 TF | **+5.6%** |
| full kernel | 722.6 TF | 779.1 TF | **+7.8%** |

**Deadlock trap:** The B tile splits into `BLOCKS = BN/64` 64-wide pieces; each CTA in the
cluster takes `BLOCKS/CLUSTER_M`. If `BN=64`, `BLOCKS=1 < CLUSTER_M=2` → each CTA gets
**zero** blocks, B never arrives, consumers hang forever on `bar_full`. Guard at compile time:

```cpp
static_assert(C_::BLOCKS >= CLUSTER_M_ && C_::BLOCKS % CLUSTER_M_ == 0,
              "B tile must split evenly across the cluster, else CTAs load nothing and hang");
// Choose CM at compile time so the bad instantiation is never formed:
constexpr int CM = (C_::BLOCKS >= CLUSTER_M && C_::BLOCKS % CLUSTER_M == 0) ? CLUSTER_M : 1;
```

---

## Step 4: Consume B in its native row-major layout (no transpose kernel)

Row-major `B[K,N]` is **N-major** — `B[k][n]` lives at `k*N+n`, so `n` is contiguous. wgmma's
default B operand is K-major, which is why a naive port materializes `B^T`.

**For fp16 (and bf16): `trans_b=1` reads N-major B in place.** tf32/fp8 have no transpose
immediate and must use TN.

Complication: TMA's innermost box under 128B swizzle is capped at 128 B = 64 halves, so a
BN=256 N-major tile arrives as 4 separate `[BK][64]` blocks. **Place them contiguously in
smem and the stride along N becomes uniform:**

```
global B[k][n],  n →
     ┌──────────┬──────────┬──────────┬──────────┐
k ↓  │ n  0..63 │ n 64..127│ n128..191│ n192..255│  ← BK rows per block
     └──────────┴──────────┴──────────┴──────────┘
         blk0        blk1       blk2       blk3

smem  [   blk0   ][   blk1   ][   blk2   ][   blk3   ]
      └ BK*128 B ┘└ BK*128 B ┘└ BK*128 B ┘
               ↑ constant stride = BK rows × 128 B = LBO
One descriptor covers all 4 blocks → a single wgmma.m64n256k16 works.
```

```cpp
// producer: B arrives as BLOCKS contiguous [BK][64] blocks in its native layout
for (int j = 0; j < BLOCKS_PER_CTA; ++j) {
  const int blk = rank * BLOCKS_PER_CTA + j;
  tma_2d_multicast(bdst + blk * B_BLOCK_ELEMS, &tmap_b,
                   tile_n * BN + blk * 64, kt * BK, &bar_full[s], MC_MASK);
}

// consumer: A advances +32B per k16 step; B crosses 16 rows of 128B = +2048B
wgmma_m64n256k16<1>(d, make_desc(a_base + ks * 32),
                       make_desc_mn(b_base + ks * MN_K_STRIDE));  // MN_K_STRIDE = 2048
// wgmma immediates: scale_d=1, scale_a=1, scale_b=0 (unused), trans_a=0, trans_b=1
```

**Measured gains vs separate transpose kernel:**

| shape | with transpose | N-major | gain |
|---|---|---|---|
| 4096³ | 617 TF | **710** | **+15%** |
| 8192³ | 745 | **810** | **+8.7%** |
| 4096×4096×8192 | 637 | **738** | **+16%** |

**Side-effect:** TMA's 16B row-stride requirement moved from `K%8` to `N%8`. Odd N falls off
the fast path — see Step 5.

---

## Step 5: Stride pad-copy for ragged operands (never fall back to scalar)

TMA needs 16B-aligned row strides: `K%8` for A, `N%8` for B. The scalar fallback is fatal:

```
4096 × 4095 × 4096:   scalar fallback = 6 TF   vs  aligned = 708 TF  (118× cliff)
```

Fix: restride the offending operand into scratch with one streaming copy. Only live columns
are copied; TMA's `globalDim` clamp ensures pad columns are never read:

```cpp
__global__ void pad_copy_kernel(const half *src, half *dst, int rows, int cols, int ld) {
  const int c = blockIdx.x * 256 + threadIdx.x;
  if (c >= cols) return;
  for (int r = blockIdx.y; r < rows; r += gridDim.y)
    dst[(long long)r * ld + c] = src[(long long)r * cols + c];
}

// at dispatch time:
const int lda = (K + 7) & ~7, ldb = (N + 7) & ~7;  // round up to 16B alignment
const bool padA = (lda != K), padB = (ldb != N);
// alloc scratch, run pad_copy_kernel for each, then pass lda/ldb as leading dims to TMA
```

**Measured:** N=4095: **6 → 517 TFLOPS (+86×)**.

C cannot be pad-fixed (it's the caller's buffer), so `N%8 ≠ 0` keeps the shuffle epilogue and
costs ~40% vs fully aligned. This is still 2–3× faster than cuBLAS, which abandons its Hopper
kernel for an sm_75 CUTLASS align1 fallback on the same shapes.

---

## Step 6: Persistent kernel with monotonic pipeline counter

**Problem:** 8192³ launched 2048 CTAs on 132 SMs (15.5 waves). The last wave ran half-empty
and every CTA re-paid the prologue. nsys showed cuBLAS launching exactly 132 CTAs.

**Fix:** Size the grid to residency; each CTA/cluster strides through the tile list.

```cpp
// grid: one CTA per SM (clamped to a multiple of CLUSTER_M)
const int grid = min(total_ctiles * CLUSTER_M, (sm_count / CLUSTER_M) * CLUSTER_M);

// tile loop — monotonic `it` across ALL tiles this CTA handles:
int it = 0;
for (int w = my_cluster; w < total_ctiles; w += num_clusters) {
  decode(w, tile_m, tile_n);
  for (int kt = 0; kt < num_k_tiles; ++kt, ++it) {
    const int s = it % STAGES;   // ← it, NOT kt — does not reset per tile
    if (it >= STAGES) bar_wait(&bar_empty[s], ((it / STAGES) - 1) & 1);
    // ... issue TMA
  }
}
```

`it` being monotonic is the key: stage/phase bookkeeping carries across tile boundaries, so
the producer streams the next tile's k-tiles while consumers are still in the current epilogue.

**Measured gains:**

| shape | per-tile grid | persistent | gain |
|---|---|---|---|
| 8192×8192×**2048** | 575 TF | **657** | **+14%** |
| 4096³ | 708 | **749** | +6% |
| 16384×8192×4096 | 714 | **747** | +4.6% |
| 8192³ | 825 | **843** | +2% |

---

## Step 7: Epilogue — isolate before patching

**Isolate the epilogue cost first.** Suppress stores with a runtime-false condition so the
compiler cannot dead-code them. The delta is the exact epilogue cost; don't model it.

In the work behind this skill, after step 6 the **mainloop alone** was 1.7–9.6% *faster* than
cuBLAS's whole kernel. The epilogue was the entire remaining gap.

### 7a. Lane exchange (coalescing)

The wgmma accumulator layout scatters a warp's 32 lanes across 8 rows, contributing only
16 contiguous bytes per row — against a 32-byte sector, 50% wasted:

```
row 0  [16B]·····················  lanes  0- 3   col 0..7
row 1  ·······[16B]·············  lanes  4- 7   col 0..7   (N*2 bytes later)
⋮
row 7  ·················[16B]···  lanes 28-31   col 0..7
128 useful bytes spread over 8 sectors = 256 B traffic → 50% waste
```

The other half of each sector is the *next* g iteration — it's in a different lane. A 2-way
`__shfl_xor` collects it without a wider store:

```cpp
const float a0 = alpha * d[i0], a1 = alpha * d[i0 + 1];   // my g   pair
const float e0 = alpha * d[i1], e1 = alpha * d[i1 + 1];   // my g+1 pair

// Odd lanes exchange their g pair; even lanes exchange their g+1 pair.
const float r0 = __shfl_xor_sync(0xffffffffu, odd ? a0 : e0, 1);
const float r1 = __shfl_xor_sync(0xffffffffu, odd ? a1 : e1, 1);

float v0, v1, v2, v3;
if (odd) { v0 = r0; v1 = r1; v2 = e0; v3 = e1; }  // partner's g, then mine
else     { v0 = a0; v1 = a1; v2 = r0; v3 = r1; }  // mine, then partner's g+1
// → each lane now holds 4 contiguous halves; the quad tiles one full 32-byte sector

union { uint2 u; half2 h2[2]; } out;
out.h2[0] = __floats2half2_rn(v0, v1);
out.h2[1] = __floats2half2_rn(v2, v3);
*reinterpret_cast<uint2 *>(dst) = out.u;
```

**Implementation trap:** Any `d[runtime_index]` spills the accumulator array to local memory.
Every `d[]` and result index must be a compile-time constant; use ternaries instead of
array indexing. Verify with `-Xptxas -v`: **0 bytes stack frame, 0 spills**.

**Measured:**

| shape | before shuffle | after | epilogue share |
|---|---|---|---|
| 8192×8192×**2048** | 657 TF | **796** (+21%) | 27.5% → **13.1%** |
| 4096³ | 749 | **836** (+12%) | 16.8% → **7.1%** |
| 8192³ | 752 | **813** (+8%) | 10.0% → **2.7%** |

### 7b. TMA store (overlap)

After the lane exchange, the implied write bandwidth hit **~2.9 TB/s** against the 3.35 TB/s
HBM3 peak — two shapes measured above peak (L2 hits). The store is efficient but
**serialized** against the mainloop. Widening buys nothing; asynchrony does.

Stage a 64×64 chunk into smem in 128B-swizzled layout, fire the TMA engine, return immediately.
Double-buffer so the register→smem fill of one chunk overlaps the engine draining the previous:

```cpp
for (int j = 0; j < BN / EPI_COLS; ++j) {       // EPI_COLS = 64
  half *buf = sEpi + (cwg * 2 + (j & 1)) * EPI_CHUNK_ELEMS;  // double-buffer

  // Reclaim buffer: ≤1 group outstanding → the store from 2 chunks ago has finished reading it.
  if (lane_in_wg == 0) tma_store_wait<1>();
  wg_barrier(bar_id);   // per-WG named barrier so WG0 and WG1 stage independently

  // Write accumulator → smem in 128B-swizzled layout (XOR bank index with row%8)
  for (int h = 0; h < 2; ++h) {
    const int r  = r_loc + 8 * h;
    const int sw = r & 7;        // swizzle: XOR chunk index with row%8
    for (int a = 0; a < 8; ++a) {
      const int off = r * EPI_COLS + ((a ^ sw) * 8) + 2 * b;
      *reinterpret_cast<half2*>(buf + off) = __floats2half2_rn(alpha * d[i], alpha * d[i+1]);
    }
  }
  asm volatile("fence.proxy.async.shared::cta;");  // generic → async proxy visibility
  wg_barrier(bar_id);
  if (lane_in_wg == 0) {
    tma_store_2d(tmap_c, tile_n * BN + EPI_COLS * j, row_base, buf);
    tma_store_commit();
  }
}
if (lane_in_wg == 0) tma_store_wait<0>();  // drain all before leaving
```

Sizing: 2 WGs × 2 phases × 8 KB = **32 KB** — fits within the 34.9 KB left over from a
4-stage pipeline. Dropping to 3 stages to make room would cost 6.6%; chunking into the
leftover 34.9 KB instead costs 0%.

**Measured:**

| shape | shuffle only | + TMA store | gain |
|---|---|---|---|
| 8192×8192×4096 | 848 TF | **899** | **+6.1%** |
| 16384×8192×4096 | 779 | **846** | **+8.6%** |
| 8192×8192×2048 | 790 | **843** | **+6.7%** |

**Restrictions:** `beta ≠ 0` keeps shuffle path (bulk store cannot read-modify-write).
`N%8 ≠ 0` keeps shuffle path (TMA needs 16B-aligned stride on C; caller's buffer, cannot
be restrided).

---

## Step 8: Tile dispatch — for shapes that don't fill the machine

A 128×256 tile at 1024³ produces 8×4 = 32 tiles on 132 SMs (24% occupancy) — idle, not
math-limited. Template on `(BM, BN)` and pick per call.

**The unit of parallelism is consumer warpgroups, not CTAs:**

```
1024³:
  BM=128, BN=256 →  32 CTAs × 2 consumer WGs =  64 WGs → 140 TF
  BM=128, BN=128 →  64 CTAs × 2 consumer WGs = 128 WGs → 205 TF
  BM=64,  BN=128 → 128 CTAs × 1 consumer WG  = 128 WGs → 231 TF  ← same WGs, no gain
  BM=128, BN=64  → 128 CTAs × 2 consumer WGs = 256 WGs → 314 TF  ← the lever
```

`BM=64` was measured for **every** shape and **never won**. Half the CTA is idle producer;
it costs arithmetic intensity without adding math capacity. Keep `BM=128` throughout; narrow
`BN` to get more tiles. Narrow `BN` also buys extra pipeline stages (same smem budget,
cheaper stages).

**Dispatch rule** (threshold measured from the sweep table, not assumed):

```cpp
auto tiles = [&](int bm, int bn) { return ((M+bm-1)/bm) * ((N+bn-1)/bn); };
const int need = (sm_count * 2 + 2) / 3;  // ~88 on a 132-SM H100
const int bn = (tiles(128,256) >= need) ? 256 : (tiles(128,128) >= need) ? 128 : 64;
```

"Fewer tiles than SMs" as the threshold is too aggressive — regresses 2048³ (0.87×) and
3000×1000×2000 (0.75×) because narrower tiles have ~1.2× worse arithmetic intensity.

**Measured gains from tile dispatch vs fixed 128×256:**

| shape | 128×256 | dispatched | cuBLAS | before → after ratio |
|---|---|---|---|---|
| 1024³ | 140 TF | **317** | 306 | 0.46× → **1.04×** |
| 384×2048×2048 | 141 | **342** | 354 | 0.40× → **0.97×** |
| 1024×1023×1024 | 75 | **133** | 71 | 1.06× → **1.88×** |

---

## Step 9: L2 rasterization (threadblock swizzle / GROUP_M)

Walk `GROUP_M` tile-rows before advancing in N. Resident CTAs share B operand tiles in L2.

```
GROUP_M=4, tiles_n=6 → grouped order:
   tile_n →
   ┌───┬───┬───┬───┬───┬───┐
t  │ 0 │ 4 │ 8 │12 │16 │20 │   w walks ↓ (M fast), then → (N slow)
i  │ 1 │ 5 │ 9 │13 │17 │21 │
l  │ 2 │ 6 │10 │14 │18 │22 │
e  │ 3 │ 7 │11 │15 │19 │23 │
_  └───┴───┴───┴───┴───┴───┘
m
```

Decode (in cluster units; `rank` picks the CTA within the cluster):

```cpp
const int gid      = w / group_sz;
const int in_grp   = w - gid * group_sz;
const int first_cm = gid * cgroup_m;
const int gcm      = min(ctiles_m - first_cm, cgroup_m);  // handles truncated last group
tile_m = (first_cm + in_grp % gcm) * CLUSTER_M + rank;   // M fast
tile_n = in_grp / gcm;                                    // N slow
```

**Measured across all 31 shapes:**

| shape | best GROUP_M | spread | stable? |
|---|---|---|---|
| 16384³ | 64 | 18.9% (GM=2 was bad; GM=8 within 0.3% of best) | not worth a table |
| 16384×8192×128 | 2 | 9.1% | YES |
| 8192×8192×2048 | 16 | 7.7% | noisy |
| 4096×512×4096 | 16 | 7.6% | YES |
| 8192³ | 8 | 3.2% (noise) | N/A |

**GROUP_M is strongly shape-dependent.** Sweeping one shape and generalizing is the mistake.

---

## Step 10: Per-shape GROUP_M policy — structural rules beat lookup tables

**A knob that silently changes two things is worse than two knobs.** A GROUP_M policy arm
looked like it was choosing a rasterization; it was actually switching off the 2-CTA cluster,
because the cluster condition included `group_m % CLUSTER_M == 0` and any odd GROUP_M failed
it. Every measurement behind that arm was correct and its *attribution* was wrong — and no
amount of re-measuring would have caught it. Only reading the launch path did. When a tuning
knob has a surprising effect, check what else it gates before theorising about mechanism.

**Cluster on/off is a shape decision with a large dynamic range.** Isolated properly (build-
level `CLUSTER_M=1` at matched M-grouping), the 2-CTA cluster + B multicast measured **−16% to
+28%** across shapes. Its cost is per tile (cluster launch, `cluster_sync`, mbarriers
collecting `CONSUMERS × CLUSTER_M` arrivals); its benefit is per k-tile (halving B's L2→SM
traffic) and needs enough M-tiles for the multicast to be reused. The rule that fit:

```
  cluster  iff  (K >= 2048 and tiles_m >= 32)  or  tiles_n <= 4
```

K alone does not separate — at K=2048 it is +7.6% at tiles_m=256 and −8.4% at tiles_m=16. The
narrow-N override matters: with few N-tiles B is re-read once per M-row, so halving it pays
regardless of K (one shape lost 14.6pp without it).

**Decide CGA with one number: predicted L2 read demand.**

```
  demand = unclustered FLOP/s x (BM + BN) / (BM * BN)      <- M, N, K cancel

  demand < ~7 TB/s (H100)  ->  do NOT cluster
  demand > ~7 TB/s         ->  cluster is likely to help
```

**Caveat learned the hard way: this criterion separated 12/12 of the shapes it was built from
and 5/11 held-out.** It is an explanation of the mechanism, not a predictor — do not ship a
dispatch rule based on it without a held-out test. The measurements underneath it also failed
to reproduce: one shape's cluster effect read +10.8% in one session and −2.9% in another, same
method. `CLUSTER_M` is compile-time so cluster on/off cannot be interleaved in-process, and
process-level alternation was not sufficient.

CGA is a **bandwidth** optimisation, never a compute one: its sole benefit is two CTAs sharing
one B tile by multicast, cutting B-side L2→SM traffic 50% but *total* L2 traffic only 1/3
(each CTA still loads its own A). Its cost is unconditional — GPC co-residency, launch/retire
as a unit, and `CLUSTER_M` remote mbarrier arrivals *per k-tile*. Below the wall you pay all of
it for nothing.

State the threshold in **TB/s**, not TFLOPS: it is tile-independent in bandwidth but not in
throughput (~597 TFLOPS at 128x256, ~448 at 128x128; at 128x64 a cluster is impossible because
`BLOCKS = BN/64 = 1`).

**The cluster pays only when L2 bandwidth is the wall.** Its cost is unconditional — CTAs lose
scheduling freedom (GPC co-residency, launched and retired as a unit, so 66 independent
clusters instead of 132 CTAs). Its benefit — multicast halving B's L2→SM traffic — only helps
if that bandwidth binds. Tabulating demand against the measured effect separates **12 of 12**
shapes at ~7 TB/s, which is not a fitted threshold but H100's sustained L2 read bandwidth:

```
  3.2 / 3.6 / 3.7 / 5.2 TB/s  ->  -7.0 / -7.0 / -16.0 / -6.7 %
  5.9 / 6.2 / 6.5 / 7.0 TB/s  ->  -0.2 / -1.2 / -0.4 /  -8.4 %
  ----------------------------  ~7 TB/s = L2 read bandwidth
  7.5 / 7.6 / 7.8 / 8.1 TB/s  ->  +0.9 / +2.6 / +10.8 / +1.1 %
```

And for a fixed tile the demand is proportional to *throughput*, because L2 traffic per FLOP is
shape-independent:

```
  L2 bytes = M·N·K·2·(BM+BN)/(BM·BN),  FLOPs = 2MNK
  bytes/FLOP = (BM+BN)/(BM·BN) = 0.0117 at 128x256   (85 FLOP/byte)
```

So "cluster when above ~7 TB/s" is "cluster when the shape already runs above ~600 TFLOPS" —
circular for a dispatcher, which is why the shipped rule uses shape proxies (K, tiles_m,
tiles_n) instead. Compute this number for your tile before adding cluster heuristics; it tells
you which half of the space you are even in.

**Do not turn the cluster up.** If multicast is the benefit, `CLUSTER_M=4` should save 3/4 of
B's L2→SM traffic instead of 1/2. Measured: **−9% to −42%** across six shapes. Check where the
synchronisation lives before assuming it amortises — if the release sits inside the k-tile loop
and issues `CLUSTER_M` remote arrivals, both sides scale per k-tile and the cost scales faster:

```
  benefit / k-tile  ~ (1 - 1/CM)·B_bytes    CM=2: 1/2   CM=4: 3/4   (×1.5)
  sync cost / k-tile ~ CM remote arrivals    CM=2: 2     CM=4: 4     (×2.0)
```

That predicts the sign but badly understates the magnitude, so a second cost is also in play —
a 4-CTA cluster must be GPC-co-resident and halves the count of independent clusters (66 → 33),
shrinking the pool of work available to hide latency. **The cluster buys L2→SM bandwidth and
pays in synchronisation and scheduling freedom.**

**Re-derive every tuned constant when you fix your measurement setup.** A GROUP_M policy
fitted under memset data + hot L2 + sequential timing was wrong once all three were corrected:
`K<=512 -> GROUP_M=4` should have been `1` for short-K and small-grid shapes, worth **+19%** on
one shape and +6..9% on four more. The kernel had not changed; only the ability to measure it
had. Tuned constants inherit the validity of the harness that produced them, and fixing the
harness silently invalidates them all — most will still be right, and you will not know which.

Two arms that did survive scrutiny, and one warning: `K <= 128 -> GROUP_M=1` and
`tiles_m <= 16 -> GROUP_M=1` (rasterize straight down N when the k-loop is too short for L2
reuse to pay for grouping, or when there are too few M-tiles for a group of 4 to mean
anything). Keep it narrow — at 16384³, GROUP_M=1 measures **374 TFLOPS against 710**.

**Get the work hierarchy straight before reasoning about rasterization.** Four levels, easy to
conflate:

| level | what it is |
|---|---|
| **CTA** | one `BM×BN` output tile — the unit the hardware schedules onto an SM |
| **cluster** | `CLUSTER_M` CTAs stacked in M at the *same* `tile_n`, sharing one multicast B tile. **The unit of work assignment** — schedulers hand out clusters, not CTAs |
| **`w`** | linear index into the flat cluster-tile list, `(tiles_m/CLUSTER_M) × tiles_n` long |
| **group** | band of the tile grid, `GROUP_M/CLUSTER_M` cluster-rows tall × all `tiles_n` wide. A numbering device only |

The persistent loop `for (w = my_cluster; w < total; w += num_clusters)` means **the resident
clusters always hold `num_clusters` consecutive `w`**. That single fact is why ordering
controls L2 residency: `GROUP_M = g` makes the concurrent tiles a `g × (SM/g)` rectangle,
needing `g` A-tiles and `SM/g` B-tiles at once.

**The optimal rasterization group is shape-independent — derive it, don't fit it.** Writing
out DRAM traffic under GROUP_M=g with SM concurrent CTAs (resident window `g` × `SM/g` tiles):

```
  traffic(g) = A·tiles_n·g/SM + B·tiles_m/g + C
  g* = sqrt(SM · B · tiles_m / (A · tiles_n)) = sqrt(SM · BN / BM)     <- M, N, K cancel
```

C contributes nothing to the optimum: it is written once and its resident footprint is
`SM·BM·BN·2`, both g-independent — even when C is 98% of all bytes moved. For BM=128:
**16.2 at BN=256, 8.1 at BN=64**, and that matched the measured best on 5/5 shapes where
re-fetching actually happens, including the one on a different tile.

It goes silent where **A+B fits in L2** — no re-fetching to optimise, so traffic stops
depending on g. In the worked example every shape whose optimum was `g=1` had `A+B ≤ 16 MB`
against a 50 MB L2, five for five with no exceptions. That is the root cause in one line: both
operands are co-resident, nothing is re-fetched, so there is no re-fetch traffic to reduce.

**Test `A+B`, not `A+B+C`.** C is write-once — it never needs residency, it just streams out.
Three of those five shapes carried 64–256 MB of C and plainly did not fit; including C would
have wrongly predicted grouping for all three.

**It is necessary, not sufficient.** Six other shapes also fit (6–32 MB) and still preferred
grouping. As a predictor it scores 9/15 against 10/15 for "always group" — so use it to explain
*why grouping cannot help*, never to decide the value. Fitting is not staying: a 64–160 MB C
write stream flows through the same L2 and evicts the operands, so real residency depends on
reuse distance, not operand size. That is why empirical arms are still needed there, and why
they are only ever local facts.

**Sweeping N configurations is not the same experiment as sweeping 2.** The same GROUP_M=4
measured 265 when rotated against GM=1 and 290–297 when rotated against 5 values, a 12% swing
that no L2 flush removed. Report which rotation produced a number, and prefer the narrowest
one that answers your question.

**Before hard-coding any winner, search each shape 3 independent times.** Only 16 of 31
shapes picked the same best GROUP_M all 3 times — for the other 15 the "winner" was noise.
Of the 16 stable shapes, only 6 were worth more than 2%.

```cpp
// group_m == 0 → auto; explicit value (autotuner, benchmark) overrides.
if (group_m == 0) {
  if (bn == 64)      group_m = 2;   // narrow tile / tiny problems  (+2.3%, stable ×3)
  else if (K <= 512) group_m = 4;   // short k-loop, many tiles     (+3.3%, stable ×3)
  else if (N <= 512) group_m = 16;  // narrow N, long k-loop        (+4.5%, stable ×3)
  else               group_m = GROUP_M;  // default=8; within noise for the rest
}
```

This rule generalises; an exact `(M,N,K)` table encodes noise as fact. Leave noise-band shapes
at the default explicitly — that is itself a decision.

**Make GROUP_M a runtime kernel argument** when it is used once per tile in the rasterization
decode (not in the k-loop) — no extra instantiations, zero measurable cost, autotuner can
sweep it without rebuilding.

**Bug pattern:** a comparison returning ~1.000× across all shapes means both arms run the same
code. Here: baseline passed `group_m=8`, policy triggered on `group_m == GROUP_M` (= 8) —
identical. Use a sentinel (`0` = auto) that cannot collide with a legal value.

**Measured — policy vs pinned GM=8:**

| shape | GM=8 | policy | GM chosen | ratio |
|---|---|---|---|---|
| 4096×512×4096 | 531 TF | **575** | 16 | **+8.2%** |
| 4096×8192×128 | 265 | **271** | 4 | **+2.4%** |
| 1024×1023×1024 | 133 | **136** | 2 | **+2.3%** |
| 1024³ | 320 | **327** | 2 | **+2.2%** |

**Autotuner:** `autotune()` sweeps 3 tiles × 6 GROUP_M values = 18 configs, cached per
`(M,N,K)`. Cost: ~1.5 s on a large shape. Keep out of the default entry point; use
`launch_tuned()` explicitly when the caller can absorb the search cost.

---

## Step 11: Cross-check against CUTLASS

Build the same contract from CUTLASS `CollectiveBuilder` primitives as an independent check.
Result (tuning both **per shape**; random data, L2 flushed per launch, interleaved, one shape
per process): **parity** — 18 of 27 at or above CUTLASS, 16 of those 27 statistical ties, and
only `1024³` clearly exceeds 5%. CUTLASS is bit-exact against cuBLAS on every one. Those
counts move ±3 between measurement sessions from identical code, so report the ties and the
direction rather than the score.

**Four measurement choices dominated every optimization in the project.** Operand data is
worth up to 19% (all-zero 906 TFLOPS vs full-entropy random 761 on identical code — zero
operands draw less tensor-core power so clocks boost); sequential A-then-B timing does not
cancel clock drift (1.10× sequential vs 1.01× interleaved on the same pair); and sweeping
many shapes in one process throttles the GPU into a systematic error that *three repeats
agreed on* (1.11× in-sweep vs 1.01× in a fresh process); and any working set under the 50 MB
L2 is measured from cache unless you flush (1.13× hot vs 1.08× cold at 1024³, while
2048×2048×512 went 0.91× → 0.86×). Get all four right before believing any margin — each was
individually larger than the kernel work in this skill.

**Give it the same tuning freedom your dispatcher has, or your score is fiction.** A kernel
that picks from a three-rung tile ladder at runtime, compared against CUTLASS pinned to two
tiles, measures your dispatcher. Sweeping tile × cluster × schedule × swizzle per shape moved
one comparison **20/27 → 17/27 → 14/27** across three harness generations — every correction
in the same direction, which is what a systematically favourable setup looks like.
`128×128×64` **pingpong** turned out to be CUTLASS's best config on essentially every thin-K
shape: a schedule that had been dismissed on a single measurement at 1024³, where it is 5.5%
worse. Third time in this project that tuning on one shape produced a wrong general rule.

**When attributing a win to one design axis, vary only that axis.** Pingpong (each consumer
warpgroup owns a whole tile, the two serialized by an `OrderedSequenceBarrier<2,2>` so one
warpgroup's epilogue hides behind the other's mainloop) was compared as *their 128×128 pingpong
vs our 128×256 cooperative*, and the entire difference charged to the schedule. That produced a
confident, wrong story — a concurrency-vs-hiding trade that should win only in a middle K band
and lose at short K. Re-running it as CUTLASS-against-itself at a **fixed** 128×128 tile
inverted the result: pingpong wins at every K, largest at the *shortest* (1.25× at 4 k-tiles →
1.04× at 64). That monotone decay is the fingerprint of a fixed per-tile cost being hidden and
then amortized, and it says there is no trade at all — `wgmma` is async, one warpgroup issuing
back-to-back already saturates the tensor core, so serializing the mainloops gives up no
throughput. If a framework exposes the axis as a config flag, the isolating experiment is
usually one line; run it before writing down a mechanism.

**A per-tile cost model for consumer scheduling, and the one asymmetry it turns on.** Take `W` =
tensor-core time for one output tile's whole k-loop, `P` = epilogue time for one warpgroup
draining that tile alone. The asymmetry that makes scheduling matter at all: **tensor-core time
does not halve across warpgroups** (the SM's tensor cores are shared and `wgmma` engages all of
them), but **epilogue time does** (LSU/TMA work, genuinely parallel). So cooperative costs
`W + P/2` per tile with the tensor cores *idle* during the `P/2`, and pingpong costs
`max(W, P)` by hiding one warpgroup's epilogue under the other's mainloop. Gain =
`1 + P/(2W)` while `P < W`, saturating and decaying once `P > W` (tensor cores start starving).
Fitting only `P` reproduced a measured curve to ~1% across an 8× range in K. Note `P ≥ W` is
where the benefit *saturates*, not where it begins — a common misreading.

**The criterion reduces to K alone, because tile area cancels.** The epilogue is paid once per
output tile, and tile count is `(M/Mt)·(N/Nt)` — independent of K — while mainloop work per
tile scales with K. Since `P ∝ tile area` and `W ∝ tile area × K`, `P/W ∝ 1/K` *for any tile
shape*. So "epilogue-hiding schedules win at small K" is exact rather than a tile-specific
heuristic, and past `K ≈ 2500` (here, epilogue < 5% of tile time) no schedule can profitably
chase it. **Corollary for split-K:** splitting K into `S` chunks multiplies the epilogue count
by `S` and adds a reduction — it buys occupancy in exactly the currency that is most expensive
at the low K where you were tempted to use it.

**When a phase won't overlap, find the resource hazard before blaming the synchronization.** A
natural reading of "why doesn't cooperative reach `max(W, P/2)`?" is that the consumers are
coupled by shared barriers. That was a symptom. The consumers were halves of the *same* tile,
so they necessarily occupied the same phase; separate barriers would have bought nothing. The
real blocker was a register WAR hazard *across tiles within one warpgroup* — a single
accumulator array, so tile `i+1`'s first `wgmma` targets registers tile `i`'s epilogue is still
draining. Seen that way, pingpong is just **a way to buy the overlap using the other
warpgroup's registers as the second accumulator buffer**, at identical register cost to
double-buffering and without intra-warpgroup software pipelining. Ask what *resource* the two
phases contend for; the barrier is usually downstream of it.

**Accumulator capacity is what decides whether you can copy a schedule.** Accumulators are
per-thread registers: a warpgroup owning an `M×N` tile needs `M·N/128` fp32 per thread against
a 232-register `setmaxnreg` cap. Pingpong doubles the tile per warpgroup, so it is free at
`128×128` (128 regs) and impossible at `128×256` (256 regs, measured spill `STACK:2016`).
Check this arithmetic *before* the profiling — it converts "why is their schedule better" into
"their schedule is only reachable at half our tile width," which is a different decision.

**Tune its tile scheduler, not just its tile.** `max_swizzle_size` defaults to **1** — no
threadblock swizzle at all. Against a kernel with a GROUP_M rasterization policy that is not
a like-for-like test: at 16384³ the default costs CUTLASS 23% (691 → 849 TFLOPS), which was
the entire content of an apparent 1.22× win. A fixed cap overfits the other way (8 everywhere
costs −38% on 384×2048×2048). Sweep `{1,2,4,8}` per shape and keep the best.

Five traps that make the comparison misleading. The first invalidated an entire round of
measurement:

**1. `-DNDEBUG` is mandatory for CUTLASS device code — it is not a cosmetic release flag.**
Without it, device-side asserts block inlining, ptxas cannot keep the wgmma group open across
the resulting call boundary, and it emits warning **C7510**. The SASS then fences and *fully
drains* around every single `wgmma`:

```
  correctly built                     built WITHOUT -DNDEBUG
  WARPGROUP.ARRIVE                    WARPGROUP.ARRIVE
  HGMMA.64x64x16.F32  ×4              HGMMA.64x64x16.F32  ×1
  WARPGROUP.DEPBAR.LE gsb0, 0x1       WARPGROUP.DEPBAR.LE gsb0, 0x0
                      ↑ 1 group in                        ↑ fully drained,
                        flight                              zero overlap
```

Cost: 4096³ 776 → **869**, 1024³ 131 → **289**. That is enough to invert most of a comparison
table. If you are benchmarking against CUTLASS and see C7510, stop and fix the build — the
warning names the mechanism exactly.

**2. CUTLASS has no runtime tile heuristic** — its device API fixes the tile per instantiation.
Comparing your dispatcher against one fixed CUTLASS tile measures the dispatcher, not the
kernels. Sweep tile × cluster × kernel schedule first (all figures `-DNDEBUG`):

```
1024³:  128×64×64 c1×1 coop  289 TF   vs   128×256×64 c2×1  143 TF   ← 2.0× from tuning alone
4096³:  128×64×64 c1×1 coop  374 TF   vs   128×256×64 c2×1  857 TF   ← 2.3× the other way
```
Small and large shapes want *opposite* tiles, sharply.

**3. Tune the cluster shape on more than one size.** `c1×2` beat `c2×1` by 1.4% at 4096³ and
shipped on that basis — then measured **1.6× worse** at 16384³ (429 vs 693 TF). One shape is
never enough evidence to fix a config; this is the same overfit as tuning a swizzle threshold
on a single problem size.

**4. `EpilogueScheduleAuto` selects `DefaultEpilogue`**, not the TMA one — measured **35%
slower** (572 vs 774 TF at 4096³). Pair a cooperative mainloop with an explicit
`TmaWarpSpecializedCooperative` epilogue, as CUTLASS's own examples do.

**5. CUTLASS validates alignment against the problem *extent*, not the leading dimension.**
Padding the stride doesn't satisfy `can_implement`; it rejects, and if you ignore the return
value C is silently left untouched. Symptom: `rel_l2 ≈ 1.0` on exactly the ragged shapes.
A hand-written TMA kernel runs them in place because TMA zero-fills out-of-range elements.

**6. Put a sanity floor on any config sweep.** `can_implement` accepting a config ≠ viable:
a 256×256×64 tile built, ran, and delivered **7 TFLOPS** (~100× slow). Check absolute
throughput, not just success.

---

## Step 12: Hoist the wgmma descriptor out of the inner loop

Count SASS instructions between consecutive `wgmma` issues. If it is much above ~8, you are
probably rebuilding the shared-memory descriptor per issue.

Only bits `[13:0]` (`addr>>4`) change with the k-step, and the step is a compile-time
constant, so the whole 64-bit field does not need reconstructing:

```cpp
// per issue (13.4 inst between issues)        // hoisted (10.6)
make_desc(a_base + ks*32)                      make_desc(a_base)    + ks*2
make_desc_mn(b_base + ks*MN_K_STRIDE)          make_desc_mn(b_base) + ks*(MN_K_STRIDE/16)
```

The base descriptor stays in a uniform register pair and each issue becomes one
`UIADD3.64 URn, URbase, imm`. Field-construction ops in the BN=256 wgmma region drop from
26 `ULOP3` + 17 `USHF` to 8 + 7; the region shrinks 214 -> 159 instructions.

Worth **+1.6%** at 1024^3 and **+1.7%** at 4096x8192x256, ~0 on large shapes -- it removes
per-k-tile overhead, so it pays where the mainloop is a large fraction of a short kernel.

**Do not "solve" this by moving parameters into constant memory.** That is what CUTLASS does
(1280 B `Params` re-read via `ULDC`), and it costs 2.25x the uniform constant loads and 3.3x
the `imc_miss` stalls. Registers are the right home. A descriptor holds a runtime shared
address, so constant memory is not even available for it.

---

## When a persistent kernel loses: the tiles-per-CTA break-even

A persistent kernel front-loads work into the prologue and amortises it across tiles. That
trade has a break-even, and below it you lose to a lighter kernel.

Measure it by holding K fixed and scaling tiles-per-CTA. If the deficit *reverses* rather
than scaling, it is a fixed per-CTA cost:

```
  delta(ours - theirs)  =  P  -  a x (tiles per CTA)

  measured: P = 1.38 us fixed prologue,  a = 0.59 us per-tile advantage
  break-even ~2.3 tiles/CTA;  predicted -7.81 us at 15.5 tiles/CTA, measured -7.90
```

That one constant predicted every shape the comparison kernel won. Shapes at ~1 wave
(tiles ~= SM count) have nothing to amortise the prologue against, which is exactly the
regime where a non-persistent launch or a lighter prologue path would win.

The distinguishing test is cheap: a K-sweep at fixed M,N. **Ratio flat, absolute gap
growing** = per-iteration cost. **Ratio converging, absolute gap flat** = fixed per-tile
cost. Two different root causes that look identical if you only look at one shape.

---

## L2 residency control via TMA cache hints

PTX lets each TMA state its cache intent: `createpolicy` builds a policy, and
`cp.async.bulk.tensor` takes it through `.L2::cache_hint`. In SASS it becomes an extra
operand — `UTMALDG.2D [UR12], [UR10], desc[UR6]`.

Worth **~+1.5% median** on shapes with a small reused operand, and **nothing** elsewhere. The
gate that survived measurement: pin the smaller operand only if it is `≤8 MB` **and** `≥4×`
smaller than the other.

**Every intuitive version of this loses, badly:**

| design | result |
|---|---|
| pin the reused operand, mark the other `evict_first` | **−16%** |
| non-pinned operands get explicit `evict_normal` | **−27%** at 1024³ |
| descriptor **only** on the pinned operand, plain instructions elsewhere | **+1.5%** |

Two non-obvious facts behind that. Marking an operand `evict_first` destroys reuse that was
happening for free — shapes where *nothing* qualified for pinning still lost 6–8%, and those
tag nothing as resident. And an explicit `evict_normal` policy is **not** the same as no
policy: carrying the descriptor operand at all has a cost.

**The mechanism is unestablished — treat the name with suspicion.** The obvious story (the
stream evicts the small operand between reuses) is refuted: holding the pinned operand at 4 MB
and sweeping the streaming one, the gain *peaks* where the whole working set still fits in L2
(36/50 MB, +1.9%) and *shrinks* as pressure rises (+0.5% at 5x over). Confirm with
`lts__t_sector_hit_rate` and `dram__bytes_read.sum` before calling it a residency effect.

**Check whether the operand is already resident before trying to pin it.** The shape that
motivated this work (`32768×8192×2048`, B = 32 MB in a 50 MB L2) gained −0.2%, because the
N-major rasterization already sweeps B once per M-row and keeps it cached. The wins came from
shapes with 2–4 MB operands, which is a different regime entirely. Compute A, B and C in bytes
against the L2 size before writing any code.

---

## Rejected optimizations

Record them so the reasoning isn't repeated:

| approach | outcome | why |
|---|---|---|
| Overlap B transpose with mainloop (2 streams) | **1–11% slower** | transpose CTAs occupy SMs and contend; banded launches lose wave quantization and L2 reuse |
| Match cuBLAS's 320×128 tile | **0%** | our 128×256 mainloop was already faster than their 320×128 whole kernel |
| 4th warpgroup for epilogue | **rejected** | registers can't cross WGs; handoff needs smem staging, which the TMA engine already drains without 128 extra threads |
| STAGES=3 to free 48 KB for epilogue staging | **−6.6%** | more than the epilogue optimization it would have funded; chunked epilogue into leftover 34.9 KB instead |
| Split-K for starved shapes | **0.80× at 1024³** | reduction over S×M×N costs more than the parallelism saves at these problem sizes |

---

## Diagnostic recipes

| question | experiment |
|---|---|
| Is the epilogue the gap? | suppress stores at runtime (`if (sentinel) return`); delta = epilogue cost |
| Is it math-limited or parallelism-limited? | scale M at fixed N,K: does TFLOPS rise? |
| Is K latency the bottleneck? | scale K at fixed M,N: flat = math-bound, rising = pipeline-fill |
| Is the store efficient but serialized? | compute implied GB/s from epilogue delta; near HBM peak → TMA async, not wider stores |
| Is the tile wrong? | count consumer WGs = `CTAs × (BM/64)`, not raw CTAs |
| Is GROUP_M a knob here? | sweep 2–64 across at least 3–4 very different shapes |

**When `ncu` needs root (`ERR_NVGPUCTRPERM`):** Use `nsys` — no special privileges. It gives
kernel names, durations, grid/block/register/smem. cuBLAS kernel names decode the tile:
`nvjet_hsh_320x128_64x3_1x2_h_bz_coopB_NNT` → half/fp32/half, 320×128 tile, BK=64, 3 stages,
1×2 cluster. `cutlass_75_..._align1` means cuBLAS fell back from its Hopper path — your "win"
there may be its degradation, not yours.

---

## Benchmarking hygiene

1. **Run serially.** Two kernels on one GPU contend. Tell: the reference library reads
   implausibly low (cuBLAS at 265 TF when it should do 790).
2. **Medians of repeats.** cuBLAS drifted ~12% run-to-run; treat ±4% as parity.
3. **Compare within the same process**, back-to-back. Cross-process adds drift that masquerades
   as signal: a GROUP_M sweep showed 7.2% cross-process and only 2.6% in-process — the larger
   number was wrong.
4. **Verify at the sizes you report.** Check the largest shape in your table explicitly.
5. **Use a different accuracy class for the anchor.** fp16 vs cuBLAS fp16 shares blind spots.
6. **Constant fill is fine for timing** (tensor-core throughput is data-independent), but then
   verify numerics separately.
7. **~1.000× across every row is a bug**, not a result. Both arms are running the same code.
8. **`paste <(./a) <(./b)` runs them concurrently.** Use `; ` not `<( )`.

---

## Correctness for tensor-core kernels

- Expect `rel_l2` at the **output format's rounding floor**: fp16 → **2.07e-4** (= 2⁻¹¹
  half-ulp) if accumulation is fp32 with exactly one narrowing at the store. A *flat* error
  across wildly different shapes is the signal; any shape that deviates is the bug.
- Cover: ragged tiles, degenerate shapes (`1×1×K`, `M×1×8`), odd leading dimensions, both
  `beta==0` and `beta!=0` (often different epilogue paths), and **many tiles per CTA × many
  k-tiles** (exercises cross-tile pipeline state in a persistent kernel).
- Run **all four** sanitizers: `memcheck`, `racecheck`, `synccheck`, `initcheck`. `racecheck`
  matters most when async DMA out of smem is involved.
- `-Xptxas -v`: confirm **0 bytes stack frame, 0 spills**. A nonzero stack frame almost always
  means a runtime-indexed local array in the epilogue accumulator.
- Against cuBLAS on large aligned shapes (8192³, 16384³): expect **bit-exact** agreement
  (`max_abs = 0.000e+00`).

---

## Precision notes for fp32 inputs

Hopper tensor cores have **no fp32 input mode** — `.f32` in `wgmma...f32.tf32.tf32` is the
accumulator type. For fp32 data:

| approach | TFLOPS ceiling | relative error | notes |
|---|---|---|---|
| FFMA on CUDA cores | 67 | exact (fp32 rounding only) | too slow |
| TF32 | 494 | ~1.8e-2 | fails fp32-referenced checks |
| **3×TF32** | ~165 | ~5e-6…3e-5 | hi+lo split of each operand, 3 products; each tf32×tf32 product is *exact* in fp32 (11×11 = 22 ≤ 24 mantissa bits), so only the accumulation rounds |

3×TF32 is within 4–8× of true fp32 error — enough for tolerance checks, not bit-equivalent.

---

## Known limits (after all current optimizations)

| shape class | ratio | root cause | what would close it |
|---|---|---|---|
| ≤32 output tiles | <0.9× | grid starvation; even BN=64 can't help | BM=64 (one consumer WG, cheaper tile) or split-K |
| 4096×512×4096 | 0.82× | narrow N, sparse tiles | already uses GROUP_M=16; deeper dispatch needed |
| 2048×2048×512 | 0.85× | short K + pipeline fill | BN=256 already chosen; K=512 is only 8 k-tiles |
| `N%8 ≠ 0` | ~0.60× of aligned | B restride + shuffle epilogue (no TMA store) | can't fix C alignment without a copy |
| `beta ≠ 0` | −6% vs `beta==0` | bulk store can't RMW; drops to shuffle epilogue | a separate RMW-capable codepath |
