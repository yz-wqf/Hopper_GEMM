// =====================================================================================
// High-performance FP16 GEMM for NVIDIA H100 (sm_90a) -- warp-specialized.
//
//   C = alpha * (A[M,K] @ B[K,N]) + beta * C        all row-major
//   half in / half out, fp32 accumulate
//
// See OPTIMIZATION_REPORT.md for how this was built, measured, and what was rejected.
//
// Design
// ------
//   * Warp specialization: 3 warpgroups / CTA. WG0 is a pure *producer* (issues TMA
//     bulk-tensor copies, 32 regs). WG1/WG2 are *consumers* (run wgmma, 232 regs).
//     Register budget is repartitioned at runtime with `setmaxnreg`.
//   * Async pipeline: STAGES-deep circular smem buffer, mbarrier full/empty handshake.
//     Consumers keep one wgmma group in flight (`wgmma.wait_group 1`) so math overlaps
//     the next stage's arrival.
//   * TMA does all global->smem movement with 128B swizzle, so wgmma's operand reads are
//     bank-conflict free. Out-of-range elements are zero-filled by the TMA unit, which
//     makes ragged M/N/K fall out for free.
//   * Tile is chosen per shape from 128x{256,128,64} x BK=64, with a matching wgmma
//     m64n{256,128,64}k16.f32.f16.f16. GROUP_M is likewise selected at runtime.
//   * 2-CTA clusters: the pair shares one B tile, so each CTA TMA-*multicasts* half of it
//     to both, halving B's L2->SM traffic.
//   * L2-friendly grouped rasterization of the tile -> CTA mapping.
//
// Operand layouts, and why there is no transpose kernel
// -----------------------------------------------------
// "X-major" below means X is the *contiguous* dimension (CUTLASS GMMA::Major convention).
// Note row-major K x N is N-major, not K-major: B[k][n] sits at k*N+n, so n varies fastest.
//
//   A, row-major M x K  ->  k contiguous  ->  K-major   -> imm-trans-a = 0
//   B, row-major K x N  ->  n contiguous  ->  N-major   -> imm-trans-b = 1
//
// wgmma's default B operand is K-major (i.e. B^T), which is why a naive port materializes
// B^T with a transpose pass. That is unnecessary: fp16 wgmma has transpose immediates, and
// B can be consumed N-major in place. The wrinkle is that under 128B swizzle TMA's
// innermost box is capped at 128B = 64 halves, so a BN=256 N-major tile arrives as 4
// separate [BK][64] blocks. Placing those blocks *contiguously* makes the stride along N
// uniform, and the MN-major descriptor expresses exactly that: LBO becomes the N-direction
// block stride (BK rows x 128B) and SBO the K-direction atom stride (8 rows x 128B), with
// a k16 step advancing the descriptor by 16*128B. One descriptor therefore still covers
// all 4 blocks and a single wide m64n256k16 works. All three constants were established by
// probe (see make_desc_mn).
//
// Dropping the transpose was worth 9-16% end-to-end, e.g. 745 -> 810 TFLOPS at 8192^3.
//
// TMA also requires 16B-aligned row strides (K%8 for A, N%8 for B). Rather than fall back
// to the scalar kernel when that fails -- which measured ~118x slower (6 vs 708 TFLOPS at
// N=4095) -- the offending operand is restrided into scratch with one streaming copy.
//
// Accuracy: accumulation is fp32 (`wgmma...f32.f16.f16`), which is strictly better than
// the fp16 inputs and is what cuBLAS's default HGEMM does. Only the final store narrows
// to half. There is no precision tradeoff to make here: the fastest path is also the most
// accurate one. Measured relative L2 error is a flat
// 2.07e-4 on every shape tested, which is exactly the fp16 *output* rounding floor
// (2^-11 half-ulp): the arithmetic adds nothing on top of storing the answer in half.
//
// Epilogue
// --------
// Writing C straight out of the wgmma accumulator layout wastes half the write bandwidth:
// for a fixed (g, j/2) a warp's lanes cover 8 rows contributing only 16 contiguous bytes
// each, against a 32-byte sector granularity. Two rounds of work fixed it:
//
//   1. Lane-pair __shfl_xor regroups each lane into a contiguous 4-half run, so a quad
//      tiles a full 32B sector. This reached ~2.9 TB/s -- essentially peak HBM write
//      bandwidth -- but fully *serialized* against the mainloop.
//   2. Since the cost was then overlap rather than efficiency, the store moved to the TMA
//      engine: stage a 64x64 chunk into smem (128B-swizzled), fire
//      cp.async.bulk.tensor.2d.global.shared, and return to the mainloop while the engine
//      drains. Double-buffered, 2 WGs x 2 phases x 8 KB = 32 KB, which fits the 34.9 KB
//      left by a 4-stage pipeline -- dropping to 3 stages to make room would have cost 6.6%
//      at exactly the shapes this helps, so it was not worth it.
//
//   epilogue as a share of runtime:   naive -> shuffle -> TMA store
//     8192x8192x4096                  ~12%      8.4%      2.8%
//     16384x8192x4096                 12.8%     6.0%      ~0%
//     8192x8192x2048                  27.5%    13.1%      7.2%
//     4096x4096x8192                   9.3%     3.8%      1.3%
//
// The shuffle path is retained for beta != 0 (a bulk store cannot read-modify-write) and
// for N%8 != 0 (TMA needs a 16B-aligned row stride, and C is the caller's buffer so it
// cannot be restrided into scratch the way A and B are).
//
// Measured, H100 SXM 80GB @ 1980 MHz, CUDA 12.9, vs cuBLAS 12.9 (alpha=1, beta=0)
// -------------------------------------------------------------------------------------
//   shape                ours         cuBLAS      ratio
//   8192^3               849-893 TF   764-797 TF  1.08-1.13x
//   16384x8192x4096      814-821      789-800     1.02-1.03x
//   4096x4096x8192       799-810      802-813     ~1.00x
//   4096^3               863-869      870-874     0.99x
//   8192x8192x4096       769-797      779-802     0.97-1.01x
//
// cuBLAS drifts ~12% run to run on this machine, so treat anything within a few percent of
// 1.0 as parity. Structurally this kernel now matches cuBLAS's nvjet_hsh_320x128_64x3_1x2
// on grid (132 persistent CTAs), block (384), registers (168) and cluster size (2); its
// 320x128 tile and 3 stages measured no better than 128x256 with 4.
//
// Known soft spot: N%8 != 0 falls back to the shuffle epilogue *and* needs B restrided into
// scratch, landing near 500 TFLOPS. Correct, but roughly half speed.
//
// Note the `a` in sm_90a is required -- plain `-arch=sm_90` rejects wgmma.
// Build: nvcc -gencode arch=compute_90a,code=sm_90a -O3 mma_half.cu
// =====================================================================================

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <map>
#include <tuple>

namespace h100_hgemm {

// ------------------------------------------------------------------ tuning parameters
// Hoist the wgmma descriptor build out of the k16 loop (see the mainloop). Default on;
// -DCFG_HOIST_DESC=0 restores the per-issue rebuild for A/B comparison.
// L2 residency hints on the TMA descriptors (see l2_policy_* above). Default on; 0 disables.
// Minimum K for the 2-CTA cluster to pay for itself; see the derivation at its use site.
#ifndef CFG_CLUSTER_MIN_K
#define CFG_CLUSTER_MIN_K 2048
#endif
#ifndef CFG_CLUSTER_MIN_TM
#define CFG_CLUSTER_MIN_TM 32
#endif
#ifndef CFG_CLUSTER_NARROW_N
#define CFG_CLUSTER_NARROW_N 4
#endif
#ifndef CFG_L2_HINT
#define CFG_L2_HINT 1
#endif
// Largest operand we will try to pin. H100 L2 is 50 MB; leave headroom for the streaming
// operand and C's write traffic rather than trying to fill it.
// Two conditions, both measured rather than reasoned. The operand must be small in absolute
// terms *and* much smaller than the one being streamed past it:
//   4096x512x4096   B=4MB   vs A=32MB  (8x)  -> +2.1%  (ranges non-overlapping over 15 A/B)
//   4096x8192x512   A=4MB   vs B=8MB   (2x)  -> -0.4%  (nothing)
//   32768x8192x2048 B=32MB  vs A=128MB (4x)  -> -0.2%  (nothing; B is already L2-resident
//                                                       there via the N-major rasterization)
// Pinning 32 MB of a 50 MB L2 leaves no room for the stream, and a 2x ratio does not create
// enough pressure to matter.
#ifndef CFG_L2_PIN_MAX
#define CFG_L2_PIN_MAX (8u << 20)
#endif
#ifndef CFG_L2_PIN_RATIO
#define CFG_L2_PIN_RATIO 4
#endif
#ifndef CFG_HOIST_DESC
#define CFG_HOIST_DESC 1
#endif
#ifndef CFG_STAGES
#define CFG_STAGES 4
#endif
#ifndef CFG_GROUP_M
#define CFG_GROUP_M 16
#endif
#ifndef CFG_CONSUMER_REGS
#define CFG_CONSUMER_REGS 232
#endif
#ifndef CFG_PRODUCER_REGS
#define CFG_PRODUCER_REGS 32
#endif
#ifndef CFG_CLUSTER_M
#define CFG_CLUSTER_M 2
#endif

constexpr int BM = 128;   // CTA tile rows
constexpr int BN = 256;   // CTA tile cols
constexpr int BK = 64;    // k-step; BK * sizeof(half) == 128B == one swizzle atom row
constexpr int STAGES = CFG_STAGES;

constexpr int WGS = 128;                 // threads per warpgroup
constexpr int CONSUMER_WGS = 2;
constexpr int NUM_THREADS = (CONSUMER_WGS + 1) * WGS;
constexpr int WG_M = BM / CONSUMER_WGS;  // rows of C owned by one consumer WG (64)
constexpr int GROUP_M = CFG_GROUP_M;
constexpr int CLUSTER_M = CFG_CLUSTER_M;

constexpr int A_STAGE_ELEMS = BM * BK;
constexpr int B_STAGE_ELEMS = BN * BK;
constexpr int TMA_BYTES = (A_STAGE_ELEMS + B_STAGE_ELEMS) * (int)sizeof(half);

// Epilogue staging: one 64x64 half chunk per consumer WG (8 KB; 64 halves = 128 B =
// exactly one 128B-swizzle atom row), double-buffered.
constexpr int EPI_ROWS = 64, EPI_COLS = 64;
constexpr int EPI_CHUNK_ELEMS = EPI_ROWS * EPI_COLS;


// --------------------------------------------------------------- tile configurations
// Two tile widths, chosen per problem. The wide tile maximizes arithmetic intensity; the
// narrow one exists because 128x256 leaves most of the GPU idle on small problems --
// 1024^3 is only 8x4 = 32 tiles against 132 SMs, i.e. 24% occupancy. Halving BN doubles
// the tile count, and because it also halves per-stage smem it buys two extra pipeline
// stages, which is what small-K shapes need. Both land on the same smem budget:
//   BN=256, STAGES=4 : 4*(8192+16384)*2 = 192 KB + 32 KB epilogue
//   BN=128, STAGES=6 : 6*(8192+ 8192)*2 = 192 KB + 32 KB epilogue
// BM is a knob too, not just BN. BM=128 with two consumer warpgroups was inherited from
// the initial design and is right for large problems, but a 64-row tile doubles the tile count
// again -- the difference between 64 and 128 CTAs on a 132-SM machine. It costs arithmetic
// intensity ((64+128)/(64*128) vs (128+256)/(128*256) is 2x the operand traffic per output)
// and spends half the CTA on the producer instead of a third, so it is strictly a
// small-problem configuration.
template <int BM_, int BN_>
struct Cfg {
  static constexpr int BM = BM_;
  static constexpr int BN = BN_;
  static constexpr int CONSUMERS = BM_ / 64;                 // one warpgroup per 64 rows
  static constexpr int THREADS = (CONSUMERS + 1) * WGS;      // + the producer
  static constexpr int NREG = 64 * BN_ / 128;                // fp32 accumulators per thread
  static constexpr int A_STAGE = BM_ * BK;
  static constexpr int B_STAGE = BN_ * BK;
  static constexpr int BLOCKS = BN_ / 64;                    // 64 halves = 128B = one atom

  // L2 read intensity of this tile: bytes a CTA pulls from L2 per FLOP it issues. Every CTA
  // reads BM x BK of A and BK x BN of B per k-tile, and issues 2*BM*BN*BK FLOPs, so
  //     bytes/FLOP = (BM + BN) / (BM * BN)
  // M, N and K cancel -- this is a property of the tile alone. It is what makes "is this
  // kernel L2-read-bound?" answerable from throughput: L2 demand = achieved FLOP/s * this.
  static constexpr double L2_BYTES_PER_FLOP = double(BM_ + BN_) / (double(BM_) * double(BN_));
  //   128x256 -> 0.01171875 B/FLOP (85 FLOP/B), crossover ~597 TFLOPS
  //   128x128 -> 0.015625        (64 FLOP/B), crossover ~448 TFLOPS
  //   128x64  -> 0.0234375       (43 FLOP/B), but BLOCKS==1 so a cluster is impossible
  static constexpr double CGA_CROSSOVER_FLOPS = 7.0e12 / L2_BYTES_PER_FLOP;
  static constexpr int TMA_B = (A_STAGE + B_STAGE) * (int)sizeof(half);
  static constexpr int EPI = CONSUMERS * 2 * EPI_CHUNK_ELEMS;  // 2 phases per consumer WG
  // Fill the smem budget: deeper pipelines for the cheaper tiles.
  static constexpr int BUDGET = 232448 - EPI * (int)sizeof(half) - 512;
#ifdef CFG_STAGES_OVERRIDE
  static constexpr int STAGES = CFG_STAGES_OVERRIDE;
#else
  static constexpr int RAW = BUDGET / ((A_STAGE + B_STAGE) * (int)sizeof(half));
  // Cap the depth. A deeper pipeline only pays if there are k-tiles to fill it: at K=1024
  // there are just 16, so 8 stages spends half the k-loop on fill/drain. Measured cap.
  static constexpr int STAGES = (BM_ == 128 && BN_ == 256) ? CFG_STAGES : (RAW > 6 ? 6 : RAW);
#endif
  static constexpr int SMEM = STAGES * (A_STAGE + B_STAGE) * (int)sizeof(half) +
                              EPI * (int)sizeof(half) + 2 * STAGES * (int)sizeof(uint64_t);
  static_assert(SMEM <= 232448, "exceeds H100 max dynamic shared memory");
  static_assert(STAGES >= 2, "pipeline too shallow");
};

// Epilogue staging: each consumer WG stages a 64x64 half chunk (8 KB, and 64 halves = 128 B
// = exactly one 128B-swizzle atom row), double-buffered so the register->smem fill of one
// chunk overlaps the TMA store of the previous. 2 WGs x 2 phases x 8 KB = 32 KB, which fits
// the 34.9 KB left over from a 4-stage pipeline -- so no stage has to be given up.

// --------------------------------------------------------------------- ptx primitives
__device__ __forceinline__ uint32_t smem_u32(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// wgmma shared-memory matrix descriptor.
//   [13:0] addr>>4 | [29:16] leading-byte-offset>>4 | [45:32] stride-byte-offset>>4
//   [51:49] base offset | [63:62] swizzle mode (1 == 128B)
// K-major operand tiled at 128B per row under 128B swizzle: LBO is one core matrix along
// K (16B), SBO is one swizzle atom along M/N (8*128B). Verified empirically for fp16.
__device__ __forceinline__ uint64_t make_desc(uint32_t addr) {
  uint64_t d = static_cast<uint64_t>((addr >> 4) & 0x3FFFull);
  d |= static_cast<uint64_t>(1ull) << 16;   // LBO  = 16B   >> 4
  d |= static_cast<uint64_t>(64ull) << 32;  // SBO  = 1024B >> 4
  d |= static_cast<uint64_t>(1ull) << 62;   // swizzle = 128B
  return d;
}

// Descriptor for an *MN-major* operand -- B, held in its native row-major [BK][BN] form.
// Under 128B swizzle the innermost TMA box is capped at 128B = 64 halves, so the BN-wide
// tile arrives as BN/64 contiguous blocks of [BK][64]. The roles of the two offsets swap
// relative to the K-major case: LBO becomes the stride along N between those blocks
// (BK rows x 128B), and SBO the stride along K between swizzle atoms (8 rows x 128B).
// Because the blocks are placed contiguously that N stride is uniform, so one descriptor
// covers all BN/64 of them and a single m64n256k16 still works. Verified by probe.
__device__ __forceinline__ uint64_t make_desc_mn(uint32_t addr) {
  uint64_t d = static_cast<uint64_t>((addr >> 4) & 0x3FFFull);
  d |= static_cast<uint64_t>(BK * 128 / 16) << 16;  // LBO = BK rows * 128B, per 64-N block
  d |= static_cast<uint64_t>(64ull) << 32;          // SBO = 1024B >> 4
  d |= static_cast<uint64_t>(1ull) << 62;           // swizzle = 128B
  return d;
}
constexpr int B_BLOCKS = BN / 64;         // 64 halves == 128B == one swizzle atom width
constexpr int B_BLOCK_ELEMS = BK * 64;    // halves per block
constexpr int MN_K_STRIDE = 16 * 128;     // descriptor advance per k16 step, bytes

__device__ __forceinline__ void bar_init(uint64_t *bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(smem_u32(bar)), "r"(count));
}
__device__ __forceinline__ void bar_expect(uint64_t *bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(bar)),
               "r"(bytes));
}
__device__ __forceinline__ void bar_arrive(uint64_t *bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_u32(bar)));
}
__device__ __forceinline__ void bar_wait(uint64_t *bar, uint32_t phase) {
  asm volatile(
      "{ .reg .pred P;\n"
      "  WAIT_%=: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], %1;\n"
      "  @P bra DONE_%=;\n"
      "  bra WAIT_%=;\n"
      "  DONE_%=: }" ::"r"(smem_u32(bar)),
      "r"(phase));
}

// ---------------------------------------------------------------- L2 residency control
// Which operand should stay in L2 is a property of the shape, not the kernel. At
// 32768x8192x2048: A is 134 MB (streamed), B is 32 MB (fits the 50 MB L2 and is re-read by
// every one of the 256 M-tile rows), C is 537 MB written once. Left alone, C's write stream
// evicts B. `createpolicy` + `.L2::cache_hint` lets each TMA say what it wants:
//   A loads  -> evict_first  (streaming, do not displace anything)
//   B loads  -> evict_last   (keep resident, it is the reused operand)
//   C stores -> evict_first  (write-once, must not evict B)
// The policy is selected per launch from the operand sizes, so the opposite case -- A small
// enough to pin while B streams -- gets the mirrored assignment.
__device__ __forceinline__ uint64_t l2_policy_evict_last() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ uint64_t l2_policy_evict_first() {
  uint64_t p;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(p));
  return p;
}
__device__ __forceinline__ void tma_2d_hint(void *dst, const CUtensorMap *map, int c0, int c1,
                                            uint64_t *bar, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      ".L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(smem_u32(dst)),
      "l"(map), "r"(c0), "r"(c1), "r"(smem_u32(bar)), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void tma_2d_multicast_hint(void *dst, const CUtensorMap *map, int c0,
                                                      int c1, uint64_t *bar, uint16_t mask,
                                                      uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      ".multicast::cluster.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5, %6;" ::"r"(smem_u32(dst)),
      "l"(map), "r"(c0), "r"(c1), "r"(smem_u32(bar)), "h"(mask), "l"(policy)
      : "memory");
}
__device__ __forceinline__ void tma_store_2d_hint(const CUtensorMap *map, int c0, int c1,
                                                  const void *src, uint64_t policy) {
  asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint"
      " [%0, {%1, %2}], [%3], %4;" ::"l"(map),
      "r"(c0), "r"(c1), "r"(smem_u32(src)), "l"(policy)
      : "memory");
}

__device__ __forceinline__ void tma_2d(void *dst, const CUtensorMap *map, int c0, int c1,
                                       uint64_t *bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(dst)),
      "l"(map), "r"(c0), "r"(c1), "r"(smem_u32(bar))
      : "memory");
}

// Same, but the payload is broadcast into every CTA named by `mask` in the cluster; the
// destination smem offset and mbarrier address are interpreted in each receiver's window.
__device__ __forceinline__ void tma_2d_multicast(void *dst, const CUtensorMap *map, int c0, int c1,
                                                 uint64_t *bar, uint16_t mask) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      ".multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;" ::"r"(smem_u32(dst)),
      "l"(map), "r"(c0), "r"(c1), "r"(smem_u32(bar)), "h"(mask)
      : "memory");
}

// TMA store: smem -> global, drained by the TMA engine rather than by the warps. That
// asynchrony is the point -- the consumer fires it and walks straight into the next tile.
__device__ __forceinline__ void tma_store_2d(const CUtensorMap *map, int c0, int c1,
                                             const void *src) {
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];" ::"l"(
                   map),
               "r"(c0), "r"(c1), "r"(smem_u32(src))
               : "memory");
}
__device__ __forceinline__ void tma_store_commit() {
  asm volatile("cp.async.bulk.commit_group;");
}
template <int N>
__device__ __forceinline__ void tma_store_wait() {
  asm volatile("cp.async.bulk.wait_group.read %0;" ::"n"(N) : "memory");
}
// Warpgroup-local barrier (named barriers 1..2), so the two consumer WGs stage independently.
__device__ __forceinline__ void wg_barrier(int id) {
  asm volatile("bar.sync %0, %1;" ::"r"(id), "n"(WGS));
}

__device__ __forceinline__ uint32_t cta_rank_in_cluster() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ uint32_t map_to_cta(uint32_t addr, uint32_t rank) {
  uint32_t d;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(d) : "r"(addr), "r"(rank));
  return d;
}
__device__ __forceinline__ void bar_arrive_remote(uint64_t *bar, uint32_t rank) {
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(
      map_to_cta(smem_u32(bar), rank)));
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.aligned;");
  asm volatile("barrier.cluster.wait.aligned;");
}

template <int CLUSTER_M_>
__device__ __forceinline__ void release_stage(uint64_t *bar_empty, int stage) {
  if constexpr (CLUSTER_M_ == 1) {
    bar_arrive(&bar_empty[stage]);
  } else {
#pragma unroll
    for (int r = 0; r < CLUSTER_M_; ++r) bar_arrive_remote(&bar_empty[stage], r);
  }
}

template <int N>
__device__ __forceinline__ void setmaxnreg_dec() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;" ::"n"(N));
}
template <int N>
__device__ __forceinline__ void setmaxnreg_inc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;" ::"n"(N));
}

__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;"); }
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;");
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N));
}

#define WGMMA_N 256
#define WGMMA_NREG 128
// wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16
// operands after D: a-desc, b-desc, scale-d, imm-scale-a, imm-scale-b, imm-trans-a, imm-trans-b
// A is K-major (row-major M x K) so imm-trans-a = 0. B is consumed *N-major* straight out
// of its row-major K x N layout, so imm-trans-b = 1 -- no transpose pass is needed.
template <int scaleD>
__device__ __forceinline__ void wgmma_m64n256k16(float *d, uint64_t da, uint64_t db) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64,%65,%66,%67,%68,%69,%70,%71,%72,%73,%74,%75,%76,%77,%78,%79,%80,%81,%82,%83,%84,%85,%86,%87,%88,%89,%90,%91,%92,%93,%94,%95,%96,%97,%98,%99,%100,%101,%102,%103,%104,%105,%106,%107,%108,%109,%110,%111,%112,%113,%114,%115,%116,%117,%118,%119,%120,%121,%122,%123,%124,%125,%126,%127}, "
      "%128, %129, %130, 1, 1, 0, 1;"
      : "+f"(d[0]),
          "+f"(d[1]),
          "+f"(d[2]),
          "+f"(d[3]),
          "+f"(d[4]),
          "+f"(d[5]),
          "+f"(d[6]),
          "+f"(d[7]),
          "+f"(d[8]),
          "+f"(d[9]),
          "+f"(d[10]),
          "+f"(d[11]),
          "+f"(d[12]),
          "+f"(d[13]),
          "+f"(d[14]),
          "+f"(d[15]),
          "+f"(d[16]),
          "+f"(d[17]),
          "+f"(d[18]),
          "+f"(d[19]),
          "+f"(d[20]),
          "+f"(d[21]),
          "+f"(d[22]),
          "+f"(d[23]),
          "+f"(d[24]),
          "+f"(d[25]),
          "+f"(d[26]),
          "+f"(d[27]),
          "+f"(d[28]),
          "+f"(d[29]),
          "+f"(d[30]),
          "+f"(d[31]),
          "+f"(d[32]),
          "+f"(d[33]),
          "+f"(d[34]),
          "+f"(d[35]),
          "+f"(d[36]),
          "+f"(d[37]),
          "+f"(d[38]),
          "+f"(d[39]),
          "+f"(d[40]),
          "+f"(d[41]),
          "+f"(d[42]),
          "+f"(d[43]),
          "+f"(d[44]),
          "+f"(d[45]),
          "+f"(d[46]),
          "+f"(d[47]),
          "+f"(d[48]),
          "+f"(d[49]),
          "+f"(d[50]),
          "+f"(d[51]),
          "+f"(d[52]),
          "+f"(d[53]),
          "+f"(d[54]),
          "+f"(d[55]),
          "+f"(d[56]),
          "+f"(d[57]),
          "+f"(d[58]),
          "+f"(d[59]),
          "+f"(d[60]),
          "+f"(d[61]),
          "+f"(d[62]),
          "+f"(d[63]),
          "+f"(d[64]),
          "+f"(d[65]),
          "+f"(d[66]),
          "+f"(d[67]),
          "+f"(d[68]),
          "+f"(d[69]),
          "+f"(d[70]),
          "+f"(d[71]),
          "+f"(d[72]),
          "+f"(d[73]),
          "+f"(d[74]),
          "+f"(d[75]),
          "+f"(d[76]),
          "+f"(d[77]),
          "+f"(d[78]),
          "+f"(d[79]),
          "+f"(d[80]),
          "+f"(d[81]),
          "+f"(d[82]),
          "+f"(d[83]),
          "+f"(d[84]),
          "+f"(d[85]),
          "+f"(d[86]),
          "+f"(d[87]),
          "+f"(d[88]),
          "+f"(d[89]),
          "+f"(d[90]),
          "+f"(d[91]),
          "+f"(d[92]),
          "+f"(d[93]),
          "+f"(d[94]),
          "+f"(d[95]),
          "+f"(d[96]),
          "+f"(d[97]),
          "+f"(d[98]),
          "+f"(d[99]),
          "+f"(d[100]),
          "+f"(d[101]),
          "+f"(d[102]),
          "+f"(d[103]),
          "+f"(d[104]),
          "+f"(d[105]),
          "+f"(d[106]),
          "+f"(d[107]),
          "+f"(d[108]),
          "+f"(d[109]),
          "+f"(d[110]),
          "+f"(d[111]),
          "+f"(d[112]),
          "+f"(d[113]),
          "+f"(d[114]),
          "+f"(d[115]),
          "+f"(d[116]),
          "+f"(d[117]),
          "+f"(d[118]),
          "+f"(d[119]),
          "+f"(d[120]),
          "+f"(d[121]),
          "+f"(d[122]),
          "+f"(d[123]),
          "+f"(d[124]),
          "+f"(d[125]),
          "+f"(d[126]),
          "+f"(d[127])
      : "l"(da), "l"(db), "n"(scaleD));
}


// wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 -- narrow-tile variant.
template <int scaleD>
__device__ __forceinline__ void wgmma_m64n128k16(float *d, uint64_t da, uint64_t db) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, "
      "%64, %65, %66, 1, 1, 0, 1;"
      : "+f"(d[0]),
          "+f"(d[1]),
          "+f"(d[2]),
          "+f"(d[3]),
          "+f"(d[4]),
          "+f"(d[5]),
          "+f"(d[6]),
          "+f"(d[7]),
          "+f"(d[8]),
          "+f"(d[9]),
          "+f"(d[10]),
          "+f"(d[11]),
          "+f"(d[12]),
          "+f"(d[13]),
          "+f"(d[14]),
          "+f"(d[15]),
          "+f"(d[16]),
          "+f"(d[17]),
          "+f"(d[18]),
          "+f"(d[19]),
          "+f"(d[20]),
          "+f"(d[21]),
          "+f"(d[22]),
          "+f"(d[23]),
          "+f"(d[24]),
          "+f"(d[25]),
          "+f"(d[26]),
          "+f"(d[27]),
          "+f"(d[28]),
          "+f"(d[29]),
          "+f"(d[30]),
          "+f"(d[31]),
          "+f"(d[32]),
          "+f"(d[33]),
          "+f"(d[34]),
          "+f"(d[35]),
          "+f"(d[36]),
          "+f"(d[37]),
          "+f"(d[38]),
          "+f"(d[39]),
          "+f"(d[40]),
          "+f"(d[41]),
          "+f"(d[42]),
          "+f"(d[43]),
          "+f"(d[44]),
          "+f"(d[45]),
          "+f"(d[46]),
          "+f"(d[47]),
          "+f"(d[48]),
          "+f"(d[49]),
          "+f"(d[50]),
          "+f"(d[51]),
          "+f"(d[52]),
          "+f"(d[53]),
          "+f"(d[54]),
          "+f"(d[55]),
          "+f"(d[56]),
          "+f"(d[57]),
          "+f"(d[58]),
          "+f"(d[59]),
          "+f"(d[60]),
          "+f"(d[61]),
          "+f"(d[62]),
          "+f"(d[63])
      : "l"(da), "l"(db), "n"(scaleD));
}

// wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 -- narrowest tile variant.
template <int scaleD>
__device__ __forceinline__ void wgmma_m64n64k16(float *d, uint64_t da, uint64_t db) {
  asm volatile(
      "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
      "%32, %33, %34, 1, 1, 0, 1;"
      : "+f"(d[0]),
          "+f"(d[1]),
          "+f"(d[2]),
          "+f"(d[3]),
          "+f"(d[4]),
          "+f"(d[5]),
          "+f"(d[6]),
          "+f"(d[7]),
          "+f"(d[8]),
          "+f"(d[9]),
          "+f"(d[10]),
          "+f"(d[11]),
          "+f"(d[12]),
          "+f"(d[13]),
          "+f"(d[14]),
          "+f"(d[15]),
          "+f"(d[16]),
          "+f"(d[17]),
          "+f"(d[18]),
          "+f"(d[19]),
          "+f"(d[20]),
          "+f"(d[21]),
          "+f"(d[22]),
          "+f"(d[23]),
          "+f"(d[24]),
          "+f"(d[25]),
          "+f"(d[26]),
          "+f"(d[27]),
          "+f"(d[28]),
          "+f"(d[29]),
          "+f"(d[30]),
          "+f"(d[31])
      : "l"(da), "l"(db), "n"(scaleD));
}

// Scatter one warpgroup's 64xBN accumulator tile to C, applying alpha/beta, narrowing to
// half.
//
// wgmma f32 accumulator layout: reg i -> (g,j) = (i/4, i%4)
//   row = 16*warp + lane/4 + 8*(j/2),  col = 8*g + 2*(lane%4) + (j%2)
//
// Storing straight from that layout wastes half the write bandwidth. For a fixed (g,j/2) a
// warp's 32 lanes cover 8 *rows* (lane/4), contributing only 16 contiguous bytes to each --
// eight fragments, N*2 bytes apart. Global writes move in 32-byte sectors, so every
// fragment half-fills one. The other half is the next g iteration (cols +8..+15): the data
// exists, but in a different instruction, because adjacent columns of a row live in
// different lanes.
//
// Fix: exchange between lane pairs so each lane holds a *contiguous* run. Lanes b and b^1
// swap one of their two column-pairs across g and g+1, after which each lane holds 4
// contiguous halves (8 B) and the quad covers 16 halves = one full 32-byte sector:
//
//   before (g, g+1):  lane0[0,1][8,9]   lane1[2,3][10,11]  lane2[4,5][12,13] lane3[6,7][14,15]
//   after          :  lane0[0..3]  lane2[4..7]  lane1[8..11]  lane3[12..15]
//
// One __shfl_xor per value, and every d[] / result index stays a compile-time constant --
// indexing a local array with a runtime lane-derived value would spill it to local memory
// and cost far more than the shuffles save.
//
// The exchange runs on fp32, before narrowing, so alpha/beta and the half conversion are
// unchanged: still exactly one rounding.
template <int NREG>
__device__ __forceinline__ void store_c_tile(const float *d, half *__restrict__ C, int M, int N,
                                             int row_base, int col_base, int lane_in_wg,
                                             float alpha, float beta) {
  const int warp = lane_in_wg / 32, lane = lane_in_wg % 32;
  const int row0 = row_base + 16 * warp + lane / 4;

  // The 8B store needs a 4-element-aligned offset; col is always a multiple of 4 below, so
  // this reduces to N. Odd N keeps the original narrow-store path.
  if (N % 4 != 0) {
    const int col0 = col_base + 2 * (lane % 4);
#pragma unroll
    for (int g = 0; g < NREG / 4; ++g) {
      const int c = col0 + 8 * g;
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const int r = row0 + 8 * h;
        if (r >= M || c >= N) continue;
        half *dst = C + static_cast<long long>(r) * N + c;
        float v0 = alpha * d[4 * g + 2 * h];
        const bool pair = (c + 1 < N);
        float v1 = pair ? alpha * d[4 * g + 2 * h + 1] : 0.0f;
        if (beta != 0.0f) {
          v0 += beta * __half2float(dst[0]);
          if (pair) v1 += beta * __half2float(dst[1]);
        }
        dst[0] = __float2half(v0);
        if (pair) dst[1] = __float2half(v1);
      }
    }
    return;
  }

  const int b = lane & 3;              // position within the row-quad
  const bool odd = (b & 1) != 0;       // odd lanes keep the g+1 pair, even keep g
  const int lane_col = 4 * (2 * (b & 1) + (b >> 1));   // -> {0,8,4,12} for b = 0..3

#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const int r = row0 + 8 * h;
#pragma unroll
    for (int t = 0; t < NREG / 8; ++t) {
      // g = 2t and 2t+1; all four indices are compile-time constants here.
      const int i0 = 4 * (2 * t) + 2 * h;
      const int i1 = 4 * (2 * t + 1) + 2 * h;
      const float a0 = alpha * d[i0], a1 = alpha * d[i0 + 1];   // my g   pair
      const float e0 = alpha * d[i1], e1 = alpha * d[i1 + 1];   // my g+1 pair

      // Odd lanes hand over their g pair, even lanes their g+1 pair.
      // Warp-collective: must precede any divergent branch.
      const float r0 = __shfl_xor_sync(0xffffffffu, odd ? a0 : e0, 1);
      const float r1 = __shfl_xor_sync(0xffffffffu, odd ? a1 : e1, 1);

      float v0, v1, v2, v3;
      if (odd) {  // partner's g+1 pair, then mine -> cols 8g+8 .. +11 / +12 .. +15
        v0 = r0; v1 = r1; v2 = e0; v3 = e1;
      } else {    // mine, then partner's g pair -> cols 8g+0 .. +3 / +4 .. +7
        v0 = a0; v1 = a1; v2 = r0; v3 = r1;
      }

      const int c = col_base + 16 * t + lane_col;
      if (r >= M || c >= N) continue;
      half *dst = C + static_cast<long long>(r) * N + c;

      if (c + 4 <= N) {  // full 8B store: the quad's four of these tile one 32B sector
        if (beta != 0.0f) {
          const uint2 prev = *reinterpret_cast<const uint2 *>(dst);
          const half2 *ph = reinterpret_cast<const half2 *>(&prev);
          const float2 o0 = __half22float2(ph[0]), o1 = __half22float2(ph[1]);
          v0 += beta * o0.x;
          v1 += beta * o0.y;
          v2 += beta * o1.x;
          v3 += beta * o1.y;
        }
        union {
          uint2 u;
          half2 h2[2];
        } out;
        out.h2[0] = __floats2half2_rn(v0, v1);
        out.h2[1] = __floats2half2_rn(v2, v3);
        *reinterpret_cast<uint2 *>(dst) = out.u;
      } else {  // ragged tail of the last N tile
        const float vv[4] = {v0, v1, v2, v3};
#pragma unroll
        for (int p = 0; p < 4; ++p) {
          if (c + p >= N) break;
          float v = vv[p];
          if (beta != 0.0f) v += beta * __half2float(dst[p]);
          dst[p] = __float2half(v);
        }
      }
    }
  }
}

// TMA-store epilogue. Stages the accumulator into smem in the 128B-swizzled layout the
// TMA engine expects, then hands each 64x64 chunk to the engine. The warps never touch
// global memory, so they return to the mainloop while the store drains in the background --
// which is the whole point: the shuffle epilogue below already hits ~2.9 TB/s (peak write
// bandwidth), it is just fully serialized against the math.
//
// Ragged M/N need no guards here: TMA clips stores to the tensor map's globalDim.
// beta != 0 is NOT handled -- a bulk store cannot read-modify-write -- so that case keeps
// the shuffle path.
template <int BN_>
__device__ __forceinline__ void store_c_tile_tma(const float *d, const CUtensorMap *tmap_c,
                                                 half *sEpi, int row_base, int tile_n, int cwg,
                                                 int lane_in_wg, float alpha, uint64_t pol_c) {
  const int warp = lane_in_wg / 32, lane = lane_in_wg % 32;
  const int r_loc = 16 * warp + lane / 4;   // row within this WG's 64, before the +8h
  const int b = lane % 4;
  const int bar_id = 1 + cwg;

#pragma unroll
  for (int j = 0; j < BN_ / EPI_COLS; ++j) {
    half *buf = sEpi + (cwg * 2 + (j & 1)) * EPI_CHUNK_ELEMS;

    // Reclaim this buffer: with <=1 group outstanding, the store from two chunks ago (which
    // used this same buffer) has finished reading it. Groups are per-thread, so the lane
    // that issued them must be the one to wait; the rest join at the barrier.
    if (lane_in_wg == 0) tma_store_wait<1>();
    wg_barrier(bar_id);

#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int r = r_loc + 8 * h;
      const int sw = r & 7;             // 128B swizzle: XOR the 16B-chunk index with row%8
#pragma unroll
      for (int a = 0; a < 8; ++a) {     // a = g - 8j, the 8-column group within this chunk
        const int i = 4 * (8 * j + a) + 2 * h;
        const int off = r * EPI_COLS + ((a ^ sw) * 8) + 2 * b;
        *reinterpret_cast<half2 *>(buf + off) =
            __floats2half2_rn(alpha * d[i], alpha * d[i + 1]);
      }
    }

    // Make the generic-proxy smem writes visible to the async proxy before the engine reads.
    asm volatile("fence.proxy.async.shared::cta;");
    wg_barrier(bar_id);
    if (lane_in_wg == 0) {
#if CFG_L2_HINT
      if (pol_c)
        tma_store_2d_hint(tmap_c, tile_n * BN_ + EPI_COLS * j, row_base, buf, pol_c);
      else
#endif
        tma_store_2d(tmap_c, tile_n * BN_ + EPI_COLS * j, row_base, buf);
      tma_store_commit();
    }
  }
}

// ------------------------------------------------------------------------- the kernel
// PERSISTENT: the grid is sized to what the GPU holds resident (one CTA per SM, since the
// smem budget admits only one), and each CTA walks a strided slice of the output tiles.
// That removes wave quantization -- with a per-tile grid, 8192^3 launches 2048 CTAs over
// 132 SMs, so the final wave runs ~half empty and every CTA re-pays the prologue.
//
// The pipeline counter `it` is monotonic across the *whole* CTA's work, not reset per
// tile, so stage/phase bookkeeping carries across tile boundaries and the producer keeps
// streaming the next tile's k-tiles while consumers are still finishing the current one.
// Only the accumulator reset and the final wgmma_wait<0> are per-tile.
//
// CLUSTER_M_ CTAs form a cluster covering adjacent tile-rows at the same tile_n, so they
// all consume the identical B tile; each multicasts a 1/CLUSTER_M_ slice to the cluster.
template <int CLUSTER_M_, int BM_, int BN_>
__global__ __launch_bounds__(Cfg<BM_, BN_>::THREADS, 1) void gemm_kernel(
    const __grid_constant__ CUtensorMap tmap_a, const __grid_constant__ CUtensorMap tmap_b,
    const __grid_constant__ CUtensorMap tmap_c, half *__restrict__ C, int M, int N, int K,
    int tiles_m, int tiles_n, int total_ctiles, int cgroup_m, float alpha, float beta,
    bool tma_epilogue) {
  using C_ = Cfg<BM_, BN_>;
  constexpr int STAGES_ = C_::STAGES;
  constexpr int CONSUMERS_ = C_::CONSUMERS;
  static_assert(C_::BLOCKS >= CLUSTER_M_ && C_::BLOCKS % CLUSTER_M_ == 0,
                "B tile must split evenly across the cluster, else CTAs load nothing and hang");
  constexpr int BLOCKS_PER_CTA = C_::BLOCKS / CLUSTER_M_;  // B blocks this CTA multicasts
  constexpr uint16_t MC_MASK = (uint16_t)((1u << CLUSTER_M_) - 1u);

  extern __shared__ __align__(1024) uint8_t smem_raw[];
  half *sA = reinterpret_cast<half *>(smem_raw);
  half *sB = sA + STAGES_ * C_::A_STAGE;
  half *sEpi = sB + STAGES_ * C_::B_STAGE;   // 1024B-aligned: all preceding sizes are
  uint64_t *bar_full = reinterpret_cast<uint64_t *>(sEpi + C_::EPI);
  uint64_t *bar_empty = bar_full + STAGES_;

  const int tid = threadIdx.x;
  const int wg = tid / WGS;
  const int lane_in_wg = tid % WGS;
  // Pick which operand to keep resident from the actual operand sizes: the smaller one, if it
  // passes the gate. Everything else is left *untagged* -- it emits the plain instruction with
  // no descriptor at all. That is measured, not stylistic: tagging the streaming operand
  // `evict_first` cost -16%, and even an explicit `evict_normal` cost -27% at 1024^3.
  const size_t l2_bytes_a = (size_t)M * (size_t)K * sizeof(half);
  const size_t l2_bytes_b = (size_t)K * (size_t)N * sizeof(half);
  const bool l2_pin_b =
      (l2_bytes_b <= CFG_L2_PIN_MAX) && (l2_bytes_b * CFG_L2_PIN_RATIO <= l2_bytes_a);
  const bool l2_pin_a = !l2_pin_b && (l2_bytes_a <= CFG_L2_PIN_MAX) &&
                        (l2_bytes_a * CFG_L2_PIN_RATIO <= l2_bytes_b);
  const bool l2_pinning = l2_pin_a || l2_pin_b;
  // Only ever evict_last, and only for the pinned operand; the other branch never reads these.
  const uint64_t pol_a = l2_pin_a ? l2_policy_evict_last() : 0ull;
  const uint64_t pol_b = l2_pin_b ? l2_policy_evict_last() : 0ull;
  // Only push C out of the way when there is actually something to protect.
  const uint64_t pol_c = l2_pinning ? l2_policy_evict_first() : 0ull;  // 0 == use plain store
  const uint32_t rank = (CLUSTER_M_ == 1) ? 0u : cta_rank_in_cluster();

  const int ctiles_m = tiles_m / CLUSTER_M_;
  const int group_sz = (cgroup_m > 0 ? cgroup_m : 1) * tiles_n;
  const int my_cluster = blockIdx.x / CLUSTER_M_;      // this cluster's slot
  const int num_clusters = gridDim.x / CLUSTER_M_;     // stride through the work list

  if (tid == 0) {
    #pragma unroll
    for (int s = 0; s < STAGES_; ++s) {
      bar_init(&bar_full[s], 1);
      bar_init(&bar_empty[s], CONSUMERS_ * CLUSTER_M_);
    }
  }
  asm volatile("fence.proxy.async.shared::cta;");
  if constexpr (CLUSTER_M_ > 1) {
    cluster_sync();
  } else {
    __syncthreads();
  }

  const int num_k_tiles = (K + BK - 1) / BK;

  // ── The four levels of work decomposition ────────────────────────────────
  // Easy to conflate, so stated once, outermost to innermost:
  //
  //   CTA      one BM x BN output tile of C. 384 threads: 1 producer + 2 consumer
  //            warpgroups. This is the unit the hardware schedules onto an SM.
  //
  //   cluster  CLUSTER_M_ CTAs stacked along M at the SAME tile_n. They consume the
  //            identical B tile, so one TMA multicast fills all of them -- that is the
  //            whole reason clusters exist here. A cluster is the unit of *work
  //            assignment*: the loop below hands out clusters, not CTAs.
  //
  //   w        linear index into the flat list of cluster-tiles,
  //            total_ctiles = (tiles_m / CLUSTER_M_) * tiles_n. `decode(w)` turns it
  //            into (tile_m, tile_n). The persistent loop is
  //                for (w = my_cluster; w < total_ctiles; w += num_clusters)
  //            so cluster 0 takes w = 0, 66, 132..., cluster 1 takes 1, 67, 133...
  //            KEY CONSEQUENCE: at any instant the num_clusters resident clusters hold
  //            num_clusters *consecutive* values of w. The ORDER of w therefore decides
  //            which tiles are co-resident, hence what hits in L2. That is the only
  //            thing group-M rasterization controls.
  //
  //   group    a band of the C tile grid, cgroup_m cluster-rows tall by ALL tiles_n
  //            columns, where cgroup_m = GROUP_M / CLUSTER_M_. Purely a numbering
  //            device -- no hardware knows it exists. It fixes the order of w.
  //
  //   containment:  group  >  cluster  >  CTA
  //                 one group = cgroup_m * tiles_n clusters
  //                           = cgroup_m * tiles_n * CLUSTER_M_ CTAs
  //
  // Why the group height matters. With SM CTAs resident, the concurrently-executing
  // tiles form a GROUP_M x (SM/GROUP_M) rectangle of the C grid. Those CTAs together
  // need GROUP_M A-tiles (one per M row) and SM/GROUP_M B-tiles (one per N column), so
  // the bytes they share is  GROUP_M*BM*K + (SM/GROUP_M)*BN*K.  Minimising that:
  //
  //     g* = sqrt(SM * BN / BM)      -- M, N and K all cancel; see report section 16
  //        = 16.2  at BN=256,  8.1 at BN=64   (SM=132, BM=128)
  //
  // ...but only where re-fetching actually happens. When A+B fits in L2 both operands
  // stay resident, nothing is ever re-read, and the rasterization order has no traffic
  // to reduce -- which is why the short-K and small-grid arms of the policy pick 1.
  //
  // Decode a linear work index w into this CTA's (tile_m, tile_n).
  //
  // ── Group-M rasterization ────────────────────────────────────────────────
  // The output tile grid is partitioned into row-bands called groups. Each
  // group spans cgroup_m cluster-M rows and ALL tiles_n N-tile columns:
  //
  //   group_sz = cgroup_m × tiles_n    (total work indices per group)
  //
  // Within a group, M is the FAST index and N is the SLOW index. This keeps
  // the same B tile [K×n] hot in L2 across cgroup_m consecutive M-tiles:
  //
  //   w  →  gid    = w / group_sz         (which group)
  //         in_grp = w mod group_sz        (position within the group)
  //                        │
  //           ┌────────────┴─────────────┐
  //           │                          │
  //   tile_n = in_grp / gcm         M_off = in_grp % gcm        (fast)
  //   (slow, outer)                        │
  //                          tile_m = (first_cm + M_off) * CLUSTER_M_ + rank
  //
  // Concrete example — cgroup_m=2, tiles_n=3, CLUSTER_M_=2, ctiles_m=5:
  //
  //   group_sz = 2 × 3 = 6
  //
  //        n=0     n=1     n=2        (tiles_n = 3 columns)
  //      ┌───────┬───────┬───────┐
  //  cm0 │  w=0  │  w=2  │  w=4  │ ─┐
  //      ├───────┼───────┼───────┤  │ group 0  (6 work indices)
  //  cm1 │  w=1  │  w=3  │  w=5  │ ─┘
  //      ╞═══════╪═══════╪═══════╡
  //  cm2 │  w=6  │  w=8  │  w=10 │ ─┐
  //      ├───────┼───────┼───────┤  │ group 1
  //  cm3 │  w=7  │  w=9  │  w=11 │ ─┘
  //      ╞═══════╪═══════╪═══════╡
  //  cm4 │  w=12 │  w=13 │  w=14 │ ── group 2 (partial, gcm=1; see below)
  //      └───────┴───────┴───────┘
  //
  //  Traversal within group 0, rank=0  (first_cm=0, gcm=2):
  //
  //   w  in_grp  M_off  tile_n  tile_m   note
  //   0    0       0      0       0
  //   1    1       1      0       2     ← same N=0: B tile[K×0] stays in L2
  //   2    2       0      1       0
  //   3    3       1      1       2     ← same N=1: B tile[K×1] stays in L2
  //   4    4       0      2       0
  //   5    5       1      2       2     ← same N=2: B tile[K×2] stays in L2
  //
  // ── Boundary: last group may be partial (gcm < cgroup_m) ─────────────────
  // gcm = min(ctiles_m - first_cm, cgroup_m) clamps the last group when
  // ctiles_m is not a multiple of cgroup_m. With gcm=1 (group 2 above,
  // only cm4 remains), M_off = in_grp % 1 = 0 always, so tile_m is fixed
  // and only tile_n advances — the group degenerates to a single M-row:
  //
  //   w  in_grp  gcm  M_off  tile_n  tile_m
  //  12    0      1     0      0       8    ← M fixed at cm4
  //  13    1      1     0      1       8
  //  14    2      1     0      2       8    ← only N advances
  //
  // ── Cluster layout (CLUSTER_M_ = 2) ─────────────────────────────────────
  // A cluster holds CLUSTER_M_ CTAs. All decode the same w; only `rank`
  // differs (0-based CTA index within the cluster). Let p = first_cm + M_off.
  //
  //  A [M × K]                      B [K × N]
  //                                  tile_n*BN      (tile_n+1)*BN
  //  p*2*BM ┌──────────────────┐          │               │
  //         │     rank 0       │          ▼               ▼
  //         │  tile_m = p*2+0  │    ┌──────────┬──────────┐
  //         │  loads BM rows   │    │  rank 0  │  rank 1  │  ← each CTA loads
  //    +BM  ├──────────────────┤    │ loads    │ loads    │     half of the
  //         │     rank 1       │    │ its half │ its half │     B slice via TMA,
  //         │  tile_m = p*2+1  │    └──────────┴──────────┘     then multicasts
  //         │  loads BM rows   │          │               │     to the other CTA
  //  +2*BM  └──────────────────┘          ▼               ▼
  //         (independent loads,    ┌─────────────────────────┐
  //          different M rows)     │  both CTAs' smem have   │
  //                                │  the FULL B slice after │
  //                                │  multicast; HBM read    │
  //                                │  only once total        │
  //                                └─────────────────────────┘
  //
  //  Each CTA then computes its own C output tile:
  //
  //        tile_n*BN ◄──── BN ────► (tile_n+1)*BN
  //              ┌─────────────────────┐
  //   p*2*BM     │       rank 0        │  C[p*2*BM   : +BM, tile_n*BN : +BN]
  //              ├─────────────────────┤       ↑ both use the same
  //   p*2*BM+BM  │       rank 1        │  C[p*2*BM+BM: +BM, tile_n*BN : +BN]
  //              └─────────────────────┘         full B tile in smem
  auto decode = [&](int w, int &tile_m, int &tile_n) {
    const int gid     = w / group_sz;             // which group of cgroup_m cluster-M rows
    const int in_grp  = w - gid * group_sz;       // position within the group [0, group_sz)
    const int first_cm = gid * cgroup_m;          // first cluster-M index of this group
    const int gcm     = min(ctiles_m - first_cm, cgroup_m);  // actual M rows (clamped at boundary)
    tile_m = (first_cm + in_grp % gcm) * CLUSTER_M_ + rank;  // fast M index, expanded by cluster rank
    tile_n = in_grp / gcm;                        // slow N index
  };

  if (wg == 0) {
    // ------------------------------------------------------------------- producer WG
    setmaxnreg_dec<CFG_PRODUCER_REGS>();
    if (lane_in_wg == 0) {
      int it = 0;
      for (int w = my_cluster; w < total_ctiles; w += num_clusters) {
        int tile_m, tile_n;
        decode(w, tile_m, tile_n);
        for (int kt = 0; kt < num_k_tiles; ++kt, ++it) {
          const int s = it % STAGES_;
          if (it >= STAGES_) bar_wait(&bar_empty[s], ((it / STAGES_) - 1) & 1);
          bar_expect(&bar_full[s], C_::TMA_B);
#if CFG_L2_HINT
          if (l2_pin_a)
            tma_2d_hint(sA + s * C_::A_STAGE, &tmap_a, kt * BK, tile_m * BM_, &bar_full[s], pol_a);
          else
#endif
            tma_2d(sA + s * C_::A_STAGE, &tmap_a, kt * BK, tile_m * BM_, &bar_full[s]);
          // B is N-major: the tile arrives as B_BLOCKS contiguous [BK][64] blocks. Under a
          // cluster each CTA multicasts its share of the blocks to the whole cluster.
          half *bdst = sB + s * C_::B_STAGE;
#pragma unroll
          for (int j = 0; j < BLOCKS_PER_CTA; ++j) {
            const int blk = rank * BLOCKS_PER_CTA + j;
            const int ncol = tile_n * BN_ + blk * 64;
            if constexpr (CLUSTER_M_ == 1) {
#if CFG_L2_HINT
              if (l2_pin_b)
                tma_2d_hint(bdst + blk * B_BLOCK_ELEMS, &tmap_b, ncol, kt * BK, &bar_full[s], pol_b);
              else
#endif
                tma_2d(bdst + blk * B_BLOCK_ELEMS, &tmap_b, ncol, kt * BK, &bar_full[s]);
            } else {
#if CFG_L2_HINT
              if (l2_pin_b)
                tma_2d_multicast_hint(bdst + blk * B_BLOCK_ELEMS, &tmap_b, ncol, kt * BK,
                                      &bar_full[s], MC_MASK, pol_b);
              else
#endif
                tma_2d_multicast(bdst + blk * B_BLOCK_ELEMS, &tmap_b, ncol, kt * BK, &bar_full[s],
                                 MC_MASK);
            }
          }
        }
      }
    }
  } else {
    // ------------------------------------------------------------------ consumer WGs
    setmaxnreg_inc<CFG_CONSUMER_REGS>();
    const int cwg = wg - 1;
    float d[C_::NREG];
    int it = 0;

    for (int w = my_cluster; w < total_ctiles; w += num_clusters) {
      int tile_m, tile_n;
      decode(w, tile_m, tile_n);

#pragma unroll
      for (int i = 0; i < C_::NREG; ++i) d[i] = 0.0f;
      wgmma_fence();

#if CFG_L2_HINT
      (void)0;
#endif
      int prev_stage = -1;
      for (int kt = 0; kt < num_k_tiles; ++kt, ++it) {
        const int s = it % STAGES_;
        bar_wait(&bar_full[s], (it / STAGES_) & 1);

        // A rows [cwg*64, cwg*64+64) of this stage; smem rows are 128B apart.
        const uint32_t a_base = smem_u32(sA + s * C_::A_STAGE) + cwg * WG_M * BK * 2;
        const uint32_t b_base = smem_u32(sB + s * C_::B_STAGE);

        wgmma_fence();
#if CFG_HOIST_DESC
        // Hoist the descriptor build out of the k16 loop. Only bits [13:0] (addr>>4) vary
        // with ks, and the step is a compile-time constant, so one base descriptor per
        // operand plus an immediate add replaces a full field rebuild per issue:
        //   make_desc(a_base + ks*32)      == make_desc(a_base)    + ks*2
        //   make_desc_mn(b_base + ks*2048) == make_desc_mn(b_base) + ks*(MN_K_STRIDE/16)
        const uint64_t da_base = make_desc(a_base);
        const uint64_t db_base = make_desc_mn(b_base);
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
          const uint64_t da = da_base + (uint64_t)(ks * 2);
          const uint64_t db = db_base + (uint64_t)(ks * (MN_K_STRIDE / 16));
          if constexpr (BN_ == 256)
            wgmma_m64n256k16<1>(d, da, db);
          else if constexpr (BN_ == 128)
            wgmma_m64n128k16<1>(d, da, db);
          else
            wgmma_m64n64k16<1>(d, da, db);
        }
#else
#pragma unroll
        for (int ks = 0; ks < BK / 16; ++ks) {
          // A is K-major: +16 halves along K == +32B. B is N-major: a k16 step crosses 16
          // rows of a 128B-wide block == +2048B.
          if constexpr (BN_ == 256)
            wgmma_m64n256k16<1>(d, make_desc(a_base + ks * 32),
                                make_desc_mn(b_base + ks * MN_K_STRIDE));
          else if constexpr (BN_ == 128)
            wgmma_m64n128k16<1>(d, make_desc(a_base + ks * 32),
                                make_desc_mn(b_base + ks * MN_K_STRIDE));
          else
            wgmma_m64n64k16<1>(d, make_desc(a_base + ks * 32),
                               make_desc_mn(b_base + ks * MN_K_STRIDE));
        }
#endif
        wgmma_commit();

        // Retire the *previous* group only, so this stage's math overlaps the next TMA.
        wgmma_wait<1>();
        if (prev_stage >= 0 && lane_in_wg == 0) release_stage<CLUSTER_M_>(bar_empty, prev_stage);
        prev_stage = s;
      }
      wgmma_wait<0>();
      // Release before the epilogue: it only touches registers, so the producer can run
      // ahead into the next tile while this warpgroup writes C.
      if (prev_stage >= 0 && lane_in_wg == 0) release_stage<CLUSTER_M_>(bar_empty, prev_stage);

      if (tma_epilogue) {
        store_c_tile_tma<BN_>(d, &tmap_c, sEpi, tile_m * BM_ + cwg * WG_M, tile_n, cwg, lane_in_wg,
                              alpha, pol_c);
      } else {
        store_c_tile<C_::NREG>(d, C, M, N, tile_m * BM_ + cwg * WG_M, tile_n * BN_, lane_in_wg,
                               alpha, beta);
      }
    }
    // Do not retire while the engine may still be reading our staging buffers.
    if (tma_epilogue && lane_in_wg == 0) tma_store_wait<0>();
  }

  if constexpr (CLUSTER_M_ > 1) cluster_sync();
}

// ------------------------------------------- portable fallback for unsupported shapes
constexpr int FB = 16;
__global__ __launch_bounds__(FB *FB) void gemm_fallback(const half *__restrict__ A,
                                                        const half *__restrict__ B,
                                                        half *__restrict__ C, int M, int N, int K,
                                                        float alpha, float beta) {
  __shared__ float sa[FB][FB + 1], sb[FB][FB + 1];
  const int row = blockIdx.y * FB + threadIdx.y, col = blockIdx.x * FB + threadIdx.x;
  float acc = 0.0f;
  for (int t = 0; t < (K + FB - 1) / FB; ++t) {
    const int ka = t * FB + threadIdx.x, kb = t * FB + threadIdx.y;
    sa[threadIdx.y][threadIdx.x] =
        (row < M && ka < K) ? __half2float(A[static_cast<long long>(row) * K + ka]) : 0.0f;
    sb[threadIdx.y][threadIdx.x] =
        (kb < K && col < N) ? __half2float(B[static_cast<long long>(kb) * N + col]) : 0.0f;
    __syncthreads();
#pragma unroll
    for (int i = 0; i < FB; ++i) acc += sa[threadIdx.y][i] * sb[i][threadIdx.x];
    __syncthreads();
  }
  if (row < M && col < N) {
    const long long o = static_cast<long long>(row) * N + col;
    float v = alpha * acc;
    if (beta != 0.0f) v += beta * __half2float(C[o]);
    C[o] = __float2half(v);
  }
}

// Widen a row-major matrix's row stride to `ld` so TMA's 16B stride rule holds. Only the
// live columns are copied -- TMA clamps reads to globalDim, so the pad is never read.
__global__ __launch_bounds__(256) void pad_copy_kernel(const half *__restrict__ src,
                                                       half *__restrict__ dst, int rows, int cols,
                                                       int ld) {
  const int c = blockIdx.x * 256 + threadIdx.x;
  if (c >= cols) return;
  for (int r = blockIdx.y; r < rows; r += gridDim.y)
    dst[static_cast<long long>(r) * ld + c] = src[static_cast<long long>(r) * cols + c];
}

// --------------------------------------------------------------------- host plumbing
using EncodeTiledFn = CUresult (*)(CUtensorMap *, CUtensorMapDataType, cuuint32_t, void *,
                                   const cuuint64_t *, const cuuint64_t *, const cuuint32_t *,
                                   const cuuint32_t *, CUtensorMapInterleave, CUtensorMapSwizzle,
                                   CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

// Resolved through the runtime so the translation unit does not have to link libcuda.
static EncodeTiledFn get_encode_tiled() {
  static EncodeTiledFn fn = [] {
    void *p = nullptr;
#if CUDART_VERSION >= 12050
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPointByVersion("cuTensorMapEncodeTiled", &p, 12000, cudaEnableDefault, &q);
#else
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &p, cudaEnableDefault);
#endif
    return reinterpret_cast<EncodeTiledFn>(p);
  }();
  return fn;
}

// Row-major [rows, cols] half -> 2D tile map, box {box_c, box_r}, 128B swizzle.
static bool make_tensor_map(CUtensorMap *map, const void *ptr, int rows, int cols, int box_r,
                            int box_c, int ld) {
  EncodeTiledFn enc = get_encode_tiled();
  if (!enc) return false;
  uint64_t gdim[2] = {static_cast<uint64_t>(cols), static_cast<uint64_t>(rows)};
  uint64_t gstr[1] = {static_cast<uint64_t>(ld) * sizeof(half)};
  uint32_t bdim[2] = {static_cast<uint32_t>(box_c), static_cast<uint32_t>(box_r)};
  uint32_t estr[2] = {1, 1};
  return enc(map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, const_cast<void *>(ptr), gdim, gstr, bdim,
             estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
             CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
             CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

// TMA needs 16B-aligned base addresses; the 16B row-stride rule (K%8 for A, N%8 for B) is
// satisfied by pad-copying into a wider workspace when the caller's stride does not comply.
static bool tma_compatible(const half *A, const half *B, const half *C, int M, int N, int K) {
  if (reinterpret_cast<uintptr_t>(A) % 16 || reinterpret_cast<uintptr_t>(B) % 16 ||
      reinterpret_cast<uintptr_t>(C) % 16)
    return false;
  return M > 0 && N > 0 && K > 0;
}

// Scratch for stride-padded copies of A and/or B, grown on demand and reused.
static half *g_pad = nullptr;
static size_t g_pad_bytes = 0;
static half *pad_workspace(size_t bytes) {
  if (bytes > g_pad_bytes) {
    if (g_pad) cudaFree(g_pad);
    if (cudaMalloc(&g_pad, bytes) != cudaSuccess) {
      g_pad = nullptr;
      g_pad_bytes = 0;
      return nullptr;
    }
    g_pad_bytes = bytes;
  }
  return g_pad;
}

template <int BM_, int BN_>
static bool launch_tile(const half *A, const half *B, half *C, int M, int N, int K, int lda,
                        int ldb, float alpha, float beta, cudaStream_t stream, int sm_count,
                        int group_m) {
  using C_ = Cfg<BM_, BN_>;
  constexpr int SMEM = C_::SMEM;
  constexpr int THREADS = C_::THREADS;

  const int tiles_m = (M + BM_ - 1) / BM_, tiles_n = (N + BN_ - 1) / BN_;
  if (tiles_n <= 0 || tiles_m <= 0) return true;
  // The B tile is multicast as BLOCKS 64-wide pieces, one share per CTA, so a cluster needs
  // at least CLUSTER_M of them. BN=64 has exactly one: clustering it would hand every CTA
  // zero blocks, B would never arrive, and the consumers would hang on bar_full. Decide that
  // at compile time so the impossible instantiation is never even formed.
  constexpr int CM = (C_::BLOCKS >= CLUSTER_M && C_::BLOCKS % CLUSTER_M == 0) ? CLUSTER_M : 1;
  // Whether to cluster is a SHAPE decision, and until now it was an accident: the old
  // condition included `group_m % CM == 0`, so any odd GROUP_M silently took the
  // non-clustered launch below. That coupling meant "no cluster" could only be requested by
  // also giving up grouping, and the two are independent.
  //
  // The cluster's cost is per tile (cluster launch, cluster_sync, mbarriers that must collect
  // CONSUMERS*CLUSTER_M arrivals). Its benefit -- halving B's L2->SM traffic by multicasting
  // one B tile to both CTAs -- is per k-tile. So it needs a long enough k-loop to pay for
  // itself. Measured at matched M-grouping, cluster on vs off:
  //
  //   K=256   4096x8192x256      -5.2%      K=2048   32768x8192x2048   +11.7%
  //   K=512   2048x2048x512     -11.3%      K=4096   8192x8192x4096     +3.6%
  //   K=512   4096x8192x512      -2.5%      K=8192   8192x1024x8192    +27.8%
  //   K=4096  4096x4096x4096     -0.6%      K=16384  16384x16384x16384  0.0%
  //
  // But K alone does not separate: at K=2048 the cluster is +7.6% on 32768x8192x2048
  // (tiles_m=256) and -8.4% on 2048x2048x2048 (tiles_m=16). It needs enough M-tiles for the
  // multicast to be reused as well as enough k-tiles to amortise the setup, so both:
  //   K >= CFG_CLUSTER_MIN_K  and  tiles_m >= CFG_CLUSTER_MIN_TM
  const bool cluster_ok   = (CM > 1) && (tiles_m % CM == 0);
  // Third case: a narrow N. B is re-read once per M-tile row, so with few N-tiles that
  // traffic dominates and halving it pays regardless of K. 3000x1000x2000 (tiles_n=4) loses
  // 14.6pp without the cluster; 8192x1024x8192 (tiles_n=4) gains 27.8% with it.
  //
  // ALL THREE CLAUSES ARE PROXIES. The actual criterion is:
  //
  //     enable CGA only when the UNCLUSTERED kernel is already L2-read-bandwidth-bound,
  //     i.e. when its L2 read demand exceeds ~7 TB/s (H100 sustained L2 read bandwidth).
  //
  //     predicted L2 demand = unclustered FLOP/s * C_::L2_BYTES_PER_FLOP
  //     for 128x256x64 that is  TFLOPS * 0.01171875,  so the crossover is ~597 TFLOPS
  //
  // Why that is the criterion: CGA here is a *bandwidth* optimisation, not a compute one.
  // Its only benefit is that the two CTAs share one B tile by TMA multicast, cutting B-side
  // L2->SM traffic by 50% -- but only 1/3 of total L2 traffic, because each CTA still loads
  // its own A. Its cost is unconditional: the cluster is GPC-co-resident, launched and
  // retired as a unit, and synchronised for multicast (release_stage issues CLUSTER_M remote
  // arrivals *per k-tile*, and bar_empty collects CONSUMERS*CLUSTER_M of them). Below the
  // wall you pay all of that and the saved bandwidth buys nothing.
  //
  // A dispatcher cannot know the unclustered throughput before running, so the shape tests
  // below stand in for it. They separate 11 of 12 measured shapes; 4096x4096x4096 is the
  // known false positive (6.23 TB/s, just under the wall, -1.2%).
  //
  // Measured demand vs measured cluster effect, 12/12 either side of ~7 TB/s:
  //     3.2-7.0 TB/s  ->  -16.0% .. -0.2%      7.5-8.1 TB/s  ->  +0.9% .. +10.8%
  //
  // See report section 16 for the full table and derivation.
  const bool cluster_want = ((K >= CFG_CLUSTER_MIN_K) && (tiles_m >= CFG_CLUSTER_MIN_TM))
                            || (tiles_n <= CFG_CLUSTER_NARROW_N);
  const bool use_cluster  = cluster_ok && cluster_want;
  // Grouping is in cluster-rows when clustered, tile-rows when not. Round down rather than
  // rejecting odd values -- GROUP_M must not decide clustering any more.
  const int cgroup_m = use_cluster ? max(1, group_m / CM) : group_m;

  CUtensorMap tmap_a, tmap_b, tmap_c;
  if (!make_tensor_map(&tmap_a, A, M, K, BM_, BK, lda) ||
      !make_tensor_map(&tmap_b, B, K, N, BK, 64, ldb))
    return false;
  const bool tma_epi =
      (beta == 0.0f) && (N % 8 == 0) && make_tensor_map(&tmap_c, C, M, N, EPI_ROWS, EPI_COLS, N);
  if (!tma_epi) tmap_c = tmap_a;

  static bool attr_set = [] {
    bool ok = cudaFuncSetAttribute(gemm_kernel<1, BM_, BN_>,
                                   cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess;
    if (CM > 1)
      ok &= cudaFuncSetAttribute(gemm_kernel<CM, BM_, BN_>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM) == cudaSuccess;
    return ok;
  }();
  if (!attr_set) return false;

  if (!use_cluster) {
    const int total_ctiles = tiles_m * tiles_n;
    const int grid = min(total_ctiles, sm_count);
    gemm_kernel<1, BM_, BN_><<<grid, THREADS, SMEM, stream>>>(
        tmap_a, tmap_b, tmap_c, C, M, N, K, tiles_m, tiles_n, total_ctiles, cgroup_m, alpha, beta,
        tma_epi);
    return true;
  }
  const int total_ctiles = (tiles_m / CM) * tiles_n;
  const int grid = min(total_ctiles * CM, (sm_count / CM) * CM);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid, 1, 1);
  cfg.blockDim = dim3(THREADS, 1, 1);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim = {CM, 1, 1};
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, gemm_kernel<CM, BM_, BN_>, tmap_a, tmap_b, tmap_c, C, M, N, K,
                     tiles_m, tiles_n, total_ctiles, cgroup_m, alpha, beta, tma_epi);
  return true;
}

// Tile selection. The wide tile has ~7% better arithmetic intensity, so prefer it -- but
// only when it produces enough tiles to fill the machine. Below that, occupancy dominates
// and the narrow tile (2x the tiles, plus 2 extra pipeline stages) wins by a lot: 1024^3
// goes from 32 CTAs on 132 SMs to 64.
bool launch_mainloop(const half *A, const half *B, half *C, int M, int N, int K, int lda, int ldb,
                     float alpha, float beta, cudaStream_t stream, int force_bm = 0,
                     int force_bn = 0, int group_m = 0) {
  static int sm_count = [] {
    int n = 132;
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0);
    return n;
  }();
#if defined(CFG_FORCE_BM) && defined(CFG_FORCE_BN)
  // The policy below lives in the #else branch, so resolve the auto sentinel here too --
  // passing group_m == 0 through would make group_sz zero and divide by zero in the decode.
  if (group_m == 0) {
    if (CFG_FORCE_BN == 64)  group_m = 2;
    else if (K <= 128)       group_m = 1;
    else if (K <= 512)       group_m = 4;
    else if (N <= 512)       group_m = 16;
    else                     group_m = GROUP_M;
  }
  return launch_tile<CFG_FORCE_BM, CFG_FORCE_BN>(A, B, C, M, N, K, lda, ldb, alpha, beta, stream,
                                                 sm_count, group_m);
#else
  // Explicit tile, used by the autotuner to probe a candidate.
  if (force_bm == 128 && force_bn == 256)
    return launch_tile<128, 256>(A, B, C, M, N, K, lda, ldb, alpha, beta, stream, sm_count, group_m);
  if (force_bm == 128 && force_bn == 128)
    return launch_tile<128, 128>(A, B, C, M, N, K, lda, ldb, alpha, beta, stream, sm_count, group_m);
  if (force_bm == 128 && force_bn == 64)
    return launch_tile<128, 64>(A, B, C, M, N, K, lda, ldb, alpha, beta, stream, sm_count, group_m);
  // Pick the largest tile that still fills the machine. Bigger tiles have better arithmetic
  // intensity and spend a smaller fraction of the CTA on the producer, so step down only
  // when there is not enough work to go around. Thresholds measured, not assumed.
  // Widest tile that still produces enough work to fill the machine. Wider tiles have better
  // arithmetic intensity, so step down only when starved. BM stays 128 throughout: narrowing
  // BN keeps two consumer warpgroups per CTA, whereas dropping to BM=64 halves them, and the
  // quantity that actually tracks throughput is *warpgroups*, not CTAs. Measured at 1024^3:
  //   128x256 -> 64 WGs -> 140 TF   128x128 -> 128 WGs -> 205 TF
  //   64x128  -> 128 WGs -> 231 TF  128x64  -> 256 WGs -> 314 TF
  // BM=64 was measured for every shape below and never won.
  // Threshold (2/3 of the SM count) is read off that sweep, not assumed.
  auto tiles = [&](int bm, int bn) { return ((M + bm - 1) / bm) * ((N + bn - 1) / bn); };
  const int need = (sm_count * 2 + 2) / 3;
  const int bn = (tiles(128, 256) >= need) ? 256 : ((tiles(128, 128) >= need) ? 128 : 64);

  // GROUP_M policy. Derived from a stability study, not a single sweep: each shape's best
  // GROUP_M was searched three independent times, and only 16 of 31 shapes picked the same
  // winner every time. The rest are inside the noise band and are deliberately left at the
  // default rather than fitted. The three rules below cover every case that was BOTH stable
  // across all three trials AND worth more than 2%:
  //
  //   BN==64  -> 2    narrow tile / tiny problems   (1024x1023x1024: +2.6% x3)
  //   K<=512  -> 4    short k-loop, many tiles      (16384x8192x128: +3.3% x3,
  //                                                  4096x8192x512:  +2.9% x3)
  //   N<=512  -> 16   narrow N, long k-loop         (4096x512x4096:  +4.5% x3)
  //
  // CAVEAT ON PROVENANCE. The K<=512 / N<=512 / default arms were fitted with memset operand
  // data, a hot L2 and sequential timing -- all three since shown to distort results. The two
  // K<=128 / tiles_m<=16 arms were fitted under the corrected regime. The older arms have not
  // been re-derived, so they are suspect: re-tuning the whole policy under random data, cold
  // L2 and interleaved timing is open work, and the autotuner exists precisely because a
  // four-line rule cannot cover this.
  //
  // group_m == 0 means "auto"; any explicit value (from the autotuner, or a benchmark
  // pinning the old behaviour) overrides the policy.
  if (group_m == 0) {
    if (bn == 64)            group_m = 2;
    // Very short k-loop: grouping has almost nothing to amortise, and measured best is 1-4
    // across the three K=128 shapes.
    else if (K <= 128)       group_m = 1;
    else if (K <= 512)       group_m = 4;
    else if (N <= 512)       group_m = 16;
    else                     group_m = GROUP_M;
    // NOTE: an earlier `tiles_m <= 16 -> 1` arm lived here. It was fitted when an odd
    // GROUP_M silently disabled clustering, so what it actually bought was "no cluster",
    // not "no grouping". With the cluster decision decoupled (see use_cluster) the arm has
    // no rationale and measures within 1% of the default, so it is gone.
  }

  if (bn == 256)
    return launch_tile<128, 256>(A, B, C, M, N, K, lda, ldb, alpha, beta, stream, sm_count,
                                 group_m);
  if (bn == 128)
    return launch_tile<128, 128>(A, B, C, M, N, K, lda, ldb, alpha, beta, stream, sm_count,
                                 group_m);
  return launch_tile<128, 64>(A, B, C, M, N, K, lda, ldb, alpha, beta, stream, sm_count, group_m);
#endif
}

// force_bm/force_bn == 0 means "use the heuristic ladder"; the autotuner passes explicit
// values. group_m defaults to the compile-time GROUP_M.
void launch(const half *A, const half *B, half *C, int M, int N, int K, float alpha, float beta,
            cudaStream_t stream, int force_bm = 0, int force_bn = 0, int group_m = 0) {
  if (tma_compatible(A, B, C, M, N, K)) {
    // A's row stride is K halves and B's is N halves; TMA needs both 16B-aligned. When one
    // is not, restride that operand into scratch (one streaming copy) rather than dropping
    // to the scalar fallback, which is ~100x slower.
    const int lda = (K + 7) & ~7, ldb = (N + 7) & ~7;
    const bool padA = (lda != K), padB = (ldb != N);
    const half *Ause = A, *Buse = B;
    if (padA || padB) {
      const size_t na = padA ? static_cast<size_t>(M) * lda : 0;
      const size_t nb = padB ? static_cast<size_t>(K) * ldb : 0;
      half *w = pad_workspace((na + nb) * sizeof(half));
      if (w) {
        if (padA) {
          pad_copy_kernel<<<dim3((K + 255) / 256, 1024), 256, 0, stream>>>(A, w, M, K, lda);
          Ause = w;
        }
        if (padB) {
          pad_copy_kernel<<<dim3((N + 255) / 256, 1024), 256, 0, stream>>>(B, w + na, K, N, ldb);
          Buse = w + na;
        }
        if (launch_mainloop(Ause, Buse, C, M, N, K, lda, ldb, alpha, beta, stream, force_bm, force_bn,
                            group_m))
          return;
      }
    } else if (launch_mainloop(A, B, C, M, N, K, K, N, alpha, beta, stream, force_bm, force_bn,
                               group_m)) {
      return;
    }
  }
  dim3 blk(FB, FB), grd((N + FB - 1) / FB, (M + FB - 1) / FB);
  gemm_fallback<<<grd, blk, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}

// ------------------------------------------------------------------------- autotuner
// Per-shape search over (tile, GROUP_M). The heuristic ladder in launch_mainloop is a
// single global rule fitted to a handful of shapes; measuring the actual shape avoids both
// the fitting and the risk of overfitting a noise band. Result is cached per (M,N,K), so
// the search is paid once.
struct TuneKey {
  int M, N, K;
  bool operator==(const TuneKey &o) const { return M == o.M && N == o.N && K == o.K; }
};
struct TuneCfg {
  int bm, bn, group_m;
};

static std::map<std::tuple<int, int, int>, TuneCfg> &tune_cache() {
  static std::map<std::tuple<int, int, int>, TuneCfg> c;
  return c;
}

// Candidate space. Tiles are the three ladder rungs; GROUP_M spans the range that the
// per-shape sweep showed to matter (best values observed ranged from 2 to 64).
TuneCfg autotune(const half *A, const half *B, half *C, int M, int N, int K, float alpha,
                 float beta) {
  auto key = std::make_tuple(M, N, K);
  auto it = tune_cache().find(key);
  if (it != tune_cache().end()) return it->second;

  const int tiles_bn[3] = {256, 128, 64};
  const int gms[6] = {2, 4, 8, 16, 32, 64};
  TuneCfg best{128, 256, GROUP_M};
  float best_ms = 1e30f;

  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  for (int t = 0; t < 3; ++t) {
    for (int g = 0; g < 6; ++g) {
      const int bn = tiles_bn[t], gm = gms[g];
      // warm up + time a few iterations
      for (int i = 0; i < 3; ++i) launch(A, B, C, M, N, K, alpha, beta, 0, 128, bn, gm);
      if (cudaDeviceSynchronize() != cudaSuccess) continue;
      cudaEventRecord(e0);
      for (int i = 0; i < 5; ++i) launch(A, B, C, M, N, K, alpha, beta, 0, 128, bn, gm);
      cudaEventRecord(e1);
      if (cudaEventSynchronize(e1) != cudaSuccess) continue;
      float ms;
      cudaEventElapsedTime(&ms, e0, e1);
      if (ms > 0 && ms < best_ms) {
        best_ms = ms;
        best = TuneCfg{128, bn, gm};
      }
    }
  }
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  tune_cache()[key] = best;
  return best;
}

// Tuned entry point: search once per shape, then dispatch with the winner.
void launch_tuned(const half *A, const half *B, half *C, int M, int N, int K, float alpha,
                  float beta, cudaStream_t stream) {
  TuneCfg c = autotune(A, B, C, M, N, K, alpha, beta);
  launch(A, B, C, M, N, K, alpha, beta, stream, c.bm, c.bn, c.group_m);
}

}  // namespace h100_hgemm

// A, B, C are device pointers (i.e. pointers to memory on the GPU)
//
//     C_output[M,N] = alpha * (A[M,K] @ B[K,N]) + beta * C_input[M,N]
// C_output and C_input are the same memory location, say, the half* C pointer.
extern "C" void solve(const half *A, const half *B, half *C, int M, int N, int K, float alpha,
                      float beta) {
  h100_hgemm::launch(A, B, C, M, N, K, alpha, beta, 0);
  cudaDeviceSynchronize();
}
