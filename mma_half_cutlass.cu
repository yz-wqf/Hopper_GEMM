// =====================================================================================
// CUTLASS 3.x reimplementation of mma_half.cu -- same contract, library primitives.
//
//   C = alpha * (A[M,K] @ B[K,N]) + beta * C        all row-major
//   half in / half out, fp32 accumulate
//
// Same entry point and semantics as the hand-written kernel, so the two are drop-in
// interchangeable and directly comparable:
//
//   extern "C" void solve(const half*, const half*, half*, int M,int N,int K, float,float)
//
// The hand-written kernel builds warp specialization, the TMA pipeline, cluster multicast
// and the TMA-store epilogue by hand. Here the CollectiveBuilder emits the equivalent
// (KernelTmaWarpSpecializedCooperative + TmaWarpSpecialized epilogue) from a tile shape.
//
// Scope: ALIGNED SHAPES ONLY (K%8 == 0 and N%8 == 0). Hopper TMA needs 16B-aligned rows and
// CUTLASS checks that against the problem extent, so ragged leading dimensions would require
// running a padded, zero-filled problem and copying back -- more work than the kernel under
// comparison does. Call `supported()` and skip instead. mma_half.cu has no such restriction:
// TMA zero-fills out-of-range elements, so it runs those shapes in place.
//
// Build: nvcc -gencode arch=compute_90a,code=sm_90a -O3 -std=c++17 \
//        -I$CUTLASS/include -I$CUTLASS/tools/util/include \
//        --expt-relaxed-constexpr mma_half_cutlass.cu
// =====================================================================================

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace h100_hgemm_cutlass {

using namespace cute;

using ElementA = cutlass::half_t;
using ElementB = cutlass::half_t;
using ElementC = cutlass::half_t;
using ElementAcc = float;
using ElementCompute = float;

// A is M x K row-major, B is K x N row-major, C is M x N row-major.
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::RowMajor;
using LayoutC = cutlass::layout::RowMajor;

static constexpr int Align = 16 / sizeof(ElementA);  // 8 halves = 16 B, the TMA minimum

// One CUTLASS configuration, chosen the way a CUTLASS user would: a standard Hopper fp16
// tile plus `KernelScheduleAuto` / `EpilogueScheduleAuto`, letting the builder select the
// mainloop and epilogue schedules and the stage count. CUTLASS's device API fixes the tile
// per instantiation -- there is no runtime tile heuristic (that is what its offline
// profiler is for), so this is a single configuration applied to every problem size.
#ifndef CUTLASS_EPI_AUTO
// CUTLASS's own Hopper examples pair the cooperative mainloop with this epilogue; leaving it
// to EpilogueScheduleAuto selects DefaultEpilogue instead, measured 35% slower here
// (572 vs 774 TFLOPS at 4096^3).
using EpiSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;
#else
using EpiSchedule = cutlass::epilogue::collective::EpilogueScheduleAuto;
#endif

// Two configurations, picked by a sweep over tile x cluster x kernel schedule (see the
// tuning table in OPTIMIZATION_REPORT.md). CUTLASS's device API fixes the tile per
// instantiation, so this is the equivalent of what its offline profiler produces.
//
//   1024^3 : 128x256x64 -> 143 TF    128x64x64 c1x1 -> 289 TF   (2.02x)
//   4096^3 : 128x256x64 c2x1 -> 857 TF  128x64x64 c1x1 -> 374 TF
//  16384^3 : c2x1 -> 693 TF, c1x2 -> 429 TF (cluster choice matters more at scale)
//
// Same starvation story as the hand-written kernel: one fixed tile cannot serve both.
//
// MUST BE BUILT WITH -DNDEBUG. Without it, CUTLASS's device-side asserts block inlining and
// ptxas cannot keep the wgmma group open across the resulting call boundary (warning C7510).
// The SASS then fences and *fully drains* around every single wgmma
// (WARPGROUP.DEPBAR.LE gsb0, 0x0 after each HGMMA) instead of keeping one group in flight,
// costing ~10% at 4096^3 and ~2x at 1024^3.
template <class TileShape_, class ClusterShape_>
struct Cfg {
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape_, ClusterShape_,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, ElementCompute, ElementC,
      LayoutC, Align, ElementC, LayoutC, Align, EpiSchedule>::CollectiveOp;

  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA, LayoutA, Align, ElementB,
      LayoutB, Align, ElementAcc, TileShape_, ClusterShape_,
      cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(
          sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;

  using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>,
                                                          CollectiveMainloop, CollectiveEpilogue>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// Cluster 2x1, not 1x2: 1x2 measured 1.4% better at 4096^3 but 1.6x WORSE at 16384^3
// (429 vs 693 TFLOPS). Tuning on one shape and generalising is how that nearly shipped.
using Large = Cfg<Shape<_128, _256, _64>, Shape<_2, _1, _1>>;
using Small = Cfg<Shape<_128, _64, _64>, Shape<_1, _1, _1>>;
using Gemm = typename Large::Gemm;   // stride types are identical across configs

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;

// CUTLASS device-side workspace (tile scheduler etc.), kept separate and reused.
static void *g_cutlass_ws = nullptr;
static size_t g_cutlass_ws_bytes = 0;
static void *cutlass_workspace(size_t bytes) {
  if (bytes == 0) return nullptr;
  if (bytes > g_cutlass_ws_bytes) {
    if (g_cutlass_ws) cudaFree(g_cutlass_ws);
    if (cudaMalloc(&g_cutlass_ws, bytes) != cudaSuccess) {
      g_cutlass_ws = nullptr;
      g_cutlass_ws_bytes = 0;
      return nullptr;
    }
    g_cutlass_ws_bytes = bytes;
  }
  return g_cutlass_ws;
}

// ------------------------------------------------------------------------ the launcher
// Aligned shapes only, by design. Hopper TMA needs 16B-aligned rows (8 halves), and
// CUTLASS validates that against the problem *extent*, not the leading dimension -- so a
// mis-strided operand cannot be fixed by widening the stride. Supporting it would mean
// running the GEMM at a padded, zero-filled extent and copying the result back, which is a
// different kernel doing more work; that is not what we want to compare.
//
// `supported()` lets the caller skip those shapes rather than silently produce nothing.
// (mma_half.cu handles them in place, because TMA zero-fills out-of-range elements natively.)
bool supported(int M, int N, int K) {
  return M > 0 && N > 0 && K > 0 && (K % Align == 0) && (N % Align == 0);
}

bool launch(const half *A, const half *B, half *C, int M, int N, int K, float alpha, float beta,
            cudaStream_t stream) {
  if (!supported(M, N, K)) return false;

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));

  // Same starvation threshold the hand-written kernel uses: step down when the wide tile
  // cannot produce enough tiles to fill the machine.
  static int sm_count = [] {
    int n = 132;
    cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, 0);
    return n;
  }();
  const int wide_tiles = ((M + 127) / 128) * ((N + 255) / 256);
  const bool use_large = wide_tiles * 3 >= sm_count * 2;

  auto run = [&](auto tag) -> bool {
    using G = typename decltype(tag)::Gemm;
    typename G::Arguments a{cutlass::gemm::GemmUniversalMode::kGemm,
                            {M, N, K, 1},
                            {reinterpret_cast<const ElementA *>(A), stride_A,
                             reinterpret_cast<const ElementB *>(B), stride_B},
                            {{alpha, beta},
                             reinterpret_cast<const ElementC *>(C), stride_C,
                             reinterpret_cast<ElementC *>(C), stride_C}};
    G op;
    if (op.can_implement(a) != cutlass::Status::kSuccess) return false;
    void *ws = cutlass_workspace(G::get_workspace_size(a));
    if (op.initialize(a, ws, stream) != cutlass::Status::kSuccess) return false;
    return op.run(stream) == cutlass::Status::kSuccess;
  };
  return use_large ? run(Large{}) : run(Small{});
}

}  // namespace h100_hgemm_cutlass

// Same signature and semantics as mma_half.cu's solve().
#ifndef CUTLASS_NO_SOLVE
extern "C" void solve(const half *A, const half *B, half *C, int M, int N, int K, float alpha,
                      float beta) {
  if (!h100_hgemm_cutlass::launch(A, B, C, M, N, K, alpha, beta, 0))
    fprintf(stderr, "cutlass launch failed for %dx%dx%d\n", M, N, K);
  cudaDeviceSynchronize();
}
#endif  // CUTLASS_NO_SOLVE
