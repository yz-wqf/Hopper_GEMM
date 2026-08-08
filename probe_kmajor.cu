// Probe: validate TMA 128B-swizzle layout + wgmma fp16 smem descriptor encoding.
// C[m][n] = sum_k A[m][k] * B[n][k]   (TN, both operands K-major)
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define CUDA_CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
  printf("CUDA err %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); exit(1);} } while(0)
#define CU_CHECK(x) do { CUresult e=(x); if(e!=CUDA_SUCCESS){ const char*s; cuGetErrorString(e,&s); \
  printf("CU err %s @%d: %s\n", #x, __LINE__, s); exit(1);} } while(0)

static constexpr int M = 64, N = 64, K = 64;

__device__ __forceinline__ uint32_t smem_u32(const void *p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// WGMMA shared-memory matrix descriptor.
//  [13:0] start addr>>4 | [29:16] LBO>>4 | [45:32] SBO>>4 | [51:49] base offset | [63:62] swizzle
__device__ __forceinline__ uint64_t make_desc(uint32_t addr, uint32_t lbo_enc,
                                              uint32_t sbo_enc, uint32_t swizzle) {
  uint64_t d = 0;
  d |= static_cast<uint64_t>((addr >> 4) & 0x3FFF);
  d |= static_cast<uint64_t>(lbo_enc & 0x3FFF) << 16;
  d |= static_cast<uint64_t>(sbo_enc & 0x3FFF) << 32;
  d |= static_cast<uint64_t>(swizzle) << 62;
  return d;
}

__global__ void __launch_bounds__(128) probe(const __grid_constant__ CUtensorMap tmap_a,
                                             const __grid_constant__ CUtensorMap tmap_b,
                                             float *C, uint32_t lbo_enc, uint32_t sbo_enc) {
  extern __shared__ __align__(1024) uint8_t smem[];
  half *sA = reinterpret_cast<half *>(smem);              // 64 x 32 f32, 128B-swizzled
  half *sB = reinterpret_cast<half *>(smem + M * K * 2);  // 64 x 32 f32, 128B-swizzled
  __shared__ __align__(8) uint64_t bar;

  const int tid = threadIdx.x;
  if (tid == 0) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&bar)));
  }
  asm volatile("fence.proxy.async.shared::cta;");
  __syncthreads();

  if (tid == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(smem_u32(&bar)),
                 "r"(static_cast<uint32_t>((M * K + N * K) * 2)));
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(sA)),
        "l"(&tmap_a), "r"(0), "r"(0), "r"(smem_u32(&bar))
        : "memory");
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];" ::"r"(smem_u32(sB)),
        "l"(&tmap_b), "r"(0), "r"(0), "r"(smem_u32(&bar))
        : "memory");
  }
  // wait phase 0
  asm volatile("{ .reg .pred P; WAIT: mbarrier.try_wait.parity.shared::cta.b64 P, [%0], 0;"
               " @P bra DONE; bra WAIT; DONE: }" ::"r"(smem_u32(&bar)));

  float d[32];
#pragma unroll
  for (int i = 0; i < 32; i++) d[i] = 0.f;

  asm volatile("wgmma.fence.sync.aligned;");
#pragma unroll
  for (int ks = 0; ks < K / 16; ks++) {
    // advance along K by 16 half elements = 32 bytes
    uint64_t da = make_desc(smem_u32(sA) + ks * 32, lbo_enc, sbo_enc, 1);
    uint64_t db = make_desc(smem_u32(sB) + ks * 32, lbo_enc, sbo_enc, 1);
    asm volatile(
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, 1, 1, 1, 0, 0;"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
          "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(da), "l"(db));
  }
  asm volatile("wgmma.commit_group.sync.aligned;");
  asm volatile("wgmma.wait_group.sync.aligned 0;");

  // Standard wgmma f32 accumulator layout.
  const int warp = tid / 32, lane = tid % 32;
#pragma unroll
  for (int i = 0; i < 32; i++) {
    int g = i / 4, j = i % 4;
    int row = 16 * warp + (lane / 4) + 8 * (j / 2);
    int col = 8 * g + 2 * (lane % 4) + (j % 2);
    C[row * N + col] = d[i];
  }
}

static void make_map(CUtensorMap *map, void *ptr, int rows, int cols) {
  uint64_t gdim[2] = {(uint64_t)cols, (uint64_t)rows};   // dim0 = fastest = K
  uint64_t gstr[1] = {(uint64_t)cols * 2};
  uint32_t bdim[2] = {(uint32_t)cols, (uint32_t)rows};
  uint32_t estr[2] = {1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, ptr, gdim, gstr, bdim,
                                  estr, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                  CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}

int main() {
  CUDA_CHECK(cudaFree(nullptr));
  std::vector<half> hA(M * K), hB(N * K);
  std::vector<float> hC(M * N), ref(M * N);
  srand(0);
  for (auto &v : hA) v = __float2half((float)(rand() % 7 - 3));
  for (auto &v : hB) v = __float2half((float)(rand() % 5 - 2));
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      float s = 0;
      for (int k = 0; k < K; k++) s += __half2float(hA[m*K+k]) * __half2float(hB[n*K+k]);
      ref[m * N + n] = s;
    }

  half *dA, *dB; float *dC;
  CUDA_CHECK(cudaMalloc(&dA, M * K * 2));
  CUDA_CHECK(cudaMalloc(&dB, N * K * 2));
  CUDA_CHECK(cudaMalloc(&dC, M * N * 4));
  CUDA_CHECK(cudaMemcpy(dA, hA.data(), M * K * 2, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(dB, hB.data(), N * K * 2, cudaMemcpyHostToDevice));

  CUtensorMap ta, tb;
  make_map(&ta, dA, M, K);
  make_map(&tb, dB, N, K);

  size_t smem = (M * K + N * K) * 2;
  CUDA_CHECK(cudaFuncSetAttribute(probe, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

  const uint32_t cands[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
  int nhit = 0;
  for (uint32_t lbo : cands)
    for (uint32_t sbo : cands) {
      CUDA_CHECK(cudaMemset(dC, 0, M * N * 4));
      probe<<<1, 128, smem>>>(ta, tb, dC, lbo, sbo);
      if (cudaGetLastError() != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) continue;
      CUDA_CHECK(cudaMemcpy(hC.data(), dC, M * N * 4, cudaMemcpyDeviceToHost));
      double maxerr = 0;
      for (int i = 0; i < M * N; i++) maxerr = fmax(maxerr, fabs(hC[i] - ref[i]));
      if (maxerr == 0.0) {
        printf("MATCH  lbo_enc=%-5u sbo_enc=%-5u\n", lbo, sbo);
        nhit++;
      }
    }
  if (!nhit) {
    printf("no match found\n");
    // dump the best-effort case for inspection
    probe<<<1, 128, smem>>>(ta, tb, dC, 1, 64);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC.data(), dC, M * N * 4, cudaMemcpyDeviceToHost));
    printf("lbo=1,sbo=64 -> C[0][0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.0f ", hC[i]);
    printf("\n              ref[0][0..7]: ");
    for (int i = 0; i < 8; i++) printf("%.0f ", ref[i]);
    printf("\n");
  }
  return 0;
}
