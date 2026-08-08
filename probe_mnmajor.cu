
// Probe: can a single wgmma consume B *N-major* (trans_b=1) across multiple 128B-swizzle
// blocks?  C[m][n] = sum_k A[m][k]*B[k][n], A K-major (known-good desc), B row-major KxN.
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
#define CU(x) do{CUresult e=(x); if(e!=CUDA_SUCCESS){const char*s;cuGetErrorString(e,&s);printf("CU %s @%d %s\n",#x,__LINE__,s);exit(1);}}while(0)
static constexpr int M=64,N=128,K=64;          // K=64 layout, one wgmma over k=0..15
static constexpr int SMEM=200704;
__device__ __forceinline__ uint32_t su(const void*p){return (uint32_t)__cvta_generic_to_shared(p);}
__device__ __forceinline__ uint64_t mkdesc(uint32_t a,uint32_t lbo,uint32_t sbo,uint32_t sw){
  uint64_t d=(uint64_t)((a>>4)&0x3FFF);
  d|=(uint64_t)(lbo&0x3FFF)<<16; d|=(uint64_t)(sbo&0x3FFF)<<32; d|=(uint64_t)sw<<62; return d;}
__global__ __launch_bounds__(128) void probe(const __grid_constant__ CUtensorMap ta,
    const __grid_constant__ CUtensorMap tb,float*C,uint32_t lbo,uint32_t sbo){
  extern __shared__ __align__(1024) uint8_t sm[];
  half*sA=(half*)sm; half*sB=(half*)(sm+65536);
  __shared__ __align__(8) uint64_t bar;
  int tid=threadIdx.x;
  if(tid==0){asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;"::"r"(su(&bar)));}
  asm volatile("fence.proxy.async.shared::cta;"); __syncthreads();
  if(tid==0){
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"::"r"(su(&bar)),
                 "r"((uint32_t)((M*K+K*N)*2)));
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];"::"r"(su(sA)),"l"(&ta),"r"(0),"r"(0),"r"(su(&bar)):"memory");
    // B in two 64-wide N blocks, placed contiguously: block j at j*K*64 halves
    for(int j=0;j<2;j++)
      asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"::"r"(su(sB+j*K*64)),"l"(&tb),"r"(j*64),"r"(0),
        "r"(su(&bar)):"memory");
  }
  asm volatile("{.reg .pred P; W: mbarrier.try_wait.parity.shared::cta.b64 P,[%0],0;"
               " @P bra D; bra W; D:}"::"r"(su(&bar)));
  float d[64];
  #pragma unroll
  for(int i=0;i<64;i++) d[i]=0.f;
  asm volatile("wgmma.fence.sync.aligned;");
  uint64_t da=mkdesc(su(sA),1,64,1);
  uint64_t db=mkdesc(su(sB),lbo,sbo,1);
  asm volatile("wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, %64, %65, 1, 1, 1, 0, 1;"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),"+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7]),"+f"(d[8]),"+f"(d[9]),"+f"(d[10]),"+f"(d[11]),"+f"(d[12]),"+f"(d[13]),"+f"(d[14]),"+f"(d[15]),"+f"(d[16]),"+f"(d[17]),"+f"(d[18]),"+f"(d[19]),"+f"(d[20]),"+f"(d[21]),"+f"(d[22]),"+f"(d[23]),"+f"(d[24]),"+f"(d[25]),"+f"(d[26]),"+f"(d[27]),"+f"(d[28]),"+f"(d[29]),"+f"(d[30]),"+f"(d[31]),"+f"(d[32]),"+f"(d[33]),"+f"(d[34]),"+f"(d[35]),"+f"(d[36]),"+f"(d[37]),"+f"(d[38]),"+f"(d[39]),"+f"(d[40]),"+f"(d[41]),"+f"(d[42]),"+f"(d[43]),"+f"(d[44]),"+f"(d[45]),"+f"(d[46]),"+f"(d[47]),"+f"(d[48]),"+f"(d[49]),"+f"(d[50]),"+f"(d[51]),"+f"(d[52]),"+f"(d[53]),"+f"(d[54]),"+f"(d[55]),"+f"(d[56]),"+f"(d[57]),"+f"(d[58]),"+f"(d[59]),"+f"(d[60]),"+f"(d[61]),"+f"(d[62]),"+f"(d[63]) : "l"(da),"l"(db));
  asm volatile("wgmma.commit_group.sync.aligned;");
  asm volatile("wgmma.wait_group.sync.aligned 0;");
  int w=tid/32,l=tid%32;
  #pragma unroll
  for(int i=0;i<64;i++){int g=i/4,j=i%4;
    int r=16*w+(l/4)+8*(j/2), c=8*g+2*(l%4)+(j%2); C[r*N+c]=d[i];}
}
static void mkmap(CUtensorMap*m,void*p,int rows,int cols,int br,int bc){
  uint64_t gd[2]={(uint64_t)cols,(uint64_t)rows}; uint64_t gs[1]={(uint64_t)cols*2};
  uint32_t bd[2]={(uint32_t)bc,(uint32_t)br}; uint32_t es[2]={1,1};
  CU(cuTensorMapEncodeTiled(m,CU_TENSOR_MAP_DATA_TYPE_FLOAT16,2,p,gd,gs,bd,es,
     CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
     CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
}
int main(){
  CK(cudaFree(0)); srand(1);
  std::vector<half> hA(M*K),hB(K*N); std::vector<float> hC(M*N),ref(M*N);
  for(auto&v:hA) v=__float2half((float)(rand()%7-3));
  for(auto&v:hB) v=__float2half((float)(rand()%5-2));
  for(int m=0;m<M;m++)for(int n=0;n<N;n++){float s=0;
    for(int k=0;k<16;k++) s+=__half2float(hA[m*K+k])*__half2float(hB[k*N+n]); ref[m*N+n]=s;}
  half*dA,*dB; float*dC;
  CK(cudaMalloc(&dA,M*K*2)); CK(cudaMalloc(&dB,K*N*2)); CK(cudaMalloc(&dC,M*N*4));
  CK(cudaMemcpy(dA,hA.data(),M*K*2,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB,hB.data(),K*N*2,cudaMemcpyHostToDevice));
  CUtensorMap ta,tb; mkmap(&ta,dA,M,K,64,64); mkmap(&tb,dB,K,N,64,64);
  CK(cudaFuncSetAttribute(probe,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM));
  const uint32_t cand[]={1,2,4,8,16,32,64,128,256,512,1024,2048};
  int hits=0;
  for(uint32_t lbo:cand)for(uint32_t sbo:cand){
    CK(cudaMemset(dC,0,M*N*4));
    probe<<<1,128,SMEM>>>(ta,tb,dC,lbo,sbo);
    if(cudaGetLastError()!=cudaSuccess||cudaDeviceSynchronize()!=cudaSuccess){printf("fault lbo=%u sbo=%u\n",lbo,sbo);return 1;}
    CK(cudaMemcpy(hC.data(),dC,M*N*4,cudaMemcpyDeviceToHost));
    double e=0; for(int i=0;i<M*N;i++) e=fmax(e,fabs(hC[i]-ref[i]));
    if(e==0.0){printf("MATCH  lbo_enc=%-5u sbo_enc=%-5u\n",lbo,sbo);hits++;}
  }
  if(!hits) printf("no match\n");
  return 0;
}
