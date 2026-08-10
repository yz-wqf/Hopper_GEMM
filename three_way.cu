// Three-way: ours (hand-written) vs CUTLASS vs cuBLAS. Serial, same process, same data.
#include "mma_half.cu"
#include "perf_shapes.h"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
namespace h100_hgemm_cutlass {
bool supported(int, int, int);
bool launch(const half *, const half *, half *, int, int, int, float, float, cudaStream_t);
void set_swizzle(int);
void set_config(int);
int num_configs();
}
// Deterministic pseudo-random fill, on device. Performance depends materially on the
// operand data -- at 8192x4096x8192 the same kernel measures 906 TFLOPS on all-zero input,
// 795 on 0x11, and 834 on random, a 14% spread. Zero operands draw far less tensor-core
// power so the GPU sustains higher clocks. Benchmarking on memset patterns is not
// representative; random is.
__global__ void rand_fill_kernel(half *p, size_t n, unsigned seed){
  size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
  size_t stride = (size_t)gridDim.x*blockDim.x;
  for(; i<n; i+=stride){
    unsigned h = (unsigned)(i*2654435761u) ^ seed;
    h ^= h>>15; h *= 2246822519u; h ^= h>>13; h *= 3266489917u; h ^= h>>16;
    p[i] = __float2half(((float)(h & 0xffff) / 32768.0f) - 1.0f);   // [-1, 1)
  }
}
static void rand_fill(half *p, size_t n, unsigned seed){
  rand_fill_kernel<<<1024,256>>>(p,n,seed);
}
static cublasHandle_t H;
static double med(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
int main(int argc,char**argv){
  cudaFree(0); cublasCreate(&H);
  const float al=1.f,be=0.f; const int REP=5,IT=20;
  std::mt19937 rng(9); std::uniform_real_distribution<float> D(-1.f,1.f);
  printf("%-18s %9s %9s %9s %8s %8s  %s\n","M x N x K","ours","CUTLASS","cuBLAS",
         "ours/cut","ours/cuB","cutlass correctness");
  for(int i=0;i<NSHAPES;i++){
    long long M=SHAPES[i].M,N=SHAPES[i].N,K=SHAPES[i].K; double fl=2.0*M*N*K;
    bool cut_ok = h100_hgemm_cutlass::supported(M,N,K);
    half *A,*B,*C,*R;
    if(cudaMalloc(&A,(size_t)M*K*2)||cudaMalloc(&B,(size_t)K*N*2)||
       cudaMalloc(&C,(size_t)M*N*2)||cudaMalloc(&R,(size_t)M*N*2)) continue;
    // correctness check for CUTLASS on this shape (random data, vs cuBLAS).
    // Reset the swept state first: otherwise this inherits the previous shape's winning
    // config, which may not even be implementable here -- launch() then returns false, C is
    // left as memset, and the shape reports a spurious FAIL.
    h100_hgemm_cutlass::set_config(-1); h100_hgemm_cutlass::set_swizzle(1);
    char verdict[24]="skipped (N%8/K%8)";
    rand_fill(A,(size_t)M*K,0x9e3779b9u ^ (unsigned)i);
    rand_fill(B,(size_t)K*N,0x85ebca6bu ^ (unsigned)i);
    cudaDeviceSynchronize();
    if(cut_ok){
      cublasGemmEx(H,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&al,B,CUDA_R_16F,N,A,CUDA_R_16F,K,&be,R,CUDA_R_16F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
      cudaMemset(C,0,(size_t)M*N*2);
      h100_hgemm_cutlass::launch(A,B,C,M,N,K,al,be,0);
      cudaDeviceSynchronize();
      std::vector<half> ho((size_t)M*N),hr((size_t)M*N);
      cudaMemcpy(ho.data(),C,ho.size()*2,cudaMemcpyDeviceToHost);
      cudaMemcpy(hr.data(),R,hr.size()*2,cudaMemcpyDeviceToHost);
      double num=0,den=0;
      for(size_t z=0;z<ho.size();z++){double o=__half2float(ho[z]),r=__half2float(hr[z]);num+=(o-r)*(o-r);den+=r*r;}
      snprintf(verdict,sizeof verdict,"%s (%.1e)",sqrt(num/(den+1e-30))<3e-3?"OK":"FAIL",sqrt(num/(den+1e-30)));
    }
    auto ours=[&]{ h100_hgemm::launch(A,B,C,M,N,K,al,be,0); };
    auto cut =[&]{ h100_hgemm_cutlass::launch(A,B,C,M,N,K,al,be,0); };
    auto cb  =[&]{ cublasGemmEx(H,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&al,B,CUDA_R_16F,N,A,CUDA_R_16F,K,&be,C,CUDA_R_16F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT); };
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float t;
    auto time_n=[&](auto fn,int warm,int reps,int iters){ std::vector<double> v;
      for(int r=0;r<warm;r++) fn(); cudaDeviceSynchronize();
      for(int r=0;r<reps;r++){ cudaEventRecord(e0); for(int j=0;j<iters;j++) fn();
        cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&t,e0,e1); v.push_back(t/iters);}
      return fl/(med(v)*1e9); };
    auto time=[&](auto fn){ return time_n(fn,5,REP,IT); };
    // CUTLASS gets its threadblock swizzle swept per shape and keeps the best -- the
    // scheduler default is no swizzle at all, which is not a fair baseline against a
    // kernel that rasterizes for L2 (23% at 16384^3), and a fixed cap is an overfit the
    // other way (-38% on 384x2048x2048).
    // CUTLASS gets the same freedom the hand-written dispatcher has: a tile ladder
    // (256/128/64), both schedules, and a threadblock swizzle -- all swept per shape, best
    // kept. Giving it only two tiles and no swizzle was not a like-for-like test: the
    // scheduler default alone was worth 23% at 16384^3, and adding the 128x128 rung plus
    // the pingpong schedule flips 6 of the 10 shapes we previously claimed by >5%.
    //
    // The discovery pass is deliberately cheap: 2 warmup + 1x5 timed per candidate. A
    // full-weight sweep here left the GPU hot enough to cost cuBLAS 15% at 4096^3
    // (877 -> 763). Do NOT "fix" that with a settle sleep: at 1024^3 the measurement is
    // microseconds, clocks never re-boost after an idle gap, and cuBLAS reads 191 vs 308.
    int best_sw=1, best_cfg=0;
    if(cut_ok){
      auto probe=[&](int cf,int sw)->double{
        h100_hgemm_cutlass::set_config(cf); h100_hgemm_cutlass::set_swizzle(sw);
        // launch() returns false when can_implement rejects the config. Without this check
        // the "kernel" costs ~0 and the sweep crowns an unusable config with a fabricated
        // TFLOPS number.
        if(!h100_hgemm_cutlass::launch(A,B,C,M,N,K,al,be,0)) return -1;
        cudaDeviceSynchronize(); return time_n(cut,2,1,3); };
      // Two stages, not a full cross-product: pick the config at swizzle 1, then sweep the
      // swizzle for the winner. 11 candidates instead of 28. The full cross-product is
      // ~224 launches, enough to heat the GPU and depress the measurement that follows
      // (4096^3 read 825 for us against its 863 baseline).
      double bv=-1;
      for(int cf=0; cf<h100_hgemm_cutlass::num_configs(); ++cf){
        double v=probe(cf,1); if(v>bv){bv=v;best_cfg=cf;} }
      for(int sw : {2,4,8}){ double v=probe(best_cfg,sw); if(v>bv){bv=v;best_sw=sw;} }
      h100_hgemm_cutlass::set_config(best_cfg);
      h100_hgemm_cutlass::set_swizzle(best_sw);
      // Re-verify with the config that will actually be timed, not just the default one.
      if(!h100_hgemm_cutlass::launch(A,B,C,M,N,K,al,be,0))
        snprintf(verdict,sizeof verdict,"UNUSABLE cfg%d",best_cfg); }
    // Interleaved timing. Sequential A-then-B does not cancel clock drift, and on large
    // shapes the drift dwarfs the effect: at 8192x4096x8192 both kernels swing 785-912
    // TFLOPS run to run *together*, and sequential timing reported 1.10x for a pair that
    // measures 1.01x interleaved. Round-robin so drift hits all three equally.
    auto one=[&](auto fn){ cudaEventRecord(e0); for(int j=0;j<IT;j++) fn();
      cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&t,e0,e1); return (double)t/IT; };
    for(int r=0;r<5;r++){ ours(); if(cut_ok) cut(); cb(); }
    cudaDeviceSynchronize();
    std::vector<double> vo,vc,vb;
    for(int r=0;r<REP;r++){ vo.push_back(one(ours));
                            if(cut_ok) vc.push_back(one(cut));
                            vb.push_back(one(cb)); }
    double o=fl/(med(vo)*1e9), c=cut_ok?fl/(med(vc)*1e9):0.0, b=fl/(med(vb)*1e9);
    char nm[48]; snprintf(nm,sizeof nm,"%lldx%lldx%lld",M,N,K);
    if(cut_ok) printf("%-18s %9.1f %9.1f %9.1f %7.2fx %7.2fx  c%d/sw%d %s\n",nm,o,c,b,o/c,o/b,best_cfg,best_sw,verdict);
    else       printf("%-18s %9.1f %9s %9.1f %8s %7.2fx  %s\n",nm,o,"n/a",b,"n/a",o/b,verdict);
    cudaFree(A);cudaFree(B);cudaFree(C);cudaFree(R);
  }
  return 0;
}
