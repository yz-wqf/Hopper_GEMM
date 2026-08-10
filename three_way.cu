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
}
static cublasHandle_t H;
static double med(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
int main(int argc,char**argv){
  cudaFree(0); cublasCreate(&H);
  const float al=1.f,be=0.f; const int REP=3,IT=20;
  std::mt19937 rng(9); std::uniform_real_distribution<float> D(-1.f,1.f);
  printf("%-18s %9s %9s %9s %8s %8s  %s\n","M x N x K","ours","CUTLASS","cuBLAS",
         "ours/cut","ours/cuB","cutlass correctness");
  for(int i=0;i<NSHAPES;i++){
    long long M=SHAPES[i].M,N=SHAPES[i].N,K=SHAPES[i].K; double fl=2.0*M*N*K;
    bool cut_ok = h100_hgemm_cutlass::supported(M,N,K);
    half *A,*B,*C,*R;
    if(cudaMalloc(&A,(size_t)M*K*2)||cudaMalloc(&B,(size_t)K*N*2)||
       cudaMalloc(&C,(size_t)M*N*2)||cudaMalloc(&R,(size_t)M*N*2)) continue;
    // correctness check for CUTLASS on this shape (random data, vs cuBLAS)
    char verdict[24]="skipped (N%8/K%8)";
    if(cut_ok){
      std::vector<half> hA((size_t)M*K),hB((size_t)K*N);
      for(auto&v:hA) v=__float2half(D(rng));
      for(auto&v:hB) v=__float2half(D(rng));
      cudaMemcpy(A,hA.data(),hA.size()*2,cudaMemcpyHostToDevice);
      cudaMemcpy(B,hB.data(),hB.size()*2,cudaMemcpyHostToDevice);
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
    cudaMemset(A,0x11,(size_t)M*K*2); cudaMemset(B,0x11,(size_t)K*N*2);
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
    // CUTLASS's tile scheduler defaults to max_swizzle_size=1 -- no threadblock swizzle at
    // all -- which is not a fair baseline against a kernel that rasterizes for L2 (worth
    // 23% to CUTLASS at 16384^3). A fixed cap overfits the other way (-38% on
    // 384x2048x2048), so sweep it per shape and keep the winner.
    //
    // The discovery pass is deliberately cheap: 2 warmup + 1x5 timed per candidate, ~28
    // launches against ~260 for a full timing. A full-weight sweep here left the GPU hot
    // enough to cost cuBLAS 15% at 4096^3 (877 -> 763). Do NOT "fix" that with a settle
    // sleep: at 1024^3 the measurement is microseconds, the clocks never re-boost after an
    // idle gap, and cuBLAS reads 191 instead of 308.
    int best_sw=1;
    if(cut_ok){ double bv=0.0;
      for(int sw : {1,2,4,8}){ h100_hgemm_cutlass::set_swizzle(sw);
        double v=time_n(cut,2,1,5); if(v>bv){bv=v;best_sw=sw;} }
      h100_hgemm_cutlass::set_swizzle(best_sw); }
    double o=time(ours), c=cut_ok?time(cut):0.0, b=time(cb);
    char nm[48]; snprintf(nm,sizeof nm,"%lldx%lldx%lld",M,N,K);
    if(cut_ok) printf("%-18s %9.1f %9.1f %9.1f %7.2fx %7.2fx  sw=%d %s\n",nm,o,c,b,o/c,o/b,best_sw,verdict);
    else       printf("%-18s %9.1f %9s %9.1f %8s %7.2fx  %s\n",nm,o,"n/a",b,"n/a",o/b,verdict);
    cudaFree(A);cudaFree(B);cudaFree(C);cudaFree(R);
  }
  return 0;
}
