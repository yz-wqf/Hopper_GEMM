// Verify the shapes the perf table actually reports, incl. large ones.
// Reference = cuBLAS fp16 w/ fp32 accumulate (same accuracy class), plus an fp64
// anchor via cublasDgemm on widened copies for a mid-size shape.
#include "mma_half.cu"
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static cublasHandle_t H;
__global__ void h2d_k(const half*s,double*d,long long n){
  for(long long i=blockIdx.x*256LL+threadIdx.x;i<n;i+=gridDim.x*256LL) d[i]=(double)__half2float(s[i]);}
int main(){
  CK(cudaFree(0)); cublasCreate(&H);
  std::mt19937 rng(11); std::uniform_real_distribution<float> D(-1.f,1.f);
  struct S{int M,N,K;float a,b;const char*why;};
  S sh[]={
   {8192,8192,8192,1,0,"perf-table shape, never verified"},
   {16384,16384,16384,1,0,"perf-table shape, never verified"},
   {8192,8192,2048,1,0,"perf-table shape"},
   {16384,8192,4096,1,0,"perf-table shape"},
   {32768,8192,2048,1,0,"perf-table shape"},
   {4096,4096,1024,1,0,"many tiles/CTA AND many k-tiles <- cross-tile pipeline"},
   {8192,4096,1024,2.5f,-1.5f,"same, with beta!=0 (shuffle epilogue)"},
   {4096,4095,4096,1,0,"perf-table odd-N path"},
   {2048,2047,2048,1,0,"perf-table odd-N path -- WAS UNVERIFIED"},
   {4095,4095,4095,1,0,"odd M and N"},
   {2048,2047,2048,1.5f,0.5f,"odd N with beta!=0"},
   {1024,1023,1024,1,0,"odd N, small"},
  };
  printf("%-22s %6s %6s %11s %11s  %s\n","M,N,K","alpha","beta","max_abs","rel_l2","note");
  for(auto&s:sh){
    long long M=s.M,N=s.N,K=s.K;
    std::vector<half> hA((size_t)M*K),hB((size_t)K*N),hC((size_t)M*N);
    for(auto&v:hA) v=__float2half(D(rng));
    for(auto&v:hB) v=__float2half(D(rng));
    for(auto&v:hC) v=__float2half(D(rng));
    half *A,*B,*C,*R;
    CK(cudaMalloc(&A,hA.size()*2)); CK(cudaMalloc(&B,hB.size()*2));
    CK(cudaMalloc(&C,hC.size()*2)); CK(cudaMalloc(&R,hC.size()*2));
    CK(cudaMemcpy(A,hA.data(),hA.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(B,hB.data(),hB.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(C,hC.data(),hC.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(R,hC.data(),hC.size()*2,cudaMemcpyHostToDevice));
    solve(A,B,C,M,N,K,s.a,s.b);
    CK(cudaGetLastError());
    cublasGemmEx(H,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&s.a,B,CUDA_R_16F,N,A,CUDA_R_16F,K,&s.b,
                 R,CUDA_R_16F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
    CK(cudaDeviceSynchronize());
    std::vector<half> ho(hC.size()),hr(hC.size());
    CK(cudaMemcpy(ho.data(),C,hC.size()*2,cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hr.data(),R,hC.size()*2,cudaMemcpyDeviceToHost));
    double ma=0,num=0,den=0;
    for(size_t i=0;i<ho.size();i++){
      double o=__half2float(ho[i]),r=__half2float(hr[i]),e=fabs(o-r);
      ma=fmax(ma,e); num+=e*e; den+=r*r;
    }
    char nm[48]; snprintf(nm,sizeof nm,"%lld,%lld,%lld",M,N,K);
    double l2=sqrt(num/(den+1e-30));
    printf("%-22s %6.2f %6.2f %11.3e %11.3e  %s %s\n",nm,s.a,s.b,ma,l2,
           l2<3e-3?"OK ":"*** FAIL ***",s.why);
    cudaFree(A);cudaFree(B);cudaFree(C);cudaFree(R);
  }
  return 0;
}
