// Numerical check of EVERY shape investigated for mma_half.cu during this session.
// Reference: cuBLAS fp16 w/ fp32 accumulate (same accuracy class).
#include "mma_half.cu"
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static cublasHandle_t H;
struct S{int M,N,K;float a,b;const char*src;};
int main(){
  CK(cudaFree(0)); cublasCreate(&H);
  std::mt19937 rng(2024); std::uniform_real_distribution<float> D(-1.f,1.f);
  S sh[]={
   // --- original 15-shape correctness suite -------------------------------------
   {128,256,64,1,0,"suite"},{256,512,128,1,0,"suite"},{1024,1024,1024,1,0,"suite"},
   {129,260,72,1,0,"suite"},{100,300,64,1,0,"suite"},{512,512,512,2.5f,-1.5f,"suite"},
   {1024,1024,256,0.5f,1.0f,"suite"},{17,33,16,1,0,"suite"},{1,1,8,1,0,"suite"},
   {2048,2048,512,1,0,"suite"},{4096,512,4096,1,0,"suite"},{255,257,264,1,0.25f,"suite"},
   {1,4096,4096,1,0,"suite"},{4096,4096,8,1,0,"suite"},{2047,1023,512,1,0,"suite"},
   // --- sanitizer sweep shapes ---------------------------------------------------
   {3000,1000,2000,1,0,"san"},{384,2048,2048,1,0,"san"},{8,8192,8192,1,0,"san"},
   {8192,8,8192,1,0,"san"},{65,4097,136,1,0,"san"},{1000,700,304,1,0,"san"},
   {33,33,33,1,0,"san"},{2048,2047,2049,1,0,"san"},{1023,1,1025,1,0,"san"},
   {7,4095,9,1,0,"san"},{4095,4095,4095,1,0,"san"},
   {3000,1000,2000,2.5f,-1.5f,"san b!=0"},{384,2048,2048,2.5f,-1.5f,"san b!=0"},
   {33,33,33,2.5f,-1.5f,"san b!=0"},{65,4097,136,2.5f,-1.5f,"san b!=0"},
   // --- perf-table shapes --------------------------------------------------------
   {2048,2048,2048,1,0,"perf"},{4096,4096,4096,1,0,"perf"},{8192,8192,8192,1,0,"perf"},
   {16384,16384,16384,1,0,"perf"},{8192,8192,2048,1,0,"perf"},{8192,8192,4096,1,0,"perf"},
   {8192,8192,16384,1,0,"perf"},{16384,8192,4096,1,0,"perf"},{4096,4096,8192,1,0,"perf"},
   {32768,8192,2048,1,0,"perf"},{4096,4095,4096,1,0,"perf"},{2048,2047,2048,1,0,"perf"},
   // --- K-sweep / stage-sweep / epilogue-decomposition shapes ---------------------
   {8192,8192,1024,1,0,"sweep"},{4096,4094,4096,1,0,"sweep"},{4096,4096,1024,1,0,"sweep"},
   {8192,4096,1024,2.5f,-1.5f,"sweep"},{1024,1023,1024,1,0,"sweep"},
   // --- narrow-output sweep (judge-convention exploration) ------------------------
   {4096,8192,128,1,0,"narrow"},{4096,8192,256,1,0,"narrow"},{4096,8192,512,1,0,"narrow"},
   {4096,8192,1024,1,0,"narrow"},{8192,8192,128,1,0,"narrow"},{16384,8192,128,1,0,"narrow"},
   {8192,1024,8192,1,0,"narrow"},{8192,4096,8192,1,0,"narrow"},{4096,8192,8192,1,0,"narrow"},
   {16384,4096,8192,1,0,"narrow"},
  };
  int nfail=0, n=sizeof(sh)/sizeof(sh[0]);
  printf("%-22s %6s %6s %11s %11s %-8s %s\n","M,N,K","alpha","beta","max_abs","rel_l2","src","");
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
    double l2=sqrt(num/(den+1e-30));
    bool ok=l2<3e-3;
    if(!ok) nfail++;
    char nm[48]; snprintf(nm,sizeof nm,"%lld,%lld,%lld",M,N,K);
    printf("%-22s %6.2f %6.2f %11.3e %11.3e %-8s %s\n",nm,s.a,s.b,ma,l2,s.src,ok?"OK":"*** FAIL ***");
    cudaFree(A);cudaFree(B);cudaFree(C);cudaFree(R);
  }
  printf("\n%d shapes, %d failures\n",n,nfail);
  return nfail?1:0;
}
