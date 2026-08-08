#include "mma_half.cu"
#include "perf_shapes.h"
#include <cublas_v2.h>
#include <algorithm>
#include <cstdio>
#include <vector>
static cublasHandle_t H;
static double med(std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];}
int main(){
  cudaFree(0); cublasCreate(&H);
  const float al=1.f,be=0.f; const int REP=3,IT=20;
  printf("%-20s %9s %9s %8s %9s %s\n","M x N x K","ours TF","cuBLAS","ratio","%peak","flag");
  for(int i=0;i<NSHAPES;i++){
    long long M=SHAPES[i].M,N=SHAPES[i].N,K=SHAPES[i].K; double fl=2.0*M*N*K;
    half *A,*B,*C;
    if(cudaMalloc(&A,(size_t)M*K*2)||cudaMalloc(&B,(size_t)K*N*2)||cudaMalloc(&C,(size_t)M*N*2)){printf("alloc fail\n");continue;}
    cudaMemset(A,0x11,(size_t)M*K*2); cudaMemset(B,0x11,(size_t)K*N*2);
    auto ours=[&]{ h100_hgemm::launch(A,B,C,M,N,K,al,be,0); };
    auto cb=[&]{ cublasGemmEx(H,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&al,B,CUDA_R_16F,N,A,CUDA_R_16F,K,
                              &be,C,CUDA_R_16F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT); };
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float t;
    std::vector<double> vo,vc;
    for(int r=0;r<5;r++){ours();cb();} cudaDeviceSynchronize();
    for(int r=0;r<REP;r++){
      cudaEventRecord(e0); for(int j=0;j<IT;j++) ours();
      cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&t,e0,e1); vo.push_back(t/IT);
      cudaEventRecord(e0); for(int j=0;j<IT;j++) cb();
      cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&t,e0,e1); vc.push_back(t/IT);
    }
    double to=med(vo),tc=med(vc),tfo=fl/(to*1e9),tfc=fl/(tc*1e9),r=tfo/tfc;
    char nm[48]; snprintf(nm,sizeof nm,"%lldx%lldx%lld",M,N,K);
    printf("%-20s %9.1f %9.1f %7.2fx %8.1f%% %s\n",nm,tfo,tfc,r,100*tfo/989.4,
           r<0.90?"<-- LOSING":(r<0.97?"<-- behind":""));
    cudaFree(A);cudaFree(B);cudaFree(C);
  }
  return 0;
}
