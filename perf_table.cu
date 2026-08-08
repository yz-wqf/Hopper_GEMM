#include "mma_half.cu"
#include <cublas_v2.h>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <string>
static cublasHandle_t H;
static double med(std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }
int main(){
  cudaFree(0); cublasCreate(&H);
  struct S{int M,N,K;const char*tag;};
  S sh[]={{2048,2048,2048,""},{4096,4096,4096,""},{8192,8192,8192,""},{16384,16384,16384,""},
          {8192,8192,2048,""},{8192,8192,4096,""},{8192,8192,16384,""},
          {16384,8192,4096,""},{4096,4096,8192,""},{32768,8192,2048,""},
          {4096,4095,4096,"N%8!=0"},{2048,2047,2048,"N%8!=0"}};
  const float al=1.f,be=0.f; const int REP=5, IT=30;
  printf("%-20s %10s %10s %9s %10s %9s  %s\n",
         "M x N x K","ours TF","cuBLAS TF","ratio","ours ms","%of peak","note");
  printf("%s\n",std::string(88,'-').c_str());
  for(auto&s:sh){
    long long M=s.M,N=s.N,K=s.K; double fl=2.0*M*N*K;
    half *A,*B,*C;
    if(cudaMalloc(&A,(size_t)M*K*2)||cudaMalloc(&B,(size_t)K*N*2)||cudaMalloc(&C,(size_t)M*N*2)){
      printf("  %-18s  (alloc failed)\n",""); continue; }
    cudaMemset(A,0x11,(size_t)M*K*2); cudaMemset(B,0x11,(size_t)K*N*2);
    auto ours=[&]{ h100_hgemm::launch(A,B,C,M,N,K,al,be,0); };
    auto cb=[&]{ cublasGemmEx(H,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&al,B,CUDA_R_16F,N,A,CUDA_R_16F,K,
                              &be,C,CUDA_R_16F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT); };
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float t;
    std::vector<double> vo,vc;
    for(int i=0;i<10;i++){ours();cb();} cudaDeviceSynchronize();
    for(int r=0;r<REP;r++){
      cudaEventRecord(e0); for(int i=0;i<IT;i++) ours();
      cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&t,e0,e1); vo.push_back(t/IT);
      cudaEventRecord(e0); for(int i=0;i<IT;i++) cb();
      cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&t,e0,e1); vc.push_back(t/IT);
    }
    double to=med(vo),tc=med(vc), tfo=fl/(to*1e9), tfc=fl/(tc*1e9);
    char nm[48]; snprintf(nm,sizeof nm,"%lld x %lld x %lld",M,N,K);
    printf("%-20s %10.1f %10.1f %8.2fx %10.3f %8.1f%%  %s\n",
           nm,tfo,tfc,tfo/tfc,to,100*tfo/989.4,s.tag);
    cudaFree(A);cudaFree(B);cudaFree(C);
  }
  printf("\nH100 SXM5 FP16 tensor-core dense peak = 989.4 TFLOPS\n");
  return 0;
}
