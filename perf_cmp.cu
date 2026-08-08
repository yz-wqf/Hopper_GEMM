#include "mma_half.cu"
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
int main(int argc,char**argv){
  int M=atoi(argv[1]),N=atoi(argv[2]),K=atoi(argv[3]),it=argc>4?atoi(argv[4]):30;
  cublasHandle_t H; cublasCreate(&H);
  half *A,*B,*C;
  cudaMalloc(&A,(size_t)M*K*2); cudaMalloc(&B,(size_t)K*N*2); cudaMalloc(&C,(size_t)M*N*2);
  cudaMemset(A,0x11,(size_t)M*K*2); cudaMemset(B,0x11,(size_t)K*N*2);
  const float al=1.f,be=0.f; double fl=2.0*M*N*K;
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float t1,t2;
  auto ours=[&]{ h100_hgemm::launch(A,B,C,M,N,K,al,be,0); };
  auto cb=[&]{ cublasGemmEx(H,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&al,B,CUDA_R_16F,N,A,CUDA_R_16F,K,
                            &be,C,CUDA_R_16F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT); };
  for(int i=0;i<10;i++){ours();cb();} cudaDeviceSynchronize();
  cudaEventRecord(e0); for(int i=0;i<it;i++) ours();
  cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&t1,e0,e1); t1/=it;
  cudaEventRecord(e0); for(int i=0;i<it;i++) cb();
  cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&t2,e0,e1); t2/=it;
  printf("%5dx%5dx%-6d ours %7.1f TF   cuBLAS %7.1f TF   ratio %.2fx\n",
         M,N,K,fl/(t1*1e9),fl/(t2*1e9),t2/t1);
  return 0;
}
