#include "mma_half.cu"
#include <cstdio>
#include <cstdlib>
int main(int argc,char**argv){
  int M=atoi(argv[1]),N=atoi(argv[2]),K=atoi(argv[3]),it=atoi(argv[4]);
  half *A,*B,*C;
  cudaMalloc(&A,(size_t)M*K*2); cudaMalloc(&B,(size_t)K*N*2); cudaMalloc(&C,(size_t)M*N*2);
  cudaMemset(A,0x11,(size_t)M*K*2); cudaMemset(B,0x11,(size_t)K*N*2);
  auto g=[&]{ h100_hgemm::launch(A,B,C,M,N,K,1.f,0.f,0); };
  for(int i=0;i<5;i++) g(); cudaDeviceSynchronize();
  cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float t;
  cudaEventRecord(e0); for(int i=0;i<it;i++) g();
  cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&t,e0,e1); t/=it;
  printf("%7.3f ms %7.1f TF  %s\n",t,2.0*M*N*K/(t*1e9),cudaGetErrorString(cudaGetLastError()));
  return 0;
}
