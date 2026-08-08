#include "mma_half.cu"
#include <cstdio>
int main(){
  struct{int M,N,K;float a,b;} s[]={{8192,8192,8192,1,0},{16384,8192,4096,1,0},
                                    {32768,8192,2048,1,0},{8192,8192,4096,2.5f,-1.5f}};
  for(auto&x:s){
    half *A,*B,*C;
    if(cudaMalloc(&A,(size_t)x.M*x.K*2)||cudaMalloc(&B,(size_t)x.K*x.N*2)||
       cudaMalloc(&C,(size_t)x.M*x.N*2)){printf("alloc fail\n");return 1;}
    cudaMemset(A,0x11,(size_t)x.M*x.K*2); cudaMemset(B,0x11,(size_t)x.K*x.N*2);
    cudaMemset(C,0,(size_t)x.M*x.N*2);
    solve(A,B,C,x.M,x.N,x.K,x.a,x.b);
    cudaError_t e=cudaDeviceSynchronize();
    printf("  %dx%dx%d beta=%.1f : %s\n",x.M,x.N,x.K,x.b,cudaGetErrorString(e));
    cudaFree(A);cudaFree(B);cudaFree(C);
  }
  return 0;
}
