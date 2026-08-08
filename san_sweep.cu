#include "mma_half.cu"
#include <cstdio>
static void chk(int M,int N,int K,float a,float b){
  half *A,*B,*C;
  cudaMalloc(&A,(size_t)M*K*2); cudaMalloc(&B,(size_t)K*N*2); cudaMalloc(&C,(size_t)M*N*2);
  cudaMemset(A,0,(size_t)M*K*2); cudaMemset(B,0,(size_t)K*N*2); cudaMemset(C,0,(size_t)M*N*2);
  solve(A,B,C,M,N,K,a,b);
  cudaError_t e=cudaDeviceSynchronize();
  if(e!=cudaSuccess) printf("  FAIL %d,%d,%d: %s\n",M,N,K,cudaGetErrorString(e));
  cudaFree(A);cudaFree(B);cudaFree(C);
}
int main(){
  int s[][3]={{128,256,64},{1024,1024,1024},{129,260,72},{100,300,64},{17,33,16},{1,1,8},
              {2048,2048,512},{4096,512,4096},{255,257,264},{1,4096,4096},{4096,4096,8},
              {2047,1023,512},{3000,1000,2000},{384,2048,2048},{8,8192,8192},{8192,8,8192},
              {65,4097,136},{512,512,512},{1000,700,304},{33,33,33},{4096,4095,4096},{2048,2047,2049},{4095,4095,4095},
              {1023,1,1025},{7,4095,9}};
  for(auto&x:s){ chk(x[0],x[1],x[2],1.f,0.f); chk(x[0],x[1],x[2],2.5f,-1.5f); }
  printf("done, %d shapes x2 (beta=0 and beta!=0)\n",(int)(sizeof(s)/sizeof(s[0])));
  return 0;
}
