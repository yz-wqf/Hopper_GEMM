#include "mma_half.cu"
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#define CK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e));exit(1);}}while(0)
static cublasHandle_t H;
// row-major C[M,N] = alpha*A[M,K]@B[K,N] + beta*C, fp32 compute, via col-major cuBLAS
static void cb(const half*A,const half*B,half*C,int M,int N,int K,float al,float be){
  cublasGemmEx(H,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&al,B,CUDA_R_16F,N,A,CUDA_R_16F,K,&be,
               C,CUDA_R_16F,N,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
}
int main(int argc,char**argv){
  CK(cudaFree(0)); cublasCreate(&H);
  std::mt19937 rng(3); std::uniform_real_distribution<float> D(-2.f,2.f);
  printf("== correctness (vs fp64 ref; fp16 output => ~1e-3 is the format floor) ==\n");
  printf("%-22s %7s %7s %12s %12s\n","M,N,K","alpha","beta","max_abs","rel_l2");
  struct S{int M,N,K;float a,b;};
  S sh[]={{128,256,64,1,0},{256,512,128,1,0},{1024,1024,1024,1,0},{129,260,72,1,0},
          {100,300,64,1,0},{512,512,512,2.5f,-1.5f},{1024,1024,256,0.5f,1.0f},
          {17,33,16,1,0},{1,1,8,1,0},{2048,2048,512,1,0},{4096,512,4096,1,0},
          {255,257,264,1.f,0.25f},{1,4096,4096,1,0},{4096,4096,8,1,0},{2047,1023,512,1,0}};
  for(auto&s:sh){
    int M=s.M,N=s.N,K=s.K;
    std::vector<half> hA((size_t)M*K),hB((size_t)K*N),hC((size_t)M*N),hO((size_t)M*N);
    std::vector<float> fA(hA.size()),fB(hB.size()),fC(hC.size());
    for(size_t i=0;i<hA.size();i++){fA[i]=D(rng); hA[i]=__float2half(fA[i]); fA[i]=__half2float(hA[i]);}
    for(size_t i=0;i<hB.size();i++){fB[i]=D(rng); hB[i]=__float2half(fB[i]); fB[i]=__half2float(hB[i]);}
    for(size_t i=0;i<hC.size();i++){fC[i]=D(rng); hC[i]=__float2half(fC[i]); fC[i]=__half2float(hC[i]);}
    half *A,*B,*C; CK(cudaMalloc(&A,hA.size()*2)); CK(cudaMalloc(&B,hB.size()*2)); CK(cudaMalloc(&C,hC.size()*2));
    CK(cudaMemcpy(A,hA.data(),hA.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(B,hB.data(),hB.size()*2,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(C,hC.data(),hC.size()*2,cudaMemcpyHostToDevice));
    solve(A,B,C,M,N,K,s.a,s.b);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hO.data(),C,hO.size()*2,cudaMemcpyDeviceToHost));
    double ma=0,num=0,den=0;
    for(int i=0;i<M;i++)for(int j=0;j<N;j++){
      double acc=0; for(int k=0;k<K;k++) acc+=(double)fA[(size_t)i*K+k]*fB[(size_t)k*N+j];
      double r=s.a*acc+s.b*(double)fC[(size_t)i*N+j];
      double e=fabs(__half2float(hO[(size_t)i*N+j])-r);
      ma=fmax(ma,e); num+=e*e; den+=r*r;
    }
    char bf[64]; snprintf(bf,sizeof bf,"%d,%d,%d",M,N,K);
    double l2=sqrt(num/(den+1e-30));
    printf("%-22s %7.2f %7.2f %12.3e %12.3e  %s\n",bf,s.a,s.b,ma,l2,l2<5e-3?"OK":"*** FAIL ***");
    cudaFree(A);cudaFree(B);cudaFree(C);
  }
  printf("\n== performance (alpha=1, beta=0) ==\n");
  printf("%-18s %10s %10s %10s %10s %8s\n","shape","ours ms","ours TF","cuBLAS ms","cuBLAS TF","ratio");
  int bs[][3]={{4096,4096,4096},{8192,8192,8192},{8192,8192,4096},{4096,4096,8192},{16384,8192,4096}};
  int it=argc>1?atoi(argv[1]):20;
  for(auto&x:bs){
    long long M=x[0],N=x[1],K=x[2];
    half *A,*B,*C; CK(cudaMalloc(&A,(size_t)M*K*2)); CK(cudaMalloc(&B,(size_t)K*N*2)); CK(cudaMalloc(&C,(size_t)M*N*2));
    CK(cudaMemset(A,0x11,(size_t)M*K*2)); CK(cudaMemset(B,0x11,(size_t)K*N*2));
    double fl=2.0*M*N*K; cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float t1,t2;
    for(int i=0;i<5;i++) h100_hgemm::launch(A,B,C,M,N,K,1.f,0.f,0); CK(cudaDeviceSynchronize());
    cudaEventRecord(e0); for(int i=0;i<it;i++) h100_hgemm::launch(A,B,C,M,N,K,1.f,0.f,0);
    cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&t1,e0,e1); t1/=it;
    for(int i=0;i<5;i++) cb(A,B,C,M,N,K,1.f,0.f); CK(cudaDeviceSynchronize());
    cudaEventRecord(e0); for(int i=0;i<it;i++) cb(A,B,C,M,N,K,1.f,0.f);
    cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&t2,e0,e1); t2/=it;
    char bf[64]; snprintf(bf,sizeof bf,"%lldx%lldx%lld",M,N,K);
    printf("%-18s %10.3f %10.1f %10.3f %10.1f %7.2fx\n",bf,t1,fl/(t1*1e9),t2,fl/(t2*1e9),t2/t1);
    cudaFree(A);cudaFree(B);cudaFree(C);
  }
  return 0;
}
