#include "mma_half.cu"
#include <cstdio>
int main(){
  int maxOptin=0, perSM=0;
  cudaDeviceGetAttribute(&maxOptin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  cudaDeviceGetAttribute(&perSM, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
  using namespace h100_hgemm;
  int used = STAGES*(A_STAGE_ELEMS+B_STAGE_ELEMS)*2 + 2*STAGES*8;
  printf("max dynamic smem/block : %d B\n", maxOptin);
  printf("smem per SM            : %d B\n", perSM);
  printf("pipeline uses          : %d B  (STAGES=%d x %d B)\n", used, STAGES,
         (A_STAGE_ELEMS+B_STAGE_ELEMS)*2);
  printf("spare                  : %d B  (%.1f KB)\n", maxOptin-used, (maxOptin-used)/1024.0);
  printf("one 128x256 half tile  : %d B  (%.1f KB)  <- too big to stage whole\n",
         128*256*2, 128*256*2/1024.0);
  printf("one 128x64 half chunk  : %d B  (%.1f KB)\n", 128*64*2, 128*64*2/1024.0);
  return 0;
}
