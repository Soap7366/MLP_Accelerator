/* Separate read-only transport. The signed-off inference timer is unchanged. */
#include <stdint.h>
#include <time.h>
static uint64_t ns(void) {
 struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
 return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec;
}
int mlp_read_existing(volatile uint32_t *dma,volatile uint32_t *gpio,
 uint32_t tx,uint32_t rx,uint32_t *received) {
 gpio[0]=0;__sync_synchronize();
 dma[0x48/4]=rx;__sync_synchronize();dma[0x58/4]=176;
 gpio[0]=1;__sync_synchronize();
 dma[0x18/4]=tx;__sync_synchronize();dma[0x28/4]=4;
 uint64_t start=ns();unsigned loops=0;
 for(;;) {
  uint32_t send=dma[1],recv=dma[0x34/4];
  if((send|recv)&0x70)return -1;
  if((send&2)&&(recv&2))break;
  if((++loops&255)==0 && ns()-start>1000000000ull)return -2;
 }
 __sync_synchronize();*received=dma[0x58/4];
 gpio[0]=0;__sync_synchronize();
 return (*received==12 || *received==176)?0:-3;
}
