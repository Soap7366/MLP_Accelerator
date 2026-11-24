#include <stdint.h>
#include <time.h>
extern int mlp_batch_quiet(volatile uint32_t *,volatile uint32_t *,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t,uint32_t *);
int mlp_measure_native(volatile uint32_t *dma,volatile uint32_t *gpio,
 uint32_t tx,uint32_t rx,uint32_t length,uint32_t count,uint32_t repeats,uint64_t *elapsed) {
 if(!repeats||repeats>1000||!elapsed)return -5;
 struct timespec begin,end;
 clock_gettime(CLOCK_MONOTONIC,&begin);
 for(uint32_t n=0;n<repeats;n++) {
  uint32_t completed=0;
  int rc=mlp_batch_quiet(dma,gpio,tx,rx,length,count,0,&completed);
  if(rc||completed!=count)return rc?rc:-6;
 }
 clock_gettime(CLOCK_MONOTONIC,&end);
 *elapsed=((uint64_t)end.tv_sec*1000000000ull+end.tv_nsec)-
          ((uint64_t)begin.tv_sec*1000000000ull+begin.tv_nsec);
 return 0;
}
