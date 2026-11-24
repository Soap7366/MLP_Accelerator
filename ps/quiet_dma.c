/* Isolated R103 transport experiment. No per-request timestamp or division.
 * Same MMIO launch ordering, memory barriers, error and length checks.
 * CPU frequency independent wall-clock watchdog remains enabled. */
#include <stdint.h>
#ifndef MLP_READ
#include <time.h>
static uint64_t wall_ns(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec;
}
#define MLP_READ(base,index) ((base)[index])
#define MLP_WRITE(base,index,value) ((base)[index]=(value))
#define MLP_BARRIER() __sync_synchronize()
#define MLP_NOW() wall_ns()
#endif

int mlp_batch_quiet(volatile uint32_t *dma,volatile uint32_t *gpio,
        uint32_t tx,uint32_t rx,uint32_t length,uint32_t count,
        uint32_t receive_first,uint32_t *completed) {
    if(!completed)return -5;
    *completed=0;
    if(!dma||!gpio||!count||(tx&7)||(rx&7)||receive_first>1||
       (length!=16&&length!=176)||
       (uint64_t)tx+(uint64_t)count*520>(1ull<<32)||
       (uint64_t)rx+(uint64_t)count*176>(1ull<<32))return -5;
    /* Start the watchdog on the first256-poll stall, as in the released
       transport; normal transactions need no timestamp reads. */
    for(uint32_t i=0;i<count;i++) {
        MLP_WRITE(gpio,0,0);MLP_BARRIER();
        MLP_WRITE(dma,0x48/4,rx+i*176);MLP_BARRIER();
        MLP_WRITE(dma,0x58/4,length);
        MLP_WRITE(gpio,0,1);MLP_BARRIER();
        MLP_WRITE(dma,0x18/4,tx+i*520);MLP_BARRIER();
        MLP_WRITE(dma,0x28/4,516);
        unsigned loops=0;uint64_t stalled_since=0;
        for(;;) {
            if(!receive_first) {
                uint32_t send=MLP_READ(dma,1);
                uint32_t recv=MLP_READ(dma,0x34/4);
                if((send|recv)&0x70)return -1;
                if((send&2)&&(recv&2))break;
            } else {
                uint32_t recv=MLP_READ(dma,0x34/4);
                if(recv&0x70)return -1;
                if((recv&2)||!(loops&15)) {
                    uint32_t send=MLP_READ(dma,1);
                    if(send&0x70)return -1;
                    if((send&2)&&(recv&2))break;
                }
            }
            if(!(++loops&255)) {
                uint64_t now=MLP_NOW();
                if(!stalled_since)stalled_since=now;
                if(now-stalled_since>1000000000ull)return -2;
            }
        }
        MLP_BARRIER();
        if(MLP_READ(dma,0x58/4)!=length)return -3;
        *completed=i+1;
    }
    return 0;
}
