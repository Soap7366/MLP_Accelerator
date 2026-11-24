#define _FILE_OFFSET_BITS 64
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
// Simple-mode AXI DMA transaction; MMIO mappings are provided by Python.
#include <stdint.h>
#include <time.h>
static uint64_t now_ns(void) {
 struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
 return (uint64_t)t.tv_sec*1000000000ull+t.tv_nsec;
}
static void barrier(void) { __sync_synchronize(); }
int mlp_transfer_monotonic(volatile uint32_t *dma, volatile uint32_t *gpio,
 uint32_t tx, uint32_t rx, uint32_t length, uint64_t *elapsed) {
 gpio[0]=0; barrier();
 uint64_t begin=now_ns();
 dma[0x48/4]=rx; barrier(); dma[0x58/4]=length;
 gpio[0]=1; barrier();
 dma[0x18/4]=tx; barrier(); dma[0x28/4]=516;
 unsigned loops=0;
 for(;;) {
  uint32_t send=dma[1],recv=dma[0x34/4];
  if((send|recv)&0x70) return -1;
  if((send&2) && (recv&2)) break;
  if((++loops&255)==0 && now_ns()-begin>1000000000ull) return -2;
 }
 barrier(); *elapsed=now_ns()-begin;
 return dma[0x58/4]==length ? 0 : -3;
}
int mlp_batch_monotonic(volatile uint32_t *dma, volatile uint32_t *gpio,
 uint32_t tx, uint32_t rx, uint32_t length, uint32_t count, uint64_t *times) {
 for(uint32_t i=0;i<count;i++) {
  int status=mlp_transfer_monotonic(dma,gpio,tx+i*520,rx+i*176,length,&times[i]);
  if(status) return status;
 }
 return 0;
}

// Explicit O_SYNC buffer aliases are noncached/write-combined. Complete all
// CPU stores before kicking DMA, and order CPU reads after DMA completion.
void mlp_memory_fence(void) {
#if defined(__arm__)
 __asm__ __volatile__("dsb sy" ::: "memory");
#else
 __sync_synchronize();
#endif
}

// Appended to an unchanged, renamed copy of the signed-off C transport.
// The counter is mapped read-only; kernel timer registers are never written.
static volatile const uint32_t *mlp_gt;
static uint32_t mlp_gt_hz, mlp_gt_control;
static int mlp_gt_enabled;
static uint64_t fast_calls,fallback_calls,timeout_calls;

static uint64_t mlp_gt_ticks(void) {
    uint32_t high=mlp_gt[1],low,previous;
    do { previous=high;low=mlp_gt[0];high=mlp_gt[1]; } while(high!=previous);
    return ((uint64_t)high<<32)|low;
}

int mlp_timer_open(uint32_t expected_hz) {
    if(mlp_gt) return expected_hz==mlp_gt_hz?0:-5;
    int fd=open("/dev/mem",O_RDONLY|O_SYNC);
    if(fd<0)return -2;
    void *memory=mmap(NULL,4096,PROT_READ,MAP_SHARED,fd,0xf8f00000);
    close(fd);
    if(memory==MAP_FAILED)return -3;
    mlp_gt=(volatile const uint32_t *)((char *)memory+0x200);
    mlp_gt_control=mlp_gt[2]&0xff01u;
    if(!(mlp_gt_control&1u)||!expected_hz) {
        mlp_gt=NULL;munmap(memory,4096);return -4;
    }
    mlp_gt_hz=expected_hz;
    return 0;
}

uint32_t mlp_timer_frequency(void) { return mlp_gt_hz; }
uint32_t mlp_timer_enabled(void) { return !!mlp_gt_enabled; }
uint64_t mlp_timer_fast_calls(void) { return fast_calls; }
uint64_t mlp_timer_fallback_calls(void) { return fallback_calls; }
uint64_t mlp_timer_timeout_calls(void) { return timeout_calls; }
uint64_t mlp_timer_ticks(void) { return mlp_gt?mlp_gt_ticks():0; }
uint32_t mlp_timer_control(void) { return mlp_gt?mlp_gt[2]:0; }
int mlp_timer_enable(uint32_t enable) {
    if(enable && (!mlp_gt||(mlp_gt[2]&0xff01u)!=mlp_gt_control))return -1;
    mlp_gt_enabled=!!enable;
    return 0;
}

static uint64_t mlp_ticks_ns(uint64_t elapsed) {
    // Quotient/remainder avoid overflowing the tick*1e9 intermediate.
    return elapsed/mlp_gt_hz*1000000000ull+
           (elapsed%mlp_gt_hz*1000000000ull+mlp_gt_hz/2)/mlp_gt_hz;
}

int mlp_transfer(volatile uint32_t *dma,volatile uint32_t *gpio,
 uint32_t tx,uint32_t rx,uint32_t length,uint64_t *elapsed) {
    // A changed/disabled counter selects the original transport for the whole
    // transaction. Its timestamp epoch is never mixed with counter ticks.
    if(!mlp_gt_enabled||!mlp_gt||(mlp_gt[2]&0xff01u)!=mlp_gt_control) {
        fallback_calls++;
        return mlp_transfer_monotonic(dma,gpio,tx,rx,length,elapsed);
    }
    fast_calls++;
    gpio[0]=0;barrier();
    uint64_t begin=mlp_gt_ticks();
    dma[0x48/4]=rx;barrier();dma[0x58/4]=length;
    gpio[0]=1;barrier();
    dma[0x18/4]=tx;barrier();dma[0x28/4]=516;
    unsigned loops=0;
    uint64_t deadline=0;
    for(;;) {
        uint32_t send=dma[1],recv=dma[0x34/4];
        if((send|recv)&0x70)return -1;
        if((send&2)&&(recv&2))break;
        if((++loops&255)==0) {
            // Keep a monotonic wall-clock watchdog, including if the kernel
            // stops/reconfigures the global counter during a stalled DMA.
            uint64_t current=now_ns();
            if(!deadline)deadline=current+1000000000ull;
            else if(current>deadline){timeout_calls++;return -2;}
        }
    }
    barrier();
    uint64_t end=mlp_gt_ticks();
    if((mlp_gt[2]&0xff01u)!=mlp_gt_control)return -4;
    *elapsed=mlp_ticks_ns(end-begin);
    return dma[0x58/4]==length?0:-3;
}

int mlp_batch(volatile uint32_t *dma,volatile uint32_t *gpio,
 uint32_t tx,uint32_t rx,uint32_t length,uint32_t count,uint64_t *times) {
    for(uint32_t i=0;i<count;i++) {
        int status=mlp_transfer(dma,gpio,tx+i*520,rx+i*176,length,&times[i]);
        if(status)return status;
    }
    return 0;
}
