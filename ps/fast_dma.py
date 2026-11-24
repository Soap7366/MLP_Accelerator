"""Native polling transport, retaining PYNQ allocation/cache management."""
import ctypes, mmap, os
from pathlib import Path

class FastDMA:
    def __init__(self,dma,gpio):
        parameters=dma.description.get('parameters',{})
        for name in ('C_M_AXI_MM2S_ADDR_WIDTH','C_M_AXI_S2MM_ADDR_WIDTH'):
            if name not in parameters or int(parameters[name])!=32:
                raise ValueError('Native MLP transport requires verified 32-bit DMA addresses: '+name)
        self.lib=ctypes.CDLL(str(Path(__file__).with_name('dma_fast.so')))
        self.lib.mlp_transfer.argtypes=[ctypes.c_void_p,ctypes.c_void_p,
             ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32,ctypes.POINTER(ctypes.c_uint64)]
        self.lib.mlp_transfer.restype=ctypes.c_int
        self.lib.mlp_batch.argtypes=[ctypes.c_void_p,ctypes.c_void_p,
             ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_uint32,ctypes.POINTER(ctypes.c_uint64)]
        self.lib.mlp_batch.restype=ctypes.c_int
        self.fd=os.open('/dev/mem',os.O_RDWR|os.O_SYNC)
        self.maps=[mmap.mmap(self.fd,4096,flags=mmap.MAP_SHARED,
                    prot=mmap.PROT_READ|mmap.PROT_WRITE,offset=int(ip.mmio.base_addr)) for ip in (dma,gpio)]
        self.buffers=[(ctypes.c_uint32*1024).from_buffer(m) for m in self.maps]
        self.native_ns=0
        self.lib.mlp_memory_fence.argtypes=[]
        self.lib.mlp_memory_fence.restype=None
        self.buffer_allocations=[]
        from timer_backend import configure
        self.timer_metadata=configure(self.lib)
    def timer_statistics(self):
        from timer_backend import statistics
        return statistics(self.lib,self.timer_metadata)
    def allocate(self,shape,dtype='u4'):
        from uncached_buffer import allocate
        buffer,details=allocate(shape,dtype,self.lib.mlp_memory_fence)
        self.buffer_allocations.append(details)
        return buffer
    @staticmethod
    def _address(buffer):
        address=int(buffer.physical_address)
        if address%8:
            raise ValueError('DMA memory addresses must be 8-byte aligned')
        if address<0 or address+int(buffer.nbytes)>1<<32:
            raise ValueError('DMA buffer exceeds the supported 32-bit address range')
        return address
    def transfer(self,tx,rx,length):
        tx_address=self._address(tx);rx_address=self._address(rx)
        tx.flush()
        elapsed=ctypes.c_uint64()
        result=self.lib.mlp_transfer(self.buffers[0],self.buffers[1],
                 tx_address,rx_address,length,ctypes.byref(elapsed))
        if result:raise RuntimeError(f'Native DMA failed: {result}')
        rx.invalidate()
        self.native_ns=elapsed.value
    def batch(self,tx,rx,length,count):
        if tx.shape!=(count,130) or rx.shape!=(count,44):
            raise ValueError('Batch strides must be 520 input and 176 output bytes')
        tx_address=self._address(tx);rx_address=self._address(rx)
        times=(ctypes.c_uint64*count)()
        tx.flush()
        result=self.lib.mlp_batch(self.buffers[0],self.buffers[1],
                 tx_address,rx_address,length,count,times)
        if result:raise RuntimeError(f'Native DMA batch failed: {result}')
        rx.invalidate()
        return list(times)
