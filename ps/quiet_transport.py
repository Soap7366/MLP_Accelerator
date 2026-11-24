"""Low-overhead R103 DMA batches; returns completed count, not timestamp lists.

Use inherited transfer() when individual transaction timing is required.
Compile quiet_dma.c into quiet_dma.so next to this file before use.
"""
import ctypes
from pathlib import Path
from fast_dma import FastDMA
from uncached_buffer import OwnedBuffer


class QuietDMA(FastDMA):
    def __init__(self, dma, gpio):
        super().__init__(dma, gpio)
        self.quiet = ctypes.CDLL(str(Path(__file__).with_name('quiet_dma.so')))
        self.quiet.mlp_batch_quiet.argtypes = [ctypes.c_void_p, ctypes.c_void_p] + [ctypes.c_uint32] * 5 + [ctypes.POINTER(ctypes.c_uint32)]
        self.quiet.mlp_batch_quiet.restype = ctypes.c_int
        self.completed = ctypes.c_uint32()

    def batch(self, tx, rx, length, count, receive_first=False):
        """Synchronously execute a batch. Buffers must remain live during use.

        This object, like FastDMA, must not be shared by concurrent callers.
        All samples retain DMA status, timeout and returned-length checks.
        """
        if count <= 0 or tx.shape != (count, 130) or rx.shape != (count, 44):
            raise ValueError('Batch strides must be 520 input and 176 output bytes')
        # OwnedBuffer always creates a contiguous ndarray and deliberately
        # exposes only guarded methods; PYNQ's fallback exposes NumPy flags.
        if (tx.dtype.itemsize != 4 or rx.dtype.itemsize != 4 or
                (not isinstance(tx, OwnedBuffer) and not tx.flags.c_contiguous) or
                (not isinstance(rx, OwnedBuffer) and not rx.flags.c_contiguous)):
            raise ValueError('Buffers must contain contiguous 32-bit words')
        if length not in (16, 176):
            raise ValueError('Expected normal or full-feature output length')
        tx_address, rx_address = self._address(tx), self._address(rx)
        tx.flush()
        rc = self.quiet.mlp_batch_quiet(self.buffers[0], self.buffers[1], tx_address, rx_address,
                                      length, count, int(receive_first), ctypes.byref(self.completed))
        rx.invalidate()
        if rc or self.completed.value != count:
            raise RuntimeError(f'Native quiet DMA failed: {rc}; completed {self.completed.value}/{count}')
        return self.completed.value

    def prepare(self, tx, rx, length, count):
        """Prepare repeated calls on fixed owned buffers; return a callable.

        Validate layout/addresses once; retain live buffer ownership and check
        closed state on every call. Contents may change between calls.
        """
        if not isinstance(tx, OwnedBuffer) or not isinstance(rx, OwnedBuffer):
            raise ValueError('Prepared calls require guarded OwnedBuffer allocations; use batch() for fallback buffers')
        if count <= 0 or tx.shape != (count,130) or rx.shape != (count,44):
            raise ValueError('Invalid batch shape')
        if tx.dtype.itemsize != 4 or rx.dtype.itemsize != 4 or length not in (16,176):
            raise ValueError('Expected 32-bit buffers and normal/full-feature output length')
        tx_address,rx_address=self._address(tx),self._address(rx)
        completed=ctypes.c_uint32()
        args=(ctypes.c_void_p(ctypes.addressof(self.buffers[0])),
              ctypes.c_void_p(ctypes.addressof(self.buffers[1])),
              ctypes.c_uint32(tx_address),ctypes.c_uint32(rx_address),
              ctypes.c_uint32(length),ctypes.c_uint32(count),ctypes.c_uint32(0),ctypes.byref(completed))
        def execute():
            # Holding self/tx/rx keeps mappings alive; these guards also reject
            # explicit close() calls made while a prepared object still exists.
            rx._open()
            tx.flush()
            rc=self.quiet.mlp_batch_quiet(*args)
            rx.invalidate()
            if rc or completed.value != count:
                raise RuntimeError(f'Prepared DMA failed: {rc}; completed {completed.value}/{count}')
            return completed.value
        return execute
