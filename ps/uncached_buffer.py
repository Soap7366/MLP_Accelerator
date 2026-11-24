"""Owned CMA buffers with guarded ARM O_SYNC aliases and view-safe lifetime."""
import functools,gzip,hashlib,mmap,os,platform,re,weakref
from pathlib import Path
import numpy as np

_stats={'created':0,'released':0}

@functools.lru_cache(maxsize=1)
def capability():
    result={'kernel':platform.release(),'machine':platform.machine()}
    if result['machine']!='armv7l':return {**result,'supported':False,'reason':'requires ARMv7 Linux'}
    try:config=gzip.decompress(Path('/proc/config.gz').read_bytes()).decode()
    except (OSError,ValueError):return {**result,'supported':False,'reason':'kernel configuration is unavailable'}
    result['kernel_config_sha256']=hashlib.sha256(config.encode()).hexdigest()
    required=('CONFIG_ARM=y','CONFIG_ARM_DMA_MEM_BUFFERABLE=y','CONFIG_DEVMEM=y')
    if not all(x in config.splitlines() for x in required) or '# CONFIG_STRICT_DEVMEM is not set' not in config:
        return {**result,'supported':False,'reason':'kernel mapping conditions are not satisfied'}
    return {**result,'supported':True,'memory_type':'ARM pgprot_writecombine / L_PTE_MT_BUFFERABLE','fence':'DSB SY'}

def _is_ram(address,size):
    try:lines=Path('/proc/iomem').read_text().splitlines()
    except OSError:return False
    for line in lines:
        m=re.match(r'\s*([0-9a-fA-F]+)-([0-9a-fA-F]+)\s*:\s*System RAM\s*$',line)
        if m and int(m[1],16)<=address and address+size-1<=int(m[2],16):return True
    return False

def _release(owner):
    owner.freebuffer();_stats['released']+=1

def lifetime_statistics():return dict(_stats)

class OwnedBuffer:
    def __init__(self,owner,memory,fence,details):
        self._array=np.ndarray(owner.shape,dtype=owner.dtype,buffer=memory)
        self._physical=int(owner.physical_address);self._memory=memory;self._fence=fence
        self.details=details;self._closed=False
        # mmap is retained by every exported NumPy base/view, including plain
        # ndarray views; the CMA owner outlives all of them through this callback.
        weakref.finalize(memory,_release,owner);_stats['created']+=1
    def _open(self):
        if self._closed:raise RuntimeError('DMA buffer is closed')
    @property
    def physical_address(self):self._open();return self._physical
    @property
    def device_address(self):return self.physical_address
    @property
    def shape(self):self._open();return self._array.shape
    @property
    def nbytes(self):self._open();return self._array.nbytes
    @property
    def dtype(self):self._open();return self._array.dtype
    def __len__(self):self._open();return len(self._array)
    def __getitem__(self,key):self._open();return self._array[key]
    def __setitem__(self,key,value):self._open();self._array[key]=value
    def __array__(self,dtype=None,copy=None):
        self._open();a=np.asarray(self._array,dtype=dtype)
        return a.copy() if copy else a
    def view(self,*args,**kwargs):self._open();return self._array.view(*args,**kwargs)
    def copy(self,*args,**kwargs):self._open();return self._array.copy(*args,**kwargs)
    def flush(self):self._open();self._fence()
    def invalidate(self):self._open();self._fence()
    def freebuffer(self):
        if not self._closed:
            self._closed=True;self._array=None;self._memory=None
    close=freebuffer
    def __enter__(self):self._open();return self
    def __exit__(self,*args):self.freebuffer()

def allocate(shape,dtype,fence):
    from pynq import allocate as pynq_allocate
    owner=pynq_allocate(shape,dtype=dtype,cacheable=False)
    info=capability()
    details={**info,'mode':'xrt_sync','shape':list(owner.shape),'bytes':int(owner.nbytes),
             'physical_address':int(owner.physical_address)}
    address,size=int(owner.physical_address),int(owner.nbytes)
    if not info['supported'] or address%mmap.PAGESIZE or size<=0 or not _is_ram(address,size):
        details['fallback_reason']=info.get('reason','owned allocation is not a page-aligned System RAM buffer')
        return owner,details
    if address<0 or address+size>2**32:
        owner.freebuffer();raise ValueError('Owned DMA allocation exceeds32-bit address space')
    try:
        # Initialize coherence once; the old alias is not touched again. Only
        # the separately mapped noncached alias is exposed to the caller.
        owner.flush();owner.invalidate();fence()
        fd=os.open('/dev/mem',os.O_RDWR|os.O_SYNC)
        try:memory=mmap.mmap(fd,size,flags=mmap.MAP_SHARED,prot=mmap.PROT_READ|mmap.PROT_WRITE,offset=address)
        finally:os.close(fd)
    except OSError as exc:
        details['fallback_reason']='O_SYNC mapping unavailable: '+str(exc)
        return owner,details
    try:
        details['mode']='owned_osync_cma'
        return OwnedBuffer(owner,memory,fence,details),details
    except BaseException:
        memory.close();owner.freebuffer();raise
