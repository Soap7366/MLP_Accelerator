"""Interactive R108 video demo. Uses the frozen board runtime and real DMA results."""
import argparse, ctypes, hashlib, json, os, platform, subprocess, time, uuid
from pathlib import Path
import numpy as np
from pynq import Clocks
from clock_config import load_overlay
from quiet_transport import QuietDMA

HERE=Path(__file__).resolve().parent

def emit(text=''):
    print(text,flush=True)

def heading(title):
    emit('\n'+'='*78);emit(title);emit('='*78)

def read_nonempty(prompt):
    # Some serial terminals send CRLF; a Linux TTY can deliver an extra empty
    # line after CR-to-LF conversion. Keep waiting instead of consuming a choice.
    while True:
        value=input(prompt).strip()
        if value:
            return value

class Demo:
    def __init__(self):
        self.cpu=float(Clocks.cpu_mhz)
        assert self.cpu==325
        candidate=json.loads((HERE/'r108_readonly/release.json').read_text())
        for name in ('mlp.bit','mlp.hwh'):
            assert hashlib.sha256((HERE/'r108_readonly'/name).read_bytes()).hexdigest()==candidate[name]
        self.ol,self.clocks=load_overlay(HERE/'r108_readonly/mlp.bit')
        self.last_sample=None
        self.reader=ctypes.CDLL(str(HERE/'read_existing.so'))
        self.read_existing=self.reader.mlp_read_existing
        self.read_existing.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.POINTER(ctypes.c_uint32)]
        self.read_existing.restype=ctypes.c_int
        gpio=self.ol.axi_gpio_0;gpio.write(4,0);gpio.write(0,0)
        self.dma=QuietDMA(self.ol.axi_dma_0,gpio)
        self.native=self.dma.quiet.mlp_measure_native
        self.native.argtypes=[ctypes.c_void_p,ctypes.c_void_p]+[ctypes.c_uint32]*5+[ctypes.POINTER(ctypes.c_uint64)]
        self.native.restype=ctypes.c_int
        with np.load(HERE/'test_vectors.npz',allow_pickle=False) as z:
            self.data={k:z[k].copy() for k in ('inputs','features','labels')}
        assert self.data['inputs'].shape==(42,512) and self.data['features'].shape==(42,162)
        self.out=HERE/'menu_runs'/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
        self.out.mkdir(parents=True);self.sequence=0
        self.pl_mhz=float(Clocks.fclk1_mhz)
        emit(f'LIVE CPU={float(Clocks.cpu_mhz):g} MHz | PL={self.pl_mhz:.6f} MHz | DMA={float(Clocks.fclk0_mhz):g} MHz')
        emit('Model: 512 -> 128 -> 32 -> 2 | INT8 PTQ | zero point = 0')
        emit('Result directory: '+str(self.out))

    def pack(self,tx,indices,feature):
        tx[:,0]=0x4d4c5000|int(feature)
        tx[:,1:129]=self.data['inputs'][indices].view('<u4').reshape(len(indices),128)

    def request(self,tx,rx,feature,count=1,repeats=1):
        tx.flush();elapsed=ctypes.c_uint64()
        rc=self.native(self.dma.buffers[0],self.dma.buffers[1],self.dma._address(tx),self.dma._address(rx),
                       176 if feature else 16,count,repeats,ctypes.byref(elapsed))
        rx.invalidate()
        if rc:raise RuntimeError('Native DMA error: '+str(rc))
        return elapsed.value/1000/(count*repeats)

    def decode(self,rx,indices,feature):
        n=162 if feature else 2
        headers=np.asarray(rx[:,0]).copy().astype(np.uint32)
        assert np.all((headers & np.uint32(0xfffffffe))==np.uint32(0x4d4c0000|(int(feature)<<1))), 'Invalid header/status'
        cycles=np.asarray(rx[:,1]).copy().astype(np.uint32)
        assert np.all(cycles==713) and np.all(rx[:,2]==n), 'Invalid cycles or payload length'
        values=rx.view(np.int8).reshape(len(indices),176)[:,12:12+n].copy()
        gold=self.data['features'][indices] if feature else self.data['features'][indices,160:]
        predictions=(headers & 1).astype(np.uint8)
        logits=values[:,160:] if feature else values
        assert np.array_equal(predictions,np.argmax(logits,axis=1)), 'Classification/output mismatch'
        return predictions,cycles,values,gold

    def save(self,kind,result):
        self.sequence+=1
        path=self.out/f'{self.sequence:02d}_{kind}.json'
        result.update({'CPU_MHz':float(Clocks.cpu_mhz),'PL_MHz':self.pl_mhz,'mode':kind,
                       'live':True,'bit_sha256':hashlib.sha256((HERE/'r108_readonly/mlp.bit').read_bytes()).hexdigest()})
        path.write_text(json.dumps(result,indent=2)+'\n');emit('Saved: '+str(path))

    def inference(self):
        heading('MODE 1 : INFERENCE / CLASSIFICATION - LIVE BOARD RUN')
        emit('42 samples | input: 516 bytes | normal output: 16 bytes')
        emit('Rows: SINGLE-SHOT round trip with two timer reads for EACH request.')
        emit('Use the PRIMARY BENCHMARK at the bottom for the original ~10 us performance metric.')
        emit('Packing, Python, outer buffer sync, comparison and printing are outside timing.')
        rows=[]
        with self.dma.allocate((1,130),'u4') as tx,self.dma.allocate((1,44),'u4') as rx:
            self.pack(tx,[0],False)
            for _ in range(10):self.request(tx,rx,False)
            for i in range(42):
                self.pack(tx,[i],False);us=self.request(tx,rx,False)
                pred,cycles,values,gold=self.decode(rx,[i],False)
                correct=int(pred[0])==int(self.data['labels'][i]);match=bool(np.array_equal(values,gold))
                status='PASS' if correct and match else 'FAIL'
                emit(f'[Group {i:02d}] pred={int(pred[0])} label={int(self.data["labels"][i])} {status:4s} | single_rt={us:8.3f} us | PL={int(cycles[0])/self.pl_mhz:.3f} us')
                rows.append({'sample':i,'prediction':int(pred[0]),'label':int(self.data['labels'][i]),'correct':correct,
                             'logits_match':match,'logits':values[0].astype(int).tolist(),'PS_PL_PS_us':us,'PL_cycles':int(cycles[0])})
        correct=sum(r['correct'] for r in rows)
        emit(f'\nAccuracy: {correct}/42 = {100*correct/42:.2f}% (provided calibration dataset)')
        emit(f'Single-request timing mean: {np.mean([r["PS_PL_PS_us"] for r in rows]):.3f} us')
        emit('\nRunning the original performance boundary: 10 warm-up groups + 100 measured groups, 420 serial requests/group...')
        timings=[]
        with self.dma.allocate((42,130),'u4') as tx,self.dma.allocate((42,44),'u4') as rx:
            self.pack(tx,list(range(42)),False)
            for group in range(110):
                us=self.request(tx,rx,False,42,10)
                pred,cycles,values,gold=self.decode(rx,list(range(42)),False)
                if not np.array_equal(values,gold):raise RuntimeError('Benchmark logits differ from integer reference')
                if group>=10:timings.append(us)
        native={k:float(v) for k,v in [('mean',np.mean(timings)),('median',np.median(timings)),('p95_group_mean',np.percentile(timings,95))]}
        emit('--- PRIMARY BENCHMARK: PS -> PL -> PS (same boundary as the original ~10 us result) ---')
        emit('Two timer reads per 420-request group; 100 measured groups after 10 warm-up groups.')
        emit(f'PS -> PL -> PS mean   : {native["mean"]:.3f} us/sample')
        emit(f'PS -> PL -> PS median : {native["median"]:.3f} us/sample')
        emit(f'PL processing        : {int(cycles[0])/self.pl_mhz:.3f} us ({int(cycles[0])} cycles)')
        passed=correct==42 and all(r['logits_match'] for r in rows)
        self.save('inference',{'passed':passed,'correct':correct,'total':42,'samples':rows,'native_DMA_roundtrip_us':native,
                  'timing_scope':'C CLOCK_MONOTONIC; excludes Python, packing, outer buffer sync and printing. Row timing brackets each request; benchmark brackets each 420-request group.'})
        self.last_sample=41
        emit('MODE 1: '+('PASS' if passed else 'FAIL'))
        return passed

    def features(self,index=None,full=False):
        index=self.last_sample
        heading('MODE 2 : READ EXISTING PL FEATURES - NO INFERENCE')
        with self.dma.allocate((1,2),'u4') as tx,self.dma.allocate((1,44),'u4') as rx:
            tx[0,0]=0x4d4c5001;tx.flush();received=ctypes.c_uint32()
            rc=self.read_existing(self.dma.buffers[0],self.dma.buffers[1],self.dma._address(tx),self.dma._address(rx),ctypes.byref(received))
            rx.invalidate()
            if rc:raise RuntimeError('Read-only DMA failed: '+str(rc))
            header=int(rx[0,0])
            assert (header & 0xfffffff8)==0x4d4c0000 and (header & 2), 'Invalid read response'
            if header & 4:
                assert received.value==12 and int(rx[0,1])==0 and int(rx[0,2])==0
                emit('No completed inference since FPGA reset. Run mode 1 first. No inference was started.')
                return False
            assert received.value==176 and index is not None, 'Unknown source sample'
            pred,cycles,values,gold=self.decode(rx,[index],True)
        emit(f'Source: PL retained feature RAM | last completed sample: {index:02d}')
        emit('Sent: 4-byte read command only; no input vector; received: 176 bytes.')
        layers=[]
        for name,begin,end in [('FC1 + ReLU',0,128),('FC2 + ReLU',128,160),('FC3 logits',160,162)]:
            got=values[0,begin:end].astype(int);ref=gold[0,begin:end].astype(int)
            show=len(got) if full or len(got)<=32 else 16
            emit(f'\n{name}: {len(got)} values (showing {show}; checking ALL {len(got)})')
            emit('FPGA : '+str(got[:show].tolist()))
            emit('REF  : '+str(ref[:show].tolist()))
            errors=np.abs(got-ref);matched=int(np.sum(errors==0));maximum=int(errors.max())
            emit(f'Match: {matched}/{len(got)} | max absolute error: {maximum} | '+('PASS' if maximum==0 else 'FAIL'))
            for j in np.flatnonzero(errors)[:8]:emit(f'MISMATCH index={j}: FPGA={got[j]}, REF={ref[j]}')
            layers.append({'layer':name,'hardware':got.tolist(),'reference':ref.tolist(),'matched':matched,'max_absolute_error':maximum})
        matches=int(np.sum(values==gold));passed=matches==162
        emit(f'\nAll layer features: {matches}/162 match | prediction={int(pred[0])} | label={int(self.data["labels"][index])}')
        emit(f'Previous inference: {int(cycles[0])} cycles. This read does not run the compute core.')
        emit('This is a PL memory read, not a new inference timing measurement.')
        self.save('features',{'passed':passed,'sample':index,'matches':matches,'layers':layers,'prediction':int(pred[0]),
                             'label':int(self.data['labels'][index]),'source':'retained_PL_RAM','inference_started':False,'previous_PL_cycles':int(cycles[0])})
        emit('MODE 2: '+('PASS' if passed else 'FAIL'));return passed

    def all_features(self):
        with self.dma.allocate((42,130),'u4') as tx,self.dma.allocate((42,44),'u4') as rx:
            self.pack(tx,list(range(42)),True);self.request(tx,rx,True,42)
            pred,cycles,values,gold=self.decode(rx,list(range(42)),True)
        assert np.array_equal(values,gold) and np.array_equal(pred,self.data['labels'])
        self.save('all_features',{'passed':True,'samples':42,'values_checked':6804})
        emit('All 42 samples / 6804 features: PASS')

def main():
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');a=p.parse_args()
    os.chdir(HERE)
    manifest=json.loads((HERE/'video_manifest.json').read_text())
    assert os.geteuid()==0,'Run with sudo'
    assert platform.release()==manifest['kernel'],'Kernel mismatch'
    for name,expected in manifest['files'].items():
        path=Path(name) if name.startswith('/') else HERE/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,'Runtime mismatch: '+name
    subprocess.run(['sh',str(HERE/'compile.sh')],check=True)
    policy=Path('/sys/devices/system/cpu/cpufreq/policy0');loaded=False;old=None
    original=float(Clocks.cpu_mhz);assert original in (325,650)
    affinity=os.sched_getaffinity(0)
    try:
        if not policy.exists():
            assert original==650
            subprocess.run(['insmod',str(HERE/'cpufreq/mlp_fixed_cpufreq.ko')],check=True);loaded=True
        assert (policy/'scaling_driver').read_text().strip()=='mlp-fixed'
        old={n:(policy/n).read_text().strip() for n in ('scaling_governor','scaling_cur_freq')}
        (policy/'scaling_governor').write_text('userspace');(policy/'scaling_setspeed').write_text('325000')
        time.sleep(.2);assert float(Clocks.cpu_mhz)==325
        os.sched_setaffinity(0,{0});demo=Demo()
        if a.self_test:
            assert demo.features() is False
            assert demo.inference()
            assert demo.features(full=True) and demo.features(full=True)
        else:
            while True:
                heading('R108 DEMO MENU | CPU 325 MHz')
                emit('1. INFERENCE: classify 42 samples + measure PS-PL-PS and PL time')
                emit('2. READ PL FEATURES: last completed sample; no inference')
                emit('0. EXIT: restore the original CPU frequency')
                choice=read_nonempty('Select [0/1/2]: ')
                if choice=='0':break
                if choice=='1':demo.inference()
                elif choice=='2':demo.features(full=True)
                else:emit('Please choose 0, 1 or 2.')
    finally:
        os.sched_setaffinity(0,affinity)
        if old is not None:
            (policy/'scaling_setspeed').write_text(old['scaling_cur_freq'])
            (policy/'scaling_governor').write_text(old['scaling_governor'])
            assert float(Clocks.cpu_mhz)==original
        if loaded:subprocess.run(['rmmod','mlp_fixed_cpufreq'],check=True)
        emit(f'CPU restored to {float(Clocks.cpu_mhz):g} MHz. R108 read-only candidate bitstream remains loaded.')
    if a.self_test:emit('R108_MENU_SELFTEST_PASS')

if __name__=='__main__':main()
