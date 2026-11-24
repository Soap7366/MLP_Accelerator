"""Calibrate a read-only A9 global counter; otherwise retain monotonic stamps."""
import ctypes,platform,time
from pynq import Clocks


def configure(lib):
    for name in ('mlp_timer_open','mlp_timer_enable'):
        getattr(lib,name).argtypes=[ctypes.c_uint32]
        getattr(lib,name).restype=ctypes.c_int
    for name in ('mlp_timer_control','mlp_timer_frequency','mlp_timer_enabled'):
        getattr(lib,name).argtypes=[];getattr(lib,name).restype=ctypes.c_uint32
    for name in ('mlp_timer_ticks','mlp_timer_fast_calls','mlp_timer_fallback_calls','mlp_timer_timeout_calls'):
        getattr(lib,name).argtypes=[];getattr(lib,name).restype=ctypes.c_uint64
    lib.mlp_timer_enable(0)
    result={'mode':'clock_monotonic','machine':platform.machine(),
            'cpu_mhz':float(Clocks.cpu_mhz),'timer_register_writes':0}
    try:
        if result['machine']!='armv7l' or result['cpu_mhz']!=650:
            raise RuntimeError('Counter calibration profile requires ARMv7 CPU650MHz')
        expected_hz=162500000
        if lib.mlp_timer_open(expected_hz):raise RuntimeError('Read-only timer mapping is unavailable')
        control=int(lib.mlp_timer_control())
        if control&0xff01!=0x101:raise RuntimeError('Counter enable/prescaler differs from the verified profile')
        rates=[]
        for _ in range(5):
            begin=time.perf_counter_ns();first=lib.mlp_timer_ticks()
            time.sleep(.1)
            last=lib.mlp_timer_ticks();end=time.perf_counter_ns()
            rate=(last-first)*1e9/(end-begin)
            rates.append(rate)
        accepted=sum(abs(rate/expected_hz-1)<=.002 for rate in rates)
        if accepted<3:raise RuntimeError('Counter rate does not match the independent monotonic reference')
        if lib.mlp_timer_control()&0xff01!=control&0xff01:
            raise RuntimeError('Counter configuration changed during calibration')
        if lib.mlp_timer_enable(1):raise RuntimeError('Counter configuration changed before enabling timestamps')
        result.update({'mode':'read_only_global_timer','counter_hz':expected_hz,
                       'measured_hz':rates,'accepted_calibration_samples':accepted,'control_before':control})
    except (OSError,RuntimeError) as exc:
        lib.mlp_timer_enable(0);result['fallback_reason']=str(exc)
    return result


def statistics(lib,metadata):
    return {**metadata,'enabled_now':bool(lib.mlp_timer_enabled()),
            'control_now':int(lib.mlp_timer_control()),
            'fast_calls':int(lib.mlp_timer_fast_calls()),
            'fallback_calls':int(lib.mlp_timer_fallback_calls()),
            'timeout_calls':int(lib.mlp_timer_timeout_calls()),
            'counter_hz_now':int(lib.mlp_timer_frequency())}
