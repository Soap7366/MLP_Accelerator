"""Load experimental100/142.857MHz overlay against the board's actual 1 GHz IO PLL.

PYNQ applies HWH divisors without reprogramming the boot-time PS PLL. Prepare
the clock dictionary before download, then verify both actual PL clocks and
that CPU/DDR/IO PLL registers remain unchanged.
"""
from copy import deepcopy
from pynq import Clocks, MMIO, Overlay


def load_overlay(bit):
    regs = MMIO(0xf8000000, 0x1000)
    offsets = (0x100, 0x104, 0x108, 0x120, 0x124)
    before = {hex(o): regs.read(o) for o in offsets}
    cpu_before = float(Clocks.cpu_mhz)
    clock0 = regs.read(0x170)
    assert (clock0 >> 4) & 3 == 0, 'This overlay expects the IO PLL clock source'
    div0, div1 = (clock0 >> 8) & 63, (clock0 >> 20) & 63
    assert div0 and div1
    io_mhz = float(Clocks.fclk0_mhz) * div0 * div1
    assert abs(io_mhz - 1000) < .01, f'Expected the experiment IO PLL at 1000 MHz; found {io_mhz}'
    overlay = Overlay(str(bit), download=False)
    original = deepcopy(overlay.clock_dict)
    for index, total_divisor in ((0,10),(1,7)):
        cfg=overlay.clock_dict[index]
        assert int(cfg['enable'])==1
        assert int(cfg['divisor0'])*int(cfg['divisor1'])==total_divisor, 'Unexpected HWH divider;refuse automatic repair'
    overlay.download()
    after = {hex(o): regs.read(o) for o in offsets}
    assert before == after, 'Overlay changed a CPU/DDR/PLL register'
    assert float(Clocks.cpu_mhz) == cpu_before
    assert abs(float(Clocks.fclk0_mhz) - 100) < .001
    assert abs(float(Clocks.fclk1_mhz) - 1000/7) < .001
    return overlay, {'io_pll_mhz': io_mhz, 'cpu_mhz_unchanged': cpu_before,
                     'original_hwh_clock_dict': original,
                     'applied_clock_dict': deepcopy(overlay.clock_dict),
                     'metadata_divisors_overridden': original != overlay.clock_dict,
                     'preserved_ps_registers': before,
                     'actual_fclk0_mhz': float(Clocks.fclk0_mhz),
                     'actual_fclk1_mhz': float(Clocks.fclk1_mhz)}
