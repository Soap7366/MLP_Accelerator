# MLP_Accelerator — 最终 PL 特征只读版

PYNQ-Z2 / Zynq-7020，512→128→32→2 INT8 PTQ 网络。128 个共享 MAC PE，分组广播流水计算，2 个 DSP 用于共享重新量化。CPU 325 MHz，PL 142.857 MHz，DMA 100 MHz。

本仓库只发布这一最终版本。模式 2 直接读取 **PL 中上一次推理留下的 162 个特征**，不发送输入向量、不重新推理，也不是读取 PS 缓存。原版推理协议和测速路径保留。

## 下载运行

下载 Release ZIP，解压到板卡任意目录，进入 `MLP_Accelerator_Final` 后执行：

```sh
sh ps/run.sh
```

使用现有已验证的 PYNQ Linux 环境：内核 `6.6.10-xilinx-v2024.1-g3c0eca68c652`，PYNQ、NumPy、gcc、sudo。程序校验文件、编译三个 C 库、加载 bit/HWH、设置 CPU 325 MHz；正常退出时恢复原 CPU 频率。附带的 CPU 驱动只适配上述内核，源代码位于 `ps/cpufreq`。这不是裸板启动镜像，需要已有 PYNQ SD 卡系统。

菜单 `1`：42 个样本分类，10 组预热 + 100 组计时，每组 420 次串行推理。菜单 `2`：直接读取最后一次推理结果；模式 1 结束后对应样本 41。可以重复读取。程序刚启动时没有有效结果，会提示先运行模式 1。菜单 `0`：退出。自测：`sh ps/run.sh --self-test`。

## 最终结果

| 指标 | 结果 |
|---|---:|
| 分类准确率 | 42/42（100%，提供的校准样本，非独立测试集） |
| PS→PL→PS 平均 / 中位数 | 9.948 / 9.985 μs（菜单实测） |
| PL | 713 周期 / 4.991 μs |
| Vivado 总芯片功耗估计 | 1.512 W（包含 PS，非整板电源实测） |
| LUT / LUT as Memory | 4568 / 402 |
| FF / BRAM / DSP | 4893 / 24.5 / 130 |
| Setup / Hold slack | +0.588 / +0.020 ns |

PS 时间不含外层 Python、输入打包、外层缓冲区同步和串口打印；每行 single_rt 与底部 PRIMARY BENCHMARK 的分组平均口径不同。模式 2 返回的 713 周期属于之前的推理，不能当作本次读取时间。历史 R108 的 1.503 W 不适用于此 bitstream。

## 演示视频

1. 电脑 PowerShell：`python -m serial.tools.miniterm COM11 115200 --eol CR --raw`（端口按实际修改）。进入板卡终端。
2. 进入解压目录并运行 `sh ps/run.sh`，展示 CPU/PL 频率。
3. 选 `1`，展示 42/42、PRIMARY BENCHMARK 平均时间与 PL 4.991 μs。
4. 选 `2`，展示 `NO INFERENCE`、四字节只读命令、最后样本 41、162/162 一致；再选 `2` 重复读取。两模式之间不要重启程序。
5. 在电脑 Vivado 2023.2 空白窗口执行 `source <解压目录>/scripts/open_reports.tcl`。Power 报告 Summary 展示 1.512 W；Utilization 报告左侧 Summary 展示资源表格和图形。
6. 回串口选 `0`，恢复 CPU 频率；Ctrl+] 退出串口。

## 工程与复现

- `rtl/`、`hardware_data/`：最终 RTL 与符号/幅值编码权重 ROM。
- `model/`：原始 FP32、PTQ 脚本、量化模型、输入与标签；`data/` 为参考仿真数据。
- `ps/`：可移植启动入口、Python/C 源码、CPU 驱动、匹配的 bit/HWH。
- `hardware/routed.dcp`、`reports/`：最终布局布线检查点及报告。
- `sim/tb_readonly.sv`：42 样本、84 次原请求、171 次只读请求，检查不启动核心、不改写特征、复位、背压及结果一致性。
- `evidence/`：RTL 仿真、板卡自测与实际菜单日志；日志中的历史本机路径仅作记录。
- 在仓库根目录执行 `python scripts/verify_release.py` 校验完整性。
- Vivado 2023.2 执行 `source <仓库>/scripts/recreate_project.tcl` 可重建工程；随后打开 `build/vivado/MLP_Accelerator.xpr`，运行 Implementation / Generate Bitstream。部署直接使用随包的已验证 bit/HWH。

为减少资源，复用已有特征 RAM、回传状态机和周期寄存器。只读命令直接进入结果回传流程。固定模型计算 RTL 和权重未变；相比旧 R108，LUT 减少 10，FF/BRAM/DSP 不变，功耗估计增加 0.009 W。详细协议见 `docs/protocol.md`。

发布验证范围：最终硬件和原菜单程序已上板通过。本包将启动路径改为相对路径并增加自动编译；发布时 COM11 不可用，打包后的独立启动入口未重新上板测试。文件哈希已检查。

