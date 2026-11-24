set root [file normalize [file join [file dirname [info script]] ..]]
open_checkpoint $root/hardware/routed.dcp
report_utilization -name Final_Utilization
report_power -name Final_Power
report_timing_summary -delay_type min_max -name Final_Timing
