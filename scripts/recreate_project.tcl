# Recreate only the qualified R108 block design with portable ROM paths.
set root [file normalize [file join [file dirname [info script]] ..]]
cd $root
if {[version -short] ne "2023.2"} {error "Use Vivado 2023.2 for the qualified IP versions"}
set project_dir $root/build/vivado
if {[file exists $project_dir/MLP_Accelerator.xpr]} {error "Project exists;open it instead of overwriting it"}
create_project MLP_Accelerator $project_dir -part xc7z020clg400-1
set_property target_language Verilog [current_project]
set_property XPM_LIBRARIES {XPM_CDC XPM_MEMORY XPM_FIFO} [current_project]
add_files -norecurse [glob $root/rtl/*.v $root/rtl/*.sv]
set_property file_type SystemVerilog [get_files *.sv]
set ::mlp_rom_dir [file normalize $root/hardware_data]
source $root/hardware/mlp_system.tcl
if {[get_property CONFIG.P [get_bd_cells npu]]!=128} {error "R108 requires128 PEs"}
if {abs([get_property CONFIG.PCW_FPGA1_PERIPHERAL_FREQMHZ [get_bd_cells ps]]-142.857143)>.001} {error "R108 clock mismatch"}
if {abs([get_property CONFIG.PCW_ACT_APU_PERIPHERAL_FREQMHZ [get_bd_cells ps]]-325)>.001} {error "R108 CPU must be 325 MHz"}
validate_bd_design
save_bd_design
generate_target all [get_files mlp_system.bd]
set wrapper [make_wrapper -files [get_files mlp_system.bd] -top]
add_files -norecurse $wrapper
set_property top mlp_system_wrapper [current_fileset]
update_compile_order -fileset sources_1
set_property STEPS.POWER_OPT_DESIGN.IS_ENABLED false [get_runs impl_1]
set_property STEPS.WRITE_BITSTREAM.ARGS.BIN_FILE true [get_runs impl_1]
puts "R108_PROJECT_READY $project_dir/MLP_Accelerator.xpr"
close_project
