// A held PS GPIO level crosses to the compute domain; one rise arms a packet.
module mlp_arm_sync(
 (* X_INTERFACE_INFO="xilinx.com:signal:clock:1.0 clk CLK" *) input wire clk,
 input wire src_in,output wire dest_out
);
 xpm_cdc_single #(.DEST_SYNC_FF(2),.INIT_SYNC_FF(1),.SIM_ASSERT_CHK(0),.SRC_INPUT_REG(0))
 sync_arm(.src_clk(1'b0),.src_in(src_in),.dest_clk(clk),.dest_out(dest_out));
endmodule
