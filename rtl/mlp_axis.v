`timescale 1ns/1ps
// Verilog top for Vivado module-reference integration; core uses SystemVerilog.
// AXIS packet protocol documented in docs/protocol.md. All transfers 32 bits.
module mlp_axis #(parameter P=64,parameter DATA_DIR="hardware_data")(
 (* X_INTERFACE_INFO="xilinx.com:signal:clock:1.0 aclk CLK", X_INTERFACE_PARAMETER="ASSOCIATED_BUSIF S_AXIS:M_AXIS, ASSOCIATED_RESET aresetn" *) input wire aclk,
 (* X_INTERFACE_INFO="xilinx.com:signal:reset:1.0 aresetn RST", X_INTERFACE_PARAMETER="POLARITY ACTIVE_LOW" *) input wire aresetn,
 input wire arm,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 S_AXIS TDATA" *) input wire [31:0] s_axis_tdata,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 S_AXIS TKEEP" *) input wire [3:0] s_axis_tkeep,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 S_AXIS TVALID" *) input wire s_axis_tvalid,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 S_AXIS TREADY" *) output wire s_axis_tready,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 S_AXIS TLAST" *) input wire s_axis_tlast,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 M_AXIS TDATA" *) output reg [31:0] m_axis_tdata,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 M_AXIS TKEEP" *) output wire [3:0] m_axis_tkeep,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 M_AXIS TVALID" *) output wire m_axis_tvalid,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 M_AXIS TREADY" *) input wire m_axis_tready,
 (* X_INTERFACE_INFO="xilinx.com:interface:axis:1.0 M_AXIS TLAST" *) output wire m_axis_tlast,
 output wire busy,output reg [31:0] last_cycles
);
 localparam IDLE=0,CMD=1,INPUT_WORD=2,UNPACK=3,LAUNCH=4,COMPUTE=5,
 HEADER=6,SEND=7,READ_REQ=8,READ_WAIT=9,READ_CAPTURE=10,DRAIN=11;
 reg [3:0] state;
 reg arm_d,mode,error;
 reg [31:0] held;
 reg [8:0] input_index;
 reg [1:0] byte_index,header_index;
 reg [7:0] payload_index;
 reg [31:0] packed_word;
 reg tx_last,prediction;
 wire core_busy,core_done,core_pred;wire [31:0] core_cycles;
 wire [7:0] feature_data;
 wire [7:0] feature_addr=mode?payload_index:8'd160+payload_index;
 wire [7:0] payload_count=error?0:(mode?162:2);
 wire input_accept=state==INPUT_WORD && s_axis_tvalid && s_axis_tkeep==15 && s_axis_tlast==(input_index==508);
 assign busy=state!=IDLE;
 assign s_axis_tready=state==CMD || state==INPUT_WORD || state==DRAIN;
 assign m_axis_tvalid=state==SEND;
 assign m_axis_tkeep=4'hf;
 assign m_axis_tlast=tx_last;
 mlp_core #(.P(P),.DATA_DIR(DATA_DIR)) core(.clk(aclk),.rst(!aresetn),.start(state==LAUNCH),
  .input_we(1'b0),.input_addr(9'd0),.input_data(8'd0),
  .word_we(input_accept),.word_addr(input_index[8:2]),.word_data(s_axis_tdata),
  .busy(core_busy),.done(core_done),.prediction(core_pred),.cycles(core_cycles),
  .feature_addr(feature_addr),.feature_data(feature_data),.trace_valid(),.trace_index(),.trace_data(),
  .acc_valid(),.acc_index(),.acc_data());
 always @(posedge aclk) begin
  if(!aresetn) begin
   state<=IDLE;arm_d<=0;mode<=0;error<=0;held<=0;input_index<=0;byte_index<=0;
   header_index<=0;payload_index<=0;packed_word<=0;tx_last<=0;prediction<=0;last_cycles<=0;m_axis_tdata<=0;
  end else begin
   arm_d<=arm;
   case(state)
    IDLE:if(arm && !arm_d) begin state<=CMD;error<=0;header_index<=0;tx_last<=0;end
    CMD:if(s_axis_tvalid) begin
     mode<=s_axis_tdata[0];input_index<=0;
     if(s_axis_tkeep!=15 || (s_axis_tdata & 32'hfffffffe)!=32'h4d4c5000) begin
      error<=1;state<=s_axis_tlast?HEADER:DRAIN;
     end else if(s_axis_tlast) begin
      // One-word feature command: read the retained PL result, never LAUNCH.
      // A nonzero last_cycles marks a completed inference since reset.
      error<=!s_axis_tdata[0] || last_cycles==0;state<=HEADER;
     end else state<=INPUT_WORD;
    end
    INPUT_WORD:if(s_axis_tvalid) begin
     if(s_axis_tkeep!=15 || s_axis_tlast!=(input_index==508)) begin error<=1;state<=s_axis_tlast?HEADER:DRAIN;end
     else begin input_index<=input_index+9'd4;if(input_index==508) state<=LAUNCH;end
    end
    UNPACK:begin
     input_index<=input_index+1'b1;
     if(byte_index==3) state<=input_index==511?LAUNCH:INPUT_WORD;
     else byte_index<=byte_index+1'b1;
    end
    LAUNCH:state<=COMPUTE;
    COMPUTE:if(core_done) begin last_cycles<=core_cycles;prediction<=core_pred;state<=HEADER;end
    DRAIN:if(s_axis_tvalid && s_axis_tlast) state<=HEADER;
    HEADER:begin
     case(header_index)
      0:m_axis_tdata<=32'h4d4c0000 | {29'd0,error,mode,prediction};
      1:m_axis_tdata<=error?0:last_cycles;
      2:m_axis_tdata<={24'd0,payload_count};
     endcase
     tx_last<=error && header_index==2;state<=SEND;
    end
    SEND:if(m_axis_tready) begin
     if(tx_last) state<=IDLE;
     else if(header_index<2) begin header_index<=header_index+1'b1;state<=HEADER;end
     else begin
      if(header_index==2) begin payload_index<=0;header_index<=3;end
      byte_index<=0;packed_word<=0;state<=READ_REQ;
     end
    end
    READ_REQ:state<=READ_WAIT;
    READ_WAIT:state<=READ_CAPTURE;
    READ_CAPTURE:begin
     packed_word[byte_index*8+:8]<=feature_data;
     payload_index<=payload_index+1'b1;
     if(byte_index==3 || payload_index==payload_count-1) begin
      case(byte_index)
       0:m_axis_tdata<={24'd0,feature_data};
       1:m_axis_tdata<={16'd0,feature_data,packed_word[7:0]};
       2:m_axis_tdata<={8'd0,feature_data,packed_word[15:0]};
       3:m_axis_tdata<={feature_data,packed_word[23:0]};
      endcase
      tx_last<=payload_index==payload_count-1;state<=SEND;
     end else begin byte_index<=byte_index+1'b1;state<=READ_REQ;end
    end
   endcase
  end
 end
endmodule
