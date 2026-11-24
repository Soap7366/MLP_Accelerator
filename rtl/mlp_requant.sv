`timescale 1ns/1ps
// II=1. Parameters and tags travel with every result; no shared mutable config.
module mlp_requant(
 input wire clk,rst,valid,
 input wire signed [31:0] sum,bias,multiplier,
 input wire [7:0] index,
 input wire relu,shift39,
 output reg out_valid,
 output reg [7:0] out_index,
 output reg signed [7:0] out_data,
 output wire debug_valid,
 output wire signed [31:0] debug_acc,
 output wire [7:0] debug_index
);
 reg [6:0] v;
 reg [7:0] tags[0:6];
 reg [6:0] rel,sh;
 reg signed [31:0] acc,mult;
 reg signed [23:0] acc_pipe;
 reg signed [16:0] lo_pipe,hi_pipe;
 // Frozen-model worst-case |acc| < 2^23 (checked at export).
 // Two 24x17 products fit individual DSP48 multipliers. A separate stage sums.
 (* use_dsp="yes" *) reg signed [40:0] product_lo,product_hi;
 reg signed [63:0] product;
 reg [63:0] magnitude;
 reg neg2,neg3;
 reg [25:0] rounded;
 reg signed [7:0] quant;
 assign debug_valid=v[0];assign debug_acc=acc;assign debug_index=tags[0];
 integer i;
 always @(posedge clk) begin
  // Validity is architectural state. Other pipeline bits are observed only
  // with valid and are overwritten before the first post-reset valid output.
  if(rst) begin v<=0;out_valid<=0;end
  else begin v<={v[5:0],valid};out_valid<=v[6];end
   rel<={rel[5:0],relu};sh<={sh[5:0],shift39};
   tags[0]<=index;for(i=1;i<7;i=i+1) tags[i]<=tags[i-1];
   if(valid) begin acc<=sum+bias;mult<=multiplier;end
   if(v[0]) begin
    acc_pipe<=acc[23:0];lo_pipe<={1'b0,mult[15:0]};hi_pipe<={1'b0,mult[31:16]};
   end
   if(v[1]) begin product_lo<=acc_pipe*lo_pipe;product_hi<=acc_pipe*hi_pipe;end
   if(v[2]) product<={{23{product_lo[40]}},product_lo}+({{23{product_hi[40]}},product_hi}<<16);
   if(v[3]) begin magnitude<=product[63]?(~product+64'd1):product;neg2<=product[63];end
   if(v[4]) begin
    rounded<=sh[4]?({1'b0,magnitude[63:39]}+magnitude[38]):({2'b0,magnitude[63:40]}+magnitude[39]);
    neg3<=neg2;
   end
   if(v[5]) begin
    if(rel[5] && neg3) quant<=0;
    else if(rounded>127) quant<=neg3?-8'sd127:8'sd127;
    else quant<=neg3?-$signed(rounded[7:0]):$signed(rounded[7:0]);
   end
   if(v[6]) begin out_data<=quant;out_index<=tags[6];end
 end
endmodule
