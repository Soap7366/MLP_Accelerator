`timescale 1ns/1ps
module mlp_pe #(parameter ACC_W=32,SKIP_ZERO=0,RESET_DATAPATH=1)(
 input wire clk,rst,clear,input wire signed [7:0] x,w,
 input wire valid,first,last,output wire signed [31:0] sum,output reg done
);
 wire reset=rst|clear;
 // A zero first product must still initialize P. Completion tokens never skip.
 wire consume=valid && (!SKIP_ZERO || first || (x!=0 && w!=0));
 wire [7:0] abs_x=x[7]?(~x+8'd1):x;
 wire [7:0] abs_w={1'b0,w[6:0]};
 reg negative;
 reg xv,xf,pv;reg tv,tl,lv,ll;
 wire signed [47:0] result;
 assign sum=$signed({~result[ACC_W-1],result[ACC_W-2:0]});
 always @(posedge clk) begin
  if(reset) begin negative<=0;xv<=0;xf<=0;pv<=0;tv<=0;tl<=0;lv<=0;ll<=0;done<=0;end
  else begin
   xv<=consume;xf<=first;pv<=xv;negative<=x[7]^w[7];
   tv<=valid;tl<=last;lv<=tv;ll<=tl;done<=lv && ll;
  end
 end
 DSP48E1 #(
  .AREG(1),.BREG(1),.MREG(1),.PREG(1),.ACASCREG(1),.BCASCREG(1),
  .CREG(0),.DREG(1),.ADREG(1),.OPMODEREG(1),.ALUMODEREG(1),
  .CARRYINREG(0),.CARRYINSELREG(0),.INMODEREG(0),
  .USE_DPORT("FALSE"),.USE_MULT("MULTIPLY"),.USE_SIMD("ONE48"),.USE_PATTERN_DETECT("NO_PATDET")
 ) mac(
  .CLK(clk),.A({22'b0,abs_x}),.B({10'b0,abs_w}),.C(48'd1 << (ACC_W-1)),.D(25'b0),
  .ACIN(30'b0),.BCIN(18'b0),.PCIN(48'b0),.CARRYCASCIN(1'b0),.MULTSIGNIN(1'b0),
  .ALUMODE(negative?4'b0011:4'b0000),.INMODE(5'b0),.CARRYIN(1'b0),.CARRYINSEL(3'b0),
  .OPMODE(xf?7'b0110101:7'b0100101),
  .CEA1(1'b0),.CEA2(consume),.CEB1(1'b0),.CEB2(consume),.CEM(xv),.CEP(pv),.CECTRL(xv),
  .CEC(1'b0),.CED(1'b0),.CEAD(1'b0),.CEALUMODE(xv),.CEINMODE(1'b0),.CECARRYIN(1'b0),
  .RSTA(1'b0),.RSTB(1'b0),.RSTM(RESET_DATAPATH?reset:1'b0),.RSTP(RESET_DATAPATH?reset:1'b0),.RSTCTRL(reset),
  .RSTC(1'b0),.RSTD(1'b0),.RSTALUMODE(reset),.RSTINMODE(1'b0),.RSTALLCARRYIN(1'b0),
  .P(result),.ACOUT(),.BCOUT(),.PCOUT(),.CARRYCASCOUT(),.CARRYOUT(),
  .MULTSIGNOUT(),.OVERFLOW(),.UNDERFLOW(),.PATTERNDETECT(),.PATTERNBDETECT()
 );
endmodule
