`timescale 1ns/1ps
// Layer-local output-stationary chains. Quantized results feed the next chain
// immediately; weights and control are independent for all three layers.
module mlp_stream_layer #(parameter L=0, NI=512, NO=128, BASE=0, DATA_DIR="hardware_data")(
 input wire clk,rst,clear,valid,input wire [7:0] x,
 output wire [7:0] pe_x[0:NO-1],pe_w[0:NO-1],
 output wire [NO-1:0] pe_valid,pe_first,pe_last
);
 localparam STRIDE=L==0?1:8;
 reg [$clog2(NI)-1:0] input_k;
 reg [7:0] a[0:NO-1];
 reg [NO-1:0] av,af,al;

 integer i;
 always @(posedge clk) begin
  if(rst) begin input_k<=0;av<=0;end
  else begin
   // Full power-of-two vectors wrap the address; reset handles aborts.
   av[0]<=valid;
   if(valid) begin a[0]<=x;input_k<=input_k+1'b1;end
   for(i=1;i<NO;i=i+1) begin
    av[i]<=av[i-1];
   end
  end
  // Data/control bits without valid are unobservable. Keeping their shifts
  // free of reset and per-stage enables maps the delays into compact SRLs.
  af[0]<=input_k==0;al[0]<=input_k==NI-1;
  for(i=1;i<NO;i=i+1) begin
   a[i]<=a[i-1];af[i]<=af[i-1];al[i]<=al[i-1];
  end
 end
 genvar g,r;
 generate for(g=0;g<(NO+7)/8;g=g+1) begin: groups
  localparam HOPS=L==0?g/4:g*STRIDE;
  wire en;wire [9:0] k;
  if(HOPS==0) begin assign en=valid;assign k=input_k;end
  else begin: local_address
   // Advance only on a consumed activation. NI is a power of two, so a full
   // vector wraps to zero without a high-fanout start reset. Unlike an ungated
   // address SRL, this preserves BRAM internal read enables after synthesis.
   reg [$clog2(NI)-1:0] read_index;
   assign en=av[HOPS-1];assign k=read_index;
   always @(posedge clk) begin
    if(rst) read_index<=0;
    else if(en) read_index<=read_index+1'b1;
   end
  end
  (* rom_style = "block" *) reg [63:0] weights[0:NI-1];
  initial begin
   if(L==0) $readmemh($sformatf("%s/w128_g%0d.mem",DATA_DIR,g),weights);
   else if(L==1) $readmemh($sformatf("%s/fc2_g%0d.mem",DATA_DIR,g),weights);
   else $readmemh({DATA_DIR,"/fc3_g0.mem"},weights);
  end
  reg [63:0] q;
  always @(posedge clk) if(en) q<=weights[k];
  for(r=0;r<8 && g*8+r<NO;r=r+1) begin: lanes
   // All eight rows consume the same activation and ROM word together.
   // Completed rows remain in their DSPs while the collector serializes them.
   assign pe_x[g*8+r]=a[HOPS];assign pe_w[g*8+r]=q[r*8+:8];
   assign pe_valid[g*8+r]=av[HOPS];
   assign pe_first[g*8+r]=af[HOPS];assign pe_last[g*8+r]=al[HOPS];
  end
 end endgenerate
endmodule

module mlp_core #(parameter P=64, DATA_DIR="hardware_data")(
 input wire clk,rst,start,
 input wire input_we,input wire [8:0] input_addr,input wire [7:0] input_data,
 input wire word_we,input wire [6:0] word_addr,input wire [31:0] word_data,
 output wire busy,done,prediction,output wire [31:0] cycles,
 input wire [7:0] feature_addr,output wire [7:0] feature_data,
 output wire trace_valid,output wire [7:0] trace_index,trace_data,
 output wire acc_valid,output wire [7:0] acc_index,output wire signed [31:0] acc_data
);
 generate if(P==64) begin: reuse
  mlp_core_reuse #(.P(P),.DATA_DIR(DATA_DIR)) impl(.*);
 end else begin: cascade
  mlp_cascade #(.DATA_DIR(DATA_DIR)) impl(.*);
 end endgenerate
endmodule

module mlp_cascade #(parameter DATA_DIR="hardware_data")(
 input wire clk,rst,start,
 input wire input_we,input wire [8:0] input_addr,input wire [7:0] input_data,
 input wire word_we,input wire [6:0] word_addr,input wire [31:0] word_data,
 output reg busy,done,prediction,output reg [31:0] cycles,
 input wire [7:0] feature_addr,output reg [7:0] feature_data,
 output wire trace_valid,output wire [7:0] trace_index,trace_data,
 output wire acc_valid,output wire [7:0] acc_index,output wire signed [31:0] acc_data
);
 (* ram_style="distributed" *) reg [7:0] inputs0[0:127],inputs1[0:127],inputs2[0:127],inputs3[0:127];
 (* ram_style="distributed" *) reg [7:0] features[0:161];
 reg [9:0] tick;reg [7:0] x;
 wire clear=start && !busy;
 // Keep transport, input/feature RAM and the cycle counter on the parent clock.
 // Start opens the compute clock before the first busy cycle. Two drain cycles
 // clear output-valid tokens after completion before the clock is stopped.
 reg [1:0] clock_drain;
 always @(posedge clk) begin
  if(rst) clock_drain<=0;
  else clock_drain<={clock_drain[0],busy};
 end
 wire compute_clk;
 reg compute_enable=1'b1;
 reg compute_reset=1'b1;
 always @(posedge clk) compute_reset<=rst;
 wire post_trace_valid,post_acc_valid;
 assign trace_valid=post_trace_valid && !compute_reset && !rst;
 assign acc_valid=post_acc_valid && !compute_reset && !rst;
 always @(posedge clk) begin
  if(rst) compute_enable<=1'b1;
  else compute_enable<=clear || busy || (|clock_drain);
 end
 BUFGCE compute_clock(.I(clk),.CE(compute_enable),.O(compute_clk));
 wire [2:0] v;wire [7:0] oi[0:2],od[0:2];
 wire [7:0] px0[0:127],pw0[0:127],px1[0:31],pw1[0:31],px2[0:1],pw2[0:1];
 wire [127:0] pv0,pf0,pl0,pd0;wire [31:0] pv1,pf1,pl1,pd1;wire [1:0] pv2,pf2,pl2,pd2;
 wire signed [31:0] sums0[0:127],sums1[0:31],sums2[0:1];
 always @* case(tick[1:0])
  0:x=inputs0[tick[8:2]];1:x=inputs1[tick[8:2]];2:x=inputs2[tick[8:2]];3:x=inputs3[tick[8:2]];
 endcase
 mlp_stream_layer #(.L(0),.NI(512),.NO(128),.BASE(0),.DATA_DIR(DATA_DIR)) fc1
  (.clk(compute_clk),.rst(compute_reset),.clear(clear),.valid(busy && tick<512),.x(x),
   .pe_x(px0),.pe_w(pw0),.pe_valid(pv0),.pe_first(pf0),.pe_last(pl0));
 mlp_stream_layer #(.L(1),.NI(128),.NO(32),.BASE(128),.DATA_DIR(DATA_DIR)) fc2
  (.clk(compute_clk),.rst(compute_reset),.clear(clear),.valid(v[0]),.x(od[0]),
   .pe_x(px1),.pe_w(pw1),.pe_valid(pv1),.pe_first(pf1),.pe_last(pl1));
 mlp_stream_layer #(.L(2),.NI(32),.NO(2),.BASE(160),.DATA_DIR(DATA_DIR)) fc3
  (.clk(compute_clk),.rst(compute_reset),.clear(clear),.valid(v[1]),.x(od[1]),
   .pe_x(px2),.pe_w(pw2),.pe_valid(pv2),.pe_first(pf2),.pe_last(pl2));
 genvar p;
 generate for(p=0;p<128;p=p+1) begin: shared_pes
  wire b,c;wire [7:0] xb,wb,xc,wc;wire fb,lb,fc,lc;
  if(p<32) begin
   assign b=pv1[p];assign xb=px1[p];assign wb=pw1[p];assign fb=pf1[p];assign lb=pl1[p];
  end else begin
   assign b=0;assign xb=0;assign wb=0;assign fb=0;assign lb=0;
  end
  if(p<2) begin
   assign c=pv2[p];assign xc=px2[p];assign wc=pw2[p];assign fc=pf2[p];assign lc=pl2[p];
  end else begin
   assign c=0;assign xc=0;assign wc=0;assign fc=0;assign lc=0;
  end
  wire [1:0] owner=c?2:b?1:0;
  reg [1:0] owner0,owner1,owner2;
  always @(posedge compute_clk) begin
   if(compute_reset) begin owner0<=0;owner1<=0;owner2<=0;end
   else begin owner0<=owner;owner1<=owner0;owner2<=owner1;end
  end
  wire done_pe;wire signed [31:0] sum_pe;
  // Frozen model bound, over every valid input: |partial sum| <= 4231040.
  // Signed 24 bits suffice; external debug and bias interfaces stay INT32.
  mlp_pe #(.ACC_W(24),.RESET_DATAPATH(0)) pe(.clk(compute_clk),.rst(compute_reset),.clear(1'b0),
   .x(c?xc:b?xb:px0[p]),.w(c?wc:b?wb:pw0[p]),
   .valid(pv0[p] | b | c),.first(c?fc:b?fb:pf0[p]),.last(c?lc:b?lb:pl0[p]),
   .sum(sum_pe),.done(done_pe));
  assign sums0[p]=sum_pe;assign pd0[p]=done_pe && owner2==0;
  if(p<32) begin assign sums1[p]=sum_pe;assign pd1[p]=done_pe && owner2==1;end
  if(p<2) begin assign sums2[p]=sum_pe;assign pd2[p]=done_pe && owner2==2;end
  `ifndef SYNTHESIS
  always @(posedge compute_clk) if(!compute_reset && ((pv0[p] && b) || (pv0[p] && c) || (b && c)))
   $fatal(1,"Shared PE input collision at lane %0d",p);
  `endif
 end endgenerate
 // All three completion windows use one collector over the physical array.
 reg collecting;reg [7:0] next_lane;reg [1:0] active_layer;
 wire first_done=pd0[0] | pd1[0] | pd2[0];
 wire [1:0] event_layer=pd0[0]?0:pd1[0]?1:2;
 wire [1:0] layer=collecting?active_layer:event_layer;
 wire [7:0] lane=next_lane;
 wire [7:0] base=layer==0?0:layer==1?128:160;
 wire collect=collecting | first_done;
 wire last_lane=lane==(layer==0?127:layer==1?31:1);
 reg [31:0] bias[0:161],mult[0:161];
 initial begin $readmemh({DATA_DIR,"/bias.mem"},bias);$readmemh({DATA_DIR,"/mult.mem"},mult);end
 reg signed [23:0] group_sum[0:15];
 reg select_valid,cv;reg [7:0] selected_lane,selected_index,ci;
 reg signed [31:0] selected_bias,selected_mult,cs,cb,cm;
 genvar cg;
 generate for(cg=0;cg<16;cg=cg+1) begin: result_groups
  always @(posedge compute_clk)
   if(collect && lane[7:3]==cg) group_sum[cg]<=sums0[cg*8+lane[2:0]];
 end endgenerate
 always @(posedge compute_clk) begin
  if(compute_reset || clear) begin collecting<=0;next_lane<=0;active_layer<=0;select_valid<=0;cv<=0;end
  else begin
   select_valid<=collect;cv<=select_valid;
   if(collect) begin
    collecting<=!last_lane;next_lane<=last_lane?8'd0:lane+1'b1;active_layer<=layer;
    selected_lane<=lane;selected_index<=base+lane;
    selected_bias<=bias[base+lane];selected_mult<=mult[base+lane];
   end
   if(select_valid) begin
    cs<=group_sum[selected_lane[7:3]];cb<=selected_bias;cm<=selected_mult;ci<=selected_index;
   end
  end
 end
 `ifndef SYNTHESIS
 always @(posedge compute_clk) if(!compute_reset && collecting && first_done)
  $fatal(1,"Shared collector completion collision");
 `endif
 mlp_requant shared_post(.clk(compute_clk),.rst(compute_reset),.valid(cv),
  .sum(cs),.bias(cb),.multiplier(cm),.index(ci),.relu(ci<160),.shift39(ci>=160),
  .out_valid(post_trace_valid),.out_index(trace_index),.out_data(trace_data),
  .debug_valid(post_acc_valid),.debug_index(acc_index),.debug_acc(acc_data));
 assign v[0]=trace_valid && trace_index<128;
 assign v[1]=trace_valid && trace_index>=128 && trace_index<160;
 assign v[2]=trace_valid && trace_index>=160;
 genvar route_layer;
 generate for(route_layer=0;route_layer<3;route_layer=route_layer+1) begin
  assign oi[route_layer]=trace_index;assign od[route_layer]=trace_data;
 end endgenerate
 always @(posedge clk) begin
  if(!busy) begin
   if(word_we) begin inputs0[word_addr]<=word_data[7:0];inputs1[word_addr]<=word_data[15:8];inputs2[word_addr]<=word_data[23:16];inputs3[word_addr]<=word_data[31:24];end
   else if(input_we) case(input_addr[1:0])
    0:inputs0[input_addr[8:2]]<=input_data;1:inputs1[input_addr[8:2]]<=input_data;2:inputs2[input_addr[8:2]]<=input_data;3:inputs3[input_addr[8:2]]<=input_data;
   endcase
  end
  feature_data<=features[feature_addr<162?feature_addr:0];
  if(trace_valid) features[trace_index]<=trace_data;
  if(rst) begin busy<=0;done<=0;prediction<=0;cycles<=0;tick<=0;end
  else begin
   done<=0;
   if(clear) begin busy<=1;cycles<=0;tick<=0;end
   if(busy) begin cycles<=cycles+1'b1;if(tick<512) tick<=tick+1'b1;end
   if(v[2] && oi[2]==161) begin busy<=0;done<=1;prediction<=($signed(features[160])>=$signed(od[2]))?0:1;end
  end
 end
endmodule
