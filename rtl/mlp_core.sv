`timescale 1ns/1ps
// Fixed 512->128->32->2 PTQ graph. P output neurons, one activation stream.
module mlp_core_reuse #(parameter P=64, parameter DATA_DIR="hardware_data")(
 input wire clk,rst,start,
 input wire input_we,input wire [8:0] input_addr,input wire [7:0] input_data,
 input wire word_we,input wire [6:0] word_addr,input wire [31:0] word_data,
 output wire busy,output reg done,output reg prediction,
 output reg [31:0] cycles,
 input wire [7:0] feature_addr,output reg [7:0] feature_data,
 output wire trace_valid,output wire [7:0] trace_index,output wire [7:0] trace_data,
 output wire acc_valid,output wire [7:0] acc_index,output wire signed [31:0] acc_data
);
 localparam IDLE=0,INIT=1,RUN=2,WAIT_POST=3;
 reg [1:0] state=IDLE,layer;
 reg tile;
 reg [9:0] tick;
 reg [7:0] received;
 // Four byte banks accept a complete DMA word each cycle and replay one byte.
 (* ram_style="distributed" *) reg [7:0] inputs0[0:127],inputs1[0:127],inputs2[0:127],inputs3[0:127];
 (* ram_style="distributed" *) reg [7:0] features[0:161];
 reg signed [31:0] bias_rom[0:161],mult_rom[0:161];
 initial begin
  if(P!=64 && P!=128) $error("P must be 64 or 128");
  $readmemh({DATA_DIR,"/bias.mem"},bias_rom);
  $readmemh({DATA_DIR,"/mult.mem"},mult_rom);
 end
 assign busy=state!=IDLE;
 wire clear=state==INIT;
 wire [9:0] kcount=layer==0?10'd512:layer==1?10'd128:10'd32;
 wire [7:0] nactive=layer==0?P:layer==1?32:2;
 wire [7:0] base=layer==0?(tile?8'd64:8'd0):layer==1?8'd128:8'd160;
 always @(posedge clk) begin
  if(!busy) begin
   if(word_we) begin
    inputs0[word_addr]<=word_data[7:0];inputs1[word_addr]<=word_data[15:8];
    inputs2[word_addr]<=word_data[23:16];inputs3[word_addr]<=word_data[31:24];
   end else if(input_we) case(input_addr[1:0])
    0:inputs0[input_addr[8:2]]<=input_data;
    1:inputs1[input_addr[8:2]]<=input_data;
    2:inputs2[input_addr[8:2]]<=input_data;
    3:inputs3[input_addr[8:2]]<=input_data;
   endcase
  end
  feature_data<=features[feature_addr<162?feature_addr:0];
 end
 reg [7:0] a[0:P-1];
 reg [P-1:0] av,af,al;
 integer j;
 always @(posedge clk) begin
  if(rst || clear) begin av<=0;af<=0;al<=0;end
  else begin
   av[0]<=state==RUN && tick<kcount;
   af[0]<=tick==0;al[0]<=tick==kcount-1;
   if(state==RUN && tick<kcount) begin
    if(layer==0) case(tick[1:0])
     0:a[0]<=inputs0[tick[8:2]];
     1:a[0]<=inputs1[tick[8:2]];
     2:a[0]<=inputs2[tick[8:2]];
     3:a[0]<=inputs3[tick[8:2]];
    endcase
    else if(layer==1) a[0]<=features[tick[6:0]];
    else a[0]<=features[128+tick[4:0]];
   end
   for(j=1;j<P;j=j+1) begin
    av[j]<=av[j-1];af[j]<=af[j-1];al[j]<=al[j-1];
    if(av[j-1]) a[j]<=a[j-1];
   end
  end
 end
 wire [P-1:0] pe_done;
 wire signed [31:0] sums[0:P-1];
 genvar g,r;
 generate for(g=0;g<P/8;g=g+1) begin: groups
  localparam DEPTH=(128/P)*512;
  (* rom_style="block" *) reg [63:0] w1[0:DEPTH-1];
  initial $readmemh($sformatf("%s/w%0d_g%0d.mem",DATA_DIR,P,g),w1);
  wire read_en=state==RUN && tick>=g*8 && tick<g*8+kcount && g*8<nactive;
  wire [9:0] k=tick-g*8;
  reg [63:0] q1;
  always @(posedge clk) if(read_en && layer==0) q1<=w1[(tile?512:0)+k];
  wire [63:0] q2,q3;
  if(g<4) begin: fc2
   (* rom_style="distributed" *) reg [63:0] mem[0:127];reg [63:0] q;
   initial $readmemh($sformatf("%s/fc2_g%0d.mem",DATA_DIR,g),mem);
   always @(posedge clk) if(read_en && layer==1) q<=mem[k[6:0]];
   assign q2=q;
  end else assign q2=0;
  if(g==0) begin: fc3
   reg [63:0] mem[0:31];reg [63:0] q;
   initial $readmemh({DATA_DIR,"/fc3_g0.mem"},mem);
   always @(posedge clk) if(read_en && layer==2) q<=mem[k[4:0]];
   assign q3=q;
  end else assign q3=0;
  wire [63:0] qw=layer==0?q1:layer==1?q2:q3;
  for(r=0;r<8;r=r+1) begin: lanes
   wire signed [7:0] weight;
   if(r==0) assign weight=qw[r*8+:8];
   else begin: skew
    reg [7:0] d[0:r-1];integer s;
    always @(posedge clk) begin
     d[0]<=qw[r*8+:8];for(s=1;s<r;s=s+1) d[s]<=d[s-1];
    end
    assign weight=d[r-1];
   end
   mlp_pe pe(.clk(clk),.rst(rst),.clear(clear),.x(a[g*8+r]),.w(weight),
    .valid(av[g*8+r] && g*8+r<nactive),.first(af[g*8+r]),.last(al[g*8+r]),
    .sum(sums[g*8+r]),.done(pe_done[g*8+r]));
  end
 end endgenerate
 // A registered result selection precedes the parameter ROM read/post pipeline.
 reg collect_valid;reg signed [31:0] collect_sum,collect_bias,collect_mult;
 reg [7:0] collect_index,collect_lane;reg collect_relu,collect_shift39;
 wire post_valid;wire [7:0] post_index;wire signed [7:0] post_data;
 always @(posedge clk) begin
  if(rst || clear) begin collect_valid<=0;collect_lane<=0;end
  else begin
   collect_valid<=0;
   // Completions are consecutive: after lane zero, consume one held sum/cycle.
   // Avoid a 128:1 completion mux feeding parameter-ROM clock enables.
   if(collect_lane<nactive && (collect_lane!=0 || pe_done[0])) begin
    collect_valid<=1;collect_sum<=sums[collect_lane];collect_index<=base+collect_lane;
    collect_bias<=bias_rom[base+collect_lane];collect_mult<=mult_rom[base+collect_lane];
    collect_relu<=layer!=2;collect_shift39<=layer==2;collect_lane<=collect_lane+1'b1;
   end
  end
 end
 mlp_requant post(.clk(clk),.rst(rst),.valid(collect_valid),.sum(collect_sum),
  .bias(collect_bias),.multiplier(collect_mult),.index(collect_index),
  .relu(collect_relu),.shift39(collect_shift39),.out_valid(post_valid),
  .out_index(post_index),.out_data(post_data),.debug_valid(acc_valid),.debug_acc(acc_data),.debug_index(acc_index));
 assign trace_valid=post_valid;assign trace_index=post_index;assign trace_data=post_data;
 always @(posedge clk) begin
  if(rst) begin state<=IDLE;layer<=0;tile<=0;tick<=0;received<=0;done<=0;cycles<=0;prediction<=0;end
  else begin
   done<=0;
   if(busy) cycles<=cycles+1'b1;
   if(post_valid) begin features[post_index]<=post_data;received<=received+1'b1;end
   case(state)
    IDLE:if(start) begin state<=INIT;layer<=0;tile<=0;cycles<=0;received<=0;end
    INIT:begin state<=RUN;tick<=0;end
    RUN:begin
     tick<=tick+1'b1;
     if(pe_done[nactive-1]) state<=WAIT_POST;
    end
    WAIT_POST:if(post_valid && post_index==base+nactive-1) begin
     if(layer==0 && P==64 && !tile) begin tile<=1;state<=INIT;end
     else if(layer!=2) begin layer<=layer+1'b1;tile<=0;state<=INIT;end
     else begin
      prediction<=($signed(features[160]) >= post_data)?1'b0:1'b1;
      done<=1;state<=IDLE;
     end
    end
   endcase
  end
 end
endmodule
