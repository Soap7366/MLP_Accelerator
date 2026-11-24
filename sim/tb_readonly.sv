`timescale 1ns/1ps
module tb_axis;
 `ifdef P128
 parameter P=128;
 `else
 parameter P=64;
 `endif
 reg clk=0;always #2.5 clk=~clk; // 200 MHz selected candidate clock.
 reg resetn=0,arm=0,sv=0,sl=0,mr=0;
 reg [31:0] sd=0;reg [3:0] sk=15;wire sr,mv,ml,busy;wire [31:0] md,cycles;wire [3:0] mk;
 `ifdef GATE
 mlp_axis dut(.aclk(clk),.aresetn(resetn),.arm(arm),.s_axis_tdata(sd),.s_axis_tkeep(sk),
 `else
 mlp_axis #(.P(P),.DATA_DIR("hardware_data")) dut(.aclk(clk),.aresetn(resetn),.arm(arm),.s_axis_tdata(sd),.s_axis_tkeep(sk),
 `endif
 .s_axis_tvalid(sv),.s_axis_tready(sr),.s_axis_tlast(sl),.m_axis_tdata(md),.m_axis_tkeep(mk),
 .m_axis_tvalid(mv),.m_axis_tready(mr),.m_axis_tlast(ml),.busy(busy),.last_cycles(cycles));
 reg [7:0] inputs[0:21503],gold[0:6803],labels[0:41];
 integer sample,mode,rx,k,b,idx,timeout; reg readonly_active=0; integer reads=0;reg active=0,stalled=0;reg [31:0] saved;
 reg savedlast;reg expect_error=0;
 reg [31:0] packet_cycles;
 always @(negedge clk) if(resetn) mr<=($urandom_range(0,3)!=0);
 always @(posedge clk) if(resetn) begin
  if(readonly_active && (dut.core_busy || dut.core.start || dut.input_accept))
   $fatal(1,"Read-only request started compute or wrote input");
  if(readonly_active && dut.core.cascade.impl.post_trace_valid)
   $fatal(1,"Read-only request generated a feature write");
  if(stalled && (!mv || md!==saved || ml!==savedlast)) $fatal(1,"AXIS changed during stall");
  stalled<=mv && !mr;saved<=md;savedlast<=ml;
  if(mv && mr) begin
   if(!active || mk!=15) $fatal(1,"unexpected output");
   if(rx==0) begin
    if(md[31:16]!=16'h4d4c || md[2]!==expect_error || (!expect_error && (md[0]!==labels[sample][0] || md[1]!==mode[0]))) $fatal(1,"header");
   end else if(rx==1) begin
    if(!expect_error && md!=713) $fatal(1,"zero cycles");
    `ifdef ROUTED_SDF
    if(!expect_error && md!==713) $fatal(1,"routed packet cycle count");
    `endif
    packet_cycles=md;
   end
   else if(rx==2) begin if(md!==(expect_error?0:mode?162:2)) $fatal(1,"payload length");end
   else for(b=0;b<4;b=b+1) begin
    idx=(rx-3)*4+b;
    if(idx<(mode?162:2)) begin
     if(md[b*8+:8]!==gold[sample*162+(mode?idx:160+idx)]) $fatal(1,"payload sample=%0d mode=%0d index=%0d",sample,mode,idx);
    end else if(md[b*8+:8]!==0) $fatal(1,"padding");
   end
   if(ml!== (rx==(expect_error?2:mode?43:3))) $fatal(1,"TLAST position");
   rx=rx+1;
  end
 end
 task beat(input [31:0] data,input last);
 begin
  repeat($urandom_range(0,2)) @(negedge clk);
  sd=data;sl=last;sv=1;
  @(posedge clk);while(!sr) @(posedge clk);
  @(negedge clk);sv=0;sl=0;
 end endtask
 task await_result;
 begin
  timeout=0;
  `ifdef ROUTED_SDF
  // Full-system implementation prunes the unused debug busy/cycles ports.
  // The AXIS packet retains the architectural cycle count and completion.
  while(rx<(expect_error?3:mode?44:4) && timeout<10000) begin @(negedge clk);timeout=timeout+1;end
  if(rx!=(expect_error?3:mode?44:4)) $fatal(1,"timeout/count rx=%0d",rx);
  `else
  while(busy && timeout<10000) begin @(negedge clk);timeout=timeout+1;end
  if(busy || rx!=(expect_error?3:mode?44:4)) $fatal(1,"timeout/count rx=%0d",rx);
  `endif
  active=0;
 end endtask
 task read_existing(input integer expected_sample,input bit invalid);
 begin
  sample=expected_sample;mode=1;expect_error=invalid;readonly_active=1;
  arm=0;@(negedge clk);rx=0;active=1;arm=1;@(negedge clk);
  beat(32'h4d4c5001,1);await_result();
  readonly_active=0;expect_error=0;reads=reads+1;
 end
 endtask
 initial begin
  $readmemh("data/inputs.mem",inputs);$readmemh("data/features.mem",gold);$readmemh("data/labels.mem",labels);
  repeat(30) @(negedge clk);resetn=1;
  read_existing(0,1);
  for(sample=0;sample<42;sample=sample+1) for(mode=0;mode<2;mode=mode+1) begin
   arm=0;@(negedge clk);rx=0;active=1;arm=1;@(negedge clk);
   beat(32'h4d4c5000|mode,0);
   for(k=0;k<128;k=k+1) beat({inputs[sample*512+4*k+3],inputs[sample*512+4*k+2],inputs[sample*512+4*k+1],inputs[sample*512+4*k]},k==127);
   await_result();
   begin : check_read
    integer old_mode;
    old_mode=mode;
    read_existing(sample,0);read_existing(sample,0);
    mode=old_mode;
   end
   `ifdef ROUTED_SDF
   $display("AXIS P=%0d sample=%0d mode=%0d cycles=%0d",P,sample,mode,packet_cycles);
   `else
   $display("AXIS P=%0d sample=%0d mode=%0d cycles=%0d",P,sample,mode,cycles);
   `endif
  end
  expect_error=1;arm=0;@(negedge clk);rx=0;active=1;arm=1;@(negedge clk);
  beat(32'hdeadbeef,1);await_result();
  read_existing(41,0); // malformed packet must not discard previous features
  resetn=0;repeat(30) @(negedge clk);resetn=1;
  read_existing(41,1); // stale feature RAM must be inaccessible after reset
  $display("READ_ONLY_PASS requests=%0d; no launch, no writes, exact retained features",reads);
  $display("PASS tb_axis P=%0d: 84 packets, random input bubbles/output stalls, all features, error packet",P);$finish;
 end
endmodule
