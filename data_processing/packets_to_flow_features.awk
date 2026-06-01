BEGIN{
  FS=","; OFS=";"

  NBINS=8

  print "flow_id;proto;duration;bytes_fwd;pkts_fwd;bytes_bwd;pkts_bwd" \
        ";fwd_size_bin1;fwd_size_bin2;fwd_size_bin3;fwd_size_bin4;fwd_size_bin5;fwd_size_bin6;fwd_size_bin7;fwd_size_bin8" \
        ";bwd_size_bin1;bwd_size_bin2;bwd_size_bin3;bwd_size_bin4;bwd_size_bin5;bwd_size_bin6;bwd_size_bin7;bwd_size_bin8" \
        ";fwd_ipt_bin1;fwd_ipt_bin2;fwd_ipt_bin3;fwd_ipt_bin4;fwd_ipt_bin5;fwd_ipt_bin6;fwd_ipt_bin7;fwd_ipt_bin8" \
        ";bwd_ipt_bin1;bwd_ipt_bin2;bwd_ipt_bin3;bwd_ipt_bin4;bwd_ipt_bin5;bwd_ipt_bin6;bwd_ipt_bin7;bwd_ipt_bin8"
}

function bin8(x){
  if (x <= 15) return 1
  else if (x <= 31) return 2
  else if (x <= 63) return 3
  else if (x <= 127) return 4
  else if (x <= 255) return 5
  else if (x <= 511) return 6
  else if (x <= 1024) return 7
  else return 8
}

function proto_name(p){
  if (p == 6) return "tcp"
  if (p == 17) return "udp"
  return "ip" p
}

function normalize_flow(src,dst,sp,dp,pnum,  A,B,prot){
  prot = proto_name(pnum)
  A = src ":" sp
  B = dst ":" dp
  if (A < B){
    FLOW_ID = prot "|" A "-" B
    DIR = "fwd"
    FLOW_PROTO = prot
  } else {
    FLOW_ID = prot "|" B "-" A
    DIR = "bwd"
    FLOW_PROTO = prot
  }
}

NR==1 { next }  # header tshark

{
  ts=$1+0
  src=$2; dst=$3
  tcp_sp=$4; tcp_dp=$5
  udp_sp=$6; udp_dp=$7
  len=$8+0
  pnum=$9+0

  sp = (tcp_sp!="") ? tcp_sp : udp_sp
  dp = (tcp_dp!="") ? tcp_dp : udp_dp
  if (sp=="" || dp=="") next

  normalize_flow(src,dst,sp,dp,pnum)
  f=FLOW_ID
  d=DIR

  if (!(f in start_ts) || ts < start_ts[f]) start_ts[f]=ts
  if (!(f in end_ts)   || ts > end_ts[f])   end_ts[f]=ts

  bytes[f,d] += len
  pkts[f,d]++

  sb = bin8(len)
  size_hist[f,d,sb]++

  key = f SUBSEP d
  if (key in last_ts_dir){
    dt_ms = (ts - last_ts_dir[key]) * 1000.0
    if (dt_ms < 0) dt_ms = 0
    ib = bin8(dt_ms)
    ipt_hist[f,d,ib]++
  }
  last_ts_dir[key]=ts

  prot[f]=FLOW_PROTO
}

END{
  for (f in start_ts){
    dur = end_ts[f] - start_ts[f]
    if (dur < 0) dur = 0

    bf = bytes[f,"fwd"]+0
    pf = pkts[f,"fwd"]+0
    bb = bytes[f,"bwd"]+0
    pb = pkts[f,"bwd"]+0

    printf "%s;%s;%.6f;%d;%d;%d;%d", f, prot[f], dur, bf, pf, bb, pb

    for(i=1;i<=NBINS;i++) printf ";%d", (size_hist[f,"fwd",i]+0)
    for(i=1;i<=NBINS;i++) printf ";%d", (size_hist[f,"bwd",i]+0)

    for(i=1;i<=NBINS;i++) printf ";%d", (ipt_hist[f,"fwd",i]+0)
    for(i=1;i<=NBINS;i++) printf ";%d", (ipt_hist[f,"bwd",i]+0)

    printf "\n"
  }
}
