#!/usr/bin/env bash
# queue_pool1.sh — cho GPU ranh roi chay POOL1 tren 161 (GPU dung chung, phai co cong VRAM).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec 8>/tmp/mvd_pool1.lock || exit 1
flock -n 8 || { echo "DA CO queue_pool1 dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }
w=0
while true; do
  free=0; flock -n /tmp/multivd_opt1.lock -c true 2>/dev/null && free=1
  use=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits|head -1)
  avail=$(( 16376 - ${use:-16376} ))
  (( free == 1 && avail >= 11500 )) && { echo "$(ts) | GPU ranh (${avail}MiB)"; break; }
  sleep 60; w=$((w+60)); (( w % 600 == 0 )) && echo "$(ts) | cho... lock=$free VRAM=${avail}MiB"
done
# Thu tu nguon co CHU Y: o quyet dinh (pur12 — cung tinh khiet voi `full` nhung 1/8 co)
# chay TRUOC, de biet cau tra loi som nhat co the.
POOLS="pur12_n930 pur100_n930 pur50_n930 pur100_n232 pur25_n930 pur75_n930 pur100_n465" \
FOLD_LIST="1 2 3" SEED=42 PYTHON=/home/ntat/miniconda3/envs/vdenv/bin/python \
  bash run/pool1.sh 8>&-
echo "########## QUEUE_POOL1 xong $(ts) ##########"
