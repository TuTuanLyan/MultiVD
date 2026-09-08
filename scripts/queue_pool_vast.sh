#!/usr/bin/env bash
# queue_pool_vast.sh — CHAY TREN MAY VAST. Cho het job train roi moi chay POOL1.
# run/pool1.sh goi THANG matrix.sh, ma matrix.sh KHONG giu lock (chi opt1.sh giu), nen
# phong thang no se chay chong len job dang co -> hai job mot GPU -> OOM. Da suyt xay ra
# 08/09 tren ntat2: job asam da chay 197s va job pool1 vua khoi dong cung luc.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec 8>/tmp/mvd_poolvast.lock || exit 1
flock -n 8 || { echo "DA CO queue_pool_vast dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }
w=0
while true; do
  jobs=$(ps -eo args --no-headers | grep -c '[s]rc/train_[a-z]*\.py')
  wl=$(ps -eo args --no-headers | grep -c '[v]ast_worklist.sh')
  tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|head -1)
  use=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits|head -1)
  avail=$(( ${tot:-0} - ${use:-0} ))
  if (( jobs == 0 && wl == 0 && avail >= 11500 )); then echo "$(ts) | ranh (VRAM ${avail}MiB) — chay POOL1"; break; fi
  sleep 60; w=$((w+60)); (( w % 600 == 0 )) && echo "$(ts) | cho... jobs=$jobs worklist=$wl VRAM=${avail}MiB"
done
POOLS="${POOLS:-pur25_n930 pur75_n930 pur100_n465 pur12_n930}" FOLD_LIST="${FOLD_LIST:-1 2 3}" \
  SEED=42 PYTHON=/venv/main/bin/python bash run/pool1.sh 8>&-
echo "########## QUEUE_POOL_VAST xong $(ts) ##########"
