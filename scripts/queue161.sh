#!/usr/bin/env bash
# queue161.sh — hang doi cho may 161 (local), chay TUAN TU khong can nguoi trong.
#
#   setsid nohup bash scripts/queue161.sh > log/queue161.log 2>&1 < /dev/null &
#
# 161 giu FOLD 1 va 2 (chia theo fold tron ven, CLAUDE.md muc 4). Hang doi:
#
#   0. (dang chay khi phong) OPT1 CONFIGS mac dinh: 4cwe+com roi full   <- cho xong
#   1. ME10 fold 1,2 — bo sung cho fold 3 cua 158, du 3 fold cho bac 1  <- ~2,9 h
#
# Viec 1 tra loi cau hoi ma early stopping dang chan: moi o co lambda@epoch5 ~ 0 deu
# dung o DUNG epoch 7 (patience 5, min_epochs 3) va sap duoi F1 0.6, con o co
# lambda@epoch5 = 0.99 chay 13-18 epoch va khong sap. Nang min_epochs len 10 cho CA
# nhanh chinh LAN doi chung nen phep so doi dung mot bien.
#
# 161 DUNG CHUNG GPU voi nguoi khac: truoc moi viec, cho toi khi VRAM trong >= 13 GB.
# Khong bao gio phong de len job cua nguoi khac (CLAUDE.md muc 9, memory vast-and-local-gpu-rules).
#
# BAC 1 — KIEM CHUNG, seed 42 (CLAUDE.md muc 1). Khong phai ket qua cuoi.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/home/ntat/miniconda3/envs/vdenv/bin/python}"
LOCK=/tmp/multivd_opt1.lock
exec 8>/tmp/mvd_queue161.lock || exit 1
flock -n 8 || { echo "DA CO queue161 dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }

wait_free(){ # cho driver opt1 nha lock VA cho GPU du cho
  local waited=0
  while true; do
    if flock -n "$LOCK" -c true 2>/dev/null; then
      local used free
      used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
      free=$(( 16376 - ${used:-16376} ))
      if (( free >= 13000 )); then sleep 10; return 0; fi
      (( waited % 1800 == 0 )) && echo "$(ts) | lock trong nhung VRAM chi con ${free}MiB — NHUONG, cho tiep"
    fi
    sleep 120; waited=$((waited+120))
    (( waited % 1800 == 0 )) && echo "$(ts) | cho... ${waited}s"
  done
}

echo "########## QUEUE161 bat dau $(ts) ##########"

wait_free
echo "########## $(ts) | VIEC 1: ME10 fold 1,2 (min_epochs=10) ##########"
MIN_EP=10 \
SOURCES="4cwe com" FOLD_LIST="1 2" PYTHON="$PY" \
CONFIGS="c5000_t0p05_me10|recadam|--sam_rho 0
c5000_t0p2k02_me10|recadam|--sam_rho 0 --anneal_t0_ratio 0.2 --anneal_k 0.02
c50_t0p2k02_me10|recadam|--sam_rho 0 --pretrain_cof 50 --anneal_t0_ratio 0.2 --anneal_k 0.02
c5000_t0p5_me10|recadam|--sam_rho 0 --anneal_t0_ratio 0.5" \
bash run/opt1.sh 8>&-
echo "########## QUEUE161 xong $(ts) ##########"
