#!/usr/bin/env bash
# queue_spd1.sh — cho 161 ranh roi chay SPD1. Khong giet gi, khong xoa gi.
#
# 161 dang chay bac 2 (queue_bac2) toi ~10:00 UTC 07/09. Script nay giu lock RIENG,
# doi den khi driver opt1 tha lock VA VRAM trong >= 13 GB (161 dung chung GPU), roi chay.
#   setsid nohup bash scripts/queue_spd1.sh > log/queue_spd1.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/home/ntat/miniconda3/envs/vdenv/bin/python}"
LOCK=/tmp/multivd_opt1.lock
exec 8>/tmp/mvd_queue_spd1.lock || exit 1
flock -n 8 || { echo "DA CO queue_spd1 dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }
w=0
while true; do
  if flock -n "$LOCK" -c true 2>/dev/null; then
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)
    free=$(( 16376 - ${used:-16376} ))
    if (( free >= 13000 )); then echo "$(ts) | GPU ranh (${free}MiB) — chay"; sleep 10; break; fi
    (( w % 1800 == 0 )) && echo "$(ts) | lock trong nhung VRAM ${free}MiB — NHUONG"
  fi
  sleep 120; w=$((w+120)); (( w % 1800 == 0 )) && echo "$(ts) | cho... ${w}s"
done
FOLD_LIST="${FOLD_LIST:-1 2 3}" SOURCES_LIST="${SOURCES_LIST:-4cwe com}" SEED=42 PYTHON="$PY" \
  bash run/spd1.sh 8>&-
echo "########## QUEUE_SPD1 xong $(ts) ##########"
