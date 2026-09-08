#!/usr/bin/env bash
# queue_int1.sh — chay INT1 khi GPU THAT SU ranh. Dung tren ca 161 va 158.
#
# VI SAO CO FILE NAY (08/09/2026): da phong `run/int1.sh` TRUC TIEP, bo qua cong VRAM.
# Ca hai may deu dang co nguoi khac dung (161: 7,0 GB cua ba tien trinh la; 158: 8,9 GB),
# nen moi o deu chet vi CUDA OOM va driver van in "xong" — 0/10 o tren 161, 1/5 tren 158.
# GPU o day la GPU DUNG CHUNG: lock cua rieng minh khong noi gi ve viec nguoi khac
# dang chiem bao nhieu. Phai xem VRAM.
#
#   FOLD_LIST="1 2" setsid nohup bash scripts/queue_int1.sh > log/queue_int1.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec 8>/tmp/mvd_queue_int1.lock || exit 1
flock -n 8 || { echo "DA CO queue_int1 dang chay"; exit 3; }
NEED_FREE="${NEED_FREE:-11500}"      # o t5p do duoc dung ~8,8 GB luc OOM; chua bien 2,7 GB
LOCK=/tmp/multivd_opt1.lock
ts(){ date -u '+%F %T'; }
w=0
while true; do
  free_ok=0
  if flock -n "$LOCK" -c true 2>/dev/null; then
    tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    use=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [[ -n "${tot:-}" && -n "${use:-}" ]]; then
      free=$(( tot - use ))
      (( free >= NEED_FREE )) && free_ok=1
      (( w % 1800 == 0 )) && echo "$(ts) | VRAM trong ${free}MiB / can ${NEED_FREE} — $( ((free_ok)) && echo CHAY || echo NHUONG )"
    fi
  fi
  (( free_ok )) && { echo "$(ts) | GPU ranh — chay INT1"; sleep 10; break; }
  sleep 120; w=$((w+120))
done
FOLD_LIST="${FOLD_LIST:-1 2 3}" SOURCES_LIST="${SOURCES_LIST:-4cwe com}" SEED=42 bash run/int1.sh 8>&-
echo "########## QUEUE_INT1 xong $(ts) ##########"
