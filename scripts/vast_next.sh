#!/usr/bin/env bash
# vast_next.sh — CHAY TREN MAY VAST. Cho viec hien tai xong roi chay khoi tiep theo.
# Tong quat hoa vast_stage2.sh: nhan ca NGUON va FOLD, va cho ca CHECKPOINT Pha 1 co mat
# (no co the dang duoc day len song song).
#
#   NEXT_SRC=full NEXT_FOLDS="1 2 4" setsid nohup bash scripts/vast_next.sh > log/next.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEXT_SRC="${NEXT_SRC:?dat NEXT_SRC}"
NEXT_FOLDS="${NEXT_FOLDS:?dat NEXT_FOLDS}"
PY="${PYTHON:-/venv/main/bin/python}"
LOCK=/tmp/multivd_opt1.lock
exec 8>/tmp/mvd_next.lock || exit 1
flock -n 8 || { echo "DA CO vast_next dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }
case "$NEXT_SRC" in 4cwe) CK=model/n48/phase1/t5p__latent_bottleneck_4cwe_l0p05/seed_42/best.pt;;
  com) CK=model/n48/phase1/t5p__latent_bottleneck_com_l0p05/seed_42/best.pt;;
  full) CK=model/n48/phase1/t5p__latent_bottleneck_full_l0p05/seed_42/best.pt;;
  *) echo "nguon la: $NEXT_SRC"; exit 1;; esac

echo "$(ts) | cho: lock nha + het job train + co $CK"
w=0
while true; do
  free=0; flock -n "$LOCK" -c true 2>/dev/null && free=1
  jobs=$(ps -eo args --no-headers | grep -c '[s]rc/train_[a-z]*\.py')
  ck=0; [ -s "$CK" ] && ck=1
  if (( free == 1 && jobs == 0 && ck == 1 )); then
    echo "$(ts) | san sang — chay nguon $NEXT_SRC, fold $NEXT_FOLDS"; break
  fi
  sleep 60; w=$((w+60)); (( w % 600 == 0 )) && echo "$(ts) | cho... ${w}s (lock=$free jobs=$jobs ckpt=$ck)"
done
sleep 5
FOLD_LIST="$NEXT_FOLDS" SOURCES_LIST="$NEXT_SRC" SEED=42 PYTHON="$PY" bash run/int1.sh 8>&-
echo "########## VAST_NEXT xong $(ts) | $NEXT_SRC fold $NEXT_FOLDS ##########"
