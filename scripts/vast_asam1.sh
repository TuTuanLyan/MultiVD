#!/usr/bin/env bash
# vast_asam1.sh — CHAY TREN MAY VAST. Cho INT1 xong roi chay ngay khoi ASAM1.
# Nguoi dung 08/09: "Khong duoc de trong may nhat la vast."
#   FOLDS="1 2 4" setsid nohup bash scripts/vast_asam1.sh > log/asam1_wait.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FOLDS="${FOLDS:?dat FOLDS}"
PY="${PYTHON:-/venv/main/bin/python}"
LOCK=/tmp/multivd_opt1.lock
exec 8>/tmp/mvd_asam1.lock || exit 1
flock -n 8 || { echo "DA CO vast_asam1 dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }
echo "$(ts) | cho INT1 xong (lock nha + het job train)"
w=0
while true; do
  free=0; flock -n "$LOCK" -c true 2>/dev/null && free=1
  jobs=$(ps -eo args --no-headers | grep -c '[s]rc/train_[a-z]*\.py')
  wait_others=$(ps -eo args --no-headers | grep -cE '[v]ast_next.sh|[v]ast_stage2.sh')
  # Chi chay khi khong con job VA khong con script noi tiep nao dang cho phien cua no
  if (( free == 1 && jobs == 0 && wait_others == 0 )); then echo "$(ts) | san sang — chay ASAM1"; break; fi
  sleep 60; w=$((w+60)); (( w % 600 == 0 )) && echo "$(ts) | cho... ${w}s (lock=$free jobs=$jobs khac=$wait_others)"
done
sleep 5
FOLD_LIST="$FOLDS" SOURCES_LIST="4cwe com" SEED=42 PYTHON="$PY" bash run/asam1.sh 8>&-
echo "########## VAST_ASAM1 xong $(ts) ##########"
