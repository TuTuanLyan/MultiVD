#!/usr/bin/env bash
# vast_stage2.sh — CHAY TREN MAY VAST. Cho giai doan 1 xong roi chay tiep giai doan 2
# ngay lap tuc, de may khong bao gio nam khong.
#
# Nguoi dung 08/09: "Khong duoc de trong may nhat la vast." May vast tinh tien theo gio
# nen mot khoang trong 40 phut giua hai giai doan la 40 phut tra tien cho GPU nhan roi.
# Monitor phong lai driver CHET thi khong du — phai co san viec KE TIEP.
#
# Giai doan 2 = leo INT1 tu bac 1 (3 fold) len bac 2 (5 fold). Day la duong leo bac da
# ghi trong CLAUDE.md muc 1, khong phai thi nghiem moi.
#   ntat  : fold 1,2 -> fold 4
#   ntat2 : fold 3   -> fold 5
#
#   FOLD_NEXT=4 setsid nohup bash scripts/vast_stage2.sh > log/stage2.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FOLD_NEXT="${FOLD_NEXT:?dat FOLD_NEXT}"
PY="${PYTHON:-/venv/main/bin/python}"
LOCK=/tmp/multivd_opt1.lock
exec 8>/tmp/mvd_stage2.lock || exit 1
flock -n 8 || { echo "DA CO stage2 dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }

echo "$(ts) | cho giai doan 1 xong (lock $LOCK va khong con job train)"
w=0
while true; do
  free=0
  flock -n "$LOCK" -c true 2>/dev/null && free=1
  # dem job train bang cot args, KHONG dung pgrep -f (no tu khop chinh minh)
  jobs=$(ps -eo args --no-headers | grep -c '[s]rc/train_[a-z]*\.py')
  if (( free == 1 && jobs == 0 )); then echo "$(ts) | giai doan 1 xong — chay fold $FOLD_NEXT"; break; fi
  sleep 60; w=$((w+60)); (( w % 1800 == 0 )) && echo "$(ts) | cho... ${w}s (lock_free=$free jobs=$jobs)"
done
sleep 5
FOLD_LIST="$FOLD_NEXT" SOURCES_LIST="4cwe com" SEED=42 PYTHON="$PY" bash run/int1.sh 8>&-
echo "########## STAGE2 xong $(ts) | fold $FOLD_NEXT ##########"
