#!/usr/bin/env bash
# 08/10 22:5x: driver 161 tự dừng (cổng run.sh: "đang có 1 tiến trình det_launch khác") vì một phiên Claude khác chạy thử khói
# smoke_adan bằng CPU trên 161. Không đụng tiến trình đó: chờ tới khi KHÔNG còn det_launch nào (2 lần kiểm liên tiếp cách 60 s),
# rồi exec run.sh với phần còn lại. run.sh tự kiểm lock + det_launch lần nữa trước mỗi fold.
#   setsid nohup bash state/wait_no_detlaunch_then_run.sh <host> "<run:fold ...>" > state/queue_<host>_wait.out 2>&1 < /dev/null &
set -uo pipefail
HOST=$1; RUNLIST=$2
cd "$(dirname "$0")/.." || exit 1
n_det () { ps -eo args --no-headers | awk '$2 ~ /det_launch\.py$/' | wc -l; }
echo "[$(date '+%F %T')] chờ hết det_launch rồi chạy: $RUNLIST"
ok=0
while [ "$ok" -lt 2 ]; do
  if [ "$(n_det)" = 0 ]; then ok=$((ok + 1)); else ok=0; fi
  sleep 60
done
echo "[$(date '+%F %T')] không còn det_launch - phóng run.sh"
exec bash scripts/run.sh "$HOST" "$RUNLIST"
