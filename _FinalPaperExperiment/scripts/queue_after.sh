#!/usr/bin/env bash
# Xếp hàng trên CÙNG máy mà không sửa driver đang chạy: chờ driver run.sh (theo PID) thoát rồi mới chạy run.sh cho run tiếp theo.
#   cd <out> && setsid nohup bash scripts/queue_after.sh <host> <pid driver> "<run ...>" ["<fold ...>"] > state/queue_<host>.out 2>&1 < /dev/null &
# Chờ theo PID + giờ khởi động của chính tiến trình đó (PID có thể bị cấp lại sau khi driver thoát), KHÔNG chờ theo pgrep
# (pgrep -f tự khớp chính nó — CLAUDE.md §13). run.sh tự giữ lock và tự dừng nếu còn tiến trình det_launch khác, nên không phóng trùng.
# Nối chuỗi: <pid> cũng có thể là một queue_after.sh KHÁC đang chờ — nó sẽ `exec` thành run.sh với CÙNG PID và giờ khởi động,
# nên vòng chờ theo (PID, lstart) chờ qua cả lúc chờ lẫn lúc driver của nó chạy, rồi mới phóng (28/09).
set -uo pipefail
HOST=$1; PID=$2; RUNLIST=$3; FOLDLIST=${4:-}
# danh sách run trong MỘT đối số có nháy (xem run.sh); kiểm TRƯỚC khi chờ để lỗi lộ ngay lúc xếp hàng, không phải lúc phóng
[ $# -le 4 ] || { echo "!! queue_after.sh <host> <pid> \"<run ...>\" [\"<fold ...>\"]: nhận $# đối số - gộp danh sách run trong một cặp nháy"; exit 2; }
case "$FOLDLIST" in *[!0-9\ ,]*) echo "!! FOLDLIST '$FOLDLIST' không phải danh sách fold - gộp danh sách run trong một cặp nháy"; exit 2 ;; esac
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPTS/.." || exit 1

ARGS=$(ps -o args= -p "$PID" 2>/dev/null)
case "$ARGS" in
  "bash scripts/run.sh "*|"bash "*"/scripts/run.sh "*|"bash scripts/queue_after.sh "*|"bash "*"/scripts/queue_after.sh "*) ;;   # 161 phóng bằng đường dẫn tuyệt đối
  *) echo "!! PID $PID không phải driver run.sh hay queue_after.sh (args: '${ARGS:-không còn}') — không xếp hàng"; exit 2 ;;
esac
START=$(ps -o lstart= -p "$PID")
echo "[$(date '+%F %T')] chờ driver $PID ($ARGS, khởi động $START) thoát rồi chạy: $RUNLIST ${FOLDLIST:+(fold $FOLDLIST)}"
while [ "$(ps -o lstart= -p "$PID" 2>/dev/null)" = "$START" ]; do sleep 60; done
echo "[$(date '+%F %T')] driver $PID đã thoát — phóng run.sh"
exec bash scripts/run.sh "$HOST" "$RUNLIST" "$FOLDLIST"
