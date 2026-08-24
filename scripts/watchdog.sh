#!/usr/bin/env bash
# Watchdog chạy TRÊN CHÍNH MÁY vast: cứ vài phút kiểm hàng đợi còn sống không,
# chết thì bật lại.
#
# Vì sao nó phải nằm trên máy chứ không phải phía tôi: đã có tiền lệ GPU nằm
# không cả đêm vì lịch nhắc phía ngoài không được đặt lại. Watchdog này không
# biết gì về session, không cần mạng ra ngoài, và sống cùng máy — máy còn chạy
# thì nó còn canh.
#
# Bật lại một cách mù quáng chỉ an toàn vì overnight.sh idempotent: mỗi bước có
# cờ .done riêng, matrix.sh bỏ qua Phase 1 và fold đã có, và Phase 1 chỉ được
# công bố khi chạy xong (ghi .partial rồi mới mv). Lần bật lại tệ nhất mất đúng
# một job đang dở.
#
# Nó DỪNG khi hàng đợi báo ALL_DONE. Không có ALL_DONE mà không có tiến trình thì
# đó là sự cố, và bật lại là đúng.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_NAME="${RUN_NAME:?can RUN_NAME}"
STATE="${STATE:-/workspace/overnight_${RUN_NAME}}"
QUEUE_LOG="${QUEUE_LOG:-/workspace/overnight_${RUN_NAME}.log}"
INTERVAL="${INTERVAL:-300}"
WD_LOG="${WD_LOG:-/workspace/watchdog_${RUN_NAME}.log}"
export PYTHON HF_HOME BACKBONES RUN_NAME SEED SEED2 FOLDS STATE

log() { echo "[$(date -u '+%F %T')] $*" >> "$WD_LOG"; }

log "watchdog bat dau — kiem moi ${INTERVAL}s, hang doi $RUN_NAME"

STALE_COUNT=0
while true; do
  if [[ -f "$STATE/ALL_DONE" ]]; then
    log "hang doi bao ALL_DONE — watchdog dung"
    exit 0
  fi

  if pgrep -f "run/overnight.sh" > /dev/null; then
    # Sống. Ghi lại tiến độ để sáng ra đọc được đường đi của cả đêm.
    DONE=$(ls "$STATE"/*.done 2>/dev/null | wc -l)
    RES=$(find results -name 'fold*.json' 2>/dev/null | wc -l)
    UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
    log "song — buoc xong=$DONE ket_qua=$RES gpu=$UTIL"
    STALE_COUNT=0
  else
    STALE_COUNT=$((STALE_COUNT + 1))
    log "KHONG THAY hang doi (lan thu $STALE_COUNT) — don mo coi roi bat lai"

    # Dọn tiến trình huấn luyện mồ côi TRƯỚC khi bật lại. Không có bước này thì
    # hàng đợi mới và job mồ côi cùng chiếm GPU 16 GB và cả hai OOM — đo được:
    # khi giết hàng đợi, bash kịp sinh job kế tiếp rồi mới chết, nên mồ côi là
    # trường hợp THƯỜNG GẶP chứ không phải hiếm.
    #
    # Giết theo PID tường minh, không dùng `pkill -f`: chuỗi tìm kiếm của pkill
    # khớp cả dòng lệnh SSH đang gọi nó và tự giết phiên của chính mình.
    for ORPHAN in $(pgrep -f 'src/train_transfer\.py|src/train_baseline\.py' 2>/dev/null); do
      kill -9 "$ORPHAN" 2>/dev/null && log "  da giet job mo coi PID $ORPHAN"
    done
    sleep 5

    nohup bash run/overnight.sh >> "$QUEUE_LOG" 2>&1 &
    log "  da bat lai, PID $!"
  fi

  sleep "$INTERVAL"
done
