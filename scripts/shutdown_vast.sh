#!/usr/bin/env bash
# Tải hết về rồi HỦY cả hai instance vast. Chạy được độc lập, không cần phiên Claude.
#
# Vì sao script này tồn tại: việc hủy là BẮT BUỘC để không hết quota vast, nhưng
# mọi cơ chế tự động khác (monitor, cron) đều chết theo phiên. Đây là bản chạy tay
# một lệnh, và cũng là bản mà lớp tự động gọi vào.
#
# Bài học từ lần tắt máy hôm qua: chỉ tải `results/` và `log/` thì MẤT checkpoint
# Phase 1, và mọi phân tích trên không gian trọng số (§40, đo độ nhọn) phải chạy
# lại Phase 1 mới làm được. Lần này tải cả `model/*/source/best.pt`.
#
# KHÔNG hủy nếu đối chiếu file không khớp — thà trả tiền thêm còn hơn mất dữ liệu.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# label:id:port:host
MACHINES="${MACHINES:-ntat:48356959:56613:115.73.216.179 ntat2:48357013:48318:115.78.134.198}"
FORCE="${FORCE:-0}"

all_ok=1
for M in $MACHINES; do
  LABEL="${M%%:*}"; R="${M#*:}"; ID="${R%%:*}"; R="${R#*:}"; PORT="${R%%:*}"; HOST="${R##*:}"
  SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $PORT"
  echo "=========== $LABEL (id $ID, $HOST:$PORT) ==========="

  if ! timeout 40 $SSH root@"$HOST" 'echo ok' >/dev/null 2>&1; then
    echo "  KHONG SSH DUOC — bo qua, KHONG huy (co the mat du lieu)"
    all_ok=0; continue
  fi

  mkdir -p "results_$LABEL" "log_$LABEL" "model_$LABEL"
  ok=1
  timeout 1800 rsync -az -e "$SSH" root@"$HOST":/workspace/MultiVD/results/ "results_$LABEL/" || ok=0
  timeout 1800 rsync -az -e "$SSH" root@"$HOST":/workspace/MultiVD/log/     "log_$LABEL/"     || ok=0
  # Chi lay checkpoint NGUON. gated.sh da tu xoa checkpoint tung fold sau khi dung,
  # nen thu muc model/ chi con source — khoang 438 MB moi cai.
  timeout 3600 rsync -az -e "$SSH" --include='*/' --include='source/best.pt' --exclude='*' \
      root@"$HOST":/workspace/MultiVD/model/ "model_$LABEL/" || ok=0
  # Cac file ket qua phu nam thang o /workspace
  timeout 600 rsync -az -e "$SSH" --include='*.txt' --include='*.log' --exclude='*' \
      root@"$HOST":/workspace/ "log_$LABEL/workspace_root/" || true

  rem=$(timeout 60 $SSH root@"$HOST" 'cd /workspace/MultiVD/results 2>/dev/null && find . -name "fold*.json" -printf "%p %s\n" | sort' 2>/dev/null)
  loc=$( (cd "results_$LABEL" && find . -name 'fold*.json' -printf '%p %s\n' | sort) 2>/dev/null )
  miss=$(comm -23 <(echo "$rem") <(echo "$loc") | grep -c . || true)
  nres=$(echo "$rem" | grep -c . || true)
  nck=$(find "model_$LABEL" -name 'best.pt' 2>/dev/null | wc -l)
  echo "  rsync_ok=$ok  ket qua tren may=$nres  thieu=$miss  checkpoint nguon tai ve=$nck"

  if [[ "$ok" == "1" && "$miss" == "0" ]] || [[ "$FORCE" == "1" ]]; then
    echo "  -> HUY instance $ID"
    vastai destroy instance "$ID" -y
  else
    echo "  -> KHONG huy: doi chieu khong khop. Sua roi chay lai, hoac FORCE=1 de bo qua."
    all_ok=0
  fi
done

echo ""
echo "=========== con lai tren vast ==========="
vastai show instances --raw 2>/dev/null | python3 -c "
import json,sys
d=json.load(sys.stdin)
print('(khong con instance nao)' if not d else '')
for i in d: print(i['id'], i.get('label'), i.get('actual_status'), i.get('gpu_name'))
"
[[ "$all_ok" == "1" ]] || { echo ""; echo "CO MAY CHUA HUY — xem o tren."; exit 1; }
