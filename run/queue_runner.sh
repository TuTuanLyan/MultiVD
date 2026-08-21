#!/usr/bin/env bash
# Chạy hàng đợi tuần tự, có CHỐT GIỜ.
#
# Quy tắc: chốt giờ chỉ chặn việc KHỞI ĐỘNG job mới. Job đang chạy được chạy nốt
# dù có vượt qua mốc — dừng giữa chừng nghĩa là vứt toàn bộ GPU đã tiêu cho nó mà
# không thu lại được fold nào, vì kết quả chỉ ghi ra sau khi từng fold xong.
#
# Tham số:
#   $1  epoch UTC của mốc chốt
#   $2  file hàng đợi: mỗi dòng một lệnh, dòng trống và dòng # bị bỏ qua
#
# Trạng thái ghi ra /workspace/QUEUE_STATE để theo dõi từ ngoài mà không cần
# đọc log dài.
set -uo pipefail
cd /workspace/MultiVD
export HF_HOME=/workspace/hf

CUTOFF="${1:?can epoch UTC cua moc chot}"
QUEUE="${2:?can file hang doi}"
STATE=/workspace/QUEUE_STATE

note() {
  echo "$(date -u '+%F %T UTC') | $*"
  echo "$(date -u '+%F %T UTC') | $*" >> "$STATE"
}

note "bat dau. moc chot $(date -u -d "@$CUTOFF" '+%F %T UTC') ($(date -d "@$CUTOFF" '+%H:%M %Z'))"

# Đợi mọi job đang chạy sẵn kết thúc trước khi đụng vào hàng đợi.
while pgrep -f "run/gated.sh" > /dev/null 2>&1; do sleep 30; done
note "khong con job nao dang chay"

index=0
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ -z "${line// /}" || "${line:0:1}" == "#" ]] && continue
  index=$((index + 1))
  now=$(date -u +%s)
  if (( now >= CUTOFF )); then
    note "DA QUA MOC CHOT — khong khoi dong job #$index nua. Con lai trong hang doi:"
    note "  $line"
    while IFS= read -r rest; do
      [[ -z "${rest// /}" || "${rest:0:1}" == "#" ]] && continue
      note "  $rest"
    done
    note "DUNG."
    touch /workspace/QUEUE_STOPPED_AT_CUTOFF
    exit 0
  fi
  remain=$(( (CUTOFF - now) / 60 ))
  note "job #$index bat dau (con $remain phut truoc moc chot): $line"
  bash -c "$line"
  note "job #$index xong"
done < "$QUEUE"

note "HET HANG DOI — tat ca da chay xong truoc moc chot."
touch /workspace/QUEUE_DONE
