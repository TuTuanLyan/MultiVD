#!/usr/bin/env bash
# Chờ Pha 1 <p1_run> trên máy từ xa <host> xong (dòng "<p1_run> f1 xong rc=0" trong state/driver_<host>.log của máy đó), rồi chép checkpoint
# Pha 1 về 161 và đối chiếu md5 - để các fold Pha 2 xếp hàng trên 161 có checkpoint khi tới lượt (người dùng 02/10: "chủ động chạy task khi đang rảnh").
# Chạy từ 161:  cd <out> && setsid nohup bash scripts/copy_p1_when_done.sh <158|vast> <p1_run> [thư mục ckpt đích] > state/copy_<p1_run>.out 2>&1 < /dev/null &
# (đối số 3 chỉ để thử khói: mặc định là ckpt của 161 trong runs.json)
# md5 lệch ⇒ xoá bản chép (fold xếp hàng sẽ tự bỏ qua vì thiếu checkpoint, không chạy trên checkpoint hỏng) và thoát 1.
set -uo pipefail
HOST=$1; RUN=$2; DST=${3:-}; BASE=$(cd "$(dirname "$0")/.." && pwd)
. "$BASE/scripts/remote_env.sh" "$HOST"
ck () { python3 -c "import json;print(json.load(open('$BASE/scripts/runs.json'))['hosts']['$1']['ckpt'])"; }
RCK=$(ck "$HOST")/$RUN/seed_42/fold1/best.pt
LCK=${DST:-$(ck 161)}/$RUN/seed_42/fold1/best.pt
echo "[$(date '+%F %T')] chờ '$RUN f1 xong rc=0' trên $HOST ($RCK)"
while true; do
  L=$(timeout 60 ssh -n "${SSHO[@]}" "$SSH" "grep -c '] $RUN f1 xong rc=0' $OUT/state/driver_$HOST.log; echo __SSH_OK__" 2>/dev/null)
  case "$L" in
    *__SSH_OK__) n=$(printf '%s\n' "$L" | head -1); [ "${n:-0}" -ge 1 ] 2>/dev/null && break ;;
    *) echo "[$(date '+%F %T')] ssh lỗi, thử lại" ;;
  esac
  sleep 120
done
mkdir -p "$(dirname "$LCK")"
rsync -a -e "$RSH" "$SSH:$RCK" "$LCK" < /dev/null
a=$(md5sum "$LCK" 2>/dev/null | cut -c1-32); b=$(timeout 60 ssh -n "${SSHO[@]}" "$SSH" "md5sum $RCK" | cut -c1-32)
if [ -n "$a" ] && [ "$a" = "$b" ]; then
  echo "[$(date '+%F %T')] chép xong $LCK, md5 TRÙNG $a"
else
  echo "[$(date '+%F %T')] !! md5 LỆCH ('$a' / '$b') - xoá bản chép"; rm -f -- "$LCK"; exit 1
fi
