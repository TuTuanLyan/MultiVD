#!/usr/bin/env bash
# Bản ghi TIẾN ĐỘ của fold đang chạy trên máy từ xa (để ghi vào db artifact): lấy log về tạm rồi `fpe.py live`.
#   bash scripts/live_remote.sh <158|vast>   -> fold đang chạy theo dòng "bắt đầu" cuối của state/driver_<host>.log
set -uo pipefail
HOST=$1; BASE=$(cd "$(dirname "$0")/.." && pwd)
. "$BASE/scripts/remote_env.sh" "$HOST"
CUR=$(timeout 30 ssh -n "${SSHO[@]}" "$SSH" "awk '/ bắt đầu\$/ {r=\$3; f=\$4} / xong rc=/ {r=\"\"} END {if (r != \"\") print r, substr(f,2)}' $OUT/state/driver_$HOST.log 2>/dev/null")
read -r RUN F <<<"$CUR"
[ -z "${RUN:-}" ] && { echo none; exit 0; }
TMP=$(mktemp); trap 'rm -f "$TMP"' EXIT
timeout 60 ssh -n "${SSHO[@]}" "$SSH" "cat $OUT/logs/$RUN/fold$F.log" > "$TMP" || { echo "!! không lấy được log $RUN f$F"; exit 1; }
python3 "$BASE/scripts/fpe.py" live "$RUN" "$F" "$TMP" "$HOST"
