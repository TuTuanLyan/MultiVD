#!/usr/bin/env bash
# Chuyển log/kết quả của các fold ĐÃ XONG trên máy từ xa (158, vast) về _FinalPaperExperiment (161), kiểm checksum rồi XOÁ bản
# trên máy đó — mỗi run chỉ còn MỘT bản, trên 161. Chạy từ 161:  bash scripts/sync_remote.sh <158|vast>   (chạy lặp lại an toàn)
# Fold "xong" = sự kiện CUỐI CÙNG của fold đó trong state/driver_<host>.log là "xong rc=" — fold đang chạy lại KHÔNG bị đụng.
set -uo pipefail
HOST=$1; BASE=$(cd "$(dirname "$0")/.." && pwd)
. "$BASE/scripts/remote_env.sh" "$HOST"
# ssh lỗi thoáng qua (28/09: "banner exchange" timeout) từng làm lượt sync thoát 0 mà KHÔNG chuyển gì — trông y hệt "chưa có fold xong".
# Dấu __SSH_OK__ ở cuối phân biệt "ssh hỏng" với "log rỗng/chưa có": thiếu dấu ⇒ báo lỗi, thoát 2.
RAW=$(timeout 60 ssh -n "${SSHO[@]}" "$SSH" "cat $OUT/state/driver_$HOST.log 2>/dev/null; echo __SSH_OK__")
case "$RAW" in *__SSH_OK__) ;; *) echo "!! sync_remote $HOST: không đọc được state/driver_$HOST.log (ssh lỗi) — CHƯA chuyển gì, chạy lại"; exit 2 ;; esac
DONE=$(printf "%s\n" "${RAW%__SSH_OK__}" | python3 -c '
import re, sys
last = {}
for line in sys.stdin:
    m = re.search(r"\] (\S+) f(\d+) (bắt đầu|xong rc=(\d+))", line)
    if m:
        last[(m.group(1), m.group(2))] = m.group(4)          # None = đang chạy
for (run, f), rc in sorted(last.items()):
    if rc is not None:
        print("logs/%s/fold%s.log results/%s/fold%s.json results/%s/fold%s.probs.npz" % (run, f, run, f, run, f))
')
[ -n "$DONE" ] || exit 0
# MỘT lần ssh liệt kê file còn trên máy từ xa (27/09: gọi `ssh test -f` cho từng file của MỌI fold đã xong làm một lượt sync
# dài quá timeout khi số fold tăng ⇒ bị cắt giữa chừng, .probs.npz về muộn một lượt).
EXIST=$(timeout 60 ssh -n "${SSHO[@]}" "$SSH" "cd $OUT && ls -1 $(echo $DONE) 2>/dev/null; echo __SSH_OK__")
case "$EXIST" in *__SSH_OK__) EXIST=${EXIST%__SSH_OK__} ;; *) echo "!! sync_remote $HOST: không liệt kê được file trên máy từ xa (ssh lỗi) — CHƯA chuyển gì, chạy lại"; exit 2 ;; esac
for sub in $EXIST; do
  mkdir -p "$(dirname "$BASE/$sub")"
  rsync -a --checksum --remove-source-files -e "$RSH" "$SSH:$OUT/$sub" "$BASE/$sub" < /dev/null && echo "chuyển $sub"
done
