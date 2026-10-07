#!/usr/bin/env bash
# Phóng khối final từ 161:   bash launch.sh <161|158> "<run ...>" ["<fold ...>"]
#   - Chạy lại = XOÁ log/kết quả cũ của đúng các run × fold đó trong _FinalPaperExperiment (bản duy nhất, trên 161).
#   - 158: đồng bộ scripts/ + runs.json sang thư mục tạm của 158 rồi phóng run.sh ở đó (setsid); kết quả về 161 bằng sync_158.sh.
#   - Từ chối nếu lock của máy đích đang bị giữ (đã có driver) — không bao giờ phóng trùng.
set -uo pipefail
HOST=$1; RUNLIST=$2; FOLDLIST=${3:-}
BASE=$(cd "$(dirname "$0")/.." && pwd)
MAN=$BASE/scripts/runs.json
hget () { python3 -c "import json;print(json.load(open('$MAN'))['hosts']['$HOST'].get('$1',''))"; }
OUT=$(hget out); SSH=$(hget ssh)

for ENTRY in $RUNLIST; do                                # xoá bản cũ trên 161; mục "run" hoặc "run:fold[,fold]"
  RUN=${ENTRY%%:*}; FOLDS=$FOLDLIST
  [ "$ENTRY" != "$RUN" ] && FOLDS=${ENTRY#*:} && FOLDS=${FOLDS//,/ }
  [ -z "$FOLDS" ] && FOLDS=$(python3 -c "import json;m=json.load(open('$MAN'));r={x['id']:x for x in m['runs']}['$RUN'];print(' '.join(map(str,r.get('folds',m['block']['target_folds']))))")
  for F in $FOLDS; do
    rm -f "$BASE/logs/$RUN/fold$F.log" "$BASE/results/$RUN/fold$F.json" "$BASE/results/$RUN/fold$F.probs.npz"
  done
done

if [ "$HOST" = 161 ]; then
  mkdir -p "$OUT/state"; flock -n "$OUT/state/fpe.lock" -c true || { echo "!! 161 đang có driver (lock bị giữ)"; exit 3; }
  setsid nohup bash "$BASE/scripts/run.sh" 161 "$RUNLIST" "$FOLDLIST" > "$OUT/state/driver_161.out" 2>&1 < /dev/null &
  sleep 2; echo "đã phóng trên 161: $(ps -eo pid,args --no-headers | awk '$2=="bash" && $3 ~ /scripts\/run\.sh$/ {print $1}')"
else
  . "$BASE/scripts/remote_env.sh" "$HOST"                # SSH/OUT/cổng của máy từ xa (158, vast)
  timeout 30 ssh -n "${SSHO[@]}" "$SSH" "mkdir -p $OUT/scripts $OUT/state && flock -n $OUT/state/fpe.lock -c true" || { echo "!! $HOST đang có driver (lock bị giữ) hoặc không vào được"; exit 3; }
  rsync -a --checksum --delete --exclude __pycache__ -e "$RSH" "$BASE/scripts/" "$SSH:$OUT/scripts/" < /dev/null
  # -n + timeout: phiên ssh phóng từng treo sau khi driver đã tách (24/09); driver nằm trong session riêng nên cắt ssh không ảnh hưởng
  timeout 30 ssh -n "${SSHO[@]}" "$SSH" "cd $OUT && setsid nohup bash scripts/run.sh $HOST '$RUNLIST' '$FOLDLIST' > state/driver_$HOST.out 2>&1 < /dev/null &"
  sleep 3
  timeout 30 ssh -n "${SSHO[@]}" "$SSH" "ps -eo pid,args --no-headers | awk '\$2==\"bash\" && \$3==\"scripts/run.sh\" {print \"đã phóng trên $HOST: \" \$1}'"
fi
