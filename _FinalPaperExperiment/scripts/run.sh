#!/usr/bin/env bash
# Driver khối final trên MỘT máy: chạy lần lượt các run × fold, mỗi fold: xoá log/kết quả cũ -> train -> test.
#   bash run.sh <host 161|158> "<run ...>" ["<fold ...>"]      (fold bỏ trống = mọi fold của run)
#   bash run.sh 158 "baseline:1 mwg_assemble:1 baseline:2,3"     (mục run:fold chạy đúng fold đó, theo thứ tự ghi — fold-major)
#   PY_DRY=1  chỉ in lệnh (chạy khô)          KEEP_CK=1  giữ checkpoint Pha 2 (mặc định xoá sau khi test; Pha 1 luôn giữ)
#   VRAM_MIN  MiB trống tối thiểu trước MỖI fold (mặc định 9000)
# Log/kết quả nằm ở <out>/{logs,results}/<run>/ của máy chạy; trên 158 đó là thư mục TẠM, sync_158.sh chuyển về 161.
set -uo pipefail
HOST=$1; RUNLIST=$2; FOLDLIST=${3:-}
# danh sách run phải nằm trong MỘT đối số có nháy: truyền rời từng mục thì $3 thành FOLDLIST và các mục sau bị bỏ IM LẶNG (06/10 02:06, 158)
[ $# -le 3 ] || { echo "!! run.sh <host> \"<run ...>\" [\"<fold ...>\"]: nhận $# đối số - gộp danh sách run trong một cặp nháy"; exit 2; }
case "$FOLDLIST" in *[!0-9\ ,]*) echo "!! FOLDLIST '$FOLDLIST' không phải danh sách fold - gộp danh sách run trong một cặp nháy"; exit 2 ;; esac
SCRIPTS=$(cd "$(dirname "$0")" && pwd)
MAN=$SCRIPTS/runs.json
ROOT=$(python3 -c "import json;print(json.load(open('$MAN'))['hosts']['$HOST']['root'])")
OUT=$(python3 -c "import json;print(json.load(open('$MAN'))['hosts']['$HOST']['out'])")
cd "$ROOT" || exit 1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TZ=${TZ:-ICT-7}          # giờ Việt Nam (UTC+7) ở MỌI máy — container vast mặc định UTC; dạng POSIX, không cần tzdata
mkdir -p "$OUT/state"

exec 9>"$OUT/state/fpe.lock"                                   # một driver một máy: lock phải được GIỮ suốt lúc chạy
flock -n 9 || { echo "!! $OUT/state/fpe.lock đang bị giữ — đã có driver khác trên $HOST"; exit 3; }

gpu_free () { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$CUDA_VISIBLE_DEVICES" | head -1; }
own_trainers () { ps -eo args --no-headers | awk '$2 ~ /det_launch\.py$/' | wc -l; }

echo "=== FPE $(hostname) $(date '+%F %T') host=$HOST runs='$RUNLIST' folds='${FOLDLIST:-tất cả}' dry=${PY_DRY:-0} ==="
for ENTRY in $RUNLIST; do                                   # mục "run" (mọi fold / FOLDLIST) hoặc "run:fold[,fold]" (xếp fold-major)
  RUN=${ENTRY%%:*}; FOLDS=$FOLDLIST
  [ "$ENTRY" != "$RUN" ] && FOLDS=${ENTRY#*:} && FOLDS=${FOLDS//,/ }
  [ -z "$FOLDS" ] && FOLDS=$(python3 -c "import json;m=json.load(open('$MAN'));r={x['id']:x for x in m['runs']}['$RUN'];print(' '.join(map(str,r.get('folds',m['block']['target_folds']))))")
  for F in $FOLDS; do
    # bắt lỗi của plan TRƯỚC khi eval: `eval "$(lệnh hỏng)"` trả 0 ⇒ `|| continue` không bao giờ chạy (đã thử 26/09)
    PLAN=$(python3 "$SCRIPTS/fpe.py" plan "$HOST" "$RUN" "$F") || { echo "!! không lập được lệnh $RUN f$F"; continue; }
    unset INIT_CKPT; eval "$PLAN"
    if [ -n "$INIT_CKPT" ] && [ ! -s "$INIT_CKPT" ] && [ -z "${PY_DRY:-}" ]; then
      echo "!! $RUN f$F: thiếu checkpoint Pha 1 $INIT_CKPT — bỏ qua fold này"; continue
    fi
    if [ -n "${PY_DRY:-}" ]; then
      echo "--- $RUN f$F"; echo "ENV  $RUN_ENV"; echo "TRAIN $TRAIN_CMD"; echo "TEST  $TEST_CMD"; echo "LOG $LOG | RESULT $RESULT | CKPT $CKPT | INIT ${INIT_CKPT:-—}"
      continue
    fi
    FREE=$(gpu_free)
    while [ "${FREE:-0}" -lt "${VRAM_MIN:-9000}" ]; do echo "  [$(date +%H:%M)] chờ VRAM: còn ${FREE} MiB"; sleep 120; FREE=$(gpu_free); done
    N=$(own_trainers)
    [ "$N" != 0 ] && { echo "!! đang có $N tiến trình det_launch khác — dừng driver"; exit 4; }
    # chạy lại = xoá bản cũ của đúng fold này (log, kết quả, dự đoán)
    rm -f "$LOG" "$RESULT" "${RESULT%.json}.probs.npz" "$CKPT" "$CKPT.resource_train.json"
    mkdir -p "$(dirname "$LOG")" "$(dirname "$RESULT")" "$(dirname "$CKPT")"
    echo "  [$(date '+%F %T')] $RUN f$F bắt đầu" | tee -a "$OUT/state/driver_$HOST.log"
    { echo "# $(date '+%F %T') $(hostname) | $RUN_ENV"; echo "# TRAIN: $TRAIN_CMD"; echo "# TEST:  $TEST_CMD"; } > "$LOG"
    env $RUN_ENV bash -c "$TRAIN_CMD" >> "$LOG" 2>&1 && env $RUN_ENV bash -c "$TEST_CMD" >> "$LOG" 2>&1
    rc=$?
    [ "$STAGE" = target ] && [ -z "${KEEP_CK:-}" ] && [ -s "$RESULT" ] && rm -f "$CKPT"
    echo "  [$(date '+%F %T')] $RUN f$F xong rc=$rc $(grep -o '"test_roc_auc": [0-9.]*' "$RESULT" 2>/dev/null)" | tee -a "$OUT/state/driver_$HOST.log"
  done
done
echo "########## FPE $HOST XONG $(date '+%F %T') ##########"
