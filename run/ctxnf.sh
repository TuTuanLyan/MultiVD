#!/usr/bin/env bash
# CTXNF — SAN NHIEU cho §62. Hai lan chay CUNG max_length, chi khac HAT GIONG.
#
# §62 do duoc 1024 hon 512 la +0.0359 ROC. Nhung khong co moc so: doi chung "hang <= 256
# token" ma toi thiet ke KHONG phai san nhieu (ba model khac nhau vi train bi cat khac nhau).
# San nhieu dung nghia = chay lai DUNG mot cau hinh voi hat giong khac. Khoi nay do no.
# TU CHUA: ca hai seed chay tren CUNG may CUNG phien, khong ghep cap qua may.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${HF_HOME:-}" ] && [ -d /workspace ]; then export HF_HOME=/workspace/.hf_home; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-python3}"
FOLDS_LIST="${FOLDS:-1 2 3}"
SEEDS="${SEEDS:-42 7}"
L="${MAXLEN:-512}"
BB="t5p=Salesforce/codet5p-220m-bimodal:mean"

exec 4>/tmp/mvd_ctxnf.lock || exit 1
flock -n 4 || { echo "DA CO ctxnf dang chay"; exit 3; }
PIDFILE="${PIDFILE:-/tmp/ctxnf.pid}"; echo "$$" > "$PIDFILE" 2>/dev/null || true
trap 'rm -f "$PIDFILE"' EXIT
ts(){ date -u '+%F %T'; }

# Card dung chung voi nguoi khac. Phong vao luc sat tran se OOM CA HAI job.
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "GPU trong: ${FREE} MiB"
(( FREE >= 7500 )) || { echo "!! chi con ${FREE} MiB (<7500) — KHONG phong"; exit 1; }
"$PY" -c "import torch,transformers,sklearn;print('env OK',torch.__version__)" || exit 5

echo "########## CTXNF bat dau $(ts) | $(hostname) | max_length $L | seed: $SEEDS ##########"
for FOLD in $FOLDS_LIST; do
  for S in $SEEDS; do
    echo "===== $(ts) | FOLD $FOLD | seed $S | max_length $L ====="
    BASELINE_ONLY=1 SKIP_BASELINE=0 RUN_NAME="ctxnf$L" SEED="$S" FOLDS="$FOLD" \
    BACKBONES="$BB" MODES=none OPTIMIZERS=adamw MAX_LENGTH="$L" \
    BASELINE_EXTRA="--grad_checkpointing" \
    DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
    PYTHON="$PY" bash run/matrix.sh 4>&-
  done
done
echo "########## CTXNF xong $(ts) | $(find results/ctxnf${L}_t5p -name 'fold*.json' 2>/dev/null | wc -l)/6 o ##########"
