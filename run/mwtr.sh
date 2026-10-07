#!/usr/bin/env bash
# MW+TR — nhanh multi-window KHOI TAO TU CHECKPOINT PHA 1.
#
# Giong mwK8 TUNG THAM SO, khac DUNG MOT thu: --init_ckpt. Do la phep tach duy nhat
# cho biet rieng phan CHUYEN GIAO dang bao nhieu khi da co cua so.
#
# Dung de LAP DAY fold 4-5 cua khoi mw_assemble (fold 1-3 da chay tren vast 5060 Ti).
# Fold la vong NGOAI; moi fold xong la mot lat cat ghep cap duoc.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${HF_HOME:-}" ] && [ -d /workspace ]; then export HF_HOME=/workspace/.hf_home; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-python3}"
RN="${RUN_NAME:-mw_assemble_codebert}"
NAME="${METHOD_NAME:-mwTR}"
FOLDS_LIST="${FOLDS:-4 5}"
SEED="${SEED:-42}"
CKPT="${INIT_CKPT:-model/shuf1/phase1/codebert__none_com_real/seed_42/best.pt}"
NEED_FREE="${NEED_FREE:-7000}"        # MiB toi thieu tren card truoc khi phong
WAIT_MAX="${WAIT_MAX:-7200}"          # giay cho toi da khi card ban

exec 4>/tmp/mvd_mwtr.lock || exit 1
flock -n 4 || { echo "DA CO mwtr dang chay"; exit 3; }
PIDFILE="${PIDFILE:-/tmp/mwtr.pid}"; echo "$$" > "$PIDFILE" 2>/dev/null || true
trap 'rm -f "$PIDFILE"' EXIT
ts(){ date -u '+%F %T'; }

[ -f src/train_mw.py ] || { echo "!! THIEU src/train_mw.py"; exit 5; }
[ -f "$CKPT" ]        || { echo "!! THIEU checkpoint Pha 1: $CKPT"; exit 5; }
[ -f data/sven_python_folds_norm/fold5/test.jsonl ] || { echo "!! THIEU du lieu dich"; exit 5; }
"$PY" -c "import torch,transformers,sklearn;print('env OK',torch.__version__)" || {
  echo "!! PYTHON='$PY' khong import duoc torch. Vd: PYTHON=/home/ntat/miniconda3/envs/vdenv/bin/python"; exit 5; }
"$PY" - <<'PYCHECK' || { echo "!! KHONG nap duoc model offline"; exit 5; }
from transformers import AutoTokenizer, AutoConfig
AutoTokenizer.from_pretrained("microsoft/codebert-base"); AutoConfig.from_pretrained("microsoft/codebert-base")
print("  HF offline OK")
PYCHECK

# Cong GPU: CHO, khong gianh. Card 161 dung chung voi cuongtm/tranmanhcuong/anhnd_02/ollama.
wait_gpu(){
  local t=0 free
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [ -n "${free:-}" ] || { echo "  khong doc duoc nvidia-smi — phong lieu"; return 0; }
    (( free >= NEED_FREE )) && { echo "  GPU trong ${free} MiB — phong"; return 0; }
    (( t >= WAIT_MAX )) && { echo "  !! cho ${t}s ma card van chi con ${free} MiB — BO"; return 1; }
    echo "  $(ts) card con ${free} MiB (<${NEED_FREE}) — cho 120s"; sleep 120; t=$((t+120))
  done
}

echo "########## MWTR bat dau $(ts) | $(hostname) | fold: $FOLDS_LIST | seed $SEED ##########"
echo "  ckpt Pha 1: $CKPT"
for FOLD in $FOLDS_LIST; do
  OUT="results/${RN}/${NAME}/seed_${SEED}/fold${FOLD}.json"
  if [ -f "$OUT" ]; then echo "=== $(ts) | fold $FOLD | da co — bo qua ==="; continue; fi
  wait_gpu || exit 1
  echo "===== $(ts) | FOLD $FOLD | $NAME ====="
  mkdir -p "log/${RN}"
  "$PY" -u src/train_mw.py \
    --run_name "$RN" --method_name "$NAME" --fold "$FOLD" --seed "$SEED" \
    --init_ckpt "$CKPT" \
    --window 510 --stride 384 --max_windows 8 --max_length 512 --agg mean --window_mode slide \
    --batch_size 4 --eval_batch_size 8 --micro 16 \
    --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.10 \
    --epochs 30 --min_epochs 3 --patience 8 --selection_metric roc_auc --sam_rho 0.02 \
    >> "log/${RN}/${NAME}_fold${FOLD}.log" 2>&1
  if [ -f "$OUT" ]; then echo "  $(ts) fold $FOLD XONG"
  else echo "  !! fold $FOLD THAT BAI — log/${RN}/${NAME}_fold${FOLD}.log"
       tail -5 "log/${RN}/${NAME}_fold${FOLD}.log" | sed 's/^/       /'; fi
done
echo "########## MWTR xong $(ts) | co $(ls results/${RN}/${NAME}/seed_${SEED}/fold*.json 2>/dev/null | wc -l)/5 o ##########"
