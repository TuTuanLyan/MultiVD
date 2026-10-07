#!/usr/bin/env bash
# MW_ASSEMBLE tren t5p — DUNG co chuan §B.3, chi doi backbone.
#
#   mwK8 : multi-window, khoi tao tu codet5p-220m-bimodal goc          (= "MW")
#   mwTR : GIONG HET mwK8, khac DUNG MOT thu la --init_ckpt Pha 1      (= "MW+TR")
#
# baseline KHONG chay lai: kho local da co 26 cay du 5 fold cung chu ky sieu tham so
# (nguoi dung 17/09). Ghep cap doc tu cay baseline duoc chot TRUOC khi nhin Delta.
#
# LUU Y doc ket qua: codet5p dung relative position bucket nen KHONG bi tran 514 vi tri
# nhu codebert. Vay tren t5p, MW khong con la "vuot tran" ma la "chia khuc roi gop" —
# cau hoi khac, phai ghi ro khi bao cao.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${HF_HOME:-}" ] && [ -d /workspace ]; then export HF_HOME=/workspace/.hf_home; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-python3}"
RN="${RUN_NAME:-mw_assemble_t5p}"
MODEL="${MODEL_NAME:-Salesforce/codet5p-220m-bimodal}"
POOL="${POOLING:-mean}"
CKPT="${INIT_CKPT:-model/shuf1/phase1/t5p__none_com_real/seed_42/best.pt}"
FOLDS_LIST="${FOLDS:-1 2 3 4 5}"
SEED="${SEED:-42}"
ARMS="${ARMS:-mwK8 mwTR}"
NEED_FREE="${NEED_FREE:-9000}"
WAIT_MAX="${WAIT_MAX:-14400}"

exec 4>/tmp/mvd_mwt5p.lock || exit 1
flock -n 4 || { echo "DA CO mw_t5p dang chay"; exit 3; }
PIDFILE="${PIDFILE:-/tmp/mw_t5p.pid}"; echo "$$" > "$PIDFILE" 2>/dev/null || true
trap 'rm -f "$PIDFILE"' EXIT
ts(){ date -u '+%F %T'; }

[ -f src/train_mw.py ] || { echo "!! THIEU src/train_mw.py"; exit 5; }
[ -f "$CKPT" ]        || { echo "!! THIEU checkpoint Pha 1: $CKPT"; exit 5; }
[ -f data/sven_python_folds_norm/fold5/test.jsonl ] || { echo "!! THIEU du lieu dich"; exit 5; }
"$PY" -c "import torch,transformers,sklearn;print('env OK',torch.__version__)" || {
  echo "!! PYTHON='$PY' khong import duoc torch"; exit 5; }
MODEL="$MODEL" "$PY" - <<'PYCHECK' || { echo "!! KHONG nap duoc model offline"; exit 5; }
import os, sys; sys.path.insert(0, "src")
from transformers import AutoTokenizer
from model import build_backbone
m = os.environ["MODEL"]
AutoTokenizer.from_pretrained(m); build_backbone(m)
print("  HF offline OK:", m)
PYCHECK

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

echo "########## MW_T5P bat dau $(ts) | $(hostname) | fold: $FOLDS_LIST | arms: $ARMS ##########"
echo "  model: $MODEL | pooling: $POOL | ckpt TR: $CKPT"
for FOLD in $FOLDS_LIST; do                       # FOLD la vong NGOAI
  for NAME in $ARMS; do
    OUT="results/${RN}/${NAME}/seed_${SEED}/fold${FOLD}.json"
    if [ -f "$OUT" ]; then echo "=== $(ts) | fold $FOLD | $NAME | da co ==="; continue; fi
    wait_gpu || exit 1
    EXTRA=(); [ "$NAME" = "mwTR" ] && EXTRA=(--init_ckpt "$CKPT")
    echo "===== $(ts) | FOLD $FOLD | $NAME ${EXTRA[*]:-} ====="
    mkdir -p "log/${RN}"
    "$PY" -u src/train_mw.py \
      --run_name "$RN" --method_name "$NAME" --fold "$FOLD" --seed "$SEED" \
      --model_name "$MODEL" --pooling "$POOL" "${EXTRA[@]}" \
      --window 510 --stride 384 --max_windows 8 --max_length 512 --agg mean --window_mode slide \
      --batch_size 4 --eval_batch_size 8 --micro 16 \
      --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.10 \
      --epochs 30 --min_epochs 3 --patience 8 --selection_metric roc_auc --sam_rho 0.02 \
      >> "log/${RN}/${NAME}_fold${FOLD}.log" 2>&1
    if [ -f "$OUT" ]; then echo "  $(ts) $NAME fold $FOLD XONG"
    else echo "  !! $NAME fold $FOLD THAT BAI — log/${RN}/${NAME}_fold${FOLD}.log"
         tail -5 "log/${RN}/${NAME}_fold${FOLD}.log" | sed 's/^/       /'; fi
  done
  echo "----- $(ts) | het fold $FOLD | o: $(find results/${RN} -name 'fold*.json' 2>/dev/null | wc -l)/10 -----"
done
echo "########## MW_T5P xong $(ts) | $(find results/${RN} -name 'fold*.json' 2>/dev/null | wc -l)/10 o ##########"
