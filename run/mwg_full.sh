#!/usr/bin/env bash
# KHOI GOP DU BA: BABEL (do thi dong) + MULTI-WINDOW + TRANSFER.
# Tai lap cau hinh chot cua dong nghiep `mwgU_xAsRl_jsCjv2sven` (ho cong bo ROC 0.9341, n=5).
#
#   Pha 1: mwgp1U_jsCjv   tren data/mwsrc_jsCjv (JS common 670 + Java 4906, co pair_id)
#          pair loss BAT BUOC — thieu no Pha 1 gan nhu khong hoc (val 0.50-0.57, §10.1 cua ho)
#          --epochs 8 cat cung vi Pha 1 overfit tu epoch 5
#   Pha 2: mwgU_xAsRl_jsCjv2sven  — nap CA (encoder + MW + do thi), SAM 0.02, RecAdam-light
#
# Chay bang MA CUA HO (src_mwg/). Seed 36, dung co cua ho.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/workspace/MultiVD/.hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-/venv/main/bin/python}"
S="${SEED:-36}"; M="${MODEL_NAME:-microsoft/codebert-base}"
P1TAG="${P1TAG:-mwgp1U_jsCjv}"; SRC="${SRC:-data/mwsrc_jsCjv}"
P2TAG="${P2TAG:-mwgU_xAsRl_jsCjv2sven}"; FOLDS="${FOLDS:-1 2 3 4 5}"
U="--drop_bracket 1 --co_mode chain"
R2=data/sven_python_folds_norm

exec 4>"${LOCKF:-/tmp/mvd_mwgfull.lock}" || exit 1
flock -n 4 || { echo "DA CO mwg_full dang chay"; exit 3; }
PIDFILE="${PIDFILE:-/tmp/mwgfull.pid}"; echo "$$" > "$PIDFILE"; trap 'rm -f "$PIDFILE"' EXIT
ts(){ date -u '+%F %T'; }

for f in src_mwg/train_mwg.py "$SRC/fold1/train.jsonl" "$R2/fold1/test.jsonl"; do
  [ -f "$f" ] || { echo "!! THIEU $f"; exit 5; }
done
MODEL_NAME="$M" "$PY" - <<'PYCHK' || { echo "!! KHONG nap duoc model offline"; exit 5; }
import os, sys; sys.path.insert(0, "src_mwg")
from transformers import AutoTokenizer
from model import build_backbone
m = os.environ["MODEL_NAME"]; AutoTokenizer.from_pretrained(m); build_backbone(m)
print(f"  model offline OK: {m}")
PYCHK

COMMON="--seed $S --model_name $M --window 510 --stride 384 --max_windows 8
        --batch_size 4 --eval_batch_size 8 --micro 16 --learning_rate 2e-5 --weight_decay 0.01
        --warmup_ratio 0.10 --min_epochs 3 --patience 8 --selection_metric roc_auc
        --num_workers 0 --agg mean"

# ───────────────────────────── PHA 1 ─────────────────────────────
CK="model/$P1TAG/multiwindow/seed_$S/fold1/best.pt"
mkdir -p log/$P1TAG results/$P1TAG/multiwindow/seed_$S "$(dirname "$CK")"
if [ -f "$CK" ]; then
  echo "########## PHA 1 da co checkpoint, bo qua: $CK ##########"
else
  echo "########## PHA 1 bat dau $(ts) | $P1TAG | nguon $SRC ##########"
  "$PY" -u src_mwg/train_mwg.py --phase train --run_name "$P1TAG" --fold 1 --data_root "$SRC" \
      --checkpoint_path "$CK" $COMMON $U --epochs 8 --pair_loss 1.0 --pair_margin 1.0 --sam_rho 0 \
      > log/$P1TAG/f1.log 2>&1
  if [ ! -f "$CK" ]; then echo "!! PHA 1 KHONG ra checkpoint"; tail -6 log/$P1TAG/f1.log | sed 's/^/   /'; exit 1; fi
  echo "  $(ts) PHA 1 xong | $(grep 'Best checkpoint' log/$P1TAG/f1.log | tail -1)"
fi

# ───────────────────────────── PHA 2 ─────────────────────────────
echo "########## PHA 2 bat dau $(ts) | $P2TAG | fold: $FOLDS ##########"
mkdir -p log/$P2TAG results/$P2TAG/multiwindow/seed_$S
X2="$U --recadam 1 --pretrain_cof 500 --anneal_t0_ratio 0.01"
for F in $FOLDS; do
  OUT="results/$P2TAG/multiwindow/seed_$S/fold$F.json"
  [ -f "$OUT" ] && { echo "=== $(ts) | fold $F da co ==="; continue; }
  CK2="model/$P2TAG/multiwindow/seed_$S/fold$F/best.pt"; mkdir -p "$(dirname "$CK2")"
  echo "===== $(ts) | FOLD $F ====="
  "$PY" -u src_mwg/train_mwg.py --phase train --run_name "$P2TAG" --fold "$F" --data_root "$R2" \
      --target_lang python --checkpoint_path "$CK2" --init all --init_ckpt "$CK" \
      $COMMON --epochs 30 --sam_rho 0.02 $X2 > log/$P2TAG/f$F.log 2>&1 \
   && "$PY" -u src_mwg/train_mwg.py --phase test --run_name "$P2TAG" --fold "$F" --data_root "$R2" \
      --target_lang python --checkpoint_path "$CK2" --result_path "$OUT" \
      $COMMON --epochs 30 --sam_rho 0.02 $X2 >> log/$P2TAG/f$F.log 2>&1 \
   || { echo "  !! fold $F LOI"; tail -6 log/$P2TAG/f$F.log | sed 's/^/     /'; }
  if [ -f "$OUT" ]; then
    echo "  $(ts) fold $F XONG: $(grep -o '\"test_roc_auc\": [0-9.]*' "$OUT" | head -1)"
    rm -f "$CK2"
  fi
done
N=$(ls results/$P2TAG/multiwindow/seed_$S/fold*.json 2>/dev/null | wc -l)
echo "########## MWG_FULL xong $(ts) | $N o ##########"
