#!/usr/bin/env bash
# BASELINE cua ho: CodeBERT thuan 512, AdamW, BATCH 4 — sao y `qb4.sh`, tung co mot.
# Ho cong bo n=5: 0.8868 / 0.8936 / 0.8981 / 0.9319 / 0.8564 -> 0.8934
# Day moi la doi chung DUNG cho lop phuong phap nay; baseline batch 16 cua ta thap hon ~0.022 ROC.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/workspace/MultiVD/.hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
PY="${PYTHON:-/venv/main/bin/python}"; S=36; TAG="${TAG:-b4_sven}"
R="${R:-data/sven_python_folds_norm}"; TLANG="${TLANG:-python}"; FOLDS="${FOLDS:-1 2 3 4 5}"
M="${MODEL_NAME:-microsoft/codebert-base}"
exec 4>"${LOCKF:-/tmp/mvd_qb4.lock}" || exit 1
flock -n 4 || { echo "DA CO qb4 dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }
BASE="--seed $S --batch_size 4 --eval_batch_size 16 --max_length 512 --truncation_strategy head_middle_tail
      --weight_decay 0.01 --patience 8 --min_epochs 3 --max_grad_norm 1.0 --num_workers 0
      --model_name $M --pooling cls --selection_metric roc_auc"
mkdir -p log/$TAG results/$TAG/baseline/seed_$S
echo "########## QB4 bat dau $(ts) | $TAG | fold: $FOLDS ##########"
for F in $FOLDS; do
  OUT="results/$TAG/baseline/seed_$S/fold$F.json"
  [ -f "$OUT" ] && { echo "=== fold $F da co ==="; continue; }
  BM=model/$TAG/baseline/seed_$S/fold$F; mkdir -p "$BM"
  echo "===== $(ts) | FOLD $F ====="
  "$PY" -u src_mwg/train_baseline.py --phase train --run_name $TAG --method_name baseline --fold $F \
      --epochs 30 --learning_rate 2e-5 --checkpoint_path "$BM/best.pt" --data_root "$R" \
      --target_lang $TLANG --warmup_ratio 0.10 $BASE > log/$TAG/f$F.log 2>&1 \
   && "$PY" -u src_mwg/train_baseline.py --phase test --run_name $TAG --method_name baseline --fold $F \
      --checkpoint_path "$BM/best.pt" --data_root "$R" --target_lang $TLANG --warmup_ratio 0.10 \
      --result_path "$OUT" $BASE >> log/$TAG/f$F.log 2>&1 \
   || { echo "  !! fold $F LOI"; tail -5 log/$TAG/f$F.log | sed 's/^/     /'; }
  [ -f "$OUT" ] && { echo "  $(ts) fold $F XONG: $(grep -o '\"test_roc_auc\": [0-9.]*' "$OUT" | head -1)"; rm -f "$BM/best.pt"; }
done
echo "########## QB4 xong $(ts) | $(ls results/$TAG/baseline/seed_$S/fold*.json 2>/dev/null | wc -l) o ##########"
