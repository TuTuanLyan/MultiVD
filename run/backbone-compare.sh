#!/usr/bin/env bash
# Swap the pretrained backbone, keep the v1 method otherwise identical.
#
# Folds are interleaved: baseline and method run back to back on the same fold
# before moving to the next one, so a paired delta is readable after fold 1
# instead of after all ten runs. Same seed and same folds throughout.
#
# T5-family checkpoints contribute their encoder only and pool by mean, since
# they have no CLS token.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"
MODEL_NAME="${MODEL_NAME:-Salesforce/codet5-base}"
POOLING="${POOLING:-mean}"
SEED="${SEED:-36}"
RUN_NAME="${RUN_NAME:-backbone_codet5}"
DATA="${PHASE1_DATA_PATH:-data/train_ccpp_js.jsonl}"
FOLDS="${FOLDS:-1 2 3 4 5}"

# v1 settings, unchanged.
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-15}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-30}"
LR="${LR:-2e-5}"
LAMBDA_CWE="${LAMBDA_CWE:-0.2}"

SHARED=(
  --seed "$SEED" --model_name "$MODEL_NAME" --pooling "$POOLING"
  --batch_size "$BATCH_SIZE" --eval_batch_size "$EVAL_BATCH_SIZE"
  --max_length "$MAX_LENGTH" --truncation_strategy head_middle_tail
  --weight_decay 0.01 --patience 5 --min_epochs 3 --max_grad_norm 1.0 --num_workers 0
)

LOG="log/$RUN_NAME/seed_$SEED";      mkdir -p "$LOG"
TMODEL="model/$RUN_NAME/transfer/seed_$SEED"; mkdir -p "$TMODEL/source"
BMODEL="model/$RUN_NAME/baseline/seed_$SEED"; mkdir -p "$BMODEL"
TRES="results/$RUN_NAME/transfer/seed_$SEED"; mkdir -p "$TRES"
BRES="results/$RUN_NAME/baseline/seed_$SEED"; mkdir -p "$BRES"

echo "### backbone=$MODEL_NAME pooling=$POOLING seed=$SEED run=$RUN_NAME"

if [[ -f "$TMODEL/source/best.pt" ]]; then
  echo "=== $(date '+%F %T') | phase1 already present, skipping ==="
else
echo "=== $(date '+%F %T') | phase1 (source pretraining, once) ==="
$PYTHON -u src/train_transfer.py --phase phase1 \
  --run_name "$RUN_NAME" --method_name transfer --data_path "$DATA" \
  --aux_mode cwe --epochs "$PHASE1_EPOCHS" --learning_rate "$LR" --lambda_cwe "$LAMBDA_CWE" \
  --checkpoint_path "$TMODEL/source/best.pt" \
  "${SHARED[@]}" >> "$LOG/phase1.log" 2>&1
if [[ $? -ne 0 ]]; then echo "PHASE1 FAILED, aborting"; tail -20 "$LOG/phase1.log"; exit 1; fi
echo "phase1 done: $(grep 'Best checkpoint saved' "$LOG/phase1.log" | tail -1 | sed 's/.*| Epoch/Epoch/')"
fi

for FOLD in $FOLDS; do
  if [[ -f "$TRES/fold$FOLD.json" && -f "$BRES/fold$FOLD.json" ]]; then
    echo "=== $(date '+%F %T') | fold $FOLD | already complete, skipping ==="
    continue
  fi
  echo "=== $(date '+%F %T') | fold $FOLD | baseline ==="
  mkdir -p "$BMODEL/fold$FOLD"
  $PYTHON -u src/train_baseline.py --phase train \
    --run_name "$RUN_NAME" --method_name baseline --fold "$FOLD" \
    --epochs "$PHASE2_EPOCHS" --learning_rate "$LR" \
    --checkpoint_path "$BMODEL/fold$FOLD/best.pt" \
    "${SHARED[@]}" >> "$LOG/baseline_train_fold$FOLD.log" 2>&1 \
    && $PYTHON -u src/train_baseline.py --phase infer \
      --run_name "$RUN_NAME" --method_name baseline --fold "$FOLD" \
      --checkpoint_path "$BMODEL/fold$FOLD/best.pt" \
      "${SHARED[@]}" >> "$LOG/baseline_infer_fold$FOLD.log" 2>&1 \
    || echo "  baseline fold$FOLD FAILED"

  echo "=== $(date '+%F %T') | fold $FOLD | method (transfer + RecAdam) ==="
  mkdir -p "$TMODEL/fold$FOLD"
  $PYTHON -u src/train_transfer.py --phase phase2 \
    --run_name "$RUN_NAME" --method_name transfer --fold "$FOLD" --aux_mode cwe \
    --epochs "$PHASE2_EPOCHS" --learning_rate "$LR" \
    --source_checkpoint "$TMODEL/source/best.pt" \
    --checkpoint_path "$TMODEL/fold$FOLD/best.pt" \
    --output_dir "results/$RUN_NAME/transfer" \
    "${SHARED[@]}" >> "$LOG/transfer_phase2_fold$FOLD.log" 2>&1 \
    && $PYTHON -u src/train_transfer.py --phase test \
      --run_name "$RUN_NAME" --method_name transfer --fold "$FOLD" --aux_mode cwe \
      --checkpoint_path "$TMODEL/fold$FOLD/best.pt" \
      --output_dir "results/$RUN_NAME/transfer" \
      "${SHARED[@]}" >> "$LOG/transfer_test_fold$FOLD.log" 2>&1 \
    || echo "  method fold$FOLD FAILED"

  # Fold checkpoints are ~500MB each and are not needed once the fold's result
  # JSON exists. Keeping them all fills a 20GB disk before fold 5.
  if [[ -f "$TRES/fold$FOLD.json" ]]; then rm -f "$TMODEL/fold$FOLD/best.pt"; fi
  if [[ -f "$BRES/fold$FOLD.json" ]]; then rm -f "$BMODEL/fold$FOLD/best.pt"; fi

  # Paired delta for this fold, printed as soon as the pair exists.
  $PYTHON - "$BRES/fold$FOLD.json" "$TRES/fold$FOLD.json" "$FOLD" <<'PYEOF'
import json, sys
try:
    b = json.load(open(sys.argv[1]))
    t = json.load(open(sys.argv[2]))
except (OSError, json.JSONDecodeError) as error:
    print(f"  fold {sys.argv[3]}: no paired result yet ({error})")
    raise SystemExit(0)
for key in ("test_macro_f1_at_0.5", "test_macro_f1_at_valcal", "test_roc_auc"):
    bv, tv = b.get(key), t.get(key)
    if bv is not None and tv is not None:
        print(f"  fold {sys.argv[3]} | {key:26s} baseline={bv:.4f} method={tv:.4f} delta={tv - bv:+.4f}")
PYEOF
done

for DIR in "$TRES" "$BRES"; do
  $PYTHON -u src/summarize_results.py --input_dir "$DIR" --output_dir "$DIR" \
    >> "$LOG/summary.log" 2>&1 || echo "summarize failed for $DIR"
done
echo "=== $(date '+%F %T') | DONE $RUN_NAME | total ${SECONDS}s ==="
