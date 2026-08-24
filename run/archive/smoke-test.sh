#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "${CONFIG_FILE:-$ROOT/run/config.sh}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

SEED="${1:-42}"
LOG_DIR="log/$RUN_NAME/$TRANSFER_NAME/smoke/seed_$SEED"
SMOKE_MODEL="model/$RUN_NAME/$TRANSFER_NAME/smoke/seed_$SEED"
SMOKE_RESULT="results/$RUN_NAME/$TRANSFER_NAME/smoke/seed_$SEED"
mkdir -p "$LOG_DIR" "$SMOKE_MODEL/source" "$SMOKE_MODEL/fold1" "$SMOKE_RESULT"

COMMON=(--seed "$SEED" --epochs 1 --min_epochs 1 --patience 1 --batch_size "$BATCH_SIZE" \
  --eval_batch_size "$EVAL_BATCH_SIZE" --max_length "$SMOKE_MAX_LENGTH" \
  --truncation_strategy "$TRUNCATION_STRATEGY" \
  --num_workers "$NUM_WORKERS" --model_name "$MODEL_NAME" \
  --max_train_samples "$SMOKE_TRAIN_SAMPLES" --max_eval_samples "$SMOKE_EVAL_SAMPLES")

python -u src/train_transfer.py --phase phase1 "${COMMON[@]}" \
  --run_name "$RUN_NAME" --method_name "$TRANSFER_NAME" \
  --data_path "$PHASE1_DATA_PATH" \
  --checkpoint_path "$SMOKE_MODEL/source/best.pt" \
  --output_dir "$SMOKE_RESULT" \
  2>&1 | tee -a "$LOG_DIR/phase1.log"

python -u src/train_transfer.py --phase phase2 --fold 1 "${COMMON[@]}" \
  --run_name "$RUN_NAME" --method_name "$TRANSFER_NAME" \
  --source_checkpoint "$SMOKE_MODEL/source/best.pt" \
  --checkpoint_path "$SMOKE_MODEL/fold1/best.pt" \
  --anneal_fun "$ANNEAL_FUN" --anneal_k "$ANNEAL_K" \
  --anneal_t0_ratio "$ANNEAL_T0_RATIO" --anneal_w "$ANNEAL_W" \
  --pretrain_cof "$PRETRAIN_COF" \
  --output_dir "$SMOKE_RESULT" \
  2>&1 | tee -a "$LOG_DIR/phase2.log"

python -u src/train_transfer.py --phase test --fold 1 "${COMMON[@]}" \
  --run_name "$RUN_NAME" --method_name "$TRANSFER_NAME" \
  --source_checkpoint "$SMOKE_MODEL/source/best.pt" \
  --checkpoint_path "$SMOKE_MODEL/fold1/best.pt" \
  --result_path "$SMOKE_RESULT/fold1.json" \
  --output_dir "$SMOKE_RESULT" \
  2>&1 | tee -a "$LOG_DIR/test.log"

echo "Smoke test result: $SMOKE_RESULT/fold1.json"
