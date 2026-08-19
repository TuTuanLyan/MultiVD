#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
source "${CONFIG_FILE:-$ROOT/run/config.sh}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

SEED="${1:?usage: bash run/train-phase1.sh SEED}"
LOG_DIR="log/$RUN_NAME/$TRANSFER_NAME/seed_$SEED"
MODEL_DIR="model/$RUN_NAME/$TRANSFER_NAME/seed_$SEED"
RESULT_DIR="results/$RUN_NAME/$TRANSFER_NAME/seed_$SEED"
mkdir -p "$LOG_DIR" "$MODEL_DIR/source" "$RESULT_DIR"

python -u src/train_transfer.py \
  --phase phase1 \
  --run_name "$RUN_NAME" \
  --method_name "$TRANSFER_NAME" \
  --data_path "$PHASE1_DATA_PATH" \
  --seed "$SEED" \
  --epochs "$PHASE1_EPOCHS" \
  --min_epochs "$MIN_EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --eval_batch_size "$EVAL_BATCH_SIZE" \
  --max_length "$MAX_LENGTH" \
  --truncation_strategy "$TRUNCATION_STRATEGY" \
  --learning_rate "$PHASE1_LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --lambda_cwe "$LAMBDA_CWE" \
  --aux_mode "$AUX_MODE" \
  --num_latent "$NUM_LATENT" \
  --latent_temperature "$LATENT_TEMPERATURE" \
  --patience "$PATIENCE" \
  --max_grad_norm "$MAX_GRAD_NORM" \
  --num_workers "$NUM_WORKERS" \
  --model_name "$MODEL_NAME" \
  --output_dir "results/$RUN_NAME/$TRANSFER_NAME" \
  --checkpoint_path "$MODEL_DIR/source/best.pt" \
  2>&1 | tee -a "$LOG_DIR/phase1.log"
