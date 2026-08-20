#!/usr/bin/env bash
# Was Phase 1 the culprit? Constrain it to a rank-r update and see if the
# negative transfer on a strong backbone disappears.
#
# Phase 2 is deliberately left alone: the baseline also fine-tunes the whole
# model on the target and does fine (0.8879), so the target stage handles either
# starting point. What differs between baseline and transfer is only where
# Phase 1 left the weights.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /venv/main/bin/activate
export HF_HOME=/workspace/hf PYTHON=python
export DATA_ROOT=data/sven_python_twin
export RUN_NAME=${RUN_NAME:-t5p_twin}
export SEED=${SEED:-36}
export MODEL_NAME=${MODEL_NAME:-Salesforce/codet5p-220m}
export POOLING=${POOLING:-mean}

# Rank 8 is a low-capacity update; rank 32 lets Phase 1 teach more while still
# staying far below full fine-tuning.
for R in 8 32; do
  echo "=== LoRA rank $R ==="
  LORA_RANK="$R" FOLDS="1 2 3" MODES="cwe+lora$R" bash run/fold-major.sh
  python src/report_matrix.py
done
touch /workspace/LORA_3FOLD
echo "=== LoRA SWEEP, 3 FOLDS DONE ==="
