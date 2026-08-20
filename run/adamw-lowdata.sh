#!/usr/bin/env bash
# Is RecAdam what breaks the transfer on a small target set?
#
# At 114 target rows the transfer loses 0.105 to its own baseline while its test
# probabilities compress into [0.32, 0.83], standard deviation 0.116 against the
# baseline 0.431. That is an under-trained model rather than a collapsed one:
# 114 rows give 240 optimizer steps and the best checkpoint lands at step 56,
# while the RecAdam annealing weight -- calibrated to total_steps -- is still
# small enough to hold the weights at the source solution.
#
# The AdamW arm reuses the SAME Phase-1 checkpoint, so the optimizer is the only
# thing that differs between the two arms.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /venv/main/bin/activate
export HF_HOME=/workspace/hf PYTHON=python
export DATA_ROOT=data/sven_python_twin
export RUN_NAME=frac114_codebert-base
export SEED=${SEED:-36}
export MODEL_NAME=microsoft/codebert-base
export POOLING=cls

SRC="model/$RUN_NAME/transfer_cwe/seed_$SEED/source/best.pt"
DST="model/$RUN_NAME/transfer_cwe+adamw/seed_$SEED/source"
if [[ ! -f "$SRC" ]]; then
  echo "missing $SRC -- the RecAdam arm must run first"
  exit 1
fi
mkdir -p "$DST" && cp "$SRC" "$DST/best.pt"
echo "seeded the AdamW arm from the RecAdam arm Phase 1"

PHASE2_OPTIMIZER=adamw MAX_TRAIN_SAMPLES=114 FOLDS="1 2 3" MODES="cwe+adamw" \
  bash run/fold-major.sh
echo "=== 114 rows: RecAdam versus AdamW ==="
python src/report_matrix.py
touch /workspace/ADAMW_DONE
