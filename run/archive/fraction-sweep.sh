#!/usr/bin/env bash
# Does the transfer earn its keep when target data is scarce?
#
# Every measurement so far uses all 456 training rows per fold. The literature
# on intermediate-task transfer (Vu et al. EMNLP 2020, STILTs) reports the
# benefit concentrating in the low-resource regime, and this project's own
# per-CWE split points the same way: the gain sits on the two rare CWEs and
# vanishes on the one with 408 test samples.
#
# Baseline and transfer are both capped at the same fraction, so each fraction
# is a self-contained comparison. Folds 1-3 first, for an early read.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /venv/main/bin/activate
export HF_HOME=/workspace/hf PYTHON=python
export DATA_ROOT=data/sven_python_twin
export SEED=${SEED:-36}
export MODEL_NAME=${MODEL_NAME:-microsoft/codebert-base}
export POOLING=${POOLING:-cls}

for N in 114 228; do          # 25% and 50% of the 456 rows per fold
  export RUN_NAME="frac${N}_$(basename "$MODEL_NAME")"
  echo "=== target train rows: $N | run: $RUN_NAME ==="
  MAX_TRAIN_SAMPLES="$N" FOLDS="1 2 3" MODES="cwe" bash run/fold-major.sh
  python src/report_matrix.py
done
touch /workspace/FRAC_DONE
echo "=== FRACTION SWEEP DONE ==="
