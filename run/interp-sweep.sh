#!/usr/bin/env bash
# How much Phase 1 can a strong backbone absorb before it starts losing?
#
# Phase 1 helps CodeBERT and hurts CodeT5+, so instead of choosing between
# running it and skipping it, dial it: blend the Phase-1 backbone back toward
# its original pretrained weights at several mixing ratios and see where the
# delta crosses zero. alpha=1.0 is the plain transfer already measured; alpha=0
# would be the baseline.
#
# Every arm reuses the one Phase-1 checkpoint and the one baseline already
# measured on this machine and these folds, so the sweep costs Phase 2 only.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /venv/main/bin/activate
export HF_HOME=/workspace/hf PYTHON=python

RUN=${RUN_NAME:-t5p_twin}
SEED=${SEED:-36}
SRC="model/$RUN/transfer_cwe/seed_$SEED/source/best.pt"
if [[ ! -f "$SRC" ]]; then echo "missing source checkpoint: $SRC"; exit 1; fi

export DATA_ROOT=data/sven_python_twin
export RUN_NAME="$RUN"
export SEED
export MODEL_NAME=${MODEL_NAME:-Salesforce/codet5p-220m}
export POOLING=${POOLING:-mean}

for A in 0.75 0.50 0.25; do
  TAG="cwe+a${A/./}"
  DST="model/$RUN/transfer_$TAG/seed_$SEED/source"
  mkdir -p "$DST"
  [[ -f "$DST/best.pt" ]] || cp "$SRC" "$DST/best.pt"
  echo "=== alpha=$A -> transfer_$TAG ==="
  SOURCE_INTERPOLATION="$A" FOLDS="1 2 3" MODES="$TAG" bash run/fold-major.sh
  python src/report_matrix.py
done
touch /workspace/INTERP_3FOLD
echo "=== INTERPOLATION SWEEP, 3 FOLDS DONE ==="
python src/report_matrix.py
