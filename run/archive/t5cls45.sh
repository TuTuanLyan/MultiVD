#!/usr/bin/env bash
# Take the CodeT5+ cls arm from three folds to five.
#
# The read-out effect measured on folds 1-3 is the tightest number in the
# project: the transfer delta under cls minus the delta under mean pooling came
# out +0.0260, +0.0263, +0.0265 -- sd 0.00025, on a task whose fold-level sd is
# 0.09. Differences of differences cancel fold difficulty, which is why paired
# designs work, but that much agreement is either a very clean effect or a
# coincidence that two more folds will break.
#
# Phase 1 is already on disk, so this is two Phase-2 runs plus their baselines.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /venv/main/bin/activate
export HF_HOME=/workspace/hf PYTHON=python
export DATA_ROOT=data/sven_python_twin
export RUN_NAME=t5p_twin_cls
export SEED=${SEED:-36}
export MODEL_NAME=Salesforce/codet5p-220m
export POOLING=cls
export PHASE1_DATA_PATH=data/train_ccpp_js.jsonl

FOLDS="4 5" MODES="cwe" bash run/fold-major.sh
echo "=== CodeT5+ from <s>, all five folds ==="
python src/report_matrix.py
echo
echo "=== the same five folds under mean pooling ==="
RUN_NAME=t5p_twin python src/report_matrix.py
touch /workspace/T5CLS45_DONE
