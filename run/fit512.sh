#!/usr/bin/env bash
# Domain distance, or simply not fitting the context window?
#
# PrimeVul transfers worse than a corpus seven times smaller. Two explanations
# compete: C/C++ is far from Python, or 70% of PrimeVul functions exceed the
# 512-token limit that neither encoder can be configured past, so the model
# never sees the lines that distinguish a vulnerable function from its fix.
#
# This source keeps the domain and removes the truncation: 2,560 C/C++ rows that
# all fit, against the 2,975 rows of the "common" variant where 67% do not. Same
# language, similar size, opposite truncation. If the delta turns positive here,
# the window was the problem.
#
# The baseline is reused from the CodeBERT twin sweep on this machine and these
# folds; it never sees source data.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /venv/main/bin/activate
export HF_HOME=/workspace/hf PYTHON=python
export DATA_ROOT=data/sven_python_twin
export RUN_NAME=twin_fit512
export SEED=${SEED:-36}
export CWE_VOCAB=source
export MODEL_NAME=microsoft/codebert-base
export POOLING=cls
export PHASE1_DATA_PATH=data/ccpp_primevul_fit512.jsonl

mkdir -p "results/$RUN_NAME/baseline/seed_$SEED"
cp results/twin_ccppjs/baseline/seed_$SEED/fold*.json \
   "results/$RUN_NAME/baseline/seed_$SEED/" 2>/dev/null

FOLDS="1 2 3" MODES="cwe" bash run/fold-major.sh
echo "=== PrimeVul that fits the window, 3 folds ==="
python src/report_matrix.py
touch /workspace/FIT512_DONE
