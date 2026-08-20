#!/usr/bin/env bash
# Finish the common variant to five folds, so it can be compared to full.
#
# This pair is the sharpest result in the project and it currently sits at n=3
# against n=5:
#
#   ccpp_primevul_paired_full    9408 rows, 121 CWEs   +0.0037  (n=5)
#   ccpp_primevul_paired_common  2975 rows,  73 CWEs   +0.0372  (n=3)
#
# Same corpus, same backbone, same folds, same baseline. Deleting 6433 rows made
# the transfer ten times better, and the 48 CWEs deleted are almost entirely
# memory and manual-resource classes that cannot occur in Python at all --
# CWE-119 buffer overflow, 125 out-of-bounds read, 787 out-of-bounds write, 476
# null dereference, 416 use-after-free, 401/772 leaks, 362 race. What survives is
# language-independent logic and validation: CWE-020 input validation, 200
# information exposure, 703 unchecked exceptional conditions, 190 integer
# overflow, 022 path traversal, 078 command injection, 079 XSS.
#
# Two competing explanations are already excluded by the numbers in hand. Size
# moves the wrong way, since the smaller corpus is the better one. Source-task
# accuracy does not separate them either: Phase-1 source validation is 0.5329 for
# full and 0.5197 for common, effectively the same.
#
# Phase 1 for both modes is already on disk, so this is four Phase-2 runs.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /venv/main/bin/activate
export HF_HOME=/workspace/hf PYTHON=python
export DATA_ROOT=data/sven_python_twin
export RUN_NAME=twin_pvcommon
export SEED=${SEED:-36}
export CWE_VOCAB=source
export MODEL_NAME=microsoft/codebert-base
export POOLING=cls
export PHASE1_DATA_PATH=data/ccpp_primevul_paired_common.jsonl

FOLDS="4 5" MODES="cwe latent_bottleneck" bash run/fold-major.sh
echo "=== common, all five folds ==="
python src/report_matrix.py
echo
echo "=== full, for comparison ==="
RUN_NAME=twin_primevul python src/report_matrix.py
touch /workspace/COMMON45_DONE
