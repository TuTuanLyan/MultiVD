#!/usr/bin/env bash
# Is it the backbone, or is it the pooling I attached to the backbone?
#
# The gain does not decay across backbones, it changes sign: +0.042 on CodeBERT
# against -0.027 on CodeT5 and -0.036 on CodeT5+. Normalising by the headroom
# each baseline leaves makes the spread worse, not better (0.078 -> 0.499), so a
# ceiling does not explain it.
#
# But CodeBERT and the T5 family were never run under the same read-out. CodeBERT
# pools from CLS; T5 has no sentence-level token at position 0, so those runs pool
# by attention-masked mean. That is a choice in this repo, not a property of the
# checkpoints, and it moved together with the backbone in every run so far.
#
# Mean pooling averages over the whole window. The evidence for a vulnerability
# is a few lines, so the mean dilutes it by the function length while CLS can
# learn to attend to it -- a mechanism that would help an auxiliary objective
# under CLS and blunt it under mean.
#
# This arm changes pooling and nothing else: same CodeBERT checkpoint, same twin
# folds, same source corpus, same seed. --pooling sits in the driver's SHARED
# block, so the baseline is retrained under mean pooling too and the comparison
# stays within-condition.
#
#   gain stays positive -> pooling is not the mechanism; the backbones differ.
#   gain flips negative -> the read-out is the mechanism, and the T5 fix is to
#                          give those encoders an aggregation slot rather than
#                          to protect their weights, which already failed three
#                          times (LP-FT, interpolation, LoRA).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /venv/main/bin/activate
export HF_HOME=/workspace/hf PYTHON=python
export DATA_ROOT=data/sven_python_twin
export RUN_NAME=twin_ccppjs_meanpool
export SEED=${SEED:-36}
export MODEL_NAME=microsoft/codebert-base
export POOLING=mean
export PHASE1_DATA_PATH=data/train_ccpp_js.jsonl

FOLDS="1 2 3" MODES="cwe" bash run/fold-major.sh
echo "=== CodeBERT under mean pooling, 3 folds ==="
python src/report_matrix.py
echo
echo "=== the same folds under CLS, for reference ==="
RUN_NAME=twin_ccppjs python src/report_matrix.py
touch /workspace/POOLING_DONE
