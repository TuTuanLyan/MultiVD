#!/usr/bin/env bash
# CodeT5+ read out from position 0, the way CodeBERT always was.
#
# The T5 runs in this project pooled by attention-masked mean because the code
# asserted that T5 has no token at position 0. That assertion is false: both
# CodeT5 and CodeT5+ tokenise with RobertaTokenizerFast and both emit <s> at
# position 0, the same token CodeBERT reads. Checked directly:
#
#   microsoft/codebert-base   RobertaTokenizerFast  pos0=0  '<s>'
#   Salesforce/codet5p-220m   RobertaTokenizerFast  pos0=1  '<s>'
#   Salesforce/codet5-base    RobertaTokenizerFast  pos0=1  '<s>'
#
# So every CodeBERT-vs-T5 comparison so far changed two things at once, and the
# sign flip -- +0.042 on CodeBERT against -0.018 on CodeT5+ over the same twin
# folds -- has never been attributed to one of them.
#
# This is the T5 half of that separation; run/pooling.sh is the CodeBERT half.
# Together they fill in a 2x2 over {CodeBERT, CodeT5+} x {cls, mean}, of which
# only the diagonal has ever been run.
#
# If CodeT5+ under cls turns positive, the method is not backbone-dependent at
# all -- it is read-out dependent, and the fix is a config change rather than any
# of the three weight-preserving interventions that already failed.
#
# --pooling is in the driver's SHARED block, so the baseline is retrained under
# cls as well and the comparison stays within-condition.
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

FOLDS="1 2 3" MODES="cwe" bash run/fold-major.sh
echo "=== CodeT5+ pooled from <s>, 3 folds ==="
python src/report_matrix.py
echo
echo "=== the same folds under mean pooling, for reference ==="
RUN_NAME=t5p_twin python src/report_matrix.py
touch /workspace/T5CLS_DONE
