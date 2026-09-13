#!/usr/bin/env bash
# FUSION_CTL — nhanh DOI CHUNG cua khoi adapter-fusion, chay TRUOC vi khong can code moi.
#
# Hai nhanh doi chung phai nam CUNG MOT CAY voi nhanh fusion (CLAUDE.md muc 4):
#   baseline           khong Pha 1
#   latent_bottleneck  phuong phap CHOT hien tai (lambda=0.05, SAM tat ca hai pha)
# Nhanh fusion se ghi vao dung cay nay sau, nen Δ ghep cap duoc theo (backbone, seed, fold)
# ma khong lan chenh lech phan cung.
#
# Pha 1 DUNG LAI tu model/n48/phase1 — khong huan luyen lai (muc 5: chi doi Pha 2).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME=/opt/hf-cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SKIP_BASELINE="${SKIP_BASELINE:-1}" \
RUN_NAME=fus1 SEED=42 FOLDS="${FOLDS:-1 2 3}" \
BACKBONES="codebert=microsoft/codebert-base:cls t5p=Salesforce/codet5p-220m-bimodal:mean" \
MODES=latent_bottleneck OPTIMIZERS=adamw \
PHASE1_DATA_PATH=data/phase1_common.jsonl CWE_VOCAB=precomputed \
ARM_TAG="_com_l0p05" PHASE1_TAG="_com_l0p05" PHASE1_STORE="model/n48/phase1" \
LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
PHASE1_EXTRA="--sam_rho 0" PHASE2_EXTRA="--sam_rho 0" \
DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
PYTHON=python bash run/matrix.sh
