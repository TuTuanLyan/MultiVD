#!/usr/bin/env bash
# FUSION5 — leo `fusft` len BAC 2 (n=5 fold, seed 42). Chay NOT fold 4 va 5 cho:
#     doi chung `latent_bottleneck`  +  nhanh `fusft`
# tren CA HAI backbone, ghi vao DUNG cay `results/fus1_<bb>` cua bac 1 => 5 fold lien mach.
#
# VI SAO LEN BAC: bac 1 (n=3) cho `fusft` duong tren CA BON chi so tren CA HAI backbone,
# 3/3 fold o F1@0.5 va ROC-AUC ca hai. Do dung la cong o CLAUDE.md muc 1, va la nhanh dau
# tien qua duoc cong do ke tu §40. `fusfrz` KHONG qua (F1@val cua t5p am) nen khong leo.
#
# GC=1 cho ca hai backbone o hai fold nay: t5p + fusft OOM neu khong co. Gradient
# checkpointing cho gradient Y HET (da kiem: dau ra khop 1e-5, gradient van toi adapter
# dich va fusion) nen no khong lam lech phep so. Ghi ra day de bao cao noi that: codebert
# fold 1-3 chay KHONG checkpointing, fold 4-5 CO; t5p thi fold 1-3 da co san.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/opt/hf-cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec 7>/tmp/mvd_fusion5.lock || exit 1
flock -n 7 || { echo "DA CO fusion5 dang chay — dung"; exit 3; }

FOLDS="${FOLDS:-4 5}"
DIM="${ADAPTER_DIM:-48}"; ALR="${ADAPTER_LR:-1e-4}"
BB="codebert=microsoft/codebert-base:cls t5p=Salesforce/codet5p-220m-bimodal:mean"
ts(){ date -u '+%F %T'; }
echo "########## FUSION5 bat dau $(ts) | fold: $FOLDS ##########"

for FOLD in $FOLDS; do
  echo "===== $(ts) | FOLD $FOLD | doi chung latent_bottleneck ====="
  SKIP_BASELINE=1 RUN_NAME=fus1 SEED=42 FOLDS="$FOLD" \
  BACKBONES="$BB" MODES=latent_bottleneck OPTIMIZERS=adamw \
  PHASE1_DATA_PATH=data/phase1_common.jsonl CWE_VOCAB=precomputed \
  ARM_TAG="_com_l0p05" PHASE1_TAG="_com_l0p05" PHASE1_STORE="model/n48/phase1" \
  LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
  PHASE1_EXTRA="--sam_rho 0" PHASE2_EXTRA="--sam_rho 0" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PYTHON=python bash run/matrix.sh 7>&-

  echo "===== $(ts) | FOLD $FOLD | fusft ====="
  SKIP_BASELINE=1 RUN_NAME=fus1 SEED=42 FOLDS="$FOLD" \
  BACKBONES="$BB" MODES=latent_bottleneck OPTIMIZERS=adamw \
  PHASE1_DATA_PATH=data/phase1_common.jsonl CWE_VOCAB=precomputed \
  ARM_TAG="_com_l0p05_ad${DIM}_fusft" PHASE1_TAG="_com_l0p05_ad${DIM}" \
  PHASE1_STORE="model/fus1/phase1" \
  LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
  PHASE1_EXTRA="--sam_rho 0 --adapter_dim $DIM --adapter_lr $ALR" \
  PHASE2_EXTRA="--sam_rho 0 --phase2_fusion ft --adapter_lr $ALR --grad_checkpointing" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PYTHON=python bash run/matrix.sh 7>&-
done
N=$(find results/fus1_codebert results/fus1_t5p -name 'fold*.json' 2>/dev/null | wc -l)
echo "########## FUSION5 xong $(ts) | tong $N o ##########"
