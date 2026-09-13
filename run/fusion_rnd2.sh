#!/usr/bin/env bash
# FUSION_RND2 — DOI CHUNG quyet dinh cho FACTS §47 tren MAY MOI, ba nhanh CUNG MOT CAY.
# Khai bao truoc: records/prediction_2026-09-13_fusion_transfer_hay_suc_chua.md
#
#   doi chung  latent_bottleneck        (phuong phap chot)
#   fusft      adapter nguon DA HOC     (nhanh that)
#   fusftrnd   adapter nguon NGAU NHIEN cung thang do   (doi chung)
#
# Ba nhanh chi khac nhau o NOI DUNG cua adapter nguon — cung kien truc, cung so tham so,
# cung optimizer, cung fold, cung may. Neu `fusftrnd` van an thi loi ich la SUC CHUA chu
# khong phai tri thuc Pha 1, va huong nay bi bac.
#
# CHI codebert (t5p khong co hieu ung de giai thich). VONG NGOAI LA FOLD (muc 1).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-/venv/main/bin/python}"
exec 6>/tmp/mvd_fusion_rnd2.lock || exit 1
flock -n 6 || { echo "DA CO fusion_rnd2 dang chay — dung"; exit 3; }
FOLDS="${FOLDS:-1 2 3 4 5}"; DIM="${ADAPTER_DIM:-48}"; ALR="${ADAPTER_LR:-1e-4}"
BB="codebert=microsoft/codebert-base:cls"
ts(){ date -u '+%F %T'; }
echo "########## FUSION_RND2 bat dau $(ts) | fold: $FOLDS | chi codebert ##########"

common() {   # $1 = ARM_TAG  $2 = PHASE1_TAG  $3 = PHASE1_STORE  $4 = PHASE1_EXTRA  $5 = PHASE2_EXTRA  $6 = FOLD
  SKIP_BASELINE=1 RUN_NAME=fus2 SEED=42 FOLDS="$6" \
  BACKBONES="$BB" MODES=latent_bottleneck OPTIMIZERS=adamw \
  PHASE1_DATA_PATH=data/phase1_common.jsonl CWE_VOCAB=precomputed \
  ARM_TAG="$1" PHASE1_TAG="$2" PHASE1_STORE="$3" \
  LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
  PHASE1_EXTRA="$4" PHASE2_EXTRA="$5" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PYTHON="$PY" bash run/matrix.sh 6>&-
}

for FOLD in $FOLDS; do
  echo "===== $(ts) | FOLD $FOLD | 1/3 doi chung ====="
  common "_com_l0p05" "_com_l0p05" "model/n48/phase1" "--sam_rho 0" "--sam_rho 0" "$FOLD"
  echo "===== $(ts) | FOLD $FOLD | 2/3 fusft (adapter nguon DA HOC) ====="
  common "_com_l0p05_ad${DIM}_fusft" "_com_l0p05_ad${DIM}" "model/fus2/phase1" \
         "--sam_rho 0 --adapter_dim $DIM --adapter_lr $ALR" \
         "--sam_rho 0 --phase2_fusion ft --adapter_lr $ALR" "$FOLD"
  echo "===== $(ts) | FOLD $FOLD | 3/3 fusftrnd (adapter nguon NGAU NHIEN) ====="
  common "_com_l0p05_ad${DIM}_fusftrnd" "_com_l0p05_ad${DIM}" "model/fus2/phase1" \
         "--sam_rho 0 --adapter_dim $DIM --adapter_lr $ALR" \
         "--sam_rho 0 --phase2_fusion ft --adapter_lr $ALR --fusion_src_random" "$FOLD"
done
N=$(find results/fus2_codebert -name 'fold*.json' 2>/dev/null | wc -l)
echo "########## FUSION_RND2 xong $(ts) | $N/15 o ##########"
