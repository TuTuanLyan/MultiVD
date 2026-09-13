#!/usr/bin/env bash
# FUSION_RND — DOI CHUNG quyet dinh cho FACTS §47: `+0.0259` cua codebert la TRANSFER hay
# chi la THEM SUC CHUA? Khai bao truoc o records/prediction_2026-09-13_fusion_transfer_hay_suc_chua.md
#
# Giu NGUYEN moi thu cua `fusft` — kien truc, so tham so, optimizer, Pha 1 checkpoint, fold,
# may — CHI thay trong so adapter NGUON bang nhieu Gauss CUNG THANG DO roi dong bang y het.
# Neu loi ich van con => khong phai transfer, chi la suc chua, va huong nay bi bac.
#
# CHI codebert: t5p khong co hieu ung de giai thich (§47).
# Ghi vao DUNG cay `results/fus1_codebert` => dung lai doi chung va Pha 1 da co.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/opt/hf-cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec 6>/tmp/mvd_fusion_rnd.lock || exit 1
flock -n 6 || { echo "DA CO fusion_rnd dang chay — dung"; exit 3; }
FOLDS="${FOLDS:-1 2 3 4 5}"
DIM="${ADAPTER_DIM:-48}"; ALR="${ADAPTER_LR:-1e-4}"
GC="${GC:-}"                      # codebert vua bo nho, khong can checkpointing
ts(){ date -u '+%F %T'; }
echo "########## FUSION_RND bat dau $(ts) | fold: $FOLDS | CHI codebert ##########"
echo "  doi chung: adapter NGUON = nhieu Gauss cung thang do, dong bang"
for FOLD in $FOLDS; do
  echo "===== $(ts) | FOLD $FOLD ====="
  SKIP_BASELINE=1 RUN_NAME=fus1 SEED=42 FOLDS="$FOLD" \
  BACKBONES="codebert=microsoft/codebert-base:cls" \
  MODES=latent_bottleneck OPTIMIZERS=adamw \
  PHASE1_DATA_PATH=data/phase1_common.jsonl CWE_VOCAB=precomputed \
  ARM_TAG="_com_l0p05_ad${DIM}_fusftrnd" PHASE1_TAG="_com_l0p05_ad${DIM}" \
  PHASE1_STORE="model/fus1/phase1" \
  LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
  PHASE1_EXTRA="--sam_rho 0 --adapter_dim $DIM --adapter_lr $ALR" \
  PHASE2_EXTRA="--sam_rho 0 --phase2_fusion ft --adapter_lr $ALR --fusion_src_random ${GC:+--grad_checkpointing}" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PYTHON=python bash run/matrix.sh 6>&-
done
N=$(find results/fus1_codebert -path '*fusftrnd*' -name 'fold*.json' 2>/dev/null | wc -l)
echo "########## FUSION_RND xong $(ts) | $N/5 o ##########"
