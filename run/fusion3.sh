#!/usr/bin/env bash
# FUSION3 — ADAPTER FUSION (Pfeiffer et al. 2020, arXiv:2005.00247), bac 1: n=3 fold,
# seed 42, nguon `com` (ccpp+js), dich Python. Nguoi dung yeu cau 13/09.
#
# PHA 1  latent_bottleneck lambda=0.05 NHU CU, nhung backbone co them adapter `src`
#        chen sau MOI lop transformer; fine-tune CA backbone lan adapter lan head.
# PHA 2  them adapter `tgt` (python) + lop fusion. Adapter `src` DONG BANG o CA HAI
#        bien the — do la diem cot loi cua bai bao (non-destructive):
#          fusft   fine-tune CA backbone pretrained
#          fusfrz  KHONG dung toi backbone, chi hoc adapter dich + fusion + head
#
# OPTIMIZER: AdamW, SAM tat han o ca hai pha (nguoi dung neu 13/09: khong RecAdam,
# khong ASAM). Code cung TU CHOI chay fusion voi recadam/spd — neo cua chung khop
# theo CHI SO tham so, ma adapter dich khong co ban doi ung trong checkpoint Pha 1.
#
# Ghi vao CUNG CAY `results/fus1_<bb>` voi nhanh doi chung `latent_bottleneck` da chay
# (run/fusion_ctl.sh) => Δ ghep cap duoc theo (backbone, seed, fold), cung may cung phien.
# BASELINE hoan lai (SKIP_BASELINE=1): nguoi dung neu 13/09 — baseline chac chan tren
# 0.74, nen method nao thap hon nguong do la thua roi, khoi ton GPU chay doi chung.
#
#   bash run/fusion3.sh
#   FOLDS="1" bash run/fusion3.sh      # thu mot fold truoc
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/opt/hf-cache}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# t5p-220m + fusion cham tran 15.49 GiB o batch 16. expandable_segments go phan vo vun;
# phan con lai da go bang cach gop truoc-roi-chieu trong AdapterFusion.forward.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec 7>/tmp/mvd_fusion3.lock || exit 1
flock -n 7 || { echo "DA CO fusion3 dang chay tren may nay — dung"; exit 3; }

FOLDS="${FOLDS:-1 2 3}"
DIM="${ADAPTER_DIM:-48}"          # reduction 16 tren H=768
# GC=1 => gradient checkpointing. t5p-220m + fusion OOM o 16 GB du da go tensor thua trong
# AdapterFusion (thieu 24 MiB). Checkpointing cho gradient Y HET, chi cham ~30%%, nen no
# KHONG doi phep so — khac han viec ha batch size von doi hai bien cung luc.
GC="${GC:-1}"
ALR="${ADAPTER_LR:-1e-4}"
P1TAG="_com_l0p05_ad${DIM}"
BB="codebert=microsoft/codebert-base:cls t5p=Salesforce/codet5p-220m-bimodal:mean"
ts(){ date -u '+%F %T'; }

echo "########## FUSION3 bat dau $(ts) | $(hostname) ##########"
echo "  fold: $FOLDS | adapter dim: $DIM | adapter lr: $ALR | nguon: com | seed 42"
echo "  bien the: fusft (backbone MO) va fusfrz (backbone KHOA); adapter nguon KHOA ca hai"

# VONG NGOAI LA FOLD (CLAUDE.md muc 1): xong fold 1 la da co mot lat cat so duoc ngay
# — du de quyet dinh co chay tiep hay khong, thay vi phai doi het ca khoi.
for FOLD in $FOLDS; do
  echo "===== $(ts) | FOLD $FOLD ====="
  for V in ft frozen; do
    case "$V" in
      ft)     TAG="${P1TAG}_fusft" ;;
      frozen) TAG="${P1TAG}_fusfrz" ;;
    esac
    echo "----- $(ts) | fold $FOLD | bien the $V -----"
    SKIP_BASELINE=1 \
    RUN_NAME=fus1 SEED=42 FOLDS="$FOLD" \
    BACKBONES="$BB" MODES=latent_bottleneck OPTIMIZERS=adamw \
    PHASE1_DATA_PATH=data/phase1_common.jsonl CWE_VOCAB=precomputed \
    ARM_TAG="$TAG" PHASE1_TAG="$P1TAG" PHASE1_STORE="model/fus1/phase1" \
    LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
    PHASE1_EXTRA="--sam_rho 0 --adapter_dim $DIM --adapter_lr $ALR" \
    PHASE2_EXTRA="--sam_rho 0 --phase2_fusion $V --adapter_lr $ALR ${GC:+--grad_checkpointing}" \
    DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
    PYTHON=python bash run/matrix.sh 7>&-
  done
done

N=$(find results/fus1_codebert results/fus1_t5p -name 'fold*.json' 2>/dev/null | wc -l)
echo "########## FUSION3 xong $(ts) | tong $N o trong results/fus1_* ##########"
