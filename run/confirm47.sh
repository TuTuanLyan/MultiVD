#!/usr/bin/env bash
# XAC NHAN hai cau hinh tot nhat cua dot quet lambda x rho, tren NHIEU SEED va DU 5 FOLD.
#
#   SRC=com  SEEDS="42 7 1234" bash run/confirm47.sh
#   SRC=4cwe SEEDS="42 7 1234" bash run/confirm47.sh
#
# MUC TIEU: dua n tu 3 len 15 (5 fold x 3 seed) cho tung cau hinh, de kiem dau
# co the xuong duoi p=0.05. O n=3 san kiem dau la p=0.25 — khong o don le nao
# co the dat y nghia, ke ca khi ca ba fold cung dau.
#
# HAI CAU HINH DUOC CHON (tu bang gop hai nguon, 6 diem ghep cap moi o):
#   lambda=0.05, rho=0.5  — Delta gop cao nhat  (+0.0614, 6/6 fold)
#   lambda=0.05, rho=0.2  — on dinh nhat        (+0.0611, 6/6, lech hai nguon 0.0096)
#
# rho=0 chay kem lam DOI CHUNG. Khong co no thi chi biet "cau hinh nay tot", khong
# biet ASAM co phai la thu tao ra cai tot do khong — vi rho=0 dung chung checkpoint
# Phase 1, moi hieu so rho-vs-rho0 la mot bien duy nhat.
#
# lambda KHONG doi giua cac o nen moi seed chi can MOT checkpoint Phase 1; ba gia
# tri rho dung chung no (muc 5 CLAUDE.md).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC="${SRC:-com}"
case "$SRC" in
  com)  DATA=data/phase1_common.jsonl; VOCAB=precomputed ;;
  4cwe) DATA=data/phase1_4cwe.jsonl;   VOCAB=fixed4      ;;
  full) DATA=data/phase1_full.jsonl;   VOCAB=precomputed ;;
  *) echo "SRC khong hop le: $SRC"; exit 2 ;;
esac
SEEDS="${SEEDS:-42 7 1234}"
LAM="${LAM:-0.05}"
RHOS="${RHOS:-0 0.2 0.5}"
FOLD_LIST="${FOLD_LIST:-1 2 3 4 5}"
BB="${BB:-t5p=Salesforce/codet5p-220m-bimodal:mean}"
LTAG="l$(printf '%s' "$LAM" | tr '.' 'p')"

LOCK="${MVD_LOCK:-/tmp/multivd_confirm47_${SRC}.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO driver confirm47 ($SRC) dang chay"; exit 3; }
mkdir -p log

echo "########## CONFIRM47 bat dau $(date -u '+%F %T') ##########"
echo "  nguon $SRC | lambda $LAM | rho: $RHOS | seed: $SEEDS | fold: $FOLD_LIST"

for SEED in $SEEDS; do
  echo "===== SEED $SEED | Phase 1 (dung chung cho moi rho) ====="
  RUN_NAME=sw SEED="$SEED" FOLDS="" \
  BACKBONES="$BB" MODES="latent_bottleneck" OPTIMIZERS=recadam \
  PHASE1_DATA_PATH="$DATA" CWE_VOCAB="$VOCAB" \
  ARM_TAG="_${SRC}_${LTAG}" PHASE1_TAG="_${SRC}_${LTAG}" \
  LAMBDA_CWE="$LAM" PHASE1_EPOCHS=15 \
  PHASE1_EXTRA="--sam_rho 0" PHASE2_EXTRA="--sam_rho 0" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  bash run/matrix.sh

  CKPT="model/sw/phase1/${BB%%=*}__latent_bottleneck_${SRC}_${LTAG}/seed_${SEED}/best.pt"
  if [[ ! -f "$CKPT" ]]; then
    echo "!! THIEU checkpoint Phase 1 cho seed=$SEED ($CKPT) — bo qua seed nay"
    continue
  fi

  for RHO in $RHOS; do
    RTAG="r$(printf '%s' "$RHO" | tr '.' 'p')"
    if [[ "$RHO" == "0" ]]; then P2="--sam_rho 0"
    else P2="--sam_rho $RHO --sam_variant asam --asam_eta 0.01"; fi
    echo "===== $(date -u '+%F %T') | seed $SEED | lambda $LAM rho $RHO | fold: $FOLD_LIST ====="
    RUN_NAME=sw SEED="$SEED" FOLDS="$FOLD_LIST" \
    BACKBONES="$BB" MODES="latent_bottleneck" OPTIMIZERS=recadam \
    PHASE1_DATA_PATH="$DATA" CWE_VOCAB="$VOCAB" \
    ARM_TAG="_${SRC}_${LTAG}_${RTAG}" PHASE1_TAG="_${SRC}_${LTAG}" \
    LAMBDA_CWE="$LAM" PHASE1_EPOCHS=15 \
    PHASE1_EXTRA="--sam_rho 0" PHASE2_EXTRA="$P2" \
    DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
    bash run/matrix.sh
  done
done
echo "########## CONFIRM47 xong $(date -u '+%F %T') | nguon $SRC ##########"
