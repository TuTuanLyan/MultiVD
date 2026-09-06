#!/usr/bin/env bash
# GIAI DOAN 2 cua dot quet: luoi lambda x rho cho latent_bottleneck tren t5p-bimodal.
#
#   FOLD_LIST="1 2" bash run/sweep46_p2.sh
#
# CHIA VIEC THEO FOLD TRON VEN, khong theo lambda.
# Muc 3 CLAUDE.md: ca nhanh chinh lan nhanh doi chung cua MOT fold phai nam cung
# mot may, khi do do lech phan cung triet tieu trong Delta ghep cap. matrix.sh tu
# chay baseline cho dung nhung fold duoc giao, nen fold nao o may nao thi baseline
# cua fold do cung o day.
#
# PHASE1_TAG tach khoi ARM_TAG: doi rho chi doi Phase 2, phai dung lai dung
# checkpoint Phase 1 cua lambda tuong ung, neu khong phep so se doi hai bien.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC="${SRC:-com}"
case "$SRC" in
  com)  DATA=data/phase1_common.jsonl; VOCAB=precomputed ;;
  4cwe) DATA=data/phase1_4cwe.jsonl;   VOCAB=fixed4      ;;
  full) DATA=data/phase1_full.jsonl;   VOCAB=precomputed ;;
  *) echo "SRC khong hop le: $SRC"; exit 2 ;;
esac
LAMBDAS="${LAMBDAS:-0.01 0.05 0.2}"
RHOS="${RHOS:-0 0.05 0.2 0.5}"
FOLD_LIST="${FOLD_LIST:?can FOLD_LIST, vd: FOLD_LIST=\"1 2\"}"
BB="${BB:-t5p=Salesforce/codet5p-220m-bimodal:mean}"
LOCK="${MVD_LOCK:-/tmp/multivd_sweep46_${SRC}.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO driver sweep46 dang chay"; exit 3; }
mkdir -p log

echo "########## SWEEP46 bat dau $(date -u '+%F %T') | fold: $FOLD_LIST ##########"
echo "  lambda: $LAMBDAS   rho: $RHOS   backbone: ${BB%%=*}   nguon: $SRC   RecAdam"

for LAM in $LAMBDAS; do
  LTAG="l$(printf '%s' "$LAM" | tr '.' 'p')"
  CKPT="model/sw/phase1/${BB%%=*}__latent_bottleneck_${SRC}_${LTAG}/seed_42/best.pt"
  if [[ ! -f "$CKPT" ]]; then
    echo "!! THIEU checkpoint Phase 1 cho lambda=$LAM ($CKPT) — bo qua ca cum rho cua lambda nay"
    continue
  fi
  for RHO in $RHOS; do
    RTAG="r$(printf '%s' "$RHO" | tr '.' 'p')"
    if [[ "$RHO" == "0" ]]; then P2="--sam_rho 0"
    else P2="--sam_rho $RHO --sam_variant asam --asam_eta 0.01"; fi
    echo "===== $(date -u '+%F %T') | lambda=$LAM rho=$RHO | fold: $FOLD_LIST ====="
    RUN_NAME=sw SEED=42 FOLDS="$FOLD_LIST" \
    BACKBONES="$BB" MODES="latent_bottleneck" OPTIMIZERS=recadam \
    PHASE1_DATA_PATH="$DATA" CWE_VOCAB="$VOCAB" \
    ARM_TAG="_${SRC}_${LTAG}_${RTAG}" PHASE1_TAG="_${SRC}_${LTAG}" \
    LAMBDA_CWE="$LAM" PHASE1_EPOCHS=15 \
    PHASE1_EXTRA="--sam_rho 0" \
    PHASE2_EXTRA="$P2" \
    DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
    bash run/matrix.sh
  done
done
echo "########## SWEEP46 xong $(date -u '+%F %T') | fold: $FOLD_LIST ##########"
