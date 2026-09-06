#!/usr/bin/env bash
# GIAI DOAN 1 cua dot quet: chi huan luyen checkpoint Phase 1 cho tung lambda.
#
#   bash run/sweep46_p1.sh
#
# VI SAO HUAN LUYEN TAT CA TREN MOT MAY
# lambda nhan vao loss Phase 1 (train.py:53) nen moi lambda phai co checkpoint
# rieng. Neu chia viec huan luyen Phase 1 ra hai may thi phep so giua cac lambda
# se lan them chenh lech phan cung (~0.010 giua hai may cung loai GPU). Huan luyen
# het o mot may roi CHEP checkpoint sang may kia thi diem xuat phat cua moi lambda
# la y het nhau, va chi con Phase 2 chia theo fold.
#
# rho KHONG doi Phase 1 (no chi la buoc nhieu loan o Phase 2), nen mot checkpoint
# dung chung cho ca 4 gia tri rho — xem muc 5 CLAUDE.md.
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
BB="${BB:-t5p=Salesforce/codet5p-220m-bimodal:mean}"
# LOCK — thieu no da lam hong mot checkpoint.
# 02/09: bo giam sat phong them mot ban p1 trong khi ban cu dang chay; hai tien
# trinh huan luyen cung chiem GPU 161 -> CUDA OOM -> checkpoint lambda=0.2 cua
# nguon 4cwe khong sinh ra, ma Phase 1 van in "xong". p2 co lock tu dau, p1 thi
# khong — day chinh la cho ho.
LOCK="${MVD_LOCK:-/tmp/multivd_sweep46p1_${SRC}.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO driver sweep46_p1 ($SRC) dang chay"; exit 3; }
mkdir -p log

for LAM in $LAMBDAS; do
  LTAG="l$(printf '%s' "$LAM" | tr '.' 'p')"
  echo "===== PHASE 1 | lambda=$LAM | tag=$LTAG ====="
  RUN_NAME=sw SEED=42 FOLDS="" \
  BACKBONES="$BB" MODES="latent_bottleneck" OPTIMIZERS=recadam \
  PHASE1_DATA_PATH="$DATA" CWE_VOCAB="$VOCAB" \
  ARM_TAG="_${SRC}_${LTAG}" PHASE1_TAG="_${SRC}_${LTAG}" \
  LAMBDA_CWE="$LAM" PHASE1_EPOCHS=15 \
  PHASE1_EXTRA="--sam_rho 0" \
  PHASE2_EXTRA="--sam_rho 0" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  bash run/matrix.sh
done
echo "########## SWEEP46 PHASE1 xong $(date -u '+%F %T') ##########"
