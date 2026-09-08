#!/usr/bin/env bash
# ASAM2 — truc rho, chi so CHINH khai bao truoc: ROC-AUC. Ghi vao CUNG cay `asam1_t5p`
# nen doi chung rho=0 va rho=0.1 da co san, ghep cap duoc ngay.
#
# VI SAO rho > 0.1 (do 08/09 tu quet co san results/sw_t5p, doc lambda TU TEN NHANH vi
# truong lambda_cwe trong file truoc 06/09 ghi mac dinh CLI 0.2 bat ke lambda that):
# tai lambda = 0.05 — dung lambda da chot — dAUC so voi rho=0 TANG DAN theo rho:
#   rho 0.05 +0.0303 | 0.1 +0.0261 | 0.2 +0.0253 | 0.5 +0.0382 | 1.0 +0.0395 | 2.0 +0.0572
# tat ca deu 3/3 fold duong. Gop ba lambda thi rho=1.0 cho +0.0240, 8/9, p=0.039.
#
# NHUNG rho=2.0 lam SAP o lambda=0.2 (dAUC -0.1097), nen khoi nay dung o rho=1.0.
# Quet cu chi n=3, mot nguon (4cwe), nen day la kiem chung chu chua phai ket luan.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$( [ -x /venv/main/bin/python ] && echo /venv/main/bin/python \
  || echo /home/ntat/miniconda3/envs/vdenv/bin/python )}"
export PYTHON
SOURCES_LIST="${SOURCES_LIST:-4cwe com}"
FOLD_LIST="${FOLD_LIST:-1 2 3 4 5}"
SEED="${SEED:-42}"
echo "########## ASAM2 bat dau $(date -u '+%F %T') | $(hostname) ##########"
echo "  truc rho: 0.5 va 1.0 (doi chung 0 va 0.1 da co trong cung cay asam1_t5p)"
echo "  nguon: $SOURCES_LIST | fold: $FOLD_LIST"
for FOLD in $FOLD_LIST; do
  for SRC in $SOURCES_LIST; do
    echo "===== $(date -u '+%F %T') | fold $FOLD | nguon $SRC ====="
    RUN=asam1 SEEDS="$SEED" SOURCES="$SRC" FOLD_LIST="$FOLD" MIN_EP=3 \
    CONFIGS="r0p5|recadam|--sam_rho 0.5 --sam_variant asam
r1p0|recadam|--sam_rho 1.0 --sam_variant asam" \
    bash run/opt1.sh
  done
done
echo "########## ASAM2 xong $(date -u '+%F %T') ##########"
