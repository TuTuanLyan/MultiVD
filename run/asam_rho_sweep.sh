#!/usr/bin/env bash
# Quet rho cua ASAM tren DUNG MOT o truoc khi do hang tram job vao mot rho doan mo.
#
#   bash run/asam_rho_sweep.sh unixcoder=microsoft/unixcoder-base:cls
#
# Co dinh: nguon 4cwe, nhanh `none`, fold 1, RecAdam, seed 42. Chi doi rho.
# Kem mot nhanh doi chung rho=0 de biet moc.
#
# VI SAO PHAI QUET: rho cua ASAM do bang don vi |w|, KHONG cung thang do voi rho
# cua SAM. Kwon et al. chon 0.5-1.0 cho vision nhung 0.2 cho thi nghiem
# transformer duy nhat cua ho, va khong co so nao cho fine-tuning PLM. Sang
# 27/08 mot rho doan mo (SAM 0.05) da giet han codebert o Phase 1 — khong lap lai
# kieu do o quy mo 200 job.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BB="${1:?backbone-spec}"; LABEL="${BB%%=*}"
RHOS="${RHOS:-0.05 0.1 0.2 0.5 1.0}"
mkdir -p log
{
echo "########## QUET RHO $LABEL bat dau $(date -u '+%F %T') ##########"
echo "===== moc: doi chung rho=0 ====="
FOLDS=1 MODES=none CONTROL=1 bash run/day43.sh 4cwe "$BB"
for R in $RHOS; do
  T=$(echo "$R" | tr -d '.')
  echo "===== ASAM rho=$R ====="
  RUN_NAME=s42 SEED=42 FOLDS=1 \
  BACKBONES="$BB" MODES=none OPTIMIZERS=recadam \
  PHASE1_DATA_PATH=data/phase1_4cwe.jsonl CWE_VOCAB=fixed4 \
  ARM_TAG="_4cwe_asam_r$T" PHASE1_TAG="_4cwe" LAMBDA_CWE=0.05 \
  PHASE1_EXTRA="--sam_rho 0" \
  PHASE2_EXTRA="--sam_rho $R --sam_variant asam --asam_eta 0.01" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  bash run/matrix.sh
done
echo "########## QUET RHO $LABEL xong $(date -u '+%F %T') ##########"
} >> "log/asam_sweep_${LABEL}.log" 2>&1
