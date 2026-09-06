#!/usr/bin/env bash
# Lap lai Phase 2 tren TAP DICH SACH `sven_python_twin` (chia theo cum, khong ro ri).
#
#   bash run/day44.sh <nguon> <backbone-spec>
#
# VI SAO DAY LA PHEP CHUNG MINH DANG GIA NHAT VA RE NHAT
#
# Ket qua manh nhat hien co — latent_bottleneck + AdamW + nguon 4cwe, D +0.0227,
# 13/15 fold, sign p=0.0074 — chay tren `sven_python_folds_norm`. Chinh
# src/build_folds.py ghi ro tap do chia THEO TUNG DONG, nen ~40% hang test co ban
# da va cua chinh no nam trong train. Do la phan bac manh nhat ma mot reviewer se
# dua ra, va no dung. `sven_python_twin` chia theo CUM near-duplicate nen khong
# con ro ri.
#
# Re vi PHASE 1 KHONG PHU THUOC TAP DICH: 28 checkpoint da co dung lai nguyen
# ven, chi Phase 2 phai chay lai. Doi mot bien duy nhat — cach chia fold.
#
# RUN_NAME rieng (`s42tw`) de baseline KHONG bi coi la "da co" tu khoi cu:
# baseline tren fold sach la mot so KHAC va la moc so sanh bat buoc.
# Kho Phase 1 tro ve `model/s42/phase1` bang symlink, khong nhan ban 13 GB.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC="${1:?nguon: full | com | 4cwe}"
BB="${2:?backbone-spec}"

case "$SRC" in
  full) DATA=data/phase1_full.jsonl;   VOCAB=precomputed; M="none latent_bottleneck latent_proto" ;;
  com)  DATA=data/phase1_common.jsonl; VOCAB=precomputed; M="none latent_bottleneck latent_proto" ;;
  4cwe) DATA=data/phase1_4cwe.jsonl;   VOCAB=fixed4;      M="none cwe latent_bottleneck latent_proto" ;;
  *) echo "nguon khong hop le: $SRC"; exit 2 ;;
esac

RN="${RUN_NAME:-s42tw}"
mkdir -p "model/$RN"
[[ -e "model/$RN/phase1" ]] || ln -s "$(cd model/s42/phase1 && pwd)" "model/$RN/phase1"

RUN_NAME="$RN" \
SEED=42 \
FOLDS="${FOLDS:-1 2 3 4 5}" \
BACKBONES="$BB" \
MODES="${MODES:-$M}" \
OPTIMIZERS="${OPTIMIZERS:-adamw recadam}" \
PHASE1_DATA_PATH="$DATA" \
CWE_VOCAB="$VOCAB" \
ARM_TAG="_${SRC}" \
PHASE1_TAG="_${SRC}" \
LAMBDA_CWE=0.05 \
PHASE1_EXTRA="--sam_rho 0" \
PHASE2_EXTRA="--sam_rho 0" \
DATA_ROOT=data/sven_python_twin \
TARGET_LANG=python \
bash run/matrix.sh
