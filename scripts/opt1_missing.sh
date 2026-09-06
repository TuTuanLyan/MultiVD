#!/usr/bin/env bash
# opt1_missing.sh — dem chinh xac so o CON THIEU cua khoi OPT1, theo TUNG TAG.
#
#   bash scripts/opt1_missing.sh "<nguon...>" "<fold...>" [seed]
#   -> in mot so: so o chua co file ket qua.
#   MISSING_LIST=1 ... -> in ra danh sach o thieu thay vi con so.
#
# Vi sao khong dem tong so file: so cau hinh doi giua cac lan chay (10 -> 12 ngay
# 06/09), va vai fold con giu o cua cau hinh DA BO. Dem tong se thay "du" trong khi
# thieu dung o moi. Dem theo tag thi tu chua lanh dung cho, khong bao gio bo sot.
# Danh sach tag lay tu chinh run/opt1.sh — mot nguon su that duy nhat.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRCS="${1:-4cwe com}"; FOLDS="${2:-1 2 3}"; SEED="${3:-42}"
RES="results/opt1_t5p"; LTAG="l0p05"

# Lay CONFIGS tu run/opt1.sh ma KHONG chay no (no giu flock va se thoat 3).
CFG=$(sed -n '/^CONFIGS="\${CONFIGS:-/,/}"$/p' run/opt1.sh | sed '1s/^CONFIGS="${CONFIGS:-\\$//; $s/}"$//' | grep '|')
[[ -z "$CFG" ]] && { echo "LOI: khong doc duoc CONFIGS tu run/opt1.sh" >&2; exit 1; }

n=0
# Baseline dem MOT LAN cho moi fold, khong theo nguon: baseline khong doc du lieu nguon.
for F in $FOLDS; do
  [[ -f "$RES/baseline/seed_$SEED/fold$F.json" ]] || { n=$((n+1)); [[ "${MISSING_LIST:-0}" == 1 ]] && echo "baseline fold$F"; }
done
for S in $SRCS; do
  for F in $FOLDS; do
    while IFS='|' read -r TAG OPT _; do
      [[ -z "$TAG" ]] && continue
      sfx=""; [[ "$OPT" == "adamw" ]] && sfx="_adamw"
      f="$RES/transfer_latent_bottleneck_${S}_${LTAG}_${TAG}${sfx}/seed_$SEED/fold$F.json"
      if [[ ! -f "$f" ]]; then
        n=$((n+1)); [[ "${MISSING_LIST:-0}" == 1 ]] && echo "$S fold$F $TAG ($OPT)"
      fi
    done <<< "$CFG"
  done
done
[[ "${MISSING_LIST:-0}" == 1 ]] || echo "$n"
