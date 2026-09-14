#!/usr/bin/env bash
# SHUF2 — hai viec, ca hai deu tra loi truc tiep cho cho CON YEU cua FACTS §54.
#
# (1) LEO LEN n=5 cho chinh khoi shuf1. Bien quyet dinh dang ky truoc — `that - doi-cho-trong-cap`
#     tren F1@0.5 — ra +0.0387 nhung chi 3/6 fold, roi vao vung "KHONG KET LUAN" ma chinh toi
#     vach san. Them fold 4 va 5 dua len 10 diem ghep cap; san kiem dinh dau xuong p=0.002.
#     Pha 1 DUNG LAI nguyen ven (cung file, da co san tren may) — khong huan luyen lai.
#
# (2) Nhanh `realbest` — go mot NHIEU LOAN DO CHINH TOI TAO RA. Trong shuf1, ca ba nhanh bi ep
#     chay dung 12 epoch va lay checkpoint CUOI de khop so buoc gradient. Ket qua
#     `that - baseline = +0.0018` vi the KHONG so duoc voi +0.0092 cua §40.5, vi moi ket qua
#     da cong bo deu lay checkpoint TOT NHAT THEO VAL. Nhanh nay chay Pha 1 y het nhung voi
#     chon-theo-val + dung som binh thuong (15 epoch, patience mac dinh), roi so voi BASELINE.
#     No tra loi: "Pha 1 khong mang lai gi" la that, hay chi la hau qua cua quy tac ep epoch?
#
#     LUU Y KHI DOC: `realbest` KHONG ghep cap duoc voi ba nhanh kia (khac so epoch). No chi
#     dung de so voi BASELINE. Ba nhanh kia van ghep cap voi nhau nhu cu.
#
# Bac 2 — xac nhan (n=5), seed 42, ca hai backbone. Cay ket qua GOP vao `results/shuf1_*`:
# cung may, cung instance, Pha 1 cung file, nen ghep cap theo fold van sach (muc 4).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-/venv/main/bin/python}"
RN="${RUN_NAME:-shuf1}"
BB_LIST="${BB_LIST:-codebert=microsoft/codebert-base:cls t5p=Salesforce/codet5p-220m-bimodal:mean}"

exec 4>/tmp/mvd_shuf2.lock || exit 1
flock -n 4 || { echo "DA CO shuf2 dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }

ARMS=( "real|data/phase1_common.jsonl" "shufall|data/phase1_common_shufall.jsonl"
       "shufpair|data/phase1_common_shufpair.jsonl" )

for a in "${ARMS[@]}"; do f="${a#*|}"; [ -f "$f" ] || { echo "!! THIEU $f"; exit 5; }; done
for f in run/matrix.sh src/train_transfer.py src/train_baseline.py; do
  [ -f "$f" ] || { echo "!! THIEU $f"; exit 5; }
done
# Pha 1 cu PHAI con day du, neu khong khoi nay se am tham huan luyen lai va doi hai bien.
miss=0
for BB in $BB_LIST; do L="${BB%%=*}"; for a in "${ARMS[@]}"; do A="${a%%|*}"
  [ -f "model/shuf1/phase1/${L}__none_com_${A}/seed_42/best.pt" ] || { echo "!! THIEU Pha 1 ${L}/${A}"; miss=1; }
done; done
[ "$miss" = 0 ] || { echo "!! Pha 1 cu khong du — DUNG (neu chay tiep se huan luyen lai va doi hai bien)"; exit 6; }
echo "Pha 1 cu: du 6/6"

run_arm() { # $1=BB  $2=FOLD  $3=arm  $4=data  $5=skip_baseline  $6=extra_p1  $7=epochs
  SKIP_BASELINE="$5" RUN_NAME="$RN" SEED=42 FOLDS="$2" \
  BACKBONES="$1" MODES=none OPTIMIZERS=adamw \
  PHASE1_DATA_PATH="$4" CWE_VOCAB=precomputed \
  ARM_TAG="_com_$3" PHASE1_TAG="_com_$3" PHASE1_STORE="model/shuf1/phase1" \
  LAMBDA_CWE=0.05 PHASE1_EPOCHS="$7" PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
  PHASE1_EXTRA="--sam_rho 0 $6" PHASE2_EXTRA="--sam_rho 0" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PYTHON="$PY" bash run/matrix.sh 4>&-
}

echo "########## SHUF2 bat dau $(ts) | $(hostname) ##########"
for BB in $BB_LIST; do
  L="${BB%%=*}"
  echo "==================== BACKBONE $L ===================="
  # (1) leo len n=5: fold 4 va 5 cho ba nhanh cu + baseline
  for FOLD in 4 5; do
    first=1
    for a in "${ARMS[@]}"; do
      A="${a%%|*}"; D="${a#*|}"
      echo "===== $(ts) | $L | FOLD $FOLD | $A (leo n=5) ====="
      run_arm "$BB" "$FOLD" "$A" "$D" "$([ $first = 1 ] && echo 0 || echo 1)" "--save_last_epoch" 12
      first=0
    done
  done
  # (2) nhanh realbest: chon-theo-val binh thuong, 15 epoch, ca 5 fold
  for FOLD in 1 2 3 4 5; do
    echo "===== $(ts) | $L | FOLD $FOLD | realbest (chon theo val) ====="
    run_arm "$BB" "$FOLD" "realbest" "data/phase1_common.jsonl" 1 "" 15
  done
  echo "----- $(ts) | het $L | o: $(find results/${RN}_${L} -name 'fold*.json' 2>/dev/null | wc -l) -----"
done
T=0; for BB in $BB_LIST; do L="${BB%%=*}"; n=$(find "results/${RN}_${L}" -name 'fold*.json' 2>/dev/null | wc -l); echo "  $L: $n o"; T=$((T+n)); done
echo "########## SHUF2 xong $(ts) | tong cong $T o trong cay shuf1 (ky vong 24+26=50) ##########"
