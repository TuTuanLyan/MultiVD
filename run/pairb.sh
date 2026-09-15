#!/usr/bin/env bash
# PAIRB — De xuat B: mat mat BIEN TRONG CAP o Pha 1, thay vi chi BCE tung dong.
#
# Du doan + nguong ghi TRUOC khi do: records/prediction_2026-09-15_bien_trong_cap.md
#
# Ba nhanh, khac DUNG MOT bien (ham muc tieu Pha 1):
#   baseline  khong Pha 1
#   bce       Pha 1 BCE thuan            (aux_mode=none)
#   pairB     Pha 1 BCE + beta*bien-cap  (aux_mode=none, --pair_margin_beta 0.5 --pair_margin_m 1.0)
#
# Pha 2 THUAN: khong adapter, khong fusion, aux_mode=none, adamw, sam 0 — de co lap DUNG
# ham muc tieu Pha 1. Bo head phu vi §56 vua cho thay no khong dong gop gi (+0.0053, 5/10).
#
# Bac 1 — kiem chung, n=3 fold, seed 42, CA HAI backbone (cong 2). 24 o + 4 lan Pha 1.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-/venv/main/bin/python}"
RN="${RUN_NAME:-pairb}"
BETA="${BETA:-0.5}"
MARGIN="${MARGIN:-1.0}"
FOLDS_LIST="${FOLDS:-1 2 3}"
BB_LIST="${BB_LIST:-codebert=microsoft/codebert-base:cls t5p=Salesforce/codet5p-220m-bimodal:mean}"

exec 4>/tmp/mvd_pairb.lock || exit 1
flock -n 4 || { echo "DA CO pairb dang chay"; exit 3; }
PIDFILE="${PIDFILE:-/workspace/pairb.pid}"
echo "$$" > "$PIDFILE" 2>/dev/null || true
trap 'rm -f "$PIDFILE"' EXIT
ts(){ date -u '+%F %T'; }

for f in run/matrix.sh src/train_transfer.py src/train_baseline.py src/pairloss.py data/phase1_common.jsonl; do
  [ -f "$f" ] || { echo "!! THIEU $f"; exit 5; }
done
"$PY" -c "import torch,transformers,sklearn" || exit 5
# Cong kiem NOI DUNG: du lieu phai that su co cap, neu khong thi loss luon bang 0 ma khong bao gi
"$PY" - <<'PYEOF' || exit 5
import json,sys
sys.path.insert(0,"src")
from pairloss import build_pair_index
recs=[json.loads(l) for l in open("data/phase1_common.jsonl")]
g,r,n=build_pair_index(recs)
cov=100.0*int((g>=0).sum())/len(recs)
print(f"  cap day du: {n} | {cov:.1f}% so dong nam trong cap")
if n < 100: print("!! qua it cap — DUNG"); sys.exit(1)
PYEOF

ARMS=( "bce|" "pairB|--pair_margin_beta $BETA --pair_margin_m $MARGIN" )

echo "########## PAIRB bat dau $(ts) | beta=$BETA margin=$MARGIN | $(hostname) ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for BB in $BB_LIST; do
  L="${BB%%=*}"
  echo "==================== BACKBONE $L ===================="
  for FOLD in $FOLDS_LIST; do
    first=1
    for a in "${ARMS[@]}"; do
      A="${a%%|*}"; EXTRA="${a#*|}"
      echo "===== $(ts) | $L | FOLD $FOLD | nhanh $A ====="
      SKIP_BASELINE=$([ "$first" = 1 ] && echo 0 || echo 1) \
      RUN_NAME="$RN" SEED=42 FOLDS="$FOLD" \
      BACKBONES="$BB" MODES=none OPTIMIZERS=adamw \
      PHASE1_DATA_PATH=data/phase1_common.jsonl CWE_VOCAB=precomputed \
      ARM_TAG="_com_$A" PHASE1_TAG="_com_$A" PHASE1_STORE="model/${RN}/phase1" \
      LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_PATIENCE=10 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
      PHASE1_EXTRA="--sam_rho 0 $EXTRA" PHASE2_EXTRA="--sam_rho 0" \
      DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
      PYTHON="$PY" bash run/matrix.sh 4>&-
      first=0
    done
    echo "----- $(ts) | het $L fold $FOLD | o: $(find results/${RN}_${L} -name 'fold*.json' 2>/dev/null | wc -l) -----"
  done
done
T=0; for BB in $BB_LIST; do L="${BB%%=*}"; n=$(find "results/${RN}_${L}" -name 'fold*.json' 2>/dev/null | wc -l); echo "  $L: $n o"; T=$((T+n)); done
echo "########## PAIRB xong $(ts) | $T/18 o ##########"
