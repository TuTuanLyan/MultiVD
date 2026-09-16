#!/usr/bin/env bash
# MW_ASSEMBLE — nhieu cua so, chay DUNG co chuan cua DE_XUAT_1_LATE_FUSION.md §B.3.
#
# Hai nhanh, CUNG cay CUNG may CUNG phien:
#   mwK8 : --max_windows 8   <- DUNG cau hinh de xuat, de doi chung voi so cua dong nghiep
#   mwK1 : --max_windows 1   <- MOI CO KHAC GIONG HET. Day la phep tach duy nhat cho biet
#                               rieng phan CUA SO dang bao nhieu.
#
# Vi sao can nhanh thu hai: chinh de xuat (dong 449) ghi "hieu ung cua so thuan ~ 0 (§131),
# phan tang cua MW target-only den tu batch 4 (+2,8) va SAM (+1,9)". Co chuan di kem batch 4
# va SAM 0.02, nen so mwK8 voi baseline cu (batch 16, khong SAM) la doi BA bien.
#
# Bac 1 — kiem chung, n=3 fold, codebert (spec MW la codebert). Fold la vong NGOAI.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${HF_HOME:-}" ] && [ -d /workspace ]; then export HF_HOME=/workspace/.hf_home; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-python3}"
RN="${RUN_NAME:-mw_assemble}"
FOLDS_LIST="${FOLDS:-1 2 3}"
SEED="${SEED:-42}"
ARMS="${ARMS:-mwK8:8 mwK1:1}"

exec 4>/tmp/mvd_mw.lock || exit 1
flock -n 4 || { echo "DA CO mw_assemble dang chay"; exit 3; }
PIDFILE="${PIDFILE:-/tmp/mw_assemble.pid}"; echo "$$" > "$PIDFILE" 2>/dev/null || true
trap 'rm -f "$PIDFILE"' EXIT
ts(){ date -u '+%F %T'; }

[ -f src/train_mw.py ] || { echo "!! THIEU src/train_mw.py"; exit 5; }
[ -f data/sven_python_folds_norm/fold1/test.jsonl ] || { echo "!! THIEU du lieu dich"; exit 5; }
"$PY" -c "import torch,transformers,sklearn;print('env OK',torch.__version__)" || exit 5
"$PY" - <<'PYCHECK' || { echo "!! KHONG nap duoc model offline"; exit 5; }
from transformers import AutoTokenizer, AutoConfig
AutoTokenizer.from_pretrained("microsoft/codebert-base"); AutoConfig.from_pretrained("microsoft/codebert-base")
print("  HF offline OK")
PYCHECK
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
echo "GPU trong: ${FREE:-?} MiB"
[ -n "${FREE:-}" ] && { (( FREE >= 7000 )) || { echo "!! chi con ${FREE} MiB (<7000) — KHONG phong"; exit 1; }; }

echo "########## MW_ASSEMBLE bat dau $(ts) | $(hostname) | fold: $FOLDS_LIST | $ARMS ##########"
for FOLD in $FOLDS_LIST; do
  for A in $ARMS; do
    NAME="${A%%:*}"; K="${A##*:}"
    OUT="results/${RN}_codebert/${NAME}/seed_${SEED}/fold${FOLD}.json"
    if [ -f "$OUT" ]; then echo "=== $(ts) | fold $FOLD | $NAME | da co ==="; continue; fi
    echo "===== $(ts) | FOLD $FOLD | $NAME (max_windows=$K) ====="
    mkdir -p "log/${RN}"
    "$PY" -u src/train_mw.py \
      --run_name "${RN}_codebert" --method_name "$NAME" --fold "$FOLD" --seed "$SEED" \
      --window 510 --stride 384 --max_windows "$K" --max_length 512 --agg mean \
      --batch_size 4 --eval_batch_size 8 --micro 16 \
      --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.10 \
      --epochs 30 --min_epochs 3 --patience 8 --selection_metric roc_auc --sam_rho 0.02 \
      >> "log/${RN}/${NAME}_fold${FOLD}.log" 2>&1
    [ -f "$OUT" ] || { echo "  !! $NAME fold$FOLD THAT BAI — xem log/${RN}/${NAME}_fold${FOLD}.log"
                       tail -3 "log/${RN}/${NAME}_fold${FOLD}.log" | sed 's/^/       /'; }
  done
  echo "----- $(ts) | het fold $FOLD | o: $(find results/${RN}_codebert -name 'fold*.json' 2>/dev/null | wc -l)/6 -----"
done
echo "########## MW_ASSEMBLE xong $(ts) | $(find results/${RN}_codebert -name 'fold*.json' 2>/dev/null | wc -l)/6 o ##########"
