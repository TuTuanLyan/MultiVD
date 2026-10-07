#!/usr/bin/env bash
# TAI LAP doi chung cua dong nghiep: `mwgU_b_sven` (MW + do thi dong nhat, target-only).
# Dung DUNG co cua ho trong qg.sh: seed 36, SAM 0, va MWEXTRA="--drop_bracket 1 --co_mode chain".
# Ky vong (ho cong bo, 3 fold): 0.8936 / 0.9440 / 0.9009  -> TB 0.9128
#
# Chay bang MA CUA HO (src_mwg/), khong phai ma cua ta — muc dich la kiem xem cung may
# cung du lieu thi co ra dung so cua ho khong. Neu khong, moi la loi moi truong.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/workspace/MultiVD/.hf}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-/venv/main/bin/python}"
TAG="${TAG:-mwgU_b_sven}"
S="${SEED:-36}"
FOLDS="${FOLDS:-1 2 3}"
SAM="${SAM:-0}"
MWEXTRA="${MWEXTRA:---drop_bracket 1 --co_mode chain}"
M="${MODEL_NAME:-microsoft/codebert-base}"
R=data/sven_python_folds_norm

exec 4>"${LOCKF:-/tmp/mvd_mwg.lock}" || exit 1
flock -n 4 || { echo "DA CO mwg dang chay"; exit 3; }
PIDFILE="${PIDFILE:-/tmp/mwg.pid}"; echo "$$" > "$PIDFILE"; trap 'rm -f "$PIDFILE"' EXIT
ts(){ date -u '+%F %T'; }

[ -f src_mwg/train_mwg.py ] || { echo "!! THIEU src_mwg/train_mwg.py"; exit 5; }
[ -f "$R/fold1/test.jsonl" ] || { echo "!! THIEU du lieu $R"; exit 5; }
# CONG NAP MODEL: hong o day thi DUNG NGAY, dung dot ca 3 fold roi moi bao.
# (08:03 18/09 da mat 3 fold trong 16 giay vi thieu cong nay.)
MODEL_NAME="$M" "$PY" - <<'PYCHK' || { echo "!! KHONG nap duoc model offline (HF_HOME=$HF_HOME)"; exit 5; }
import os, sys; sys.path.insert(0, "src_mwg")
from transformers import AutoTokenizer
from model import build_backbone
m = os.environ["MODEL_NAME"]
AutoTokenizer.from_pretrained(m); build_backbone(m)
print(f"  model offline OK: {m}")
PYCHK
"$PY" -c "import sys;sys.path.insert(0,'src_mwg');import train_mwg" || { echo "!! khong import duoc"; exit 5; }

COMMON="--seed $S --model_name $M --data_root $R --target_lang python
        --window 510 --stride 384 --max_windows 8 --batch_size 4 --eval_batch_size 8 --micro 16
        --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.10 --epochs 30 --min_epochs 3
        --patience 8 --selection_metric roc_auc --num_workers 0 --sam_rho $SAM"

echo "########## MWG_REPRO bat dau $(ts) | $(hostname) | $TAG | seed $S | SAM $SAM | fold: $FOLDS ##########"
echo "  MWEXTRA: $MWEXTRA"
mkdir -p log/$TAG results/$TAG/multiwindow/seed_$S
for F in $FOLDS; do
  OUT="results/$TAG/multiwindow/seed_$S/fold$F.json"
  [ -f "$OUT" ] && { echo "=== $(ts) | fold $F da co ==="; continue; }
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
  echo "===== $(ts) | FOLD $F | GPU trong ${FREE:-?} MiB ====="
  CK=model/$TAG/multiwindow/seed_$S/fold$F/best.pt; mkdir -p "$(dirname $CK)"
  "$PY" -u src_mwg/train_mwg.py --phase train --run_name $TAG --fold $F --agg mean \
      --checkpoint_path "$CK" $COMMON $MWEXTRA > log/$TAG/f$F.log 2>&1 \
   && "$PY" -u src_mwg/train_mwg.py --phase test --run_name $TAG --fold $F --agg mean \
      --checkpoint_path "$CK" --result_path "$OUT" $COMMON $MWEXTRA >> log/$TAG/f$F.log 2>&1 \
   || { echo "  !! fold $F LOI"; tail -5 log/$TAG/f$F.log | sed 's/^/     /'; }
  if [ -f "$OUT" ]; then
    echo "  $(ts) fold $F XONG: $(grep -o '\"test_roc_auc\": [0-9.]*' "$OUT" | head -1)"
    rm -f "$CK"
  fi
done
echo "########## MWG_REPRO xong $(ts) | $(ls results/$TAG/multiwindow/seed_$S/fold*.json 2>/dev/null | wc -l) o ##########"
