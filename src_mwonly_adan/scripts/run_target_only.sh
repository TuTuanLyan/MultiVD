#!/usr/bin/env bash
# Target-only (mốc không chuyển giao): cùng kiến trúc MWG, chỉ huấn luyện trên đích SVEN v4, AdamW trần
# (không khởi tạo từ nguồn, không RecAdam, không SAM). Chạy từ thư mục refactorMWG/:
#   MODEL=<thư mục codebert-base> [SEED=36] [TAG=target_only] bash scripts/run_target_only.sh
set -euo pipefail
: "${MODEL:?đặt MODEL (vd. /path/codebert-base)}"
PY=${PY:-python}
SEED=${SEED:-36}
TAG=${TAG:-target_only}
FOLDS=${FOLDS:-"1 2 3 4 5"}
mkdir -p log/$TAG
for F in $FOLDS; do
  [ -f results/$TAG/multiwindow/seed_$SEED/fold$F.json ] && continue
  ARGS="--run_name $TAG --fold $F --data_root data/sven_python_folds_v4 --target_lang python --seed $SEED --model_name $MODEL
        --window 510 --stride 384 --max_windows 8 --batch_size 4 --eval_batch_size 8 --micro 16 --learning_rate 2e-5
        --weight_decay 0.01 --warmup_ratio 0.10 --epochs 30 --min_epochs 3 --patience 8 --selection_metric roc_auc
        --num_workers 0 --agg mean --drop_bracket 1 --co_mode chain --sam_rho 0"
  $PY -u train.py --phase train $ARGS > log/$TAG/f$F.log 2>&1
  $PY -u train.py --phase test $ARGS >> log/$TAG/f$F.log 2>&1
  rm -f model/$TAG/multiwindow/seed_$SEED/fold$F/best.pt
  echo "  [$(date +%H:%M)] $TAG f$F: $(grep -o '"test_roc_auc": [0-9.]*' results/$TAG/multiwindow/seed_$SEED/fold$F.json)"
done
$PY scripts/summarize.py results/$TAG/multiwindow/seed_$SEED
