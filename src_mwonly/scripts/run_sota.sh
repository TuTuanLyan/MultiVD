#!/usr/bin/env bash
# SOTA: Pha 1 trên nguồn JS common + Java (1 lần) -> Pha 2 trên đích SVEN v4, 5 fold, RecAdam + ASAM rho 0,5.
# Chạy từ thư mục refactorMWG/ sau scripts/build_data.sh:
#   MODEL=<thư mục codebert-base> [SEED=36] [SEED_P1=36] [TAG=sota] bash scripts/run_sota.sh
# P1_DATA / P2_DATA: pool Pha 1 (mặc định data/mwsrc_v4jsCjv_rand) / đích 5 fold (mặc định data/sven_python_folds_v4).
# Pha 1 do scripts/run_p1.sh chạy: tự khởi động lại với seed kế tiếp khi kẹt (P1_TRIES, STUCK_EPOCH, STUCK_MIN_DROP).
set -euo pipefail
: "${MODEL:?đặt MODEL (vd. /path/codebert-base)}"
PY=${PY:-python}
SEED=${SEED:-36}
SEED_P1=${SEED_P1:-$SEED}
TAG=${TAG:-sota}
FOLDS=${FOLDS:-"1 2 3 4 5"}
P2_DATA=${P2_DATA:-data/sven_python_folds_v4}
P2_PATIENCE_AFTER=${P2_PATIENCE_AFTER:-0}   # >0: Pha 2 chỉ đếm patience khi train loss < giá trị này (0 = như cũ)
COMMON="--model_name $MODEL --window 510 --stride 384 --max_windows 8 --batch_size 4 --eval_batch_size 8 --micro 16
        --learning_rate 2e-5 --weight_decay 0.01 --warmup_ratio 0.10 --min_epochs 3 --patience 8 --selection_metric roc_auc
        --num_workers 0 --agg mean --drop_bracket 1 --co_mode chain"

MODEL=$MODEL PY=$PY TAG=$TAG SEED_P1=$SEED_P1 bash scripts/run_p1.sh
CK1=$(cat model/${TAG}_p1/P1_CKPT)

P2=${TAG}_p2
mkdir -p log/$P2
for F in $FOLDS; do
  [ -f results/$P2/multiwindow/seed_$SEED/fold$F.json ] && continue
  ARGS2="--run_name $P2 --fold $F --data_root $P2_DATA --target_lang python --seed $SEED $COMMON
         --epochs 30 --recadam 1 --pretrain_cof 500 --anneal_t0_ratio 0.01 --sam_rho 0.5 --sam_variant asam --sam_eta 0.01
         --patience_after_loss $P2_PATIENCE_AFTER"
  $PY -u train.py --phase train $ARGS2 --init all --init_ckpt "$CK1" > log/$P2/f$F.log 2>&1
  $PY -u train.py --phase test $ARGS2 >> log/$P2/f$F.log 2>&1
  rm -f model/$P2/multiwindow/seed_$SEED/fold$F/best.pt
  echo "  [$(date +%H:%M)] $P2 f$F: $(grep -o '"test_roc_auc": [0-9.]*' results/$P2/multiwindow/seed_$SEED/fold$F.json)"
done
$PY scripts/summarize.py results/$P2/multiwindow/seed_$SEED
