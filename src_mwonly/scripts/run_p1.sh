#!/usr/bin/env bash
# Pha 1 SOTA trên nguồn JS common + Java (pair loss), TỰ KHỞI ĐỘNG LẠI KHI KẸT.
# Kẹt = nghiệm tầm thường (val ROC ~0,5). Cuối epoch STUCK_EPOCH (mặc định 5), nếu train loss giảm chưa tới
# STUCK_MIN_DROP (mặc định 0,02 = 2 %, tương đối) so với epoch trước thì train.py dừng (mã thoát 3) và lượt được chạy
# lại với seed kế tiếp: SEED_P1, SEED_P1+1, ... tối đa P1_TRIES lần. Chạy từ thư mục refactorMWG/:
#   MODEL=<thư mục codebert-base> [SEED_P1=36] [TAG=sota] [P1_TRIES=3] bash scripts/run_p1.sh
# Checkpoint dùng được ghi vào model/<TAG>_p1/P1_CKPT (scripts/run_sota.sh đọc file này); log lượt kẹt giữ ở
# log/<TAG>_p1/f1_seed<S>_stuck.log. Chạy lại script thì bỏ qua seed đã kẹt và dùng lại checkpoint đã có.
# KEEP_STUCK=1: giữ checkpoint lượt kẹt thành .../seed_<S>/fold1/stuck.pt (để kiểm bằng Pha 2) thay vì xoá.
# P1_DATA (mặc định data/mwsrc_v4jsCjv_rand = pool SOTA): pool Pha 1, vd. pool dựng bằng dataset/pool_from_export.py.
# P1_SELECT (mặc định roc_auc): metric chọn best.pt, train_loss = train loss nhỏ nhất.
# P1_ALSO_SELECT (vd. train_loss): giữ thêm best_<metric>.pt cạnh best.pt để so hai cách chọn trên cùng một lượt.
# 04/10: mặc định 16 epoch, warmup 0,25 (lr đạt đỉnh cuối epoch 4). Lượt học vẫn đứng ở loss ~1,9 tới hết epoch 3 rồi mới
# bứt ra ở epoch 4–5, nên luật kẹt kiểm ở epoch 5 (so với epoch 4). Đo trên Colab: warmup 0,25 thoát kẹt 10/10 lượt
# (JS 9 seed, JS+Java 1 seed); warmup 0,10 hỏng 2/6 lượt (1 kẹt cả 16 epoch, 1 sập ngược ở epoch 4).
set -euo pipefail
: "${MODEL:?đặt MODEL (vd. /path/codebert-base)}"
PY=${PY:-python}
SEED_P1=${SEED_P1:-${SEED:-36}}
TAG=${TAG:-sota}
P1_TRIES=${P1_TRIES:-3}
STUCK_EPOCH=${STUCK_EPOCH:-5}
STUCK_MIN_DROP=${STUCK_MIN_DROP:-0.02}
KEEP_STUCK=${KEEP_STUCK:-0}
P1_SELECT=${P1_SELECT:-roc_auc}
P1_DATA=${P1_DATA:-data/mwsrc_v4jsCjv_rand}
P1_ALSO_SELECT=${P1_ALSO_SELECT:-}
P1=${TAG}_p1
mkdir -p log/$P1 model/$P1
rm -f model/$P1/P1_CKPT

for ((i = 0; i < P1_TRIES; i++)); do
  S=$((SEED_P1 + i))
  CK=model/$P1/multiwindow/seed_$S/fold1/best.pt
  LOG=log/$P1/f1_seed$S.log
  STUCK_LOG=log/$P1/f1_seed${S}_stuck.log
  if [ -f "$STUCK_LOG" ]; then
    echo "  P1 seed $S: đã kẹt ở lần chạy trước, bỏ qua"
    continue
  fi
  ARGS="--run_name $P1 --fold 1 --data_root $P1_DATA --seed $S --model_name $MODEL
        --window 510 --stride 384 --max_windows 8 --batch_size 4 --eval_batch_size 8 --micro 16 --learning_rate 2e-5
        --weight_decay 0.01 --warmup_ratio 0.25 --min_epochs 3 --patience 8 --selection_metric $P1_SELECT
        --num_workers 0 --agg mean --drop_bracket 1 --co_mode chain
        --graph_lr 2e-5 --epochs 16 --pair_loss 1.0 --pair_margin 1.0 --sam_rho 0"
  if [ ! -f "$CK" ]; then
    echo "  [$(date +%H:%M)] P1 seed $S: bắt đầu (lần $((i + 1))/$P1_TRIES)"
    RC=0
    $PY -u train.py --phase train $ARGS --stuck_epoch $STUCK_EPOCH --stuck_min_drop $STUCK_MIN_DROP \
      ${P1_ALSO_SELECT:+--also_select $P1_ALSO_SELECT} > $LOG 2>&1 || RC=$?
    if [ $RC -eq 3 ]; then
      if [ "$KEEP_STUCK" = 1 ] && [ -f "$CK" ]; then
        mv "$CK" "$(dirname $CK)/stuck.pt"
      else
        rm -f "$CK"
      fi
      mv $LOG $STUCK_LOG
      echo "  [$(date +%H:%M)] P1 seed $S KẸT: $(grep -o 'Kiểm kẹt.*' $STUCK_LOG) -> chạy lại với seed khác"
      continue
    fi
    if [ $RC -ne 0 ]; then
      echo "P1 seed $S lỗi (mã thoát $RC), xem $LOG" >&2
      exit $RC
    fi
  fi
  [ -f results/$P1/multiwindow/seed_$S/fold1.json ] || $PY -u train.py --phase test $ARGS >> $LOG 2>&1
  grep -hE "Epoch [0-9]+/[0-9]+|Kiểm kẹt" $LOG | sed -E 's/.*INFO - /  P1 /'
  echo "$CK" > model/$P1/P1_CKPT
  echo "  [$(date +%H:%M)] P1 xong: seed $S, checkpoint $CK"
  exit 0
done
echo "Pha 1 kẹt cả $P1_TRIES lần (seed $SEED_P1..$((SEED_P1 + P1_TRIES - 1))), xem log/$P1/*_stuck.log" >&2
exit 1
