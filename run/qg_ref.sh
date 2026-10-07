#!/usr/bin/env bash
# Runner MW / MW+BABEL (16/09, vast cuongtm). Giong q1r.sh nhung chon duoc TRAINER.
#   TRAINER=train_mw.py|train_mwg.py  TGT=sven|js|cc|src<name>  ARMS="tag:agg ..."  FOLDS="1 2 3"
#   INIT=none|encoder|encoder_agg|all  INIT_CKPT=...  SAM=0.02  MWEXTRA="--recadam 1 ..."  KEEP_CK=1
set -uo pipefail
: "${ROOTDIR:?}" "${PY:?}" "${M:?}" "${TGT:?}" "${ARMS:?}"
cd "$ROOTDIR"; S=${SEED:-36}; INIT=${INIT:-none}; K=${K:-8}; FOLDS=${FOLDS:-"1 2 3"}; TRAINER=${TRAINER:-train_mwg.py}
case "$TGT" in
  cc) R=data/pr_prime_ccpp_folds; LANG=ccpp ;;
  js) R=data/pr_jsc_folds; LANG=js ;;
  sven) R=data/sven_python_folds_norm; LANG=python ;;
  src*) NAME=${TGT#src}; R=data/mwsrc_$NAME; FOLDS=1
        case "$NAME" in sven) LANG=python ;; ccpp) LANG=ccpp ;; *) LANG="" ;; esac ;;
  *) echo "!! TGT"; exit 2 ;;
esac
TL=""; [ -n "$LANG" ] && TL="--target_lang $LANG"
COMMON="--seed $S --model_name $M --data_root $R $TL --window 510 --stride 384
        --max_windows $K --batch_size 4 --eval_batch_size 8 --micro 16 --learning_rate 2e-5
        --weight_decay 0.01 --warmup_ratio 0.10 --epochs 30 --min_epochs 3 --patience 8
        --selection_metric roc_auc --num_workers 0 --sam_rho ${SAM:-0}"
for A in $ARMS; do TAG=${A%%:*}; AGG=${A#*:}
  mkdir -p log/$TAG results/$TAG/multiwindow/seed_$S
  for F in $FOLDS; do
    [ -f "results/$TAG/multiwindow/seed_$S/fold$F.json" ] && continue
    CK=model/$TAG/multiwindow/seed_$S/fold$F/best.pt; mkdir -p "$(dirname $CK)"
    EXTRA=""; [ "$INIT" != none ] && EXTRA="--init $INIT --init_ckpt ${INIT_CKPT:?}"
    echo "  [$(date +%H:%M)] $TAG f$F bat dau ($TRAINER)"
    $PY -u src/$TRAINER --phase train --run_name $TAG --fold $F --agg $AGG --checkpoint_path "$CK" $EXTRA $COMMON ${MWEXTRA:-} \
        > log/$TAG/f$F.log 2>&1 \
     && $PY -u src/$TRAINER --phase test --run_name $TAG --fold $F --agg $AGG --checkpoint_path "$CK" ${MWEXTRA:-} \
        --result_path results/$TAG/multiwindow/seed_$S/fold$F.json $COMMON >> log/$TAG/f$F.log 2>&1 \
     || { echo "  !! $TAG f$F LOI"; tail -4 log/$TAG/f$F.log; }
    [ -f "results/$TAG/multiwindow/seed_$S/fold$F.json" ] && [ "${KEEP_CK:-0}" != 1 ] && rm -f "$CK"
    echo "  [$(date +%H:%M)] $TAG f$F xong: $(grep -o '"test_roc_auc": [0-9.]*' results/$TAG/multiwindow/seed_$S/fold$F.json 2>/dev/null)"
  done
done
echo "########## QG $TGT $ARMS XONG $(date '+%F %T') ##########"
