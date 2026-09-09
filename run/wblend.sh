#!/usr/bin/env bash
# WBLEND — noi suy TRONG SO baseline <-> chuyen giao. Cau hoi CHUA AI TRA LOI trong du an.
#
# §25 do duoc phep TRON XAC SUAT vuot ca hai dau mut (ROC +0.0165 so voi chuyen giao
# thuan, 59/83 khoi, p=0.0002; song sot phep kiem ro ri §25.4). Nhung no ton HAI LAN
# suy luan. Neu tron duoc trong khong gian TRONG SO thi chi phi ve lai MOT mo hinh —
# do la khac biet giua mot meo ensemble va mot phuong thuc trien khai duoc.
#
# CO CHE DA KIEM TRUOC, 0 GPU (src/model.py): BaselineModel va TransferModel chia dung
# `backbone.*` + `vul_head.*` — cung ten cung shape. `latent_proj`/`cwe_head` chi co o
# ban chuyen giao va KHONG nam tren duong tinh vul_logits, nen giu nguyen. Vay noi suy
# xac dinh duoc cho moi tensor co tac dong len du doan.
#
# BAC 1 (sang loc, 3 fold, seed 42). alpha chon tren VAL, bao tren TEST.
# Khoi nay cung sinh ra o co `val_probabilities` (ban va 09/09) nen sau do chon duoc
# he so tron XAC SUAT tren val — vay la go luon gioi han so 2 cua §25.
#
# KEEP_CKPT=1: matrix.sh mac dinh `rm -rf` checkpoint sau moi o. Bat co nay de giu lai,
# VA PHAI DON TAY sau moi fold — moi checkpoint ~450MB, bon cai mot fold ~1.8GB.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$( for c in /venv/main/bin/python /data/ntat/envs/vdenv/bin/python \
      /home/ntat/miniconda3/envs/vdenv/bin/python; do
    [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { echo "$c"; break; }; done )}"
[ -n "$PYTHON" ] || { echo "!! khong tim thay python co torch"; exit 2; }
export PYTHON
RUN="${RUN:-wb}"
SEED="${SEED:-42}"
SOURCES_LIST="${SOURCES_LIST:-4cwe com full}"
FOLD_LIST="${FOLD_LIST:-1 2 3}"
BB="${BB:-t5p=Salesforce/codet5p-220m-bimodal:mean}"
LABEL="${BB%%=*}"; REST="${BB#*=}"; MODEL="${REST%%:*}"; POOL="${REST##*:}"
GRID="${GRID:-0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0}"
OUT="results/wblend_${LABEL}"
mkdir -p "$OUT" log

# Dia: moi fold giu ~1.8GB checkpoint. Dung han neu khong du cho, dung de torch.save
# ghi cut roi bi doc thanh "phuong phap kem" (§16).
FREE=$(df -Pm . | awk 'NR==2{print $4}')
(( FREE > 12000 )) || { echo "!! chi con ${FREE}MB — can >12GB de giu checkpoint. DUNG."; exit 4; }

echo "########## WBLEND bat dau $(date -u '+%F %T') | $(hostname) ##########"
echo "  $LABEL ($MODEL, pooling=$POOL) | nguon: $SOURCES_LIST | fold: $FOLD_LIST | seed $SEED"
echo "  dia con ${FREE}MB"

for FOLD in $FOLD_LIST; do
  echo "===== $(date -u '+%F %T') | FOLD $FOLD | huan luyen (giu checkpoint) ====="
  for SRC in $SOURCES_LIST; do
    KEEP_CKPT=1 RUN=$RUN SEEDS="$SEED" SOURCES="$SRC" FOLD_LIST="$FOLD" MIN_EP=3 BB="$BB" \
    CONFIGS="plain|adamw|--sam_rho 0" \
    bash run/opt1.sh
  done

  BCK="model/${RUN}_${LABEL}/baseline/seed_${SEED}/fold${FOLD}/best.pt"
  if [[ ! -f "$BCK" ]]; then
    echo "  !! thieu checkpoint baseline $BCK — bo fold $FOLD"; continue
  fi
  for SRC in $SOURCES_LIST; do
    TCK="model/${RUN}_${LABEL}/transfer_latent_bottleneck_${SRC}_l0p05_plain_adamw/seed_${SEED}/fold${FOLD}/best.pt"
    if [[ ! -f "$TCK" ]]; then
      echo "  !! thieu checkpoint chuyen giao $TCK — bo ($SRC, fold $FOLD)"; continue
    fi
    echo "----- $(date -u '+%F %T') | noi suy trong so | fold $FOLD | $SRC -----"
    $PYTHON tools/wblend.py --baseline_ckpt "$BCK" --transfer_ckpt "$TCK" --fold "$FOLD" \
      --model_name "$MODEL" --pooling "$POOL" --grid "$GRID" \
      --out "$OUT/${SRC}_seed${SEED}_fold${FOLD}.json" \
      2>&1 | grep -vE '^(INFO|WARNING|Loading)' | tail -20
  done

  # DON NGAY sau moi fold, khong doi het khoi: giu ca 3 fold cung luc la ~5.4GB.
  echo "  don checkpoint fold $FOLD"
  rm -rf "model/${RUN}_${LABEL}"/*/seed_${SEED}/fold${FOLD}
done
echo "########## WBLEND xong $(date -u '+%F %T') | $(find "$OUT" -name '*.json' | wc -l) file ##########"
