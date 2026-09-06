#!/usr/bin/env bash
# NIGHT48 — lambda=0.05, latent_bottleneck, RecAdam, codet5p-220m-bimodal.
# DU BA NGUON (4cwe / com / full) x 3 seed x 5 fold x HAI NHANH rho = 90 o Pha 2,
# cong 15 o baseline.
#
#   rho=0    doi chung — Pha 2 khong co ASAM
#   rho=0.1  nhanh chinh — ASAM, eta=0.01
#
# Hai nhanh DUNG CHUNG mot checkpoint Pha 1 (rho chi vao Pha 2, muc 5 CLAUDE.md),
# nen hieu giua chung doi dung MOT bien. Khong co rho=0 thi chi biet 'cau hinh nay
# tot', khong biet ASAM co phai la thu tao ra cai tot do khong.
#
#   SEEDS="42 7 1234" PYTHON=/venv/main/bin/python bash run/night48.sh
#
# THU TU: seed la vong ngoai cung, trong moi seed thi FOLD la vong ngoai va NGUON la
# vong trong (muc 1 CLAUDE.md). Vi vay:
#   - het fold 1 cua mot seed la da co mot lat cat so duoc ngay: baseline + ca ba
#     nguon tren cung fold cung seed cung may;
#   - het seed 42 la da co n=5 day du; seed 7 nang len n=10; seed 1234 len n=15.
# Neu chay theo nguon-truoc thi phai gan xong moi co o nao so duoc voi o nao.
#
# Pha 1 cua ca ba nguon duoc huan luyen TRUOC vong fold cua seed do, vi mot checkpoint
# Pha 1 dung chung cho ca 5 fold (fold la cach chia tap DICH, khong dinh gi den Pha 1).
#
# PHASE1_MIN_VAL=0 — CHAY HET, ke ca khi Pha 1 yeu (muc 3 CLAUDE.md). `full` la nguon
# hay lam Pha 1 sap nhat; o sap van phai co so, kem theo val Pha 1 de nguoi doc biet.
# Bo o vi Pha 1 yeu chinh la thien lech chon loc.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEEDS="${SEEDS:-42 7 1234}"
SOURCES="${SOURCES:-4cwe com full}"
FOLD_LIST="${FOLD_LIST:-1 2 3 4 5}"
LAM="${LAM:-0.05}"
RHOS="${RHOS:-0 0.1}"   # doi chung truoc, nhanh chinh sau
BB="${BB:-t5p=Salesforce/codet5p-220m-bimodal:mean}"
LTAG="l$(printf '%s' "$LAM" | tr '.' 'p')"
LABEL="${BB%%=*}"

data_of(){ case "$1" in
  4cwe) echo "data/phase1_4cwe.jsonl fixed4" ;;
  com)  echo "data/phase1_common.jsonl precomputed" ;;
  full) echo "data/phase1_full.jsonl precomputed" ;;
  *) echo "" ;; esac; }

LOCK="${MVD_LOCK:-/tmp/multivd_night48.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO driver night48 dang chay"; exit 3; }
mkdir -p log

common_env(){ # $1=src $2=data $3=vocab $4=seed $5=folds $6=armtag $7=p2extra
  RUN_NAME=n48 SEED="$4" FOLDS="$5" \
  BACKBONES="$BB" MODES="latent_bottleneck" OPTIMIZERS=recadam \
  PHASE1_DATA_PATH="$2" CWE_VOCAB="$3" \
  ARM_TAG="$6" PHASE1_TAG="_${1}_${LTAG}" \
  LAMBDA_CWE="$LAM" PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 \
  PHASE1_EXTRA="--sam_rho 0" PHASE2_EXTRA="$7" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  bash run/matrix.sh
}

echo "########## NIGHT48 bat dau $(date -u '+%F %T') ##########"
echo "  $LABEL | latent_bottleneck | recadam | lambda $LAM | rho: $RHOS (0 = doi chung)"
echo "  nguon: $SOURCES | seed: $SEEDS | fold: $FOLD_LIST"
NS=$(echo $SOURCES|wc -w); NE=$(echo $SEEDS|wc -w); NF=$(echo $FOLD_LIST|wc -w); NR=$(echo $RHOS|wc -w)
EXP=$(( NS*NE*NF*NR )); EXPB=$(( NE*NF ))
echo "  ky vong: ${NS}x${NE}x${NF}x${NR} = $EXP o Pha 2 + $EXPB baseline"

for SEED in $SEEDS; do
  echo "===== $(date -u '+%F %T') | SEED $SEED | Pha 1 cho moi nguon ====="
  for SRC in $SOURCES; do
    read -r DATA VOCAB <<< "$(data_of "$SRC")"
    CKPT="model/n48/phase1/${LABEL}__latent_bottleneck_${SRC}_${LTAG}/seed_${SEED}/best.pt"
    if [[ -f "$CKPT" ]]; then echo "  [$SRC] da co checkpoint Pha 1 — bo qua"; continue; fi
    common_env "$SRC" "$DATA" "$VOCAB" "$SEED" "" "_${SRC}_${LTAG}" "--sam_rho 0"
  done

  # Bao cao checkpoint nao co, nao thieu — TRUOC khi vao vong fold, de doc log la biet
  # ngay o nao se trong va vi sao, thay vi phai suy tu 45 dong ket qua.
  for SRC in $SOURCES; do
    CKPT="model/n48/phase1/${LABEL}__latent_bottleneck_${SRC}_${LTAG}/seed_${SEED}/best.pt"
    if [[ -f "$CKPT" ]]; then
      V=$(PYTHONWARNINGS=ignore ${PYTHON:-python} - "$CKPT" <<'PY' 2>/dev/null
import sys,torch
try:
    c=torch.load(sys.argv[1],map_location="cpu",weights_only=False)
    print("val=%.4f ep=%s"%(float(c.get("best_val_macro_f1") or 0),c.get("best_epoch")))
except Exception as e: print("KHONG DOC DUOC: %s"%e)
PY
)
      echo "  [seed $SEED / $SRC] Pha 1 OK — $V"
    else
      echo "  [seed $SEED / $SRC] !! THIEU checkpoint Pha 1 — 5 o cua nguon nay se TRONG"
    fi
  done

  # rho la vong TRONG CUNG: hai nhanh cua cung (fold, nguon) chay CANH NHAU nen hieu
  # giua chung ghep cap duoc theo dung fold do, va neu khoi dut giua chung thi cai da
  # chay xong luon la nhung cap tron ven chu khong phai mot nua nhanh chinh.
  for FOLD in $FOLD_LIST; do
    for SRC in $SOURCES; do
      read -r DATA VOCAB <<< "$(data_of "$SRC")"
      CKPT="model/n48/phase1/${LABEL}__latent_bottleneck_${SRC}_${LTAG}/seed_${SEED}/best.pt"
      [[ -f "$CKPT" ]] || continue
      for RHO in $RHOS; do
        RTAG="r$(printf '%s' "$RHO" | tr '.' 'p')"
        if [[ "$RHO" == "0" ]]; then P2="--sam_rho 0"
        else P2="--sam_rho $RHO --sam_variant asam --asam_eta 0.01"; fi
        echo "===== $(date -u '+%F %T') | seed $SEED | fold $FOLD | nguon $SRC | rho $RHO ====="
        common_env "$SRC" "$DATA" "$VOCAB" "$SEED" "$FOLD" "_${SRC}_${LTAG}_${RTAG}" "$P2"
      done
    done
  done

  N=$(ls results/n48_${LABEL}/transfer_latent_bottleneck_*_${LTAG}_r*/seed_${SEED}/fold*.json 2>/dev/null | wc -l)
  B=$(ls results/n48_${LABEL}/baseline/seed_${SEED}/fold*.json 2>/dev/null | wc -l)
  echo "########## NIGHT48 seed $SEED xong $(date -u '+%F %T') | $N/$(( NS*NF*NR )) o + $B/$NF baseline ##########"
done

TOT=$(ls results/n48_${LABEL}/transfer_latent_bottleneck_*_${LTAG}_r*/seed_*/fold*.json 2>/dev/null | wc -l)
TOTB=$(ls results/n48_${LABEL}/baseline/seed_*/fold*.json 2>/dev/null | wc -l)
echo "########## NIGHT48 xong $(date -u '+%F %T') | $TOT/$EXP o Pha 2 + $TOTB/$EXPB baseline ##########"
