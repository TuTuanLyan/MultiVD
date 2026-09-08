#!/usr/bin/env bash
# E60 — cho neo NGAN SACH GAP DOI (60 epoch thay vi 30). Bac 1: 3 fold.
#
# VI SAO: day la CHO DUY NHAT trong toan du an ma cai neo lam duoc dieu AdamW khong lam duoc.
# Khoi 06/09 (CHI fold 3, n=1 moi o) cho thay:
#   plain AdamW  best_epoch 8 va 9  -> ket qua 60 epoch TRUNG KHIT 30 epoch (+0.0000 ca F1
#                lan AUC). No da dung tu lau truoc moc 30, cho them gap doi cung khong dung.
#   co neo       best_epoch 16..39  -> van con cai thien khi duoc chay dai hon.
# Nhung n=1 va cac Delta mau thuan nhau (gamma=5 tren 4cwe -0.0801, tren com +0.0398), nen
# chua ket luan duoc gi. Khoi nay dua len 3 fold.
#
# Bo nhanh c5000_t0p5k005: da biet no sap ve 0.47/0.48, khong can do lai.
# Giu `plain` lam doi chung o DUNG 60 epoch, cung phien cung may.
#
# CONG VRAM TRUOC TUNG O — 161 la GPU dung chung, va cong chi kiem mot lan luc khoi dong da
# lam hong 14 o cua POOL1 hom nay khi mot nguoi dung khac gianh bo nho giua chung.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/home/ntat/miniconda3/envs/vdenv/bin/python}"
# 08/09: NEED=10500 KHONG DU. Cong kiem TRUOC khi chay, nhung nguoi dung khac tren 161
# gianh bo nho TRONG LUC chay -> 12/12 o Pha 2 chet vi OOM, chi 3 baseline song.
# O transfer ton hon baseline vi no giu them mot ban sao trong so lam NEO (build_recadam_anchor
# chay ke ca khi optimizer la AdamW). Nang nguong va them THU LAI: OOM la loi ha tang,
# khong phai ket qua, nen phai thu lai chu khong ghi la o hong.
NEED="${NEED:-13000}"
RETRY="${RETRY:-3}"
exec 8>/tmp/mvd_e60.lock || exit 1
flock -n 8 || { echo "DA CO e60 dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }
wait_vram(){
  local w=0
  while true; do
    local tot use avail
    tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|head -1)
    use=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits|head -1)
    avail=$(( ${tot:-0} - ${use:-0} ))
    (( avail >= NEED )) && return 0
    sleep 60; w=$((w+60)); (( w % 900 == 0 )) && echo "$(ts) | cho VRAM ${avail}MiB < $NEED ... ${w}s"
  done
}
echo "########## E60 bat dau $(ts) | $(hostname) ##########"
for FOLD in ${FOLD_LIST:-1 2 3}; do
  for SRC in ${SOURCES_LIST:-4cwe com}; do
    case "$SRC" in 4cwe) D=data/phase1_4cwe.jsonl; V=fixed4;; com) D=data/phase1_common.jsonl; V=precomputed;;
      full) D=data/phase1_full.jsonl; V=precomputed;; *) continue;; esac
    while IFS='|' read -r TAG OPT EXTRA; do
      [[ -z "$TAG" ]] && continue
      wait_vram
      echo "===== $(ts) | fold $FOLD | $SRC | $TAG ($OPT) ====="
      RES="results/e60_t5p/transfer_latent_bottleneck_${SRC}_l0p05_${TAG}$([ "$OPT" = adamw ] && echo _adamw)/seed_42/fold${FOLD}.json"
      for try in $(seq 1 "$RETRY"); do
        [[ -f "$RES" ]] && break
        (( try > 1 )) && { echo "$(ts) | THU LAI lan $try (OOM la loi ha tang, khong phai ket qua)"; wait_vram; }
      RUN_NAME=e60 SEED=42 FOLDS="$FOLD" \
      BACKBONES="t5p=Salesforce/codet5p-220m-bimodal:mean" \
      MODES="latent_bottleneck" OPTIMIZERS="$OPT" \
      PHASE1_DATA_PATH="$D" CWE_VOCAB="$V" \
      ARM_TAG="_${SRC}_l0p05_${TAG}" PHASE1_TAG="_${SRC}_l0p05" PHASE1_STORE="model/n48/phase1" \
      LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 PHASE2_EPOCHS=60 \
      PHASE1_EXTRA="--sam_rho 0" PHASE2_EXTRA="$EXTRA" \
      DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
      PYTHON="$PY" bash run/matrix.sh 8>&-
      done
    done <<< "plain|adamw|--sam_rho 0
c5000_t0p2k02|recadam|--sam_rho 0 --anneal_t0_ratio 0.2 --anneal_k 0.02"
  done
done
echo "########## E60 xong $(ts) ##########"
