#!/usr/bin/env bash
# E60B — nhu run/e60.sh nhung CHON DUOC BACKBONE. Viet file rieng thay vi sua e60.sh:
# e60.sh dang chay tren 161, ma bash doc script theo offset nen ghi de tai cho lam no doc lech
# (memory never-edit-a-running-script-in-place).
#
# Nguoi dung 08/09: "queue them cai 60 epochs, them 2 fold cua 2 backbone lay doi chung".
#   t5p      fold 4,5   (e60.sh dang chay fold 1,2,3 tren 161 -> tron 5 fold)
#   codebert fold 4,5
#
#   BB=t5p      FOLD_LIST="4 5" bash run/e60b.sh
#   BB=codebert FOLD_LIST="4 5" bash run/e60b.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BBNAME="${BB:-t5p}"
case "$BBNAME" in
  t5p)      BBSPEC="t5p=Salesforce/codet5p-220m-bimodal:mean"; P1STORE=model/n48/phase1; P1SUF="_l0p05";;
  codebert) BBSPEC="codebert=microsoft/codebert-base:cls";     P1STORE=model/s42/phase1; P1SUF="";;
  *) echo "BB phai la t5p hoac codebert"; exit 1;;
esac
PY="${PYTHON:-$( [ -x /venv/main/bin/python ] && echo /venv/main/bin/python \
  || { [ -x /data/ntat/envs/vdenv/bin/python ] && echo /data/ntat/envs/vdenv/bin/python \
  || echo /home/ntat/miniconda3/envs/vdenv/bin/python; } )}"
NEED="${NEED:-10500}"
exec 8>"/tmp/mvd_e60b_${BBNAME}.lock" || exit 1
flock -n 8 || { echo "DA CO e60b $BBNAME dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }
wait_vram(){   # cong VRAM TRUOC TUNG O — GPU dung chung, kiem mot lan luc khoi dong la khong du
  local w=0 tot use avail
  while true; do
    tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits|head -1)
    use=$(nvidia-smi --query-gpu=memory.used  --format=csv,noheader,nounits|head -1)
    avail=$(( ${tot:-0} - ${use:-0} ))
    (( avail >= NEED )) && return 0
    sleep 60; w=$((w+60)); (( w % 900 == 0 )) && echo "$(ts) | cho VRAM ${avail}MiB < $NEED ... ${w}s"
  done
}
echo "########## E60B bat dau $(ts) | $BBNAME | $(hostname) ##########"
for FOLD in ${FOLD_LIST:-4 5}; do
  for SRC in ${SOURCES_LIST:-4cwe com}; do
    case "$SRC" in 4cwe) D=data/phase1_4cwe.jsonl; V=fixed4;; com) D=data/phase1_common.jsonl; V=precomputed;;
      full) D=data/phase1_full.jsonl; V=precomputed;; *) continue;; esac
    CK="$P1STORE/${BBNAME}__latent_bottleneck_${SRC}${P1SUF}/seed_42/best.pt"
    [[ -f "$CK" ]] || { echo "  !! thieu Pha 1 $CK — bo qua, KHONG tu huan luyen"; continue; }
    while IFS='|' read -r TAG OPT EXTRA; do
      [[ -z "$TAG" ]] && continue
      wait_vram
      echo "===== $(ts) | $BBNAME | fold $FOLD | $SRC | $TAG ($OPT) ====="
      RUN_NAME=e60 SEED=42 FOLDS="$FOLD" BACKBONES="$BBSPEC" \
      MODES="latent_bottleneck" OPTIMIZERS="$OPT" \
      PHASE1_DATA_PATH="$D" CWE_VOCAB="$V" \
      ARM_TAG="_${SRC}_l0p05_${TAG}" PHASE1_TAG="_${SRC}${P1SUF}" PHASE1_STORE="$P1STORE" \
      LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 PHASE2_EPOCHS=60 \
      PHASE1_EXTRA="--sam_rho 0" PHASE2_EXTRA="$EXTRA" \
      DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
      PYTHON="$PY" bash run/matrix.sh 8>&-
    done <<< "plain|adamw|--sam_rho 0
c5000_t0p2k02|recadam|--sam_rho 0 --anneal_t0_ratio 0.2 --anneal_k 0.02"
  done
done
echo "########## E60B xong $(ts) | $BBNAME ##########"
