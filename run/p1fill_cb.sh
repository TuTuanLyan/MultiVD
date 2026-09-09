#!/usr/bin/env bash
# P1FILL_CB — huan luyen Pha 1 codebert cho BA NGUON CHUAN vao DUNG kho ma opt1.sh doc.
#
# VI SAO CAN RIENG FILE NAY: `run/opt1.sh` TU CHOI tu huan luyen Pha 1 (dung ky luat
# CLAUDE.md muc 5) — no chi bao "THIEU checkpoint ... KHONG tu huan luyen" roi bo o.
# Va `run/phase1_fill.sh` co san thi goi day42.sh, ma day42.sh dat ARM_TAG="_<src>"
# KHONG co hau to lambda, nen ghi ra `codebert__latent_bottleneck_4cwe/` trong khi
# opt1.sh doc `codebert__latent_bottleneck_4cwe_l0p05/`. Hai duong khac nhau.
#
# matrix.sh dung CUNG bieu thuc duong dan khi huan luyen (dong 238) va khi doc (dong 376):
#   $PHASE1_STORE/${LABEL}__${MODE}${PHASE1_TAG}/seed_$SEED/best.pt
# nen chi can dat dung bon bien do la ra dung file.
#
# FOLDS="" (KHONG phai FOLDS=":-") de chi chay Pha 1: matrix.sh dong 57 dung ${FOLDS-...}
# nen chuoi rong duoc giu nguyen.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$( for c in /venv/main/bin/python /data/ntat/envs/vdenv/bin/python \
      /home/ntat/miniconda3/envs/vdenv/bin/python; do
    [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { echo "$c"; break; }; done )}"
[ -n "$PYTHON" ] || { echo "!! khong tim thay python co torch"; exit 2; }
export PYTHON
BB="${BB:-codebert=microsoft/codebert-base:cls}"
LABEL="${BB%%=*}"
STORE="${STORE:-model/n48/phase1}"
SEED="${SEED:-42}"
data_of(){ case "$1" in
  4cwe) echo "data/phase1_4cwe.jsonl fixed4" ;;
  com)  echo "data/phase1_common.jsonl precomputed" ;;
  full) echo "data/phase1_full.jsonl precomputed" ;;
esac; }
echo "########## P1FILL $LABEL bat dau $(date -u '+%F %T') ##########"
for S in ${SOURCES_LIST:-4cwe com full}; do
  read -r D V <<< "$(data_of "$S")"
  TARGET="$STORE/${LABEL}__latent_bottleneck_${S}_l0p05/seed_${SEED}/best.pt"
  if [[ -f "$TARGET" ]]; then echo "=== $S | da co, bo qua: $TARGET ==="; continue; fi
  echo "===== $(date -u '+%F %T') | Pha 1 | $LABEL | $S -> $TARGET ====="
  RUN_NAME=p1fill SEED="$SEED" FOLDS="" \
  BACKBONES="$BB" MODES="latent_bottleneck" OPTIMIZERS="adamw" \
  PHASE1_DATA_PATH="$D" CWE_VOCAB="$V" \
  PHASE1_TAG="_${S}_l0p05" ARM_TAG="_${S}_l0p05" PHASE1_STORE="$STORE" \
  LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 \
  PHASE1_EXTRA="--sam_rho 0" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PYTHON="$PYTHON" bash run/matrix.sh
  [[ -f "$TARGET" ]] && echo "  => DA TAO: $TARGET" || echo "  !! VAN THIEU: $TARGET"
done
echo "########## P1FILL $LABEL xong $(date -u '+%F %T') ##########"
