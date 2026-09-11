#!/usr/bin/env bash
# SEED15 — leo §40 lên **bậc 3 (n = 5 fold × 3 seed = 15)**: đường cong Δ theo CỠ TẬP TRAIN ĐÍCH.
#
# VÌ SAO khối này chứ không phải khối khác. Luật leo bậc: một nhánh chỉ được lên bậc khi nó
# dương trên **cả bốn** chỉ số VÀ **lặp trên cả hai backbone**. Tính đến 11/09, trong toàn dự án
# **chỉ có §40 qua được cổng đó**:
#
#   ROC-AUC Δ (chuyển giao − baseline), ghép cặp theo fold, seed 42, n=5:
#     codebert  N=456 +0.0114 → 228 +0.0362 → 152 +0.0610 → 76 **+0.1940 (5/5)**
#     t5p       N=456 +0.0173 → 228 +0.0395 → 152 +0.0897 → 76 **+0.1003 (5/5)**
#   đơn điệu CHẶT trên **cả hai** backbone.
#
# Và seed 7 (codebert, hai đầu mút) đã **lặp lại**: +0.0139 → +0.1777, 5/5 fold ở N=76.
# Mọi can thiệp cơ chế khác (§38.2, §39, §43, và nút thắt 8 chiều 11/09) đều **tách theo
# backbone** — không cái nào qua cổng. Nên GPU đổ vào đây, không đổ vào cơ chế thứ sáu.
#
# MỘT NHUỴ PHẢI NÊU KHI ĐỌC, đã thấy ở seed 42 VÀ seed 7: ở N=456 (dữ liệu đầy đủ) PR-AUC của
# codebert **ÂM** (−0.0087 và −0.0081). Tức "dương trên cả bốn chỉ số" TỰ NÓ cũng là hiện tượng
# dữ liệu-ít. Đừng viết thành "phương pháp thắng ở mọi cỡ dữ liệu".
#
# Pha 1 còn thiếu thì khối này TỰ HUẤN LUYỆN (khác `opt1.sh`, vốn cố tình bỏ qua): t5p mới chỉ
# có seed 42.
#
#   BB=codebert=microsoft/codebert-base:cls      SEEDS="1234"   bash run/seed15.sh   # 161
#   BB=t5p=Salesforce/codet5p-220m-bimodal:mean  SEEDS="7 1234" bash run/seed15.sh   # 158
#   ... FOLD_LIST="4 5"  cho máy thứ ba (chia theo FOLD TRỌN VẸN — CLAUDE.md mục 4)
#
# Đọc: python3 tools/tsize_report.py
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$( for c in /venv/main/bin/python /data/ntat/envs/vdenv/bin/python \
      /home/ntat/miniconda3/envs/vdenv/bin/python; do
    [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { echo "$c"; break; }; done )}"
[ -n "$PYTHON" ] || { echo "!! khong tim thay python co torch"; exit 2; }
export PYTHON
: "${BB:?can BB=<label>=<hf-name>:<pooling>}"
SEEDS="${SEEDS:-7 1234}"
SIZES="${SIZES:-456 228 152 76}"
FOLD_LIST="${FOLD_LIST:-1 2 3 4 5}"
SRC="${SRC:-4cwe}"
STORE="${STORE:-model/n48/phase1}"
L="${BB%%=*}"

LOCK="${SEED15_LOCK:-/tmp/mvd_seed15.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO seed15 dang chay tren may nay — dung"; exit 3; }

case "$SRC" in
  4cwe) P1DATA=data/phase1_4cwe.jsonl;   VOCAB=fixed4 ;;
  com)  P1DATA=data/phase1_common.jsonl; VOCAB=precomputed ;;
  full) P1DATA=data/phase1_full.jsonl;   VOCAB=precomputed ;;
  *) echo "!! nguon la $SRC?"; exit 4 ;;
esac

NF=$(echo "$FOLD_LIST" | wc -w); NS=$(echo "$SIZES" | wc -w); NE=$(echo "$SEEDS" | wc -w)
echo "########## SEED15 bat dau $(date -u '+%F %T') | $(hostname) | $L ##########"
echo "  seed: $SEEDS | N: $SIZES | fold: $FOLD_LIST | nguon: $SRC"
echo "  ky vong $(( NE*NS*NF )) o chuyen giao + $(( NE*NS*NF )) baseline = $(( 2*NE*NS*NF )) o"

for SEED in $SEEDS; do
  CKPT="$STORE/${L}__latent_bottleneck_${SRC}_l0p05/seed_${SEED}/best.pt"
  if [[ -f "$CKPT" ]]; then
    echo "===== $(date -u '+%F %T') | seed $SEED | Pha 1 da co ($(stat -c %s "$CKPT") B) ====="
  else
    echo "===== $(date -u '+%F %T') | seed $SEED | Pha 1 THIEU -> huan luyen ====="
    # FOLDS="" (KHONG hai cham o matrix.sh) => chi chay Pha 1, khong dung Pha 2.
    RUN_NAME="p1s$SEED" SEED="$SEED" FOLDS="" \
    BACKBONES="$BB" MODES=latent_bottleneck OPTIMIZERS=adamw \
    PHASE1_DATA_PATH="$P1DATA" CWE_VOCAB="$VOCAB" \
    PHASE1_TAG="_${SRC}_l0p05" ARM_TAG="_${SRC}_l0p05" PHASE1_STORE="$STORE" \
    LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 \
    PHASE1_EXTRA="--sam_rho 0" \
    DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
    bash run/matrix.sh 9>&-
    if [[ -f "$CKPT" ]]; then
      echo "  Pha 1 seed $SEED xong ($(stat -c %s "$CKPT") B)"
    else
      # O TRONG khong viet duoc gi vao bai (CLAUDE.md muc 3) — bao that roi chay tiep seed sau.
      echo "  !! Pha 1 seed $SEED VAN THIEU sau khi huan luyen — bo qua seed nay, chay tiep"
      continue
    fi
  fi

  echo "===== $(date -u '+%F %T') | seed $SEED | duong cong N ====="
  SIZES="$SIZES" SRC="$SRC" FOLD_LIST="$FOLD_LIST" SEED="$SEED" BB="$BB" STORE="$STORE" \
  TSIZE_LOCK=/tmp/mvd_seed15_tsize.lock \
  bash run/tsize.sh 9>&-
done

TOT=0
for N in $SIZES; do TOT=$(( TOT + $(find "results/sz${N}_${L}" -name 'fold*.json' 2>/dev/null | wc -l) )); done
echo "########## SEED15 xong $(date -u '+%F %T') | $(hostname) | $L | $TOT o trong results/sz*_${L} ##########"
