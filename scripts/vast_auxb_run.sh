#!/usr/bin/env bash
# Chay tren VAST: MANH 1 cua co che head moi — lam cho head phu THAT SU HOC.
#
# §30.2 do truc tiep: head phu chi doan MOT lop, do chinh xac trung khit san doan-lop-da-so.
# Nguyen nhan: cross-entropy tran tren nguon lech 74% ve CWE-79, lambda chi 0.05.
# Sua: trong so nghich tan suat, chuan hoa ve trung binh 1 (tong do lon loss KHONG doi nen
# lambda van so sanh duoc voi moi khoi cu). Chi PHAN BO lai giua cac lop.
#
# Ba buoc, va buoc 2 la CONG: neu head van khong vuot san doan-lop-da-so thi manh 2
# (dinh tuyen quyet dinh qua nut that) VO NGHIA va phai dung lai.
#   1. Pha 1 tren 4cwe VOI --aux_class_balanced, hai backbone -> model/auxb/phase1
#   2. Probe head: phai VUOT san doan-lop-da-so, va phai dung >1 lop
#   3. Pha 2 plain AdamW, 3 fold, hai backbone, + baseline cung may cung fold
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON="${PYTHON:-/venv/main/bin/python}"
BBS="codebert=microsoft/codebert-base:cls t5p=Salesforce/codet5p-220m-bimodal:mean"
STORE=model/auxb/phase1
echo "########## VAST AUXB bat dau $(date -u '+%F %T') | $(hostname) | python=$PYTHON ##########"

for BB in $BBS; do
  L="${BB%%=*}"
  echo "===== $(date -u '+%F %T') | $L | Pha 1 CAN BANG LOP ====="
  RUN_NAME=auxb SEED=42 FOLDS="" BACKBONES="$BB" MODES=latent_bottleneck OPTIMIZERS=adamw \
  PHASE1_DATA_PATH=data/phase1_4cwe.jsonl CWE_VOCAB=fixed4 \
  PHASE1_TAG=_4cwe_l0p05_bal ARM_TAG=_4cwe_l0p05_bal PHASE1_STORE="$STORE" \
  LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 \
  PHASE1_EXTRA="--sam_rho 0 --aux_class_balanced" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  bash run/matrix.sh
done

echo "===== $(date -u '+%F %T') | CONG: head co THAT SU HOC khong ====="
for L in codebert t5p; do
  M=$([[ $L == codebert ]] && echo microsoft/codebert-base || echo Salesforce/codet5p-220m-bimodal)
  for tag in _4cwe_l0p05_bal _4cwe_l0p05; do
    ST=$([[ $tag == *bal ]] && echo "$STORE" || echo model/n48/phase1)
    C="$ST/${L}__latent_bottleneck${tag}/seed_42/best.pt"
    [[ -f "$C" ]] || { echo "  THIEU $C"; continue; }
    echo "--- $L $tag"
    PYTHONWARNINGS=ignore $PYTHON tools/aux_head_probe.py --ckpt "$C" \
      --data data/phase1_4cwe.jsonl --vocab fixed4 --device cpu 2>/dev/null \
      | grep -E "HEAD PHU|san: doan lop da so|head chi dung|val nhi phan" | sed 's/^/    /'
  done
done

echo "===== $(date -u '+%F %T') | Pha 2: co che moi vs doi chung ====="
for BB in $BBS; do
  L="${BB%%=*}"
  for FOLD in 1 2 3; do
    RUN_NAME=auxb SEED=42 FOLDS="$FOLD" BACKBONES="$BB" MODES=latent_bottleneck OPTIMIZERS=adamw \
    PHASE1_DATA_PATH=data/phase1_4cwe.jsonl CWE_VOCAB=fixed4 \
    PHASE1_TAG=_4cwe_l0p05_bal ARM_TAG=_4cwe_l0p05_bal PHASE1_STORE="$STORE" \
    LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
    PHASE1_EXTRA="--sam_rho 0 --aux_class_balanced" PHASE2_EXTRA="--sam_rho 0" \
    DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
    bash run/matrix.sh
  done
done
echo "########## VAST AUXB XONG $(date -u '+%F %T') | $(find results -path '*auxb*' -name 'fold*.json' | wc -l) o ##########"
