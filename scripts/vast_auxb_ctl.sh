#!/usr/bin/env bash
# DOI CHUNG CUNG MAY cho khoi `auxb`: nhanh Pha 1 KHONG can bang lop, cung fold cung phien.
#
# Vi sao bat buoc: `auxb` chi chay nhanh CAN BANG + baseline, nen no tra loi duoc
# "chuyen giao co hon baseline khong" (da biet tu lau) chu KHONG tra loi duoc
# "can bang lop co giup khong" — cau hoi cua chinh no. So Δ cua nhanh can bang tren vast
# voi Δ cua nhanh khong can bang tren 161 la so qua HAI LOAI GPU: san nhieu 0.028, lon hon
# ca hieu ung dang do. Phai co doi chung cung may (CLAUDE.md muc 4).
#
# Va no la mot phep kiem PHUONG PHAP LUAN dang gia: §44 do probe dac trung dong bang va
# du bao can bang lop GIUP codebert (p768 0.7700 -> 0.7827) va HAI t5p (0.6405 -> 0.6118).
# Sau khi fine-tune thi co dung nhu vay khong? Neu probe du bao duoc thi lan sau khoi phai
# dot GPU de biet.
#
# 2 backbone x 3 fold x 1 nhanh = 6 o. Baseline da co san trong cung thu muc.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON="${PYTHON:-/venv/main/bin/python}"
BBS="codebert=microsoft/codebert-base:cls t5p=Salesforce/codet5p-220m-bimodal:mean"
echo "########## AUXB-CTL bat dau $(date -u '+%F %T') | $(hostname) ##########"
for BB in $BBS; do
  L="${BB%%=*}"
  CK="model/n48/phase1/${L}__latent_bottleneck_4cwe_l0p05/seed_42/best.pt"
  if [[ ! -f "$CK" ]]; then echo "  !! THIEU $CK — bo qua $L"; continue; fi
  echo "  $L Pha 1 khong can bang: $(stat -c %s "$CK") B"
  for FOLD in 1 2 3; do
    # ARM_TAG KHAC (_unbal) nhung ghi vao CUNG cay results/auxb_<bb> => ghep cap sach
    # theo (cay, seed, fold) voi ca nhanh `bal` lan `baseline`.
    RUN_NAME=auxb SEED=42 FOLDS="$FOLD" BACKBONES="$BB" MODES=latent_bottleneck OPTIMIZERS=adamw \
    PHASE1_DATA_PATH=data/phase1_4cwe.jsonl CWE_VOCAB=fixed4 \
    PHASE1_TAG=_4cwe_l0p05 ARM_TAG=_4cwe_l0p05_unbal PHASE1_STORE=model/n48/phase1 \
    LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
    PHASE1_EXTRA="--sam_rho 0" PHASE2_EXTRA="--sam_rho 0" \
    DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
    bash run/matrix.sh
  done
done
echo "########## AUXB-CTL xong $(date -u '+%F %T') | $(find results -path '*auxb*' -name 'fold*.json' | wc -l) o auxb tong ##########"
