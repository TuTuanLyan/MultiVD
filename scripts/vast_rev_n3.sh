#!/usr/bin/env bash
# Nang khoi DAO CHIEU (nguon Python -> dich JavaScript) tu n=1 len **n=3 fold**.
#
# Vi sao dang chay: §41 / §41.1 / §41.2 deu do o **1 fold** va ca ba muc deu tu ghi
# "n=1, chi sang loc, can n=3 moi duoc viet". Chay fold 2 va 3 bien chung tu quan sat
# thanh thu viet duoc. Day la viec TRA LOI MOT CAU CHUA BIET, khong phai chay lai thu da ro.
#
# Chi lam duoc voi dich `js_4cwe` — `js_com_folds` va `js_full_folds` moi co fold1, phai
# dung them fold roi mới nang duoc, de sau.
#
# 2 backbone x 2 fold x (baseline + plain + ASAM/RecAdam) = 12 o.
# Pha 1 nguon Python dung lai, KHONG huan luyen lai (`run/rev1.sh` tach PHASE1_TAG khoi ARM_TAG).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON="${PYTHON:-/venv/main/bin/python}"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
FOLDS="${FOLDS:-2 3}"
echo "########## REV-n3 bat dau $(date -u '+%F %T') | $(hostname) | fold $FOLDS ##########"

# Cong: kiem MOI duong dan khoi se cham, truoc khi phong. Mot file thieu da tung lam
# vast nam khong 7 phut (FACTS §45).
bad=0
for f in run/rev1.sh run/matrix.sh data/phase1_python.jsonl \
         model/rev1/phase1/codebert__latent_bottleneck_py_l0p05/seed_42/best.pt \
         model/rev1/phase1/t5p__latent_bottleneck_py_l0p05/seed_42/best.pt; do
  [ -e "$f" ] || { echo "  !! THIEU $f"; bad=1; }
done
for k in $FOLDS; do
  for s in train val test; do
    [ -f "data/js_4cwe_folds/fold$k/$s.jsonl" ] || { echo "  !! THIEU data/js_4cwe_folds/fold$k/$s.jsonl"; bad=1; }
  done
done
(( bad )) && { echo "!! thieu dau vao — DUNG, khong chay"; exit 2; }
echo "  moi duong dan can thiet DA CO"

for BB in "codebert=microsoft/codebert-base:cls" "t5p=Salesforce/codet5p-220m-bimodal:mean"; do
  L="${BB%%=*}"
  echo "===== $(date -u '+%F %T') | $L | dao chieu fold $FOLDS ====="
  BB="$BB" FOLDS="$FOLDS" REV_LOCK=/tmp/mvd_rev_n3.lock bash run/rev1.sh
done
echo "########## REV-n3 xong $(date -u '+%F %T') | $(find results/rev1_codebert results/rev1_t5p -name 'fold*.json' 2>/dev/null | wc -l) o rev1 tong ##########"
