#!/usr/bin/env bash
# Mot MAY = mot BACKBONE. Chay het 3 nguon cho backbone do.
#
# KHOA flock: KHONG THE co hai driver cung chay tren mot may. Truoc day viec chan
# trung lap phu thuoc vao viec toi giet dung tien trinh; sang 27/08 lenh kill
# truot (sai cot awk) nen lan phong lai tao driver thu hai, hai job tranh mot GPU
# va OOM. Khoa bien dieu do thanh KHONG THE XAY RA thay vi "nho dung quy trinh".
#
# Vi sao chia theo backbone chu khong theo nguon: bien thien giua may do duoc la
# ~0.028 Macro-F1, lon hon phan lon hieu ung dang do. Mot backbone nam tron tren
# MOT may thi baseline cua no va moi nhanh cua no cung phan cung.
#
#   bash run/day42_machine.sh codebert=microsoft/codebert-base:cls
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BB="${1:?vi du: codebert=microsoft/codebert-base:cls}"
LABEL="${BB%%=*}"

LOCK="${MVD_LOCK:-/tmp/multivd_day42.lock}"
exec 9>"$LOCK" || { echo "khong mo duoc khoa $LOCK"; exit 1; }
if ! flock -n 9; then
  echo "DA CO driver day42 chay tren may nay (khoa $LOCK). Thoat, khong tao driver thu hai."
  exit 3
fi
# Chan lop thu hai, KHONG dua vao khoa: mot driver khoi dong tu ban script CU
# khong giu khoa nao ca, nen chi rieng flock se cho lot trong giai doan chuyen
# tiep. Dem truc tiep tien trinh dang chay moi la thu chac chan.
source scripts/proc.sh
_others=$(mvd_other_drivers | wc -l)
if [ "$_others" -gt 0 ]; then
  echo "DA CO $_others driver day42 dang chay (dem truc tiep). Thoat."
  exit 3
fi
# PYTHON phai chay duoc torch, neu khong moi job se chet o dong import va ca luot
# chay "xong" trong vai giay ma khong huan luyen gi.
"${PYTHON:-python}" -c "import torch" 2>/dev/null || {
  echo "PYTHON='${PYTHON:-python}' khong import duoc torch. Dat PYTHON cho dung roi chay lai."
  exit 4
}
echo $$ > "$LOCK.pid"
trap 'rm -f "$LOCK.pid"' EXIT

export BACKBONES="$BB"
mkdir -p log
LOG="log/day42_${LABEL}.log"

{
echo "########## $LABEL bat dau $(date -u '+%F %T') pid=$$ ##########"
# 1) Phase 1 truoc, ca 3 nguon. Khong phu thuoc tap dich nen chay som nhat co the.
for S in 4cwe full com; do
  echo "===== PHASE 1 | $LABEL | nguon $S ====="
  bash run/day42.sh "$S" phase1
done
# 2) Phase 2, tung fold mot. Trong moi fold: 4cwe (co nhanh cwe) -> full -> com.
#    matrix.sh xep adamw/recadam canh nhau trong cung fold de ghep cap duoc.
for F in 1 2 3 4 5; do
  for S in 4cwe full com; do
    echo "===== FOLD $F | $LABEL | nguon $S ====="
    FOLDS="$F" bash run/day42.sh "$S" phase2
  done
done
echo "########## $LABEL xong $(date -u '+%F %T') ##########"
} >> "$LOG" 2>&1
