#!/usr/bin/env bash
# Chay Phase 2 cua MOT backbone cho MOT fold, ca 3 nguon. Dung de "nap them"
# viec cho mot may vast da xong hang doi cua no.
#
#   bash run/day42_fold.sh unixcoder=microsoft/unixcoder-base:cls 5
#
# VI SAO CHIA THEO FOLD CHU KHONG THEO NHANH:
# Hieu ung dang do la ~+0.03, con chenh lech giua may do duoc la ~0.028 — gan
# bang nhau. Neu baseline cua mot backbone o may A con vai nhanh cua no o may B
# thi `D vs baseline` thanh hieu giua hai may, va nhieu phan cung nuot tron tin
# hieu. Chuyen TRON mot fold (baseline + moi nhanh cua fold do) thi moi phep so
# trong fold ay van cung mot may; va vi ta gop cac D DA GHEP CAP THEO FOLD, mot
# do lech co dinh cua may se TRIET TIEU trong tung D. An toan ve thong ke, khong
# chi la chap nhan duoc.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BB="${1:?vi du: unixcoder=microsoft/unixcoder-base:cls}"
FOLD="${2:?so fold, vi du 5}"
LABEL="${BB%%=*}"

LOCK="${MVD_LOCK:-/tmp/multivd_day42.lock}"
exec 9>"$LOCK" || { echo "khong mo duoc khoa $LOCK"; exit 1; }
flock -n 9 || { echo "DA CO driver day42 chay tren may nay. Thoat."; exit 3; }
source scripts/proc.sh
_o=$(mvd_other_drivers | wc -l)
[ "$_o" -gt 0 ] && { echo "DA CO $_o driver dang chay. Thoat."; exit 3; }
"${PYTHON:-python}" -c "import torch" 2>/dev/null || {
  echo "PYTHON='${PYTHON:-python}' khong import duoc torch."; exit 4; }

export BACKBONES="$BB"
mkdir -p log
{
  echo "########## $LABEL fold $FOLD bat dau $(date -u '+%F %T') pid=$$ ##########"
  for S in 4cwe full com; do
    echo "===== FOLD $FOLD | $LABEL | nguon $S ====="
    FOLDS="$FOLD" bash run/day42.sh "$S" phase2
  done
  echo "########## $LABEL fold $FOLD xong $(date -u '+%F %T') ##########"
} >> "log/day42_${LABEL}_fold${FOLD}.log" 2>&1
