#!/usr/bin/env bash
# Chi chay nhanh DOI CHUNG (khong ASAM) cho mot backbone, ca 3 nguon, 5 fold.
# Doi chung KHONG phu thuoc rho, nen chay duoc ngay trong luc con dang quet rho.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BB="${1:?backbone-spec}"; LABEL="${BB%%=*}"
LOCK="${MVD_LOCK:-/tmp/multivd_day43.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO driver day43 tren may nay"; exit 3; }
"${PYTHON:-python}" -c "import torch" 2>/dev/null || { echo "PYTHON khong co torch"; exit 4; }
mkdir -p log
{
  echo "########## DOI CHUNG $LABEL bat dau $(date -u '+%F %T') ##########"
  for F in 1 2 3 4 5; do
    for S in 4cwe full com; do
      echo "===== FOLD $F | $LABEL | nguon $S | DOI CHUNG ====="
      FOLDS="$F" CONTROL=1 bash run/day43.sh "$S" "$BB"
    done
    echo "@@@@@ XONG FOLD $F cua $LABEL (doi chung) — $(date -u '+%F %T') @@@@@"
  done
  echo "########## DOI CHUNG $LABEL xong $(date -u '+%F %T') ##########"
} >> "log/day43_${LABEL}.log" 2>&1
