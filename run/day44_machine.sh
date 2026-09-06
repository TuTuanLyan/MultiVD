#!/usr/bin/env bash
# Chay lap lai tren tap dich sach cho mot hay nhieu backbone.
#   bash run/day44_machine.sh "codebert=microsoft/codebert-base:cls"
# Thu tu: fold ngoai, nguon trong -> xong mot fold la co lat cat du de so.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
(( $# >= 1 )) || { echo "can backbone-spec"; exit 2; }
LOCK="${MVD_LOCK:-/tmp/multivd_day44.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO driver day44 tren may nay"; exit 3; }
_o=$(ps -eo pid,pgid,args --no-headers | awk -v pg="$(ps -o pgid= -p $$ | tr -d ' ')" \
      '$4 ~ /day44_machine/ && $2 != pg {print $1}' | wc -l)
(( _o > 0 )) && { echo "DA CO $_o driver day44"; exit 3; }
"${PYTHON:-python}" -c "import torch" 2>/dev/null || { echo "PYTHON khong co torch"; exit 4; }
mkdir -p log
for BB in "$@"; do
  L="${BB%%=*}"
  {
    echo "########## TAP DICH SACH $L bat dau $(date -u '+%F %T') ##########"
    for F in 1 2 3 4 5; do
      for S in 4cwe com full; do
        echo "===== FOLD $F | $L | nguon $S | tap dich SACH ====="
        FOLDS="$F" bash run/day44.sh "$S" "$BB"
      done
      echo "@@@@@ XONG FOLD $F cua $L (sach) — $(date -u '+%F %T') @@@@@"
    done
    echo "########## TAP DICH SACH $L xong $(date -u '+%F %T') ##########"
  } >> "log/day44_${L}.log" 2>&1
done
