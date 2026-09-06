#!/usr/bin/env bash
#   bash run/day45_machine.sh "codebert=microsoft/codebert-base:cls" [them backbone...]
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
(( $# >= 1 )) || exit 2
LOCK="${MVD_LOCK:-/tmp/multivd_day45.lock}"; exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO driver day45"; exit 3; }
"${PYTHON:-python}" -c "import torch" 2>/dev/null || { echo "PYTHON khong co torch"; exit 4; }
mkdir -p log
for BB in "$@"; do
  L="${BB%%=*}"
  {
    echo "########## L02 $L bat dau $(date -u '+%F %T') lambda=0.02 ASAM rho=${ASAM_RHO:-0.1} ##########"
    for S in 4cwe com full; do echo "===== PHASE 1 | $L | $S ====="; FOLDS="" bash run/day45.sh "$S" "$BB"; done
    for F in ${FOLD_LIST:-1 2 3 4 5}; do
      for S in 4cwe com full; do
        echo "===== FOLD $F | $L | $S | lambda=0.02 ASAM rho=${ASAM_RHO:-0.1} RecAdam ====="
        FOLDS="$F" bash run/day45.sh "$S" "$BB"
      done
      echo "@@@@@ XONG FOLD $F cua $L (l02) — $(date -u '+%F %T') @@@@@"
    done
    echo "########## L02 $L xong $(date -u '+%F %T') ##########"
  } >> "log/day45_${L}.log" 2>&1
done
