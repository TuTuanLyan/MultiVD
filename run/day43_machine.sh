#!/usr/bin/env bash
# Mot MAY chay khoi ASAM@Phase2 cho mot hay nhieu backbone, ca 3 nguon, 5 fold.
#
#   bash run/day43_machine.sh "codebert=microsoft/codebert-base:cls" "unixcoder=microsoft/unixcoder-base:cls"
#
# Thu tu: fold ngoai cung, nguon trong. Xong mot fold la co ngay mot lat cat DU
# moi nhanh x moi nguon cho backbone do — so duoc lien, thay vi co 5 fold cua
# mot nhanh va khong co gi cua cac nhanh khac.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
(( $# >= 1 )) || { echo "can it nhat mot backbone-spec"; exit 2; }

LOCK="${MVD_LOCK:-/tmp/multivd_day43.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO driver day43 tren may nay. Thoat."; exit 3; }
source scripts/proc.sh
_o=$(ps -eo pid,pgid,args --no-headers | awk -v pg="$(ps -o pgid= -p $$ | tr -d ' ')" \
      '$4 ~ /day43_machine/ && $2 != pg {print $1}' | wc -l)
(( _o > 0 )) && { echo "DA CO $_o driver day43 dang chay. Thoat."; exit 3; }
"${PYTHON:-python}" -c "import torch" 2>/dev/null || { echo "PYTHON khong co torch"; exit 4; }

mkdir -p log
for BB in "$@"; do
  LABEL="${BB%%=*}"
  {
    echo "########## ASAM@P2 $LABEL bat dau $(date -u '+%F %T') pid=$$ rho=${ASAM_RHO:-0.1} ##########"
    # FOLD_LIST cho phep chia fold giua hai may. Chuyen TRON tung fold: ca
    # `_asam` lan `_ctl` cua mot fold nam cung may, nen do lech phan cung triet
    # tieu trong Delta ghep cap. Chia theo NHANH thi khong duoc.
    for F in ${FOLD_LIST:-1 2 3 4 5}; do
      for S in 4cwe full com; do
        # ASAM va doi chung khong-ASAM chay LIEN NHAU trong cung fold cung may,
        # de hieu ung phan cung triet tieu trong Delta.
        echo "===== FOLD $F | $LABEL | nguon $S | ASAM rho=${ASAM_RHO:-0.1} ====="
        FOLDS="$F" bash run/day43.sh "$S" "$BB"
        # MODES_SKIP_CTL=1: nhanh doi chung KHONG phu thuoc rho nen chay lai o
        # vong rho khac la thua — dung lai ket qua _ctl da co.
        if [[ "${MODES_SKIP_CTL:-0}" != "1" ]]; then
          echo "===== FOLD $F | $LABEL | nguon $S | DOI CHUNG khong ASAM ====="
          FOLDS="$F" CONTROL=1 bash run/day43.sh "$S" "$BB"
        fi
      done
      echo "@@@@@ XONG FOLD $F cua $LABEL — $(date -u '+%F %T') @@@@@"
    done
    echo "########## ASAM@P2 $LABEL xong $(date -u '+%F %T') ##########"
  } >> "log/day43_${LABEL}.log" 2>&1
done
