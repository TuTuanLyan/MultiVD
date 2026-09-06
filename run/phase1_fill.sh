#!/usr/bin/env bash
# Bu cho DU kho Phase 1 cua mot backbone: chay Phase 1 ca 3 nguon, KHONG Phase 2.
# matrix.sh tu bo qua o da co best.pt va o da co best.pt.rejected, nen chi nhung
# o THIEU moi duoc huan luyen.
#
#   bash run/phase1_fill.sh t5p=Salesforce/codet5p-220m-bimodal:mean
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BB="${1:?backbone-spec}"; LABEL="${BB%%=*}"
LOCK="${MVD_LOCK:-/tmp/multivd_p1fill.lock}"
exec 9>"$LOCK" || exit 1
flock -n 9 || { echo "DA CO phase1_fill chay tren may nay"; exit 3; }
"${PYTHON:-python}" -c "import torch" 2>/dev/null || { echo "PYTHON khong co torch"; exit 4; }
mkdir -p log
{
  echo "########## BU PHASE 1 $LABEL bat dau $(date -u '+%F %T') ##########"
  for S in 4cwe full com; do
    echo "===== PHASE 1 | $LABEL | nguon $S ====="
    BACKBONES="$BB" bash run/day42.sh "$S" phase1
  done
  echo "########## BU PHASE 1 $LABEL xong $(date -u '+%F %T') ##########"
} >> "log/phase1_fill_${LABEL}.log" 2>&1
