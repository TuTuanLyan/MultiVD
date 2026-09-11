#!/usr/bin/env bash
# Chay DAO CHIEU (nguon Python -> dich JS) tren HAI dich lon hon: com va full.
# Pha 1 nguon Python DUNG LAI tu model/rev1/phase1, khong huan luyen lai.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${BB:?can BB}"
for spec in "rev1com data/js_com_folds" "rev1full data/js_full_folds"; do
  set -- $spec
  echo "##### $(date -u '+%F %T') | dich = $2 #####"
  RUN="$1" TGT_ROOT="$2" REV_LOCK="/tmp/mvd_$1.lock" BB="$BB" bash run/rev1.sh
done
echo "##### JSBIG XONG $(date -u '+%F %T') | $(hostname) #####"
