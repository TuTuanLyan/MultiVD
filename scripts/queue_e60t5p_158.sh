#!/usr/bin/env bash
# 158 chay e60 t5p fold 1,2,3 (ntat giu fold 4,5) — chia theo FOLD TRON VEN, moi may tu
# chay baseline cua fold minh, nen Delta ghep cap trong fold van sach.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec 8>/tmp/mvd_e60t5p158.lock || exit 1
flock -n 8 || { echo "DA CO dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }
w=0
while true; do
  j=$(ps -eo args --no-headers | grep -c '[s]rc/train_[a-z]*\.py')
  (( j == 0 )) && { echo "$(ts) | ranh — chay e60 t5p fold 1 2 3"; break; }
  sleep 60; w=$((w+60)); (( w % 900 == 0 )) && echo "$(ts) | cho... job=$j"
done
sleep 5
BB=t5p FOLD_LIST="1 2 3" SOURCES_LIST="4cwe com" NEED=9000 \
  PYTHON=/data/ntat/envs/vdenv/bin/python bash run/e60b.sh 8>&-
