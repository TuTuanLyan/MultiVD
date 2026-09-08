#!/usr/bin/env bash
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec 8>/tmp/mvd_e60q158.lock || exit 1
flock -n 8 || { echo "DA CO queue_e60_158 dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }
w=0
while true; do
  j=$(ps -eo args --no-headers | grep -c '[s]rc/train_[a-z]*\.py')
  (( j == 0 )) && { echo "$(ts) | ranh — chay E60B codebert"; break; }
  sleep 60; w=$((w+60)); (( w % 900 == 0 )) && echo "$(ts) | cho... job=$j (${w}s)"
done
sleep 5
BB=codebert FOLD_LIST="4 5" SOURCES_LIST="4cwe com" NEED=9000 \
  PYTHON=/data/ntat/envs/vdenv/bin/python bash run/e60b.sh 8>&-
