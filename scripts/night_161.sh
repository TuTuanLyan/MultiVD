#!/usr/bin/env bash
# 161 — chay tuan tu suot dem. GPU dung chung nen moi o deu qua cong VRAM cua pool1.sh/e60.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec 8>/tmp/mvd_night161.lock || exit 1
flock -n 8 || { echo "DA CO night_161 dang chay"; exit 3; }
PY=/home/ntat/miniconda3/envs/vdenv/bin/python
ts(){ date -u '+%F %T'; }
run(){ echo "########## $(ts) | $* ##########"; "$@" || echo "  (muc tren tra ma loi, di tiep)"; }
wait_free(){ local w=0; while (( $(ps -eo args --no-headers | grep -c '[s]rc/train_[a-z]*\.py') > 0 )); do sleep 60; w=$((w+60)); done; }
wait_free; run env POOLS="pur100_n930 pur75_n930 pur50_n930 pur25_n930 pur12_n930 pur100_n465 pur100_n232" FOLD_LIST="4 5" SEED=42 PYTHON=$PY bash run/pool1.sh
wait_free; run env POOLS="lm100_n930 lm75_n930 lm50_n930 lm25_n930 lm12_n930" FOLD_LIST="1 2 3" SEED=42 PYTHON=$PY bash run/pool1.sh
wait_free; run env POOLS="lm100_n930 lm75_n930 lm50_n930 lm25_n930 lm12_n930" FOLD_LIST="4 5" SEED=42 PYTHON=$PY bash run/pool1.sh
echo "########## NIGHT_161 xong $(ts) ##########"
