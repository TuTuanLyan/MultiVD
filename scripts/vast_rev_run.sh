#!/usr/bin/env bash
# Chay tren VAST: toan bo phan DAO CHIEU (nguon Python -> dich JS), hai backbone.
#   A. 4cwe fold 2,3   B. com fold 1   C. full fold 1
# Pha 1 nguon Python day san len tu local, KHONG huan luyen lai.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHON="${PYTHON:-/venv/main/bin/python}"
BBS="codebert=microsoft/codebert-base:cls t5p=Salesforce/codet5p-220m-bimodal:mean"
echo "########## VAST REV bat dau $(date -u '+%F %T') | $(hostname) ##########"
echo "  python: $PYTHON"
for BB in $BBS; do
  L="${BB%%=*}"
  echo "===== $(date -u '+%F %T') | backbone $L ====="
  RUN=rev1    TGT_ROOT=data/js_4cwe_folds FOLDS="2 3" REV_LOCK=/tmp/v_a.lock STORE=model/rev1/phase1 BB="$BB" bash run/rev1.sh
  RUN=rev1com TGT_ROOT=data/js_com_folds  FOLDS="1"   REV_LOCK=/tmp/v_b.lock STORE=model/rev1/phase1 BB="$BB" bash run/rev1.sh
  RUN=rev1full TGT_ROOT=data/js_full_folds FOLDS="1"  REV_LOCK=/tmp/v_c.lock STORE=model/rev1/phase1 BB="$BB" bash run/rev1.sh
done
echo "########## VAST REV XONG $(date -u '+%F %T') | $(find results -path '*rev1*' -name 'fold*.json' | wc -l) o ##########"
