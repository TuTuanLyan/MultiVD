#!/usr/bin/env bash
# Xep hang leo §40 len bac 3 (n=15) cho MOT may. Goi qua chain_after.sh de no doi driver
# truoc thoat han — mot GPU mot chuoi, vi `run/matrix.sh` KHONG giu lock.
#
#   WHO=161  bash scripts/queue_seed15.sh
#   WHO=158  bash scripts/queue_seed15.sh
#   WHO=vast bash scripts/queue_seed15.sh
#
# Chia viec (CLAUDE.md muc 4 — mot backbone mot may; neu chia thi chia theo FOLD TRON VEN):
#   161  codebert: seed 7 con thieu N=228,152 | seed 1234 thieu ca bon N
#   158  t5p     : seed 7 fold 1 2 3          | seed 1234 (tu huan luyen Pha 1) ca 5 fold
#   vast t5p     : seed 7 fold 4 5
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${WHO:?can WHO=161|158|vast}"
CB="codebert=microsoft/codebert-base:cls"
T5="t5p=Salesforce/codet5p-220m-bimodal:mean"
L=log/seed15_${WHO}.log
say(){ echo "$(date -u '+%F %T') | $*" | tee -a "$L"; }

case "$WHO" in
  161)
    say "codebert — seed 7 (N=228,152) roi seed 1234 (ca bon N)"
    BB="$CB" SEEDS="7"    SIZES="228 152"         bash run/seed15.sh >> "$L" 2>&1
    BB="$CB" SEEDS="1234" SIZES="456 228 152 76"  bash run/seed15.sh >> "$L" 2>&1
    ;;
  158)
    say "t5p — seed 7 fold 1 2 3, roi seed 1234 ca 5 fold (tu huan luyen Pha 1)"
    BB="$T5" SEEDS="7"    SIZES="456 228 152 76" FOLD_LIST="1 2 3"   bash run/seed15.sh >> "$L" 2>&1
    BB="$T5" SEEDS="1234" SIZES="456 228 152 76" FOLD_LIST="1 2 3 4 5" bash run/seed15.sh >> "$L" 2>&1
    ;;
  vast)
    say "t5p — seed 7 fold 4 5"
    BB="$T5" SEEDS="7"    SIZES="456 228 152 76" FOLD_LIST="4 5"     bash run/seed15.sh >> "$L" 2>&1
    ;;
  *) say "!! WHO=$WHO khong hieu"; exit 2 ;;
esac
say "XONG hang cua $WHO"
