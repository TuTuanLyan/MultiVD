#!/usr/bin/env bash
# Canh khoi _l02 chay tren may local. Khong huy gi ca — chi phong lai khi driver chet.
#
# Bay da mac: giet driver cha bang `kill <pid>` KHONG giet `matrix.sh` con. Con mo coi
# do van doc file script; neu luc ay ta ghi de matrix.sh thi bash doc lech offset va
# nem "syntax error near unexpected token". No cung giu flock nen driver moi khong vao
# duoc, va may nam khong. Vi vay o day luon giet theo CA NHOM tien trinh (kill -- -PGID).
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVERY="${EVERY:-600}"; MAX="${MAX:-200}"; NEED="${NEED:-50}"
BB='codebert=microsoft/codebert-base:cls'
LOG=log/local_watch.log; mkdir -p log
say(){ echo "$(date -u '+%F %T') $*" | tee -a "$LOG"; }
FAILS=0
ndriver(){ if flock -n /tmp/multivd_day45.lock -c true 2>/dev/null; then echo 0; else echo 1; fi; }
say "== local_watch2 bat dau, can codebert=$NEED o, chu ky ${EVERY}s =="
for ((i=0;i<MAX;i++)); do
  N=$(ls results/s42_codebert/transfer_*_l02/seed_42/fold*.json 2>/dev/null | wc -l)
  T=$(ls results/s42_*/transfer_*_l02/seed_42/fold*.json 2>/dev/null | wc -l)
  D=$(ndriver)
  say "[local dr=$D codebert=$N/$NEED] tong=$T/150"
  if (( N >= NEED )); then say "== codebert du $N/$NEED o. Tong $T/150. Ket thuc canh. =="; exit 0; fi
  if (( D == 0 )); then
    FAILS=$((FAILS+1))
    say "[local] driver CHET, con $((NEED-N)) o — PHONG LAI (lan $FAILS)."
    if (( FAILS > 6 )); then say "[local] that bai $FAILS lan, dung de nguoi xem."; exit 1; fi
    # khong xoa lock: driver moi tu lay duoc khi cai cu da chet
    PHASE1_MIN_VAL=0 ASAM_RHO=0.1 PYTHON=/home/ntat/miniconda3/envs/vdenv/bin/python \
      setsid nohup nice -n 10 bash run/day45_machine.sh "$BB" >/dev/null 2>&1 </dev/null &
    sleep 25
    say "[local] sau khoi phuc: driver=$(ndriver)"
  fi
  sleep "$EVERY"
done
say "== het $MAX vong =="
