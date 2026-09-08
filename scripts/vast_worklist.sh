#!/usr/bin/env bash
# vast_worklist.sh — CHAY TREN MAY VAST. Doc mot DANH SACH VIEC va chay tuan tu het,
# nen may khong bao gio nam khong giua hai khoi.
#
# Nguoi dung 08/09: "Co the tim kiem lien tuc va lien tuc thu de tranh bi trong GPU."
# Truoc do watchdog chi BAO DONG "may dang trong" vao log — bao dong khong phai hanh dong,
# va ntat2 nam khong 23 phut (05:53 -> 06:16) trong khi bao dong da kieu tu 06:10.
#
# Dinh dang worklist (log/worklist.txt), moi dong: <script>|<SOURCES>|<FOLDS>
#   run/asam1.sh|full|3 5
#   run/int1.sh|4cwe com|3 5
# Dong bat dau bang # bi bo qua. Xong dong nao thi ghi vao log/worklist.done.
#
#   setsid nohup bash scripts/vast_worklist.sh > log/worklist.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WL="${WL:-log/worklist.txt}"
# TU DO tim env cua may, KHONG mac dinh duong cua vast: 08/09 chay script nay tren 158 va
# no truyen /venv/main/bin/python (duong cua vast) xuong, lam moi muc bi tu choi. Cong
# chan-truoc cua matrix.sh bat duoc va dung han thay vi pha gi — nhung may nam khong 45 phut.
PY="${PYTHON:-$( for c in /venv/main/bin/python /data/ntat/envs/vdenv/bin/python \
        /home/ntat/miniconda3/envs/vdenv/bin/python; do
      [ -x "$c" ] && "$c" -c "import torch" 2>/dev/null && { echo "$c"; break; }; done )}"
[ -n "$PY" ] || { echo "!! khong tim thay python co torch tren may nay"; exit 2; }
LOCK=/tmp/multivd_opt1.lock
exec 8>/tmp/mvd_worklist.lock || exit 1
flock -n 8 || { echo "DA CO worklist dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }
touch log/worklist.done
echo "########## WORKLIST bat dau $(ts) | $(hostname) ##########"

wait_free(){
  local w=0
  while true; do
    local free=0; flock -n "$LOCK" -c true 2>/dev/null && free=1
    local jobs; jobs=$(ps -eo args --no-headers | grep -c '[s]rc/train_[a-z]*\.py')
    (( free == 1 && jobs == 0 )) && return 0
    sleep 30; w=$((w+30)); (( w % 600 == 0 )) && echo "$(ts) | cho GPU... ${w}s"
  done
}

n=0
while IFS='|' read -r SCRIPT SRCS FOLDS; do
  [[ -z "${SCRIPT:-}" || "${SCRIPT:0:1}" == "#" ]] && continue
  KEY="$SCRIPT|$SRCS|$FOLDS"
  if grep -Fqx "$KEY" log/worklist.done 2>/dev/null; then
    echo "$(ts) | BO QUA (da xong): $KEY"; continue
  fi
  [[ -f "$SCRIPT" ]] || { echo "$(ts) | !! khong co $SCRIPT — bo qua"; continue; }
  wait_free
  echo "########## $(ts) | CHAY: $KEY ##########"
  FOLD_LIST="$FOLDS" SOURCES_LIST="$SRCS" SEED=42 PYTHON="$PY" bash "$SCRIPT" 8>&- \
    && echo "$KEY" >> log/worklist.done
  n=$((n+1))
done < "$WL"
echo "########## WORKLIST xong $(ts) | da chay $n muc ##########"
