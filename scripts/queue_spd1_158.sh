#!/usr/bin/env bash
# queue_spd1_158.sh — 158 chay FOLD 3 cua khoi SPD1, ngay sau khi RET1 xong.
#
# Chia theo FOLD TRON VEN (CLAUDE.md muc 4): 161 lam fold 1 va 2, 158 lam fold 3.
# Baseline va CA HAI doi chung cua fold 3 deu chay tren 158 nen Delta ghep cap trong fold
# van sach — chenh lech phan cung triet tieu.
#
# Ly do chia: 161 mot minh chay 21 o mat ~3,8 h va se vuot moc 21:00 gio VN nguoi dung dat.
# Chia doi thi ca hai may xong truoc 20:00.
#
#   setsid nohup bash scripts/queue_spd1_158.sh > log/queue_spd1_158.log 2>&1 &
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-/data/ntat/envs/vdenv/bin/python}"
exec 8>/tmp/mvd_queue_spd1_158.lock || exit 1
flock -n 8 || { echo "DA CO queue_spd1_158 dang chay"; exit 3; }
ts(){ date -u '+%F %T'; }
w=0
# Cho RET1 tra lock. Khong dem o, khong doc log — chi bam vao lock cua chinh driver do.
while ! flock -n /tmp/mvd_ret1.lock -c true 2>/dev/null; do
  sleep 120; w=$((w+120)); (( w % 1800 == 0 )) && echo "$(ts) | cho RET1... ${w}s"
done
echo "$(ts) | RET1 da xong — chay SPD1 fold 3"
FOLD_LIST=3 SOURCES_LIST="4cwe com" SEED=42 PYTHON="$PY" bash run/spd1.sh 8>&-
echo "########## QUEUE_SPD1_158 xong $(ts) ##########"
