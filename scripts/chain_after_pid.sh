#!/usr/bin/env bash
# Cho DUNG MOT tien trinh (theo PID) thoat han roi moi phong lenh tiep theo.
#   chain_after_pid.sh <workdir> <pid> <logfile> <lenh...>
#
# VI SAO THEO PID CHU KHONG THEO PATTERN. `scripts/chain_after.sh` cho bang
# `pgrep -f "$PAT"`, ma chinh $PAT lai nam trong argv cua no => pgrep TU KHOP CHINH NO,
# vong lap khong bao gio thoat, va lenh tiep theo khong bao gio chay: GPU nam khong ma
# van tinh tien. Day la lan thu NAM cua ho loi "pgrep -f bat ca dong lenh cua minh"
# (xem memory pkill-kills-own-ssh-session). PID thi khong the tu khop.
#
# Mot GPU mot chuoi: `run/matrix.sh` KHONG giu lock nen hai chuoi cung luc la OOM.
set -u
cd "$1" || exit 1; shift
PID="$1"; LOGF="$2"; shift 2
mkdir -p "$(dirname "$LOGF")"
echo "$(date -u '+%F %T') | cho PID $PID thoat roi phong: $*" >> "$LOGF"
for i in $(seq 1 360); do              # toi da 6 gio
  kill -0 "$PID" 2>/dev/null || break
  sleep 60
done
if kill -0 "$PID" 2>/dev/null; then
  echo "$(date -u '+%F %T') | PID $PID VAN CHAY sau 6h — KHONG phong '$*'" >> "$LOGF"; exit 1
fi
sleep 30                                # de GPU nha het VRAM
echo "$(date -u '+%F %T') | PID $PID da xong, phong: $*" >> "$LOGF"
exec "$@" >> "$LOGF" 2>&1
