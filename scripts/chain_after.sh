#!/usr/bin/env bash
# Cho MOI driver GPU cua chinh minh tren MAY NAY thoat han roi moi phong lenh tiep theo.
# Mot GPU mot chuoi — `run/matrix.sh` KHONG giu lock nen hai chuoi cung luc la OOM.
#   chain_after.sh <workdir> <regex-cho-pgrep> <logfile> <lenh...>
cd "$1" || exit 1; shift
PAT="$1"; LOGF="$2"; shift 2
for i in $(seq 1 300); do            # toi da 5 gio
  pgrep -f "$PAT" >/dev/null || break
  sleep 60
done
if pgrep -f "$PAT" >/dev/null; then
  echo "$(date -u '+%F %T') | driver truoc van chay sau 5h — KHONG phong '$*'" >> "$LOGF"; exit 1
fi
sleep 30
echo "$(date -u '+%F %T') | driver truoc da xong, phong: $*" >> "$LOGF"
exec "$@" >> "$LOGF" 2>&1
