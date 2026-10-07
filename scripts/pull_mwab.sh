#!/usr/bin/env bash
# Keo ket qua mw_assemble_babel + baseline tu vast 51144271 ve local, doi chieu TUNG BYTE.
# KHONG huy may (nguoi dung dan 17/09). Chi keo results/ va log/, khong keo model/.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H=root@115.73.216.179; P=56599
SSHV="ssh -p $P -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes -i /home/ntat/.ssh/id_ed25519"
DEST="${DEST:-results_mwab_vast}"; mkdir -p "$DEST"
TAGS="mwgU_xAsRl_jsCjv2sven b4_sven mwgp1U_jsCjv"

for T in $TAGS; do
  rsync -a --info=stats1 -e "$SSHV" "$H:/workspace/MultiVD/results/$T" "$DEST/" || { echo "!! rsync results/$T LOI"; exit 1; }
  rsync -a -e "$SSHV" "$H:/workspace/MultiVD/log/$T" "$DEST/_log/" 2>/dev/null
done

# ── doi chieu TUNG BYTE: kich thuoc + md5, ca hai chieu ──
$SSHV "$H" 'cd /workspace/MultiVD && for T in '"$TAGS"'; do find results/$T -type f -printf "%s  %p\n"; done | LC_ALL=C sort' > /tmp/mwab_remote.lst
( cd "$DEST" && for T in $TAGS; do find "$T" -type f -printf "%s  results/%p\n"; done | LC_ALL=C sort ) > /tmp/mwab_local.lst

R=$(wc -l < /tmp/mwab_remote.lst); L=$(wc -l < /tmp/mwab_local.lst)
echo "== file tren vast: $R | file o local: $L =="
if [ "$R" -eq 0 ]; then echo "!! manifest RONG — KHONG coi la da keo xong"; exit 2; fi
if LC_ALL=C diff -u /tmp/mwab_remote.lst /tmp/mwab_local.lst > /tmp/mwab_diff.txt; then
  echo "== KHOP $R file, dung tung byte =="
else
  echo "!! LECH — xem /tmp/mwab_diff.txt"; head -20 /tmp/mwab_diff.txt; exit 3
fi

# ── dem o: phai du 5+5 ──
NA=$(ls "$DEST"/mwgU_xAsRl_jsCjv2sven/multiwindow/seed_36/fold*.json 2>/dev/null | wc -l)
NB=$(ls "$DEST"/b4_sven/*/seed_36/fold*.json 2>/dev/null | wc -l)
echo "== mw_assemble_babel $NA/5 | baseline $NB/5 =="
[ "$NA" -eq 5 ] && [ "$NB" -eq 5 ] || echo "!! CHUA DU O — bang se thieu fold, ghi ro o nao thieu khi bao cao"
echo "== GIU MAY vast 51144271, KHONG huy =="
