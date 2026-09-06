#!/usr/bin/env bash
# Chuyen cho lam viec MultiVD tu /home sang /drive1. CHAY MOT LENH, KHI DA XONG VIEC.
#
#   bash scripts/migrate_to_drive1.sh          # kiem tra + dong bo + chep memory (khong xoa gi)
#   CONFIRM_DELETE=1 bash scripts/migrate_to_drive1.sh   # ... roi xoa ban o /home
#
# Script nay KHONG xoa gi tru khi dat CONFIRM_DELETE=1, va chi xoa sau khi da doi
# chieu tung file tung byte.
#
# BA THU PHAI MANG SANG, khong duoc quen thu nao:
#   1. Ma nguon + du lieu + checkpoint  -> rsync
#   2. Thu muc memory cua Claude Code   -> khoa theo DUONG DAN, doi duong dan la mat
#   3. Session id (neu muon resume)     -> in ra o cuoi
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SRC=/home/ntat/workspace/MultiVD
DST=/drive1/cuongtm/ntat/MultiVD
OLDKEY=/home/ntat/.claude/projects/-home-ntat-workspace-MultiVD
NEWKEY=/home/ntat/.claude/projects/-drive1-cuongtm-ntat-MultiVD
say(){ echo "$*"; }

# ---------- 1. Khong duoc chuyen khi con job dang chay ----------
say "== 1. Kiem cong viec dang chay =="
BUSY=0
# 06/09: danh sach nay tung tro vao lock cua dot sweep46/day45 da het tu lau. Mot cong
# canh gac khong nhan ra job dang chay la cong hong — cap nhat cung luc voi viec.
for L in /tmp/multivd_night48.lock /tmp/multivd_night48_ntat.lock \
         /tmp/multivd_confirm47_com.lock /tmp/multivd_confirm47_4cwe.lock \
         /tmp/multivd_sweep46p1_com.lock /tmp/multivd_day45.lock; do
  [[ -e "$L" ]] || continue
  if ! flock -n "$L" -c true 2>/dev/null; then say "   !! con driver giu $L"; BUSY=1; fi
done
NPY=$(ps -eo comm,args --no-headers -u "$(id -un)" | awk '$1=="python" && (/src\/train_transfer/ || /src\/train_baseline/) {n++} END{print n+0}')
(( NPY > 0 )) && { say "   !! con $NPY tien trinh huan luyen tren 161"; BUSY=1; }
R158=$(timeout 30 ssh -o BatchMode=yes -o StrictHostKeyChecking=no -i /home/ntat/.ssh/id_ed25519 \
       tranmanhcuong@112.137.129.158 \
       'ps -eo comm,args --no-headers | awk "\$1==\"python\" && (/src\/train_transfer/ || /src\/train_baseline/) {n++} END{print n+0}"' 2>/dev/null | tail -1)
(( ${R158:-0} > 0 )) && say "   .. 158 con ${R158} tien trinh (khong chan: 158 co ban rieng o /data/ntat)"
if (( BUSY )); then say "   => DUNG LAI. Cho job tren 161 xong roi chay lai."; exit 1; fi
say "   161 ranh, chuyen duoc."

# ---------- 2. Tat cron de no khong phong job moi giua chung ----------
say ""
say "== 2. Tam tat cron giam sat =="
crontab -l 2>/dev/null > /tmp/mvd_crontab.bak || true
crontab -l 2>/dev/null | grep -vE 'watch48_local|watch48b|sweep46_super' | crontab - 2>/dev/null || true
say "   da go cac muc giam sat (ban luu o /tmp/mvd_crontab.bak)"

# ---------- 3. Dong bo lan cuoi ----------
say ""
say "== 3. Dong bo lan cuoi $SRC -> $DST (MANG CA results/) =="
# 06/09 doi quyet dinh cu ("de results/ trong ben drive1"): khoi NIGHT48 vua xong
# co seed 7 fold 1-2 nam trong results/n48_t5p, ba phan con lai o results_night48*/.
# Bo results/ lai la cat doi mot khoi da hoan chinh. Ca thu muc chi ~10 MB.
mkdir -p "$DST"
nice -n 19 ionice -c3 rsync -a --delete --exclude '/log/' "$SRC/" "$DST/"
say "   rsync xong (ma thoat $?)"
mkdir -p "$DST/log"

# ---------- 4. Doi chieu TUNG FILE TUNG BYTE ----------
say ""
say "== 4. Doi chieu tung file tung byte =="
A=$(cd "$SRC" && find . -type f -not -path './log/*' -printf '%p %s\n' | LC_ALL=C sort)
B=$(cd "$DST" && find . -type f -not -path './log/*' -printf '%p %s\n' | LC_ALL=C sort)
MISS=$(LC_ALL=C comm -23 <(echo "$A") <(echo "$B") | grep -c . || true)
say "   nguon: $(echo "$A" | grep -c .) file | dich: $(echo "$B" | grep -c .) file | lech: $MISS"
if (( MISS > 0 )); then
  LC_ALL=C comm -23 <(echo "$A") <(echo "$B") | head -5 | sed 's/^/     thieu: /'
  say "   => CHUA KHOP. Khong xoa gi. Chay lai script nay."
  exit 1
fi
say "   KHOP."

# ---------- 5. Chep memory sang khoa duong dan MOI ----------
# Claude Code danh khoa project theo DUONG DAN thu muc lam viec, khong theo git
# repo (da kiem: /home/ntat/workspace khong phai repo nhung van co thu muc project
# rieng). Doi cwd la doi khoa, nen memory KHONG tu theo sang.
say ""
say "== 5. Chep memory Claude Code sang khoa duong dan moi =="
if [[ -d "$OLDKEY/memory" ]]; then
  mkdir -p "$NEWKEY"
  cp -a "$OLDKEY/memory" "$NEWKEY/"
  say "   $(ls "$NEWKEY/memory"/*.md 2>/dev/null | wc -l) memory da sang $NEWKEY/memory"
else
  say "   !! khong thay $OLDKEY/memory"
fi

# ---------- 6. Xoa ban cu (chi khi duoc yeu cau ro) ----------
say ""
# Nguoi dung chot: GIU repo va results o /home de tra cuu, chi xoa file NANG
# (trong so mo hinh). Cac file do da co ban sao o $DST va da doi chieu o buoc 4.
if [[ "${CONFIRM_DELETE:-0}" == "1" ]]; then
  say "== 6. Xoa file NANG o /home (giu nguyen ma nguon va results/) =="
  NPT=$(find "$SRC/model" -type f -name '*.pt' 2>/dev/null | wc -l)
  SZ=$(du -sk "$SRC/model" 2>/dev/null | cut -f1)
  say "   se xoa: $NPT file .pt trong $SRC/model  ($(awk -v k="${SZ:-0}" 'BEGIN{printf "%.1f", k/1048576}') GB)"
  # chi xoa khi ban o drive1 that su co du so file .pt
  NDST=$(find "$DST/model" -type f -name '*.pt' 2>/dev/null | wc -l)
  if (( NDST < NPT )); then
    say "   !! $DST/model chi co $NDST/.pt so voi $NPT o nguon — KHONG xoa."
  else
    find "$SRC/model" -type f -name '*.pt' -delete
    say "   da xoa. / con: $(df -h / | tail -1 | awk '{print $4}') trong"
    say "   giu lai: ma nguon, data/, results/ ($(find "$SRC/results" -name 'fold*.json' 2>/dev/null | wc -l) file ket qua)"
  fi
else
  say "== 6. Chua xoa gi o /home (dat CONFIRM_DELETE=1 de xoa file .pt) =="
fi

# ---------- 7. Buoc tiep theo cho nguoi dung ----------
cat <<'TXT'

=====================================================================
 XONG. Mo phien moi:

   cd $DST
   claude

 Muon noi lai dung phien cu (session id cua phien 03/09):
   claude --resume a84c9fd3-bdf4-4ecf-8166-dc8953f347d2
   (ban 2.1.259 co tim session xuyen project khi duoc chi dinh id;
    `claude --resume` de trong hoac `--continue` thi KHONG thay)

 Doc theo thu tu: HANDOFF.md -> CLAUDE.md -> SERVER.md -> FACTS.md

 Bat lai cron giam sat khi can:
   crontab /tmp/mvd_crontab.bak
   (nho sua duong dan trong do tu /home/... sang $DST)
=====================================================================
TXT
