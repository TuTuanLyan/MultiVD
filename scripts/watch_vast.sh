#!/usr/bin/env bash
# watch_vast.sh — giam sat HAI may vast va PHONG LAI danh sach viec khi no chet.
# Thay watch_int1_vast.sh: cai cu chi BAO DONG "may dang trong" vao log, va bao dong
# khong phai hanh dong — ntat2 nam khong 23 phut trong khi bao dong da keu 13 phut truoc.
#
# Cron 10 phut. KHONG giet gi, KHONG xoa gi, KHONG BAO GIO tu huy may.
# Dia chi SSH giai theo NHAN moi lan goi: IP doi khi instance bi doi may chu.
#   Thu: env -i /bin/bash -c 'DRY=1 bash /drive1/cuongtm/ntat/MultiVD/scripts/watch_vast.sh'
export HOME="${HOME:-/home/ntat}"
export USER="${USER:-$(id -un)}"
export PATH="/home/ntat/.local/bin:/home/ntat/miniconda3/envs/vdenv/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
set -u
ROOT=/drive1/cuongtm/ntat/MultiVD
cd "$ROOT" || exit 1
source scripts/endpoints.sh
LOG=log/watch_vast.log
R=/workspace/MultiVD
MAXPASS="${MAXPASS:-30}"
DRY="${DRY:-0}"
ts(){ date -u '+%F %T'; }
say(){ echo "$(ts) | $*" >> "$LOG"; }

for L in ntat ntat2; do
  VAST_CACHE_TTL=1 read -r H P <<< "$(vast_endpoint "$L" 2>/dev/null)"
  if [[ -z "${H:-}" || "${P:-None}" == "None" ]]; then say "$L | KHONG giai duoc dia chi"; continue; fi
  SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -p $P"
  out=$(timeout 90 $SSH root@"$H" "cd $R 2>/dev/null || exit 1
echo job=\$(ps -eo args --no-headers | grep -c '[s]rc/train_[a-z]*\.py')
echo wl=\$(ps -eo args --no-headers | grep -c '[v]ast_worklist.sh')
echo todo=\$(grep -vc '^\s*#\|^\s*$' log/worklist.txt 2>/dev/null)
echo done=\$(grep -xF -f <(grep -v '^\s*#\|^\s*\$' log/worklist.txt) log/worklist.done 2>/dev/null | grep -c .)
echo vram=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo last=\$(tail -1 log/worklist.log 2>/dev/null | cut -c1-100)" 2>/dev/null)
  if [[ -z "$out" ]]; then say "$L | KHONG SSH DUOC ($H:$P) — KHONG ket luan may chet"; continue; fi
  g(){ sed -n "s/^$1=//p" <<<"$out" | head -1; }
  job=$(g job); wl=$(g wl); todo=$(g todo); dn=$(g done); vram=$(g vram); last=$(g last)
  say "$L | job=${job:-?} worklist=${wl:-?} xong=${dn:-0}/${todo:-?} vram=${vram:-?}MiB | $last"

  # ba tinh huong, ba xu ly — gop lai thi hoac bo lo may trong hoac phong lai nham
  if (( ${wl:-0} > 0 )); then continue; fi                       # con danh sach viec dang chay
  if (( ${job:-0} > 0 )); then say "$L | worklist chet nhung job con chay — cho"; continue; fi
  # `dn` phai la SO GIAO giua worklist.done va worklist.txt, khong phai so dong tho.
  # 09/09/2026 00:15: ntat2 co worklist.done 6 dong nhung BA dong thuoc danh sach CU
  # (da xep lai). Dem tho cho 6/6 nen cong nay se ket luan "DA XONG HET" va KHONG phong
  # lai — trong khi ba muc moi chua chay. May vast se nam khong ma van tinh tien.
  if (( ${dn:-0} >= ${todo:-0} )) && (( ${todo:-0} > 0 )); then
    say "$L | *** DA XONG HET ${todo} MUC — MAY TRONG, can them viec vao log/worklist.txt hoac huy may ***"
    continue
  fi
  pass=$(cat "log/vast_${L}_passes" 2>/dev/null || echo 0)
  if (( pass >= MAXPASS )); then say "$L | da phong lai $pass lan — DUNG, can nguoi xem"; continue; fi
  if [[ "$DRY" == 1 ]]; then say "$L | [DRY] se phong lai worklist lan $((pass+1))"; continue; fi
  echo $((pass+1)) > "log/vast_${L}_passes"
  timeout 40 $SSH root@"$H" "cd $R && setsid nohup bash scripts/vast_worklist.sh >> log/worklist.log 2>&1 </dev/null & disown" >/dev/null 2>&1
  say "$L | PHONG LAI worklist lan $((pass+1))"
done

# keo ket qua ve, moi khoi mot thu muc rieng theo may
for L in ntat ntat2; do
  VAST_CACHE_TTL=1 read -r H P <<< "$(vast_endpoint "$L" 2>/dev/null)"
  [[ -n "${H:-}" && "${P:-None}" != "None" ]] || continue
  SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -p $P"
  # DANH SACH CU CUNG `int1 asam1` DA BO SOT: 09/09 cac khoi asamaw / poolcb / wblend
  # deu ghi vao cay khac va watchdog KHONG keo ve. Voi quy tac "mat log => khong huy",
  # mot cay khong duoc keo ve la mot cay co the mat khi huy may. Gio HOI MAY XA xem no
  # co nhung cay nao roi keo het, thay vi giu danh sach cung tay.
  RUNS=$(timeout 40 $SSH root@"$H" "ls -1 $R/results 2>/dev/null" 2>/dev/null | tr -d '\r')
  [[ -n "$RUNS" ]] || { say "$L | khong liet ke duoc results/ — BO QUA lan keo nay"; continue; }
  pulled=""
  for RUN in $RUNS; do
    # CAT hau to backbone: cay tren may xa ten <run>_<backbone> nhung quy uoc thu muc
    # local o day la results_<run>_<may> (results_pool1_ntat2, results_e60_ntat...).
    # Giu nguyen ca ten se tao thu muc TRUNG ben canh cay cu -> moi cong cu glob
    # `results_*` dem MOT o thanh HAI. Da mac dung loi nay luc 04:20 va da hoa giai.
    BASE="${RUN%_t5p}"; BASE="${BASE%_t5pe}"; BASE="${BASE%_codebert}"
    BASE="${BASE%_unixcoder}"; BASE="${BASE%_roberta}"
    D="results_${BASE}_${L}"
    mkdir -p "$D"
    rsync -az -e "$SSH" "root@$H:$R/results/${RUN}/" "$D/" 2>/dev/null
    n=$(find "$D" -name 'fold*.json' 2>/dev/null | wc -l)
    # thu muc rong thi don di cho khoi rac cay ket qua
    if (( n == 0 )); then rmdir "$D" 2>/dev/null; else pulled="$pulled ${BASE}=$n"; fi
  done
  say "$L | keo ve:${pulled:- (khong co o nao)}"
done
