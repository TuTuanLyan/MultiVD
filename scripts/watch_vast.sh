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
echo done=\$(grep -c . log/worklist.done 2>/dev/null)
echo vram=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo last=\$(tail -1 log/worklist.log 2>/dev/null | cut -c1-100)" 2>/dev/null)
  if [[ -z "$out" ]]; then say "$L | KHONG SSH DUOC ($H:$P) — KHONG ket luan may chet"; continue; fi
  g(){ sed -n "s/^$1=//p" <<<"$out" | head -1; }
  job=$(g job); wl=$(g wl); todo=$(g todo); dn=$(g done); vram=$(g vram); last=$(g last)
  say "$L | job=${job:-?} worklist=${wl:-?} xong=${dn:-0}/${todo:-?} vram=${vram:-?}MiB | $last"

  # ba tinh huong, ba xu ly — gop lai thi hoac bo lo may trong hoac phong lai nham
  if (( ${wl:-0} > 0 )); then continue; fi                       # con danh sach viec dang chay
  if (( ${job:-0} > 0 )); then say "$L | worklist chet nhung job con chay — cho"; continue; fi
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
  for RUN in int1 asam1; do
    mkdir -p "results_${RUN}_${L}"
    rsync -az -e "$SSH" "root@$H:$R/results/${RUN}_t5p/" "results_${RUN}_${L}/" 2>/dev/null
  done
  say "$L | keo ve: int1 $(find results_int1_$L -name 'fold*.json' 2>/dev/null|wc -l) | asam1 $(find results_asam1_$L -name 'fold*.json' 2>/dev/null|wc -l)"
done
