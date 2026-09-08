#!/usr/bin/env bash
# watch_int1_vast.sh — giam sat INT1 tren hai may vast (ntat = fold 1,2 | ntat2 = fold 3).
# Cron 10 phut. CHI ba viec: ghi trang thai, keo ket qua ve, phong lai driver da chet
# khi con o thieu. KHONG giet gi, KHONG xoa gi, KHONG bao gio tu huy may.
#
# Dia chi SSH giai theo NHAN moi lan goi (scripts/endpoints.sh): IP cua instance DOI khi
# no bi doi may chu, va doc cai timeout do thanh "may chet" la cach de huy nham nhat.
# Cron khong co moi truong: tu dat HOME/USER/PATH.
#   Thu: env -i /bin/bash -c 'DRY=1 bash /drive1/cuongtm/ntat/MultiVD/scripts/watch_int1_vast.sh'
export HOME="${HOME:-/home/ntat}"
export USER="${USER:-$(id -un)}"
export PATH="/home/ntat/.local/bin:/home/ntat/miniconda3/envs/vdenv/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"
set -u
ROOT=/drive1/cuongtm/ntat/MultiVD
cd "$ROOT" || exit 1
source scripts/endpoints.sh
LOG=log/watch_int1_vast.log
R=/workspace/MultiVD
PY=/venv/main/bin/python
MAXPASS="${MAXPASS:-3}"
DRY="${DRY:-0}"
ts(){ date -u '+%F %T'; }
say(){ echo "$(ts) | $*" >> "$LOG"; }

check(){ # $1=nhan  $2=danh sach fold  $3=so o ky vong
  local L="$1" FOLDS="$2" NEED="$3"
  VAST_CACHE_TTL=1 read -r H P <<< "$(vast_endpoint "$L" 2>/dev/null)"
  if [[ -z "${H:-}" || "${P:-None}" == "None" ]]; then say "$L | KHONG giai duoc dia chi"; return; fi
  local SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -p $P"
  local out
  out=$(timeout 90 $SSH root@"$H" "cd $R 2>/dev/null || exit 1
echo n=\$(ls results/int1_t5p/*/seed_42/fold*.json 2>/dev/null | wc -l)
echo alive=\$(ps -eo args --no-headers | grep -c 'src/train_[a-z]*\.py.*int1')
echo drv=\$(ps -eo args --no-headers | grep -c '[r]un/int1.sh')
echo vram=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo last=\$(grep -E '^=====|THAT BAI|INT1 xong' log/int1_${L}.log 2>/dev/null | tail -1 | cut -c1-110)" 2>/dev/null)
  if [[ -z "$out" ]]; then say "$L | KHONG SSH DUOC ($H:$P) — KHONG ket luan may chet"; return; fi
  # khong dinh nghia ham long trong ham: bash khong cho `local g(){...}`
  local n alive drv vram last
  n=$(sed -n 's/^n=//p' <<<"$out" | head -1)
  alive=$(sed -n 's/^alive=//p' <<<"$out" | head -1)
  drv=$(sed -n 's/^drv=//p' <<<"$out" | head -1)
  vram=$(sed -n 's/^vram=//p' <<<"$out" | head -1)
  last=$(sed -n 's/^last=//p' <<<"$out" | head -1)
  say "$L | o=${n:-?}/$NEED | job=${alive:-?} | driver=${drv:-?} | vram=${vram:-?}MiB | $last"

  mkdir -p "results_int1_${L}"
  local got
  got=$(rsync -az -e "$SSH" "root@$H:$R/results/int1_t5p/" "results_int1_${L}/" 2>/dev/null && \
        find "results_int1_${L}" -name 'fold*.json' | wc -l)
  say "$L | keo ve results_int1_${L}/ : ${got:-0} file"

  if [[ "${n:-0}" -lt "$NEED" && "${drv:-0}" -eq 0 && "${alive:-0}" -eq 0 ]]; then
    local pass; pass=$(cat "log/int1_${L}_passes" 2>/dev/null || echo 0)
    if (( pass >= MAXPASS )); then say "$L | da phong lai $pass lan van thieu — DUNG, can nguoi xem"; return; fi
    if [[ "$DRY" == 1 ]]; then say "$L | [DRY] se phong lai lan $((pass+1))"; return; fi
    echo $((pass+1)) > "log/int1_${L}_passes"
    timeout 60 $SSH root@"$H" "cd $R && FOLD_LIST='$FOLDS' SOURCES_LIST='4cwe com' SEED=42 PYTHON=$PY \
      setsid nohup bash run/int1.sh >> log/int1_${L}.log 2>&1 </dev/null & disown" >/dev/null 2>&1
    say "$L | PHONG LAI lan $((pass+1))"
  fi
}

check ntat  "1 2" 10
check ntat2 "3"    5
