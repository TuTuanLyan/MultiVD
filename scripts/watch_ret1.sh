#!/usr/bin/env bash
# watch_ret1.sh — giam sat khoi RET1 (do giu lai tri thuc nguon) tren 158, cron 10 phut.
#
# CHI ba viec: (1) ghi mot dong trang thai, (2) phong lai driver khi no chet ma con o
# thieu, (3) keo ket qua ve results_ret1_158/. KHONG BAO GIO giet tien trinh, KHONG xoa gi.
#
# Dem HIEN VAT (so fold*.json theo tung tag), khong doc dong log — "xong" trong log khong
# co nghia la viec da thanh (memory one-driver-one-lock-and-count-artifacts).
# Cron khong co moi truong (memory cron-has-no-environment): tu dat HOME/USER/PATH.
#   Thu:  env -i /bin/bash -c 'DRY=1 bash /drive1/cuongtm/ntat/MultiVD/scripts/watch_ret1.sh'
#   Thu cong dem:  bash scripts/watch_ret1.sh --test-count
export HOME="${HOME:-/home/ntat}"
export USER="${USER:-$(id -un)}"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
set -u
ROOT=/drive1/cuongtm/ntat/MultiVD
cd "$ROOT" || exit 1
LOG=log/watch_ret1.log
R158="tranmanhcuong@112.137.129.158"; ROOT158=/data/ntat/MultiVD
PY158=/data/ntat/envs/vdenv/bin/python
FOLDS="${FOLDS:-1 2 3}"; SRCS="${SRCS:-4cwe com}"
TAGS="plain_adamw c5000_t0p05 c50_t0p05"
MAXPASS="${MAXPASS:-3}"
DRY="${DRY:-0}"
ts(){ date -u '+%F %T'; }
say(){ echo "$(ts) | $*" >> "$LOG"; }

# Lenh dem chay TREN 158. In mot so: so o con thieu.
count_cmd(){
  echo "cd $ROOT158 || exit 1; n=0
for F in $FOLDS; do
  [ -f results/ret1_t5p/baseline/seed_42/fold\$F.json ] || n=\$((n+1))
  for S in $SRCS; do for T in $TAGS; do
    [ -f results/ret1_t5p/transfer_latent_bottleneck_\${S}_l0p05_\${T}/seed_42/fold\$F.json ] || n=\$((n+1))
  done; done
done; echo \$n"
}

if [[ "${1:-}" == "--test-count" ]]; then
  # Cong hai chieu: dem that, roi dem voi mot tag BIA phai ra so LON HON.
  a=$(ssh -o BatchMode=yes -o ConnectTimeout=15 "$R158" "$(count_cmd)")
  b=$(TAGS="$TAGS tag_khong_ton_tai" ssh -o BatchMode=yes -o ConnectTimeout=15 "$R158" "$(TAGS="$TAGS tag_khong_ton_tai" count_cmd)")
  echo "dem that=$a | dem voi tag bia=$b"
  [[ -n "$a" && -n "$b" && "$b" -gt "$a" ]] && echo "cong dem OK" || { echo "cong dem SAI"; exit 1; }
  exit 0
fi

out=$(timeout 90 ssh -o BatchMode=yes -o ConnectTimeout=15 "$R158" "cd $ROOT158 || exit 1
{ flock -n /tmp/mvd_ret1.lock -c true && echo alive=no || echo alive=yes; }
echo miss=\$($(count_cmd) 2>/dev/null | tail -1)
echo used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo procs=\$(ps -eo args --no-headers | grep -c 'src/train_[a-z]*\.py.*ret1')
echo last=\$(grep -E '^=====|^=== .*\| fold|THAT BAI|RET1' log/ret1_158.log 2>/dev/null | tail -1 | cut -c1-120)" 2>/dev/null)

if [[ -z "$out" ]]; then say "158 | KHONG SSH DUOC"; exit 0; fi
g(){ sed -n "s/^$1=//p" <<<"$out" | head -1; }
alive=$(g alive); miss=$(g miss); used=$(g used); procs=$(g procs); last=$(g last)
say "ret1 | driver=$alive | thieu=${miss:-?}/21 | vram=${used:-?}MiB | train_procs=${procs:-?} | $last"

if [[ "$alive" == no && "${miss:-1}" != 0 ]]; then
  pass=$(cat log/ret1_passes 2>/dev/null || echo 0)
  if (( pass >= MAXPASS )); then say "ret1 | da phong lai $pass lan ma van thieu ${miss} — DUNG, can nguoi xem"
  elif (( ${procs:-0} > 0 )); then say "ret1 | con ${procs} job train — cho"
  elif [[ "$DRY" == 1 ]]; then say "ret1 | [DRY] se phong lai lan $((pass+1))"
  else
    echo $((pass+1)) > log/ret1_passes
    ssh -o BatchMode=yes -o ConnectTimeout=15 "$R158" "cd $ROOT158 && FOLD_LIST='$FOLDS' SOURCES_LIST='$SRCS' SEED=42 PYTHON=$PY158 setsid nohup flock -n -o /tmp/mvd_ret1.lock bash run/ret1.sh >> log/ret1_158.log 2>&1 < /dev/null &" >/dev/null 2>&1
    say "ret1 | PHONG LAI lan $((pass+1))"
  fi
fi

mkdir -p results_ret1_158
n=$(rsync -az --stats "$R158:$ROOT158/results/ret1_t5p/" results_ret1_158/ 2>/dev/null \
    | sed -n 's/^Number of regular files transferred: //p')
say "ret1 | rsync -> results_ret1_158/ : $(find results_ret1_158 -name 'fold*.json' | wc -l) file (moi: ${n:-0})"
