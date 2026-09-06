#!/usr/bin/env bash
# Giam sat dot VA seed 7 (fold 3,4,5) tren may vast `ntat` 49951040. Cron 10 phut:
#   */10 * * * * ARM=destroy /usr/bin/flock -n -o /tmp/mvd_w48b.lock /bin/bash .../scripts/watch48b.sh
#
# Ky vong: 3 fold x 3 nguon x 2 rho = 18 o Pha 2 + 3 baseline. Pha 1 KHONG chay
# (checkpoint seed 7 day tu 161 len, night48.sh thay co san thi bo qua).
#
# Cac bai hoc da ap dung, moi cai deu tung lam hong mot dot chay:
#   cron khong co bien moi truong; `pgrep -f` tu khop chinh no; `flock` thieu -o thi
#   con chau giu lock; dieu kien "xong" phai la dong driver tu in; cong dem "xong" phai
#   reset o moi "bat dau"; va "xong" mot minh chua du — thieu o thi phong lai co chan.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HOME="${HOME:-/home/ntat}"
export USER="${USER:-$(id -un)}"
export PATH="/home/ntat/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

ARM="${ARM:-none}"
R=/workspace/MultiVD
PY=/venv/main/bin/python
NEED=18; NEEDB=3
LOG=log/watch48b.log; mkdir -p log results_night48b
say(){ echo "$(date -u '+%F %T') $*" >> "$LOG"; }

source scripts/endpoints.sh 2>/dev/null || true
read -r H P <<< "$(vast_endpoint ntat 2>/dev/null)"
if [[ -z "${H:-}" || -z "${P:-}" || "$P" == "None" ]]; then
  say "[ntat-b] chua giai duoc dia chi (loading, hoac da huy) — bo qua vong nay"; exit 0
fi
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p $P"
sshx(){ timeout 60 $SSH root@"$H" "$1" 2>/dev/null; }

rsync -az -e "$SSH" --include='n48_*/' --include='n48_*/**' --exclude='*' \
      "root@$H:$R/results/" results_night48b/ 2>/dev/null
rsync -az -e "$SSH" --include='night48.log' --exclude='*' "root@$H:$R/log/" log/vast48b/ 2>/dev/null

FIN=$(sshx "awk '/NIGHT48 bat dau/{c=0} /NIGHT48 xong/{c++} END{print c+0}' $R/log/night48.log 2>/dev/null"); FIN="${FIN:-0}"
BUSY=$(sshx "flock -n /tmp/multivd_night48.lock -c true 2>/dev/null && echo 0 || echo 1"); BUSY="${BUSY:-1}"
N=$(sshx "ls $R/results/n48_t5p/transfer_latent_bottleneck_*_l0p05_r*/seed_7/fold[345].json 2>/dev/null | wc -l"); N="${N:-0}"
NB=$(sshx "ls $R/results/n48_t5p/baseline/seed_7/fold[345].json 2>/dev/null | wc -l"); NB="${NB:-0}"
DISK=$(sshx "df -P / | tail -1 | awk '{print \$4}'"); DISK="${DISK:-0}"
say "[ntat-b :$P] $N/$NEED o + $NB/$NEEDB baseline | driver=$BUSY xong=$FIN | dia $(( DISK/1024 ))MB"
(( DISK < 3145728 )) && say "[ntat-b] !! DIA CON $(( DISK/1024 ))MB"

# CHOT: khong bao gio phong driver khi chua du CA BA checkpoint Pha 1 cua seed 7.
# night48.sh thay thieu checkpoint la TU HUAN LUYEN Pha 1 moi — fold 3-5 se xuat phat
# tu nguon khac fold 1-2 va seed 7 tu mau thuan. Day dung la canh cron suyt lam luc
# 10:33 ngay 05/09, khi rsync moi day duoc 1/3 checkpoint.
launch(){
  local ck
  ck=$(sshx "ls $R/model/n48/phase1/*/seed_7/best.pt 2>/dev/null | wc -l"); ck="${ck:-0}"
  if (( ck < 3 )); then
    say "[ntat-b] CHUA phong: moi $ck/3 checkpoint Pha 1 seed 7 tren may (dang day len?)"
    return 1
  fi
  sshx "cd $R && SEEDS='7' FOLD_LIST='3 4 5' PYTHON=$PY setsid nohup bash run/night48.sh >> log/night48.log 2>&1 </dev/null & disown" >/dev/null
}

if (( FIN == 0 )) && (( BUSY == 0 )); then
  say "[ntat-b] driver CHET ma chua in dong ket thuc — phong lai"; launch; exit 0
fi

PASSF=log/n48b_passes; [[ -f "$PASSF" ]] || echo 0 > "$PASSF"
PASSES=$(cat "$PASSF" 2>/dev/null); PASSES="${PASSES:-0}"
if (( FIN >= 1 )) && (( BUSY == 0 )) && (( N < NEED )) && (( PASSES < 2 )); then
  echo $(( PASSES + 1 )) > "$PASSF"
  say "[ntat-b] driver in xong nhung moi $N/$NEED o — phong lai lan $(( PASSES + 1 ))/2"
  launch; exit 0
fi

if (( FIN >= 1 )) && (( BUSY == 0 )); then
  sshx "cd $R/results && find n48_* -name '*.json' -printf '%p\t%s\n' | LC_ALL=C sort" > /tmp/n48b_rem.txt
  ( cd results_night48b && find n48_* -name '*.json' -printf '%p\t%s\n' 2>/dev/null | LC_ALL=C sort ) > /tmp/n48b_loc.txt
  RN=$(wc -l < /tmp/n48b_rem.txt); MISS=$(LC_ALL=C comm -23 /tmp/n48b_rem.txt /tmp/n48b_loc.txt | wc -l)
  say "[ntat-b] XONG. may $RN file, lech $MISS"
  if (( RN > 0 )) && (( MISS == 0 )); then
    if [[ "$ARM" == "destroy" ]]; then
      VID=$(vastai show instances --raw 2>/dev/null | python3 -c "import json,sys;print(next((i['id'] for i in json.load(sys.stdin) if i.get('label')=='ntat'),''))")
      [[ -z "$VID" ]] && { say "[ntat-b] khong tim ra id — KHONG huy"; exit 0; }
      OUT=$(vastai destroy instance "$VID" -y 2>&1); say "[ntat-b] huy $VID: $OUT"
      echo "$OUT" | grep -qi "abort\|error\|fail" && say "[ntat-b] !! HUY THAT BAI — MAY VAN TINH TIEN"
    else
      say "[ntat-b] ARM=none nen giu may."
    fi
  else
    LC_ALL=C comm -23 /tmp/n48b_rem.txt /tmp/n48b_loc.txt | head -5 | sed 's/^/    thieu: /' >> "$LOG"
    say "[ntat-b] chua keo ve du — GIU MAY"
  fi
fi
