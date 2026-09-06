#!/usr/bin/env bash
# Giam sat NIGHT48 tren may vast nhan `ntat`. Goi tu cron moi 10 phut:
#   */10 * * * * /usr/bin/flock -n -o /tmp/mvd_w48.lock /bin/bash .../scripts/watch48.sh
#
# Khoi dang chay: lambda=0.05, latent_bottleneck, recadam, t5p-bimodal, ba nguon
# 4cwe/com/full x fold 1-5 x rho {0, 0.1}. TU 05/09 chia ba may theo SEED:
#   vast ntat -> seed 42   |   161 -> seed 7   |   158 -> seed 1234
# Script NAY chi lo may vast. Hai may local do scripts/watch48_local.sh lo.
# Chia theo seed la sach: moi seed tu co Pha 1 va baseline rieng nen moi Delta
# ghep cap nam gon trong mot may.
#
# NAM BAI HOC DA AP DUNG, moi cai deu tung lam hong mot dot chay trong du an nay:
#   1. Cron khong co bien moi truong  -> tu dat HOME/USER/PATH.
#   2. `pgrep -f`/`ps|grep` tu khop chinh no -> song bang `flock`, khong bao gio grep.
#   3. `flock` khong co -o thi con chau giu lock -> moi lan cron sau bi chan im lang.
#   4. Dieu kien "xong" phai la dong ket thuc do CHINH driver in ra, khong bao gio la
#      "dem du N o" — nguong dem co the khong bao gio dat va may cu tinh tien.
#   5. Cong dem "xong" phai reset o moi dong "bat dau", neu khong mot luot cu bi huy
#      van de lai dong "xong" va giam sat tuong da xong trong khi luot moi dang chay.
#   Va: dia day lam torch.save ghi cut -> canh bao som, KHONG tu xoa gi.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HOME="${HOME:-/home/ntat}"
export USER="${USER:-$(id -un)}"
export PATH="/home/ntat/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

ARM="${ARM:-none}"           # ARM=destroy thi moi duoc huy may
R=/workspace/MultiVD
PY=/venv/main/bin/python
# 05/09 03:20 — chia viec: vast chi con SEED 42 (30 o + 5 baseline).
# Bang phan cong phai doi CUNG LUC voi viec; lan truoc doi viec ma quen sua bang
# thi giam sat soi nham glob roi phong mot job xen ngang len may dang chay.
VSEEDS='42'; NEED=30; NEEDB=5
LOG=log/watch48.log; mkdir -p log log/vast48 results_night48
say(){ echo "$(date -u '+%F %T') $*" >> "$LOG"; }

source scripts/endpoints.sh 2>/dev/null || true
read -r H P <<< "$(vast_endpoint ntat 2>/dev/null)"
if [[ -z "${H:-}" || -z "${P:-}" || "$P" == "None" ]]; then
  say "[ntat] chua giai duoc dia chi (dang loading, hoac da huy) — bo qua vong nay"; exit 0
fi
SSH="ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p $P"
sshx(){ timeout 60 $SSH root@"$H" "$1" 2>/dev/null; }

# ---- keo ket qua ve (chi json, nho) va log driver ----
rsync -az -e "$SSH" --include='n48_*/' --include='n48_*/**' --exclude='*' \
      "root@$H:$R/results/" results_night48/ 2>/dev/null
rsync -az -e "$SSH" --include='night48.log' --exclude='*' \
      "root@$H:$R/log/" log/vast48/ 2>/dev/null

# ---- trang thai ----
FIN=$(sshx "awk '/NIGHT48 bat dau/{c=0} /NIGHT48 xong/{c++} END{print c+0}' $R/log/night48.log 2>/dev/null"); FIN="${FIN:-0}"
BUSY=$(sshx "flock -n /tmp/multivd_night48.lock -c true 2>/dev/null && echo 0 || echo 1"); BUSY="${BUSY:-1}"
N=$(sshx "ls $R/results/n48_t5p/transfer_latent_bottleneck_*_l0p05_r*/seed_*/fold*.json 2>/dev/null | wc -l"); N="${N:-0}"
NB=$(sshx "ls $R/results/n48_t5p/baseline/seed_*/fold*.json 2>/dev/null | wc -l"); NB="${NB:-0}"
DISK=$(sshx "df -P / | tail -1 | awk '{print \$4}'"); DISK="${DISK:-0}"
say "[ntat :$P] $N/$NEED o + $NB/$NEEDB baseline | driver=$BUSY xong=$FIN | dia $(( DISK/1024 ))MB"

(( DISK < 3145728 )) && say "[ntat] !! DIA CON $(( DISK/1024 ))MB — nguy co torch.save ghi cut, CAN NGUOI XU LY (watchdog khong tu xoa)"

# ---- BAN GIAO: xong seed 42 roi thi dung, vi hai seed kia da sang may local ----
# Driver duoc phong voi SEEDS='42 7 1234' truoc khi chia viec, nen xong seed 42 no se
# di tiep sang seed 7 — trung viec voi 161. Cat o day, roi phong lai voi SEEDS='42':
# lan chay moi thay 30/30 o da co, bo qua het, va tu in dong ket thuc THAT cua chinh
# no. Khong bao gio tu tay ghi dong do vao log — cong huy phai bam tin hieu do driver
# phat ra, gia mot dong la vut bo ca cong.
#
# BAY DA MAC (05/09 04:21): dieu kien duoi day KHONG phan biet duoc driver cu (3 seed)
# voi driver vua phong lai (1 seed). Ca hai deu la "seed 42 xong + dang chay + chua in
# ket thuc", nen moi vong watchdog lai giet va phong lai — lap vo han, may cu tinh tien.
# Chan bang MOT moc dat tren dia: ban giao chi duoc xay ra dung mot lan.
HANDOFF=log/n48_handoff_done
SEED42_DONE=$(sshx "grep -c 'NIGHT48 seed 42 xong' $R/log/night48.log 2>/dev/null"); SEED42_DONE="${SEED42_DONE:-0}"
if (( SEED42_DONE >= 1 )) && (( BUSY == 1 )) && (( FIN == 0 )) && [[ ! -f "$HANDOFF" ]]; then
  date -u '+%F %T' > "$HANDOFF"
  # Tim pgid qua NGUOI GIU LOCK, khong grep ps: lan truoc grep bat nham mot bash khac
  # va kill -TERM -1552 di lac, driver that (pgid 1555) van song.
  PG=$(sshx "p=\$(fuser /tmp/multivd_night48.lock 2>/dev/null | tr -s ' ' '\n' | grep -E '^[0-9]+$' | head -1); [ -n \"\$p\" ] && ps -o pgid= -p \$p | tr -d ' '")
  if [[ -n "${PG:-}" ]]; then
    say "[ntat] seed 42 xong, hai seed kia o may local — dung driver (pgid $PG) roi phong lai voi SEEDS=42"
    sshx "kill -TERM -$PG 2>/dev/null; sleep 8; kill -KILL -$PG 2>/dev/null; true"
    sshx "cd $R && SEEDS='$VSEEDS' PYTHON=$PY setsid nohup bash run/night48.sh >> log/night48.log 2>&1 </dev/null & disown" >/dev/null
  else
    say "[ntat] seed 42 xong nhung KHONG tim ra pgid driver — de nguyen, xem tay"
  fi
  exit 0
fi

# ---- driver chet giua chung -> phong lai ----
if (( FIN == 0 )) && (( BUSY == 0 )); then
  say "[ntat] driver CHET ma chua in dong ket thuc — phong lai"
  sshx "cd $R && SEEDS='$VSEEDS' PYTHON=$PY setsid nohup bash run/night48.sh >> log/night48.log 2>&1 </dev/null & disown" >/dev/null
  exit 0
fi

# ---- driver in dong ket thuc NHUNG con thieu o -> phong lai, co chan ----
# Cong huy khong duoc dem o (nguong dem co the khong bao gio dat -> may tinh tien mai).
# Nhung "driver in xong" mot minh cung khong du: neu vai checkpoint Pha 1 hong thi no
# van in xong voi 60/90 o, va huy luon la vut ca may lan o con thieu. matrix.sh bo qua
# o da co nen phong lai chi chay phan thieu. Chan bang bo dem: toi da 2 lan.
PASSF=log/n48_passes; [[ -f "$PASSF" ]] || echo 0 > "$PASSF"
PASSES=$(cat "$PASSF" 2>/dev/null); PASSES="${PASSES:-0}"
if (( FIN >= 1 )) && (( BUSY == 0 )) && (( N < NEED )) && (( PASSES < 2 )); then
  echo $(( PASSES + 1 )) > "$PASSF"
  say "[ntat] driver da in XONG nhung moi $N/$NEED o — phong lai lan $(( PASSES + 1 ))/2 de chay phan thieu"
  sshx "cd $R && SEEDS='$VSEEDS' PYTHON=$PY setsid nohup bash run/night48.sh >> log/night48.log 2>&1 </dev/null & disown" >/dev/null
  exit 0
fi
(( FIN >= 1 )) && (( BUSY == 0 )) && (( N < NEED )) && \
  say "[ntat] da phong lai $PASSES lan van chi $N/$NEED o — chap nhan, doi chieu roi huy"

# ---- xong that -> doi chieu TUNG BYTE roi moi huy ----
if (( FIN >= 1 )) && (( BUSY == 0 )); then
  # Danh sach "duong-dan<TAB>kich-thuoc" hai ben, so bang comm duoi LC_ALL=C.
  # Dem file khong du: mot lan rsync dut giua chung de lai file ngan hon ma `wc -l`
  # van thay "du".
  sshx "cd $R/results && find n48_* -name '*.json' -printf '%p\t%s\n' | LC_ALL=C sort" > /tmp/n48_remote.txt
  ( cd results_night48 && find n48_* -name '*.json' -printf '%p\t%s\n' 2>/dev/null | LC_ALL=C sort ) > /tmp/n48_local.txt
  RN=$(wc -l < /tmp/n48_remote.txt); LN=$(wc -l < /tmp/n48_local.txt)
  MISS=$(LC_ALL=C comm -23 /tmp/n48_remote.txt /tmp/n48_local.txt | wc -l)
  say "[ntat] XONG. may $RN file, local $LN file, lech $MISS"
  if (( RN > 0 )) && (( MISS == 0 )); then
    if [[ "$ARM" == "destroy" ]]; then
      VID=$(vastai show instances --raw 2>/dev/null | python3 -c "import json,sys;print(next((i['id'] for i in json.load(sys.stdin) if i.get('label')=='ntat'),''))")
      if [[ -z "$VID" ]]; then say "[ntat] khong tim ra id — KHONG huy"; exit 0; fi
      OUT=$(vastai destroy instance "$VID" -y 2>&1)
      say "[ntat] huy $VID: $OUT"
      # `vastai destroy` in "Aborted." roi thoat 0 — phai doc output, khong tin ma thoat.
      echo "$OUT" | grep -qi "abort\|error\|fail" && say "[ntat] !! HUY THAT BAI — may VAN DANG TINH TIEN"
    else
      say "[ntat] ARM=none nen giu may. Huy tay: vastai destroy instance <id> -y"
    fi
  else
    LC_ALL=C comm -23 /tmp/n48_remote.txt /tmp/n48_local.txt | head -5 | sed 's/^/    thieu: /' >> "$LOG"
    say "[ntat] chua keo ve du — GIU MAY, thu lai vong sau"
  fi
fi
