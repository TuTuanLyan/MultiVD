#!/usr/bin/env bash
# Giam sat dot xac nhan tren HAI may vast. Goi tu cron moi 10 phut:
#   */10 * * * * flock -n -o /tmp/mvd_w47.lock /bin/bash /home/ntat/workspace/MultiVD/scripts/watch47.sh
#
#   ntat  cong 39702 -> nguon common
#   ntat2 cong 35245 -> nguon 4cwe
#   moi may: 3 seed x 5 fold x 3 rho = 45 o Pha 2, cong 15 baseline
#
# BON BAI HOC DA AP DUNG, moi cai deu tung lam hong mot dot chay trong du an nay:
#   1. Cron khong co bien moi truong -> tu dat HOME/USER/PATH.
#   2. `pgrep -f`/`ps|grep` tu khop chinh no -> dung `flock` va thu thuat ngoac [c].
#   3. `flock` khong co -o thi con chau giu lock, moi lan cron sau bi chan im lang.
#   4. Dieu kien "xong" phai la dong ket thuc do driver in ra, KHONG phai dem o —
#      nguong dem co the khong bao gio dat va may cu tinh tien.
#   Va: dia day lam torch.save ghi cut, hong checkpoint ma trong nhu "phuong phap
#   kem" -> canh bao som khi dia con it.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HOME="${HOME:-/home/ntat}"
export USER="${USER:-$(id -un)}"
export PATH="/home/ntat/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

IP=202.122.49.242
# 04/09: doi phan cong — CA HAI may vast chay nguon `common`, chia theo SEED.
# Truoc do bang nay ghi 35245 la may "4cwe"; sau khi doi viec ma khong sua bang thi
# giam sat se soi glob `..._4cwe_l0p05_*` tren may do, khong thay gi, bao 0/45 roi
# PHONG LAI SRC=4cwe de len ngay seed dang chay. Chua kip xay ra vi driver con giu
# lock, nhung day la loi that: bang phan cong phai doi cung luc voi viec.
#   ten | cong | id vast | nguon | cac seed
declare -A PORT=( [ntat]=39702 [ntat2]=35245 )
declare -A VID=(  [ntat]=49840185 [ntat2]=49840024 )
declare -A MSRC=( [ntat]=com [ntat2]=com )
declare -A MSEED=( [ntat]="42 7" [ntat2]="1234" )
declare -A MNEED=( [ntat]=30 [ntat2]=15 )   # 3 rho x 5 fold x so seed
R=/workspace/MultiVD
PY=/venv/main/bin/python
ARM="${ARM:-none}"          # ARM=destroy thi moi huy may
LOG=log/watch47.log; mkdir -p log log/vast47
say(){ echo "$(date -u '+%F %T') $*" >> "$LOG"; }
sshx(){ timeout 60 ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$1" root@$IP "$2" 2>/dev/null; }

for M in ntat ntat2; do
  P=${PORT[$M]}; SRC=${MSRC[$M]}; SEEDS=${MSEED[$M]}; NEED=${MNEED[$M]}
  # keo ket qua ve (chi file json, nho)
  rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p $P" \
    --include='sw_*/' --include='sw_*/**' --exclude='*' \
    "root@$IP:$R/results/" "results_vast47_$M/" 2>/dev/null
  rsync -az -e "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -p $P" \
    --include="confirm47_*.log" --exclude='*' "root@$IP:$R/log/" "log/vast47/$M/" 2>/dev/null

  # Chi dem dong "xong" SAU lan "bat dau" gan nhat. Dem het ca file thi mot luot cu
  # da bi huy van de lai dong "xong", va giam sat tuong may da lam xong trong khi
  # luot moi con dang chay — dung loi nay tren ntat2 sang 04/09.
  FIN=$(sshx "$P" "awk '/CONFIRM47 bat dau/{c=0} /CONFIRM47 xong/{c++} END{print c+0}' $R/log/confirm47_${SRC}.log 2>/dev/null"); FIN="${FIN:-0}"
  BUSY=$(sshx "$P" "flock -n /tmp/multivd_confirm47_$SRC.lock -c true 2>/dev/null && echo 0 || echo 1"); BUSY="${BUSY:-1}"
  GLOB=""; for sd in $SEEDS; do GLOB="$GLOB $R/results/sw_t5p/transfer_latent_bottleneck_${SRC}_l0p05_*/seed_${sd}/fold*.json"; done
  N=$(sshx "$P" "ls $GLOB 2>/dev/null | wc -l"); N="${N:-0}"
  DISK=$(sshx "$P" "df -P / | tail -1 | awk '{print \$4}'"); DISK="${DISK:-0}"
  say "[$M/$SRC seed:$SEEDS] $N/$NEED o | driver=$BUSY | xong=$FIN | dia $(( DISK/1024 ))MB"

  (( DISK < 3145728 )) && say "[$M] !! DIA CON $(( DISK/1024 ))MB — nguy co torch.save ghi cut"

  if (( FIN == 0 )) && (( BUSY == 0 )); then
    say "[$M] driver CHET ma chua in dong ket thuc — phong lai"
    sshx "$P" "cd $R && SRC=$SRC SEEDS='$SEEDS' PYTHON=$PY setsid nohup bash run/confirm47.sh >> log/confirm47_$SRC.log 2>&1 </dev/null &"
  fi

  if (( FIN >= 1 )) && (( BUSY == 0 )); then
    LN=$(find "results_vast47_$M" -name 'fold*.json' 2>/dev/null | wc -l)
    RN=$(sshx "$P" "find $R/results -name 'fold*.json' | wc -l"); RN="${RN:-0}"
    say "[$M] XONG. may $RN file, da keo ve $LN file"
    if (( LN >= RN )) && (( RN > 0 )); then
      if [[ "$ARM" == "destroy" ]]; then
        OUT=$(vastai destroy instance "${VID[$M]}" -y 2>&1)
        say "[$M] huy ${VID[$M]}: $OUT"
        echo "$OUT" | grep -qi "abort\|error\|fail" && say "[$M] !! HUY THAT BAI"
      else
        say "[$M] ARM=none nen giu may."
      fi
    else
      say "[$M] chua keo ve du — giu may, thu lai vong sau."
    fi
  fi
done

# ---------- Theo doi 161 va 158: bao khi ranh de chia viec sang ----------
# Nguoi dung da nhac cuongtm; khi ho xong thi minh vao. KHONG tu dong phong job o
# day — chi ghi nhan trang thai roi de nguoi quyet.
#
# KE HOACH DA CHOT (04/09): khi 161 hoac 158 ranh thi CHUYEN NGUON `4cwe` ve do,
# giu `common` tren vast, roi HUY may vast ntat2 (id 49840024) de khoi tinh tien.
# Chia theo NGUON chu khong theo fold: moi nguon van tron ven trong mot may nen
# Delta ghep cap cua no khong dinh chenh lech gi. Rieng phep GOP hai nguon se mang
# them chenh lech torch (vast 2.11.0 vs local 2.9.1) — nguoi dung chap nhan o muc
# so sanh tuong doi, nhung phai ghi ro khi bao cao.
NEED_FREE=13000        # MB trong toi thieu de job t5p chay duoc (thuc te dung ~12.6 GB)
check_local(){          # $1 = ten, $2 = lenh chay (rong = chay tai cho)
  local name=$1 cmd=$2 used total free others
  if [[ -z "$cmd" ]]; then
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
    others=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | wc -l)
  else
    read -r used total others < <($cmd 'echo "$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ",")" "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"' 2>/dev/null)
  fi
  [[ -z "${used:-}" ]] && { say "[$name] khong doc duoc GPU"; return; }
  free=$(( ${total:-0} - ${used:-0} ))
  if (( free >= NEED_FREE )); then
    say "[$name] >>> RANH: con ${free}MB, $others tien trinh — CHUYEN NGUON 4cwe VE DAY, roi huy vast ntat2 (49840024)"
  else
    say "[$name] ban: con ${free}MB, $others tien trinh"
  fi
}
check_local 161 ""
check_local 158 "ssh -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=15 -i /home/ntat/.ssh/id_ed25519 tranmanhcuong@112.137.129.158"
