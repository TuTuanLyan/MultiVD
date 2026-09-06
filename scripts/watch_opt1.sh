#!/usr/bin/env bash
# watch_opt1.sh — giam sat khoi OPT1 tren 161 (local) va 158 (ssh), cron 10 phut/lan.
#
# CHI ba viec: (1) ghi mot dong trang thai moi may, (2) PHONG LAI driver khi no chet
# ma chua in dong ket thuc `########## OPT1 xong` (toi da MAXPASS lan, 161 chi khi VRAM
# trong >= 13 GB va khong con job train mo coi), (3) keo ket qua 158 ve results_opt1_158/.
# KHONG BAO GIO giet tien trinh, KHONG xoa gi.
#
# Cron khong co moi truong (memory cron-has-no-environment): tu dat HOME/USER/PATH.
# Thu: env -i /bin/bash -c 'DRY=1 bash /drive1/cuongtm/ntat/MultiVD/scripts/watch_opt1.sh'
# Hai chieu: LOCK161=/tmp/khong_ton_tai.lock DRY=1 ... phai ra dong "PHONG LAI" [DRY].
export HOME="${HOME:-/home/ntat}"
export USER="${USER:-$(id -un)}"
export PATH="/usr/local/bin:/usr/bin:/bin:${PATH:-}"
set -u
ROOT=/drive1/cuongtm/ntat/MultiVD
cd "$ROOT" || exit 1
LOG=log/watch_opt1.log
PY161=/home/ntat/miniconda3/envs/vdenv/bin/python
R158="tranmanhcuong@112.137.129.158"; ROOT158=/data/ntat/MultiVD; PY158=/data/ntat/envs/vdenv/bin/python
LOCK161="${LOCK161:-/tmp/multivd_opt1.lock}"
DONE_RE='^########## OPT1 xong'
MAXPASS="${MAXPASS:-3}"
DRY="${DRY:-0}"
ts(){ date -u '+%F %T'; }
say(){ echo "$(ts) | $*" >> "$LOG"; }

# ---------------- 161 ----------------
alive161=$(flock -n "$LOCK161" -c true 2>/dev/null && echo no || echo yes)
n161=$(ls results/opt1_t5p/transfer_*/seed_42/fold*.json 2>/dev/null | wc -l)
b161=$(ls results/opt1_t5p/baseline/seed_42/fold*.json 2>/dev/null | wc -l)
fin161=$(grep -c "$DONE_RE" log/opt1_161.log 2>/dev/null); fin161="${fin161:-0}"
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1); used="${used:-0}"
free=$(( 16376 - used ))
# job train mo coi cua chinh khoi nay? (loc theo cot, khong tu khop: dong lenh cua script nay la `bash scripts/watch_opt1.sh`)
orphan=$(ps -eo user,args --no-headers | awk '$1=="ntat" && /src\/train_(transfer|baseline)\.py/ && /opt1/' | wc -l)
last161=$(grep -E "^=====|^=== .*\| fold|THAT BAI|xong" log/opt1_161.log 2>/dev/null | tail -1 | cut -c1-120)
say "161 | driver=$alive161 | o=$n161/40 base=$b161/2 | fin=$fin161 | vram_used=${used}MiB | train_procs=$orphan | $last161"
if [[ "$alive161" == no && "$fin161" -eq 0 ]]; then
  pass=$(cat log/opt1_161_passes 2>/dev/null || echo 0)
  if (( pass >= MAXPASS )); then say "161 | driver chet, da phong lai $pass lan — DUNG, can nguoi xem"
  elif (( orphan > 0 )); then say "161 | driver chet nhung con $orphan job train dang chay — cho, chua phong lai"
  elif (( free < 13000 )); then say "161 | driver chet nhung VRAM trong ${free}MiB < 13000 — NHUONG, chua phong lai"
  else
    if [[ "$DRY" == 1 ]]; then say "161 | [DRY] se PHONG LAI lan $((pass+1))"
    else
      echo $((pass+1)) > log/opt1_161_passes
      FOLD_LIST="1 2" PYTHON=$PY161 setsid nohup bash run/opt1.sh >> log/opt1_161.log 2>&1 < /dev/null &
      say "161 | PHONG LAI lan $((pass+1))"
    fi
  fi
fi

# ---------------- 158 ----------------
out=$(timeout 60 ssh -o BatchMode=yes -o ConnectTimeout=15 "$R158" "cd $ROOT158 || exit 1
{ flock -n /tmp/multivd_opt1.lock -c true && echo alive=no || echo alive=yes; }
echo n=\$(ls results/opt1_t5p/transfer_*/seed_42/fold*.json 2>/dev/null | wc -l)
echo b=\$(ls results/opt1_t5p/baseline/seed_42/fold*.json 2>/dev/null | wc -l)
echo fin=\$(grep -c '$DONE_RE' log/opt1_158.log 2>/dev/null)
echo used=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
echo orphan=\$(ps -eo args --no-headers | grep -c 'src/train_[a-z]*\.py.*opt1')
echo last=\$(grep -E '^=====|THAT BAI|xong' log/opt1_158.log 2>/dev/null | tail -1 | cut -c1-120)" 2>/dev/null)
if [[ -z "$out" ]]; then
  say "158 | KHONG SSH DUOC"
else
  g(){ sed -n "s/^$1=//p" <<<"$out" | head -1; }
  alive158=$(g alive); n158=$(g n); b158=$(g b); fin158=$(g fin); fin158="${fin158:-0}"
  used158=$(g used); orphan158=$(g orphan); orphan158="${orphan158:-0}"; last158=$(g last)
  say "158 | driver=$alive158 | o=${n158:-?}/60 base=${b158:-?}/3 | fin=$fin158 | vram_used=${used158:-?}MiB | train_procs=$orphan158 | $last158"
  if [[ "$alive158" == no && "$fin158" -eq 0 ]]; then
    pass=$(cat log/opt1_158_passes 2>/dev/null || echo 0)
    if (( pass >= MAXPASS )); then say "158 | driver chet, da phong lai $pass lan — DUNG, can nguoi xem"
    elif (( orphan158 > 0 )); then say "158 | driver chet nhung con $orphan158 job train — cho"
    else
      if [[ "$DRY" == 1 ]]; then say "158 | [DRY] se PHONG LAI lan $((pass+1))"
      else
        echo $((pass+1)) > log/opt1_158_passes
        timeout 60 ssh -o BatchMode=yes "$R158" "cd $ROOT158 && (FOLD_LIST='3 4 5' PYTHON=$PY158 setsid nohup bash run/opt1.sh >> log/opt1_158.log 2>&1 < /dev/null &)" 2>/dev/null
        say "158 | PHONG LAI lan $((pass+1))"
      fi
    fi
  fi
  # keo ket qua ve (chi them, khong xoa, khong ghi de)
  if [[ "${n158:-0}" -gt 0 || "${b158:-0}" -gt 0 ]]; then
    mkdir -p results_opt1_158
    if rsync -az --ignore-existing "$R158:$ROOT158/results/opt1_t5p/" results_opt1_158/ 2>/dev/null; then
      say "158 | rsync -> results_opt1_158/ : $(ls results_opt1_158/*/seed_42/*.json 2>/dev/null | wc -l) file"
    fi
  fi
fi
