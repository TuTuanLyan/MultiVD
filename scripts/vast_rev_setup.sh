#!/usr/bin/env bash
# Cai dat vast 50569747 (nhan `ntat`) va giao TOAN BO phan DAO CHIEU (Python -> JS).
# Khong trung voi local: local dang chay seed7 roi se lam co che head moi.
#
#   bash scripts/vast_rev_setup.sh [ID]
#
# Viec giao cho vast (24 o):
#   A. rev1    dich js_4cwe_folds  fold 2,3   (fold 1 da chay xong o local)  -> 12 o
#   B. rev1com dich js_com_folds   fold 1                                    ->  6 o
#   C. rev1full dich js_full_folds fold 1                                    ->  6 o
# Moi buoc 3 nhanh: baseline + plain(AdamW) + RecAdam&ASAM o rho tot nhat cua backbone.
# Pha 1 nguon PYTHON day san len, KHONG huan luyen lai.
#
# LUAT VAST (VAST_RULES.md): chi dung nhan `ntat`. Het viec la HUY NGAY, khong stop.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ID="${1:-50569747}"
LOG=log/vast_rev_setup.log
ts(){ date -u '+%F %T'; }
say(){ echo "$(ts) | $*" | tee -a "$LOG"; }

# --- cong 1: dung nhan ---
LB=$(timeout 60 vastai show instance "$ID" --raw 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('label'))" 2>/dev/null)
if [[ "$LB" != "ntat" ]]; then
  say "!! instance $ID co nhan '$LB', KHONG phai 'ntat' — TU CHOI dung (VAST_RULES.md)"; exit 3
fi
say "instance $ID nhan 'ntat' — duoc phep dung"

# --- cho may len va ssh vao duoc (toi da 20 phut) ---
HOST=""; PORT=""
for i in $(seq 1 80); do
  st=$(timeout 60 vastai show instance "$ID" --raw 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('actual_status'),d.get('ssh_host'),d.get('ssh_port'))" 2>/dev/null)
  read -r S H P <<<"$st"
  if [[ "$S" == running && -n "${H:-}" && -n "${P:-}" ]]; then
    if timeout 30 ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -p "$P" "root@$H" true 2>/dev/null; then
      HOST="$H"; PORT="$P"; say "ssh vao duoc sau $((i*15))s: $H:$P"; break
    fi
  fi
  (( i % 8 == 0 )) && say "  [$i] trang thai=$S ssh chua vao duoc"
  sleep 15
done
[[ -n "$HOST" ]] || { say "!! 20 phut khong ssh duoc — DUNG, kiem tay"; exit 5; }
V="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 -p $PORT root@$HOST"
R="rsync -az -e 'ssh -o StrictHostKeyChecking=no -p $PORT'"

$V "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nproc" | sed 's/^/    /' | tee -a "$LOG"

# --- day ma + du lieu ---
say "day ma va du lieu"
$V "mkdir -p /workspace/MultiVD/log" || { say "!! ssh hong"; exit 6; }
for d in src run scripts tools data; do
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" "$d/" "root@$HOST:/workspace/MultiVD/$d/" \
    || { say "!! rsync $d that bai"; exit 7; }
done

# --- day Pha 1 nguon PYTHON cho CA HAI backbone ---
for bb in codebert t5p; do
  p="model/rev1/phase1/${bb}__latent_bottleneck_py_l0p05/seed_42/best.pt"
  if [[ -f "$p" ]]; then
    $V "mkdir -p /workspace/MultiVD/$(dirname "$p")"
    rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" "$p" "root@$HOST:/workspace/MultiVD/$p" \
      && say "  Pha 1 $bb day xong ($(stat -c %s "$p") B)"
  else
    say "  !! THIEU Pha 1 $bb ($p) — vast se phai tu huan luyen"
  fi
done

# --- doi chieu BYTE, khong chi dem file (CLAUDE.md muc 10) ---
say "doi chieu byte"
bad=0
for d in src data; do
  a=$(find "$d" -type f -printf '%P\t%s\n' | LC_ALL=C sort | md5sum | cut -d' ' -f1)
  b=$($V "cd /workspace/MultiVD && find $d -type f -printf '%P\t%s\n' | LC_ALL=C sort | md5sum | cut -d' ' -f1")
  if [[ "$a" == "$b" ]]; then say "  $d KHOP"; else say "  !! $d LECH ($a vs $b)"; bad=1; fi
done
for bb in codebert t5p; do
  p="model/rev1/phase1/${bb}__latent_bottleneck_py_l0p05/seed_42/best.pt"
  [[ -f "$p" ]] || continue
  a=$(stat -c %s "$p"); b=$($V "stat -c %s /workspace/MultiVD/$p 2>/dev/null || echo 0")
  if [[ "$a" == "$b" ]]; then say "  Pha 1 $bb KHOP $a B"; else say "  !! Pha 1 $bb LECH $a vs $b"; bad=1; fi
done
(( bad )) && { say "!! du lieu lech — KHONG chay. Kiem tay roi chay lai."; exit 8; }

# --- kiem moi truong python TRUOC khi phong (bay §40.1) ---
PY=$($V "for c in /venv/main/bin/python /usr/bin/python3; do [ -x \$c ] && \$c -c 'import torch,numpy,sklearn' 2>/dev/null && { echo \$c; break; }; done")
[[ -n "$PY" ]] || { say "!! tren vast khong python nao import duoc torch/numpy/sklearn — DUNG"; exit 9; }
say "python tren vast: $PY"

# --- phong ---
say "phong khoi DAO CHIEU tren vast (24 o)"
$V "cd /workspace/MultiVD && PYTHON=$PY setsid nohup bash scripts/vast_rev_run.sh >> log/vast_rev.log 2>&1 </dev/null & disown"
sleep 25
$V "cd /workspace/MultiVD && tail -8 log/vast_rev.log" | sed 's/^/    /' | tee -a "$LOG"
say "XONG cai dat. ID=$ID  ssh: -p $PORT root@$HOST"
say "NHO: het viec la HUY NGAY — vastai destroy instance $ID -y — roi xac nhan bang vastai show instances."
