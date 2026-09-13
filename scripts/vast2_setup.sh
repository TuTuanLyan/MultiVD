#!/usr/bin/env bash
# Cai dat vast 50570168 qua dia chi TRUC TIEP (`vastai ssh-url`), khong dung proxy.
# BAY: truong ssh_host/ssh_port trong API la dia chi PROXY va o may nay no TU CHOI o buoc
# bat tay. `vastai ssh-url <id>` tra dia chi TRUC TIEP va no vao duoc ngay.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ID=50570168; LOG=log/vast2_setup.log
ts(){ date -u '+%F %T'; }; say(){ echo "$(ts) | $*" | tee -a "$LOG"; }
LB=$(timeout 60 vastai show instance "$ID" --raw 2>/dev/null | python3 -c "import json,sys;print(json.load(sys.stdin).get('label'))")
[[ "$LB" == ntat ]] || { say "!! nhan '$LB' khong phai ntat — TU CHOI"; exit 3; }
URL=$(timeout 60 vastai ssh-url "$ID" 2>/dev/null | tr -d '\r')
HP=${URL#ssh://root@}; H=${HP%%:*}; P=${HP##*:}
[[ -n "$H" && -n "$P" ]] || { say "!! khong lay duoc ssh-url"; exit 4; }
say "dia chi truc tiep: $H:$P"
V="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=25 -p $P root@$H"
$V true 2>/dev/null || { say "!! ssh truc tiep khong vao duoc"; exit 5; }

say "cai goi con thieu (sklearn, scipy, transformers)"
$V '/venv/main/bin/pip install -q --no-input scikit-learn scipy transformers 2>&1 | tail -3' | sed 's/^/    /' | tee -a "$LOG"
for m in torch numpy sklearn scipy transformers; do
  v=$($V "/venv/main/bin/python -c 'import $m;print(getattr($m,\"__version__\",\"?\"))' 2>&1 | tail -1")
  say "  $m = $v"
  [[ "$v" == *Error* ]] && { say "!! thieu $m — DUNG"; exit 6; }
done

say "day ma va du lieu"
$V "mkdir -p /workspace/MultiVD/log"
for d in src run scripts tools data; do
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p $P" "$d/" "root@$H:/workspace/MultiVD/$d/" || { say "!! rsync $d hong"; exit 7; }
done
for bb in codebert t5p; do
  p="model/n48/phase1/${bb}__latent_bottleneck_4cwe_l0p05/seed_42/best.pt"
  [[ -f "$p" ]] || { say "  !! thieu $p o local"; continue; }
  $V "mkdir -p /workspace/MultiVD/$(dirname "$p")"
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p $P" "$p" "root@$H:/workspace/MultiVD/$p" && say "  Pha 1 cu $bb day xong"
done

say "doi chieu BYTE"
bad=0
for d in src data; do
  a=$(find "$d" -type f -printf '%P\t%s\n' | LC_ALL=C sort | md5sum | cut -d' ' -f1)
  b=$($V "cd /workspace/MultiVD && find $d -type f -printf '%P\t%s\n' | LC_ALL=C sort | md5sum | cut -d' ' -f1")
  [[ "$a" == "$b" ]] && say "  $d KHOP" || { say "  !! $d LECH"; bad=1; }
done
for bb in codebert t5p; do
  p="model/n48/phase1/${bb}__latent_bottleneck_4cwe_l0p05/seed_42/best.pt"
  [[ -f "$p" ]] || continue
  a=$(stat -c %s "$p"); b=$($V "stat -c %s /workspace/MultiVD/$p 2>/dev/null || echo 0")
  [[ "$a" == "$b" ]] && say "  Pha1 $bb KHOP $a B" || { say "  !! Pha1 $bb LECH $a vs $b"; bad=1; }
done
(( bad )) && { say "!! LECH — KHONG chay"; exit 8; }

say "phong khoi AUXB (co che head moi)"
$V "cd /workspace/MultiVD && PYTHON=/venv/main/bin/python setsid nohup bash scripts/vast_auxb_run.sh >> log/vast_auxb.log 2>&1 </dev/null & disown"
sleep 30
$V "cd /workspace/MultiVD && tail -10 log/vast_auxb.log" | sed 's/^/    /' | tee -a "$LOG"
say "XONG. ssh -p $P root@$H   |  het viec la HUY: vastai destroy instance $ID -y"
