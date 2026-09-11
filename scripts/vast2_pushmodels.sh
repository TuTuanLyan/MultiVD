#!/usr/bin/env bash
# HF Hub tren vast tai DUNG (blob .incomplete = 0 byte sau 3 phut). Day cache tu local len.
# Image dat HF_HOME=/workspace/.hf_home, KHONG phai ~/.cache/huggingface.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H=115.73.216.179; P=53817; LOG=log/vast2_push.log
V="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=25 -p $P root@$H"
ts(){ date -u '+%F %T'; }; say(){ echo "$(ts) | $*" | tee -a "$LOG"; }

say "dung job dang treo o buoc tai model"
for pid in $($V 'pgrep -f "src/train_transfer.py" || true' 2>/dev/null); do
  $V "kill $pid" 2>/dev/null && say "  da giet PID $pid tren vast"
done
$V 'pkill -f "bash scripts/vast_auxb_run.sh" || true; pkill -f "bash run/matrix.sh" || true' 2>/dev/null
sleep 3
$V 'rm -rf /workspace/.hf_home/hub/.locks /workspace/.hf_home/hub/models--microsoft--codebert-base' 2>/dev/null

say "day cache HF tu local (~1,8 GB)"
$V "mkdir -p /workspace/.hf_home/hub" 2>/dev/null
for m in models--microsoft--codebert-base models--Salesforce--codet5p-220m-bimodal; do
  src="$HOME/.cache/huggingface/hub/$m"
  [[ -d "$src" ]] || { say "  !! khong co $src o local"; continue; }
  for try in 1 2 3; do
    if rsync -a --info=stats1 -e "ssh -o StrictHostKeyChecking=no -p $P" "$src" "root@$H:/workspace/.hf_home/hub/" 2>&1 | grep -E "sent|total size" | sed 's/^/    /' | tee -a "$LOG"; then
      say "  $m day xong (lan $try)"; break
    fi
    say "  $m lan $try that bai, thu lai"; sleep 5
  done
done

say "doi chieu BYTE cache model"
bad=0
for m in models--microsoft--codebert-base models--Salesforce--codet5p-220m-bimodal; do
  a=$(find "$HOME/.cache/huggingface/hub/$m" -type f -printf '%s\n' 2>/dev/null | LC_ALL=C sort -n | md5sum | cut -d' ' -f1)
  b=$($V "find /workspace/.hf_home/hub/$m -type f -printf '%s\n' 2>/dev/null | LC_ALL=C sort -n | md5sum | cut -d' ' -f1" 2>/dev/null)
  [[ "$a" == "$b" ]] && say "  $m KHOP" || { say "  !! $m LECH"; bad=1; }
done
(( bad )) && { say "!! cache lech — KHONG phong lai"; exit 1; }

say "phong lai AUXB, bat che do OFFLINE de khoi cham vao HF Hub"
$V "cd /workspace/MultiVD && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHON=/venv/main/bin/python \
    setsid nohup bash scripts/vast_auxb_run.sh >> log/vast_auxb.log 2>&1 </dev/null & disown" 2>/dev/null
sleep 40
$V 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; cd /workspace/MultiVD && tail -4 log/auxb/phase1_codebert*.log 2>/dev/null | cut -c1-110' 2>&1 | grep -v "Welcome\|Have fun\|AI agents" | sed 's/^/    /' | tee -a "$LOG"
say "XONG"
