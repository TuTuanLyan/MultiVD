#!/usr/bin/env bash
# VA cho vast2_pushmodels.sh: chi day NOT phan con thieu cua cache HF roi phong lai OFFLINE.
# Khac ban goc o DUNG MOT cho: --partial --append-verify => duong truyen dut giua chung thi
# lan sau NOI TIEP, khong nem bo 800 MB da gui. (vast2_pushmodels dung rsync tran nen moi
# lan dut la mat sach phan da gui — da thay that 2 lan.)
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H=115.73.216.179; P=53817; LOG=log/vast2_fixpush.log
V="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=25 -p $P root@$H"
ts(){ date -u '+%F %T'; }; say(){ echo "$(ts) | $*" | tee -a "$LOG"; }

say "bat dau va push"
for m in models--microsoft--codebert-base models--Salesforce--codet5p-220m-bimodal; do
  src="$HOME/.cache/huggingface/hub/$m"
  for try in 1 2 3 4 5 6; do
    a=$(find "$src" -type f -printf '%s\n' | LC_ALL=C sort -n | md5sum | cut -d' ' -f1)
    b=$($V "find /workspace/.hf_home/hub/$m -type f -printf '%s\n' 2>/dev/null | LC_ALL=C sort -n | md5sum | cut -d' ' -f1" 2>/dev/null)
    [[ "$a" == "$b" ]] && { say "  $m KHOP (sau $((try-1)) lan)"; break; }
    say "  $m lan $try: day tiep"
    rsync -a --partial --append-verify --info=progress2 \
      -e "ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -p $P" \
      "$src" "root@$H:/workspace/.hf_home/hub/" >/dev/null 2>&1
  done
done

say "doi chieu BYTE lan cuoi (CA HAI CHIEU: so file VA tap kich thuoc)"
bad=0
for m in models--microsoft--codebert-base models--Salesforce--codet5p-220m-bimodal; do
  a=$(find "$HOME/.cache/huggingface/hub/$m" -type f -printf '%s\n' | LC_ALL=C sort -n | md5sum | cut -d' ' -f1)
  na=$(find "$HOME/.cache/huggingface/hub/$m" -type f | wc -l)
  read -r b nb <<<"$($V "cd /workspace/.hf_home/hub && echo \$(find $m -type f -printf '%s\n' 2>/dev/null | LC_ALL=C sort -n | md5sum | cut -d' ' -f1) \$(find $m -type f 2>/dev/null | wc -l)" 2>/dev/null)"
  if [[ "$a" == "$b" && "$na" == "$nb" ]]; then say "  $m KHOP ($na file)"; else say "  !! $m LECH local=$na/$a vast=$nb/$b"; bad=1; fi
done
(( bad )) && { say "!! cache VAN lech — KHONG phong. Can xem tay."; exit 1; }

say "don tien trinh cu (VO truoc, LA sau) roi phong OFFLINE"
$V 'for t in "bash scripts/vast_auxb_run.sh" "bash run/matrix.sh" "src/train_transfer.py" "src/train_baseline.py"; do
      for p in $(pgrep -f "$t" || true); do kill $p 2>/dev/null || true; done; sleep 1; done' 2>/dev/null
sleep 3
$V "cd /workspace/MultiVD && HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/workspace/.hf_home PYTHON=/venv/main/bin/python \
    setsid nohup bash scripts/vast_auxb_run.sh >> log/vast_auxb.log 2>&1 </dev/null & disown" 2>/dev/null
sleep 60
$V 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader; cd /workspace/MultiVD && tail -5 log/vast_auxb.log' 2>&1 \
  | grep -v "Welcome\|Have fun\|AI agents" | sed 's/^/    /' | tee -a "$LOG"
say "XONG va push"
