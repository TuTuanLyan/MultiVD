#!/usr/bin/env bash
# Thue DUNG MOT vast A4000 va cai dat de chay mot lat cua khoi n=15.
# CHI chay khi mot may local VO (xem log/chot15_state). Nguoi dung 10/09:
#   "Bi chiem thi van thue vast ngay ... Toi da co the co 3 GPU chay cung luc"
#   "neu vo ca 2 server rieng thi van chi duoc thue 1 va chi 1 vast"
#
#   bash scripts/rent_one_vast.sh <offer_id> <codebert|t5p> <seed>
#
# Chia viec de KHONG BAO GIO trung voi may local dang hoi phuc: vast nhan DUNG MOT SEED,
# may local giu seed con lai. Cay ket qua rieng (`chotv15_<bb>`) nen khong the ghep cap nham.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OFFER="${1:?can offer id — lay bang: vastai search offers 'gpu_name=RTX_A4000 num_gpus=1 rentable=true' -o dph+}"
WHICH="${2:?can codebert hoac t5p}"
SEED="${3:?can seed, vd 1234}"
IMG="${IMG:-vastai/pytorch:2.9.1-cuda-12.8.1-py310-24.04-2026-08-21}"
LOG=log/rent_vast.log
ts(){ date -u '+%F %T'; }
say(){ echo "$(ts) | $*" | tee -a "$LOG"; }

# --- CHAN: khong bao gio thue cai thu hai ---
source scripts/endpoints.sh
cur=$(vast_labels)
if [[ -n "$cur" ]]; then
  say "!! DA CO vast dang chay ($cur) — luat cho phep DUNG MOT. Tu choi thue them."; exit 3
fi
say "thue offer $OFFER cho $WHICH seed $SEED"
out=$(vastai create instance "$OFFER" --image "$IMG" --disk 60 --label ntat --ssh --direct 2>&1)
say "  $out"
ID=$(grep -oE "'new_contract': [0-9]+" <<<"$out" | grep -oE '[0-9]+')
[[ -n "$ID" ]] || { say "!! khong doc duoc id instance — KIEM TAY roi chay tiep"; exit 4; }
say "  instance id=$ID"

say "cho may len (toi da 15 phut)"
for i in $(seq 1 90); do
  sleep 10
  st=$(vastai show instance "$ID" --raw 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);print(d.get('actual_status'),d.get('ssh_host'),d.get('ssh_port'))" 2>/dev/null)
  say "  [$i] $st"
  [[ "$st" == running* ]] && break
done
read -r _ HOST PORT <<<"$st"
[[ -n "${HOST:-}" && -n "${PORT:-}" ]] || { say "!! chua co ssh_host/port — KIEM TAY"; exit 5; }
V="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -p $PORT root@$HOST"

say "day code + du lieu + Pha 1"
$V "mkdir -p /workspace/MultiVD" || { say "!! ssh khong vao duoc"; exit 6; }
for d in src run scripts tools data; do
  rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" $d/ root@$HOST:/workspace/MultiVD/$d/ \
    || { say "!! rsync $d that bai"; exit 7; }
done
BB_FULL=$([[ $WHICH == codebert ]] && echo 'codebert=microsoft/codebert-base:cls' || echo 't5p=Salesforce/codet5p-220m-bimodal:mean')
A=$([[ $WHICH == codebert ]] && echo 'r0p1|recadam|--sam_rho 0.1 --sam_variant asam' || echo 'r2p0|recadam|--sam_rho 2.0 --sam_variant asam')
for s in 4cwe com full; do
  p="model/n48/phase1/${WHICH}__latent_bottleneck_${s}_l0p05/seed_${SEED}/best.pt"
  if [[ -f "$p" ]]; then
    $V "mkdir -p /workspace/MultiVD/$(dirname $p)"
    rsync -az -e "ssh -o StrictHostKeyChecking=no -p $PORT" "$p" "root@$HOST:/workspace/MultiVD/$p"
    say "  day Pha 1 $s (da co san, khoi train lai)"
  else
    say "  Pha 1 $s CHUA CO — vast se tu huan luyen"
  fi
done

say "doi chieu byte truoc khi chay"
bad=0
for d in src data; do
  a=$(find $d -type f -printf '%P\t%s\n' | LC_ALL=C sort | md5sum | cut -d' ' -f1)
  b=$($V "cd /workspace/MultiVD && find $d -type f -printf '%P\t%s\n' | LC_ALL=C sort | md5sum | cut -d' ' -f1")
  [[ "$a" == "$b" ]] && say "  $d KHOP" || { say "  !! $d LECH"; bad=1; }
done
(( bad )) && { say "!! du lieu lech — KHONG chay, kiem tay"; exit 8; }

say "phong khoi tren vast: $WHICH seed $SEED -> results/chotv15_$WHICH"
$V "cd /workspace/MultiVD && RUN=chotv15 BB='$BB_FULL' CFG_A='$A' CFG_B='plain|adamw|--sam_rho 0' \
  SEED_LIST='$SEED' PYTHON=/venv/main/bin/python \
  setsid nohup bash run/chot15.sh >> log/chot15_vast.log 2>&1 </dev/null & disown"
sleep 20
$V "cd /workspace/MultiVD && tail -3 log/chot15_vast.log" | tee -a "$LOG"
say "XONG buoc thue+cai. NHO: canh may nay, het viec la HUY ngay (dung bo phi = bi phat)."
say "  huy: vastai destroy instance $ID -y  (roi kiem lai bang vastai show instances)"
