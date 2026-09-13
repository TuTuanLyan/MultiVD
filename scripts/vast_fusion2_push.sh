#!/usr/bin/env bash
# Day len vast `ntat` 50882617 cho khoi DOI CHUNG NGAU NHIEN (fus2).
# CHI codebert: t5p khong co hieu ung de giai thich (§47), khong day cache cua no cho nhe dia.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="${H:-115.73.216.179}"; P="${P:-56143}"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 -p $P root@$H"
RS="rsync -az --partial --append-verify -e 'ssh -o StrictHostKeyChecking=no -p $P'"
ts(){ date -u '+%F %T'; }
$SSH "mkdir -p /workspace/MultiVD/{log,model,results} /workspace/.hf_home/hub" || exit 2
for d in src run tools scripts data tests; do
  echo "$(ts) | rsync $d"; eval $RS "$d/" "root@$H:/workspace/MultiVD/$d/" || exit 3
done
eval $RS requirements-pin.txt "root@$H:/workspace/MultiVD/"
for m in models--microsoft--codebert-base models--hf-internal-testing--tiny-random-roberta \
         models--hf-internal-testing--tiny-random-t5; do
  echo "$(ts) | cache $m"
  eval $RS "$HOME/.cache/huggingface/hub/$m/" "root@$H:/workspace/.hf_home/hub/$m/" || exit 4
done
# Pha 1 KHONG adapter (cho nhanh doi chung) — dung lai, khong huan luyen lai.
ck=model/n48/phase1/codebert__latent_bottleneck_com_l0p05/seed_42/best.pt
$SSH "mkdir -p /workspace/MultiVD/$(dirname $ck)"
echo "$(ts) | Pha 1 khong-adapter ($(stat -c %s $ck) B)"
eval $RS "$ck" "root@$H:/workspace/MultiVD/$ck" || exit 5
echo "$(ts) | XONG"
