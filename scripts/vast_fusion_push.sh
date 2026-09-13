#!/usr/bin/env bash
# Day ma + du lieu + cache HF len vast `ntat` cho khoi ADAPTER FUSION.
# Endpoint = public_ipaddr + HostPort cua 22/tcp (KHONG phai ssh_host/ssh_port) —
# nham hai cai nay la "connection refused", da mat thoi gian vi no.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
H="${H:-115.73.216.179}"; P="${P:-56162}"
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 -p $P root@$H"
# --partial --append-verify: duong truyen toi vast dut giua chung la chuyen thuong,
# rsync tran vut bo toan bo phan da gui (da mat 892 MB hai lan).
RS="rsync -az --partial --append-verify -e 'ssh -o StrictHostKeyChecking=no -p $P'"
ts(){ date -u '+%F %T'; }
echo "$(ts) | bat dau day len $H:$P"
$SSH "mkdir -p /workspace/MultiVD/{log,model,results} /opt/hf-cache/hub" || exit 2
for d in src run tools scripts data; do
  echo "$(ts) | rsync $d"
  eval $RS "$d/" "root@$H:/workspace/MultiVD/$d/" || { echo "!! rsync $d that bai"; exit 3; }
done
for f in requirements-pin.txt CLAUDE.md VAST_RULES.md; do
  [ -f "$f" ] && eval $RS "$f" "root@$H:/workspace/MultiVD/$f"
done
for m in models--microsoft--codebert-base models--Salesforce--codet5p-220m-bimodal; do
  echo "$(ts) | rsync cache $m"
  eval $RS "$HOME/.cache/huggingface/hub/$m/" "root@$H:/opt/hf-cache/hub/$m/" \
    || { echo "!! rsync cache $m that bai"; exit 4; }
done
echo "$(ts) | XONG day len"
