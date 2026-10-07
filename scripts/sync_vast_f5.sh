#!/usr/bin/env bash
# Cho vast xong fold 5 roi DONG BO NGAY ve 161, de driver o 161 thay file va BO QUA fold 5.
# Neu khong: hai may chay trung mot o, ton GPU va sinh hai ban khac nhau cua cung mot o.
set -u
cd /drive1/cuongtm/ntat/MultiVD || exit 1
S="ssh -p 56599 -o StrictHostKeyChecking=no -o ConnectTimeout=25 -o BatchMode=yes -i /home/ntat/.ssh/id_ed25519"
for i in $(seq 1 90); do            # toi da 90 x 40s = 60 phut
  N=$($S root@115.73.216.179 'find /workspace/MultiVD/results/mw_assemble_babelmw -name "fold5.json" 2>/dev/null | wc -l' 2>/dev/null)
  if [ "${N:-0}" -ge 2 ]; then
    rsync -az -e "$S" root@115.73.216.179:/workspace/MultiVD/results/mw_assemble_babelmw/ results/mw_assemble_babelmw/ 2>/dev/null
    echo "$(date -u '+%F %T') | fold 5 tu vast da dong bo ve 161 ($N o)"
    exit 0
  fi
  sleep 40
done
echo "$(date -u '+%F %T') | het 60 phut ma vast chua xong fold 5 — KHONG dong bo"
exit 1
