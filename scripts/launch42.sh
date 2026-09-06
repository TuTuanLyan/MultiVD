#!/usr/bin/env bash
# Phong (hoac phong lai) mot may MOT CACH AN TOAN.
#   bash scripts/launch42.sh <local|ntat|ntat2> <backbone-spec> [--restart]
#
# Trinh tu bat buoc, moi buoc deu XAC NHAN truoc khi sang buoc sau:
#   1. neu da co driver va khong co --restart  -> khong lam gi (khoa se tu chan)
#   2. --restart: dung -> xac nhan da dung -> thu hoi VRAM mo coi -> xac nhan GPU rong
#   3. phong duoi flock -> xac nhan dung MOT driver
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET="${1:?local|ntat|ntat2}"; BB="${2:?backbone spec}"; MODE="${3:-}"
# rho cua SAM o Phase 1. Phai dat duoc THEO MAY vi rho khong chuyen duoc
# giua cac backbone (27/08: rho=0.05 giet codebert, vo hai voi t5p/unixcoder).
RHO="${SAM_RHO:-0.05}"

REMOTE_BODY='set -uo pipefail; cd /workspace/MultiVD; source scripts/proc.sh
if [ "__MODE__" = "--restart" ]; then
  mvd_stop || { echo "KHONG dung duoc het tien trinh — DUNG LAI, khong phong lai"; exit 1; }
  echo -n "VRAM sau khi thu hoi: "; mvd_reap_gpu
fi
n=$(mvd_count_drivers)
if [ "$n" -gt 0 ]; then echo "da co $n driver, khong phong them"; exit 0; fi
SAM_RHO=__RHO__ PYTHON=__PY__ setsid bash run/day42_machine.sh "__BB__" </dev/null >/dev/null 2>&1 &
sleep 6
echo -n "driver sau khi phong: "; mvd_count_drivers'

if [[ "$TARGET" == "local" ]]; then
  BODY="${REMOTE_BODY//__MODE__/$MODE}"; BODY="${BODY//__BB__/$BB}"
  BODY="${BODY//__PY__//home/ntat/miniconda3/envs/vdenv/bin/python}"; BODY="${BODY//__RHO__/$RHO}"
  BODY="${BODY//\/workspace\/MultiVD/$(pwd)}"
  BODY="${BODY//setsid bash/setsid nice -n 10 bash}"
  bash -c "$BODY"
else
  source scripts/endpoints.sh
  read -r H P <<< "$(vast_endpoint "$TARGET")"
  BODY="${REMOTE_BODY//__MODE__/$MODE}"; BODY="${BODY//__BB__/$BB}"; BODY="${BODY//__PY__/python3}"; BODY="${BODY//__RHO__/$RHO}"
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p "$P" root@"$H" "$BODY" \
    2>&1 | grep -v "Welcome\|AI agents\|Have fun\|Warning"
fi
