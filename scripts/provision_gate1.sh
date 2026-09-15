#!/usr/bin/env bash
# Day code + du lieu + Pha 1 len may vast roi phong GATE1.
#   bash scripts/provision_gate1.sh [nhan] [danh sach fold]
#   vd: bash scripts/provision_gate1.sh ntat "1 2 3"
# Chay lai duoc nhieu lan: rsync dong bo, flock trong driver chan phong trung.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/home/ntat/.local/bin:$PATH"
source scripts/endpoints.sh
LABEL="${1:-ntat}"; FOLDS_ARG="${2:-1 2 3}"
R=/workspace/MultiVD

read -r H P <<< "$(vast_endpoint "$LABEL" 2>/dev/null)"
[[ -n "${H:-}" && "${P:-None}" != "None" ]] || { echo "!! may nhan '$LABEL' chua san sang"; exit 1; }
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -p $P"
echo "== $LABEL $H:$P | fold: $FOLDS_ARG =="

$SSH root@"$H" "mkdir -p $R/{data,log,results} $R/model/shuf1/phase1" || exit 1

PY=$($SSH root@"$H" 'for c in /venv/main/bin/python /opt/conda/bin/python python3; do
  $c -c "import torch" 2>/dev/null && { echo $c; break; }; done' | tail -1)
[[ -n "$PY" ]] || { echo "!! khong tim thay python co torch"; exit 1; }
echo "python: $PY"
$SSH root@"$H" "$PY -c 'import torch,transformers,sklearn;print(\"torch\",torch.__version__,\"| tf\",transformers.__version__,\"| sk\",sklearn.__version__);import torch as t;print(\"cuda\",t.cuda.is_available(), t.cuda.get_device_name(0) if t.cuda.is_available() else \"\")'" || exit 1

for D in src run scripts tools; do rsync -az --delete -e "$SSH" "$D/" "root@$H:$R/$D/" || exit 1; done
rsync -az -e "$SSH" data/phase1_common.jsonl "root@$H:$R/data/" || exit 1
rsync -az -e "$SSH" data/sven_python_folds_norm/ "root@$H:$R/data/sven_python_folds_norm/" || exit 1

# Pha 1 — hai checkpoint ~940MB. Day TRUOC khi phong; driver co cong kiem ton tai nhung
# cong do chi doc duoc "co file", con "file DU BYTE" phai doi chieu o day.
echo "-- day Pha 1 (co the vai phut) --"
for L in codebert t5p; do
  rsync -az --info=progress2 -e "$SSH" "model/shuf1/phase1/${L}__none_com_real/" \
    "root@$H:$R/model/shuf1/phase1/${L}__none_com_real/" || exit 1
done

echo "-- doi chieu TUNG BYTE --"
{ find data/sven_python_folds_norm -type f -printf '%p %s\n'
  stat -c '%n %s' data/phase1_common.jsonl
  find model/shuf1/phase1 -name 'best.pt' -path '*_com_real*' -printf '%p %s\n'
} | LC_ALL=C sort > /tmp/gate1_local.txt
$SSH root@"$H" "cd $R && { find data/sven_python_folds_norm -type f -printf '%p %s\n'
  stat -c '%n %s' data/phase1_common.jsonl
  find model/shuf1/phase1 -name 'best.pt' -path '*_com_real*' -printf '%p %s\n'; } | LC_ALL=C sort" > /tmp/gate1_remote.txt
if LC_ALL=C comm -23 /tmp/gate1_local.txt /tmp/gate1_remote.txt | grep -q .; then
  echo "!! LECH:"; LC_ALL=C comm -23 /tmp/gate1_local.txt /tmp/gate1_remote.txt | head; exit 1
fi
N=$(wc -l < /tmp/gate1_local.txt)
(( N >= 20 )) || { echo "!! manifest chi co $N dong — nghi ngo find rong, KHONG phong"; exit 1; }
echo "   khop $N file"

echo "-- phong driver --"
$SSH root@"$H" "cd $R && FOLDS='$FOLDS_ARG' PYTHON=$PY setsid nohup bash run/gate1.sh >> log/gate1.log 2>&1 </dev/null & disown; sleep 5; tail -12 log/gate1.log"
echo "== da phong. Theo doi: bash scripts/watch_gate1.sh $LABEL =="
