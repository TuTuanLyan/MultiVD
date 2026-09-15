#!/usr/bin/env bash
# Day code + Pha 1 sang server 158 roi phong GATE1 o do.
#   bash scripts/provision_gate1_158.sh "1 2 3"
# 158: chi dung /data/ntat/ (SERVER.md). KHONG dung toi thu muc nguoi khac,
# KHONG kill tien trinh nao — card dang co job cua tranmanhcuong, ta chay CANH no.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FOLDS_ARG="${1:-1 2 3}"
H=tranmanhcuong@112.137.129.158
R=/data/ntat/MultiVD
PY=/data/ntat/envs/vdenv/bin/python
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25"

echo "== 158 | fold: $FOLDS_ARG =="
$SSH $H "$PY -c 'import torch,transformers,sklearn;print(\"torch\",torch.__version__,\"| tf\",transformers.__version__,\"| sk\",sklearn.__version__);import torch as t;print(\"cuda\",t.cuda.is_available(),t.cuda.get_device_name(0))'" || exit 1

# Bo nho TRONG phai du truoc khi phong. Card dang chay job nguoi khac; neu ta phong vao
# luc chi con vai GB thi ca HAI job cung OOM — hong viec cua minh va cua ho.
FREE=$($SSH $H "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits" | tr -d ' ')
echo "GPU trong: ${FREE} MiB"
(( FREE >= 9000 )) || { echo "!! chi con ${FREE} MiB (<9000) — KHONG phong, de khong lam hong job cua tranmanhcuong"; exit 1; }

for D in src run scripts tools; do rsync -az --delete -e "$SSH" "$D/" "$H:$R/$D/" || exit 1; done
rsync -az -e "$SSH" data/phase1_common.jsonl "$H:$R/data/" || exit 1
rsync -az -e "$SSH" data/sven_python_folds_norm/ "$H:$R/data/sven_python_folds_norm/" || exit 1
echo "-- day Pha 1 (~940 MB) --"
$SSH $H "mkdir -p $R/model/shuf1/phase1"
for L in codebert t5p; do
  rsync -az --info=progress2 -e "$SSH" "model/shuf1/phase1/${L}__none_com_real/" \
    "$H:$R/model/shuf1/phase1/${L}__none_com_real/" || exit 1
done

echo "-- doi chieu TUNG BYTE --"
{ find data/sven_python_folds_norm -type f -printf '%p %s\n'
  stat -c '%n %s' data/phase1_common.jsonl
  find model/shuf1/phase1 -name 'best.pt' -path '*_com_real*' -printf '%p %s\n'
} | LC_ALL=C sort > /tmp/g1_158_local.txt
$SSH $H "cd $R && { find data/sven_python_folds_norm -type f -printf '%p %s\n'
  stat -c '%n %s' data/phase1_common.jsonl
  find model/shuf1/phase1 -name 'best.pt' -path '*_com_real*' -printf '%p %s\n'; } | LC_ALL=C sort" > /tmp/g1_158_remote.txt
if LC_ALL=C comm -23 /tmp/g1_158_local.txt /tmp/g1_158_remote.txt | grep -q .; then
  echo "!! LECH:"; LC_ALL=C comm -23 /tmp/g1_158_local.txt /tmp/g1_158_remote.txt | head; exit 1
fi
N=$(wc -l < /tmp/g1_158_local.txt); (( N >= 20 )) || { echo "!! manifest chi $N dong — KHONG phong"; exit 1; }
echo "   khop $N file"

$SSH $H "cd $R && mkdir -p log && FOLDS='$FOLDS_ARG' PYTHON=$PY PIDFILE=/data/ntat/gate1.pid \
  setsid nohup bash run/gate1.sh >> log/gate1.log 2>&1 </dev/null & disown; sleep 5; tail -12 log/gate1.log"
echo "== da phong tren 158. Theo doi: bash scripts/watch_gate1_158.sh =="
