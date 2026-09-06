#!/usr/bin/env bash
# Day code + du lieu + CHECKPOINT PHA 1 SEED 7 len may vast `ntat`, roi chay fold 3,4,5.
#
# Vi sao phai day checkpoint len chu khong huan luyen Pha 1 moi: fold 1-2 cua seed 7 da
# chay tren 161 tu chinh ba checkpoint nay. Neu may moi huan luyen Pha 1 khac thi fold
# 3-5 xuat phat tu nguon khac fold 1-2, va seed 7 tu mau thuan voi chinh no.
# night48.sh thay checkpoint da co thi bo qua Pha 1 — dung cai ta can.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/home/ntat/.local/bin:$PATH"
source scripts/endpoints.sh
R=/workspace/MultiVD

read -r H P <<< "$(vast_endpoint ntat 2>/dev/null)"
[[ -n "${H:-}" && "${P:-None}" != "None" ]] || { echo "may chua san sang"; exit 1; }
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $P"
echo "== ntat $H:$P =="

$SSH root@"$H" "mkdir -p $R/{data,log,results} $R/model/n48/phase1" || exit 1

PY=$($SSH root@"$H" 'for c in /venv/main/bin/python /opt/conda/bin/python python3; do
  $c -c "import torch" 2>/dev/null && { echo $c; break; }; done' | tail -1)
[[ -n "$PY" ]] || { echo "!! khong tim thay python co torch"; exit 1; }
echo "python: $PY"
# Ghim transformers 4.57.1 nhu moi may khac trong du an (requirements-pin.txt)
$SSH root@"$H" "$PY -c 'import transformers' 2>/dev/null || $PY -m pip install -q --no-input transformers==4.57.1 scikit-learn scipy evaluate"
$SSH root@"$H" "$PY -c 'import torch,transformers,sklearn;print(\"torch\",torch.__version__,\"| transformers\",transformers.__version__,\"| sklearn\",sklearn.__version__)'"

for D in src run scripts tools; do rsync -az --delete -e "$SSH" "$D/" "root@$H:$R/$D/" || exit 1; done
rsync -az -e "$SSH" data/phase1_4cwe.jsonl data/phase1_common.jsonl data/phase1_full.jsonl "root@$H:$R/data/" || exit 1
rsync -az -e "$SSH" data/sven_python_folds_norm/ "root@$H:$R/data/sven_python_folds_norm/" || exit 1
echo "-- day 3 checkpoint Pha 1 seed 7 (1,25 GB) --"
rsync -a --info=progress2 -e "$SSH" --include='*/' --include='seed_7/**' --exclude='*' \
  model/n48/phase1/ "root@$H:$R/model/n48/phase1/" || exit 1

echo "-- doi chieu tung byte: du lieu VA checkpoint --"
( for f in data/phase1_4cwe.jsonl data/phase1_common.jsonl data/phase1_full.jsonl run/night48.sh run/matrix.sh; do stat -c '%n %s' "$f"; done
  find data/sven_python_folds_norm -type f -printf '%p %s\n'
  find model/n48/phase1 -path '*seed_7*' -name best.pt -printf '%p %s\n' ) | LC_ALL=C sort > /tmp/n48b_local.txt
$SSH root@"$H" "cd $R && { for f in data/phase1_4cwe.jsonl data/phase1_common.jsonl data/phase1_full.jsonl run/night48.sh run/matrix.sh; do stat -c '%n %s' \$f; done; find data/sven_python_folds_norm -type f -printf '%p %s\n'; find model/n48/phase1 -path '*seed_7*' -name best.pt -printf '%p %s\n'; } | LC_ALL=C sort" > /tmp/n48b_remote.txt
if LC_ALL=C comm -23 /tmp/n48b_local.txt /tmp/n48b_remote.txt | grep -q .; then
  echo "!! LECH:"; LC_ALL=C comm -23 /tmp/n48b_local.txt /tmp/n48b_remote.txt | head; exit 1
fi
echo "   khop $(wc -l < /tmp/n48b_local.txt) file"

echo "-- doc thu checkpoint tren may (kich thuoc khop chua du) --"
$SSH root@"$H" "cd $R && for f in model/n48/phase1/*/seed_7/best.pt; do $PY -c \"
import torch,sys,os
c=torch.load('\$f',map_location='cpu',weights_only=False)
print('   OK %-50s val=%.4f ep=%s'%('/'.join('\$f'.split('/')[-3:]),float(c.get('best_val_macro_f1') or 0),c.get('best_epoch')))\"; done"

echo "-- phong driver: seed 7, fold 3 4 5 --"
$SSH root@"$H" "cd $R && SEEDS='7' FOLD_LIST='3 4 5' PYTHON=$PY setsid nohup bash run/night48.sh >> log/night48.log 2>&1 </dev/null & disown; sleep 3; tail -4 log/night48.log"
