#!/usr/bin/env bash
# Day code + du lieu len may vast `ntat` roi phong NIGHT48.
#   bash scripts/provision48.sh
# Chay lai duoc nhieu lan: rsync dong bo, flock trong driver chan phong trung.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/home/ntat/.local/bin:$PATH"
source scripts/endpoints.sh
R=/workspace/MultiVD

read -r H P <<< "$(vast_endpoint ntat 2>/dev/null)"
[[ -n "${H:-}" && "${P:-None}" != "None" ]] || { echo "may chua san sang"; exit 1; }
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -p $P"
echo "== ntat $H:$P =="

$SSH root@"$H" "mkdir -p $R/{data,log,results,model}" || exit 1

# python nao co torch
PY=$($SSH root@"$H" 'for c in /venv/main/bin/python /opt/conda/bin/python python3; do
  $c -c "import torch" 2>/dev/null && { echo $c; break; }; done' | tail -1)
[[ -n "$PY" ]] || { echo "!! khong tim thay python co torch tren may"; exit 1; }
echo "python: $PY"
$SSH root@"$H" "$PY -c 'import torch,transformers;print(\"torch\",torch.__version__,\"| transformers\",transformers.__version__);print(\"cuda\",torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"\")'"

for D in src run scripts tools; do
  rsync -az --delete -e "$SSH" "$D/" "root@$H:$R/$D/" || exit 1
done
rsync -az -e "$SSH" data/phase1_4cwe.jsonl data/phase1_common.jsonl data/phase1_full.jsonl "root@$H:$R/data/" || exit 1
rsync -az -e "$SSH" data/sven_python_folds_norm/ "root@$H:$R/data/sven_python_folds_norm/" || exit 1

echo "-- doi chieu du lieu tung byte --"
( for f in data/phase1_4cwe.jsonl data/phase1_common.jsonl data/phase1_full.jsonl; do
    stat -c '%n %s' "$f"; done; find data/sven_python_folds_norm -type f -printf '%p %s\n' ) | LC_ALL=C sort > /tmp/n48_data_local.txt
$SSH root@"$H" "cd $R && { for f in data/phase1_4cwe.jsonl data/phase1_common.jsonl data/phase1_full.jsonl; do stat -c '%n %s' \$f; done; find data/sven_python_folds_norm -type f -printf '%p %s\n'; } | LC_ALL=C sort" > /tmp/n48_data_remote.txt
if LC_ALL=C comm -23 /tmp/n48_data_local.txt /tmp/n48_data_remote.txt | grep -q .; then
  echo "!! DU LIEU LECH:"; LC_ALL=C comm -23 /tmp/n48_data_local.txt /tmp/n48_data_remote.txt | head; exit 1
fi
echo "   khop $(wc -l < /tmp/n48_data_local.txt) file"

echo "-- phong driver --"
$SSH root@"$H" "cd $R && SEEDS='42 7 1234' PYTHON=$PY setsid nohup bash run/night48.sh >> log/night48.log 2>&1 </dev/null & disown; sleep 3; tail -5 log/night48.log"
