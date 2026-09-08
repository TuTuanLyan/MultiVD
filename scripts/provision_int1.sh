#!/usr/bin/env bash
# provision_int1.sh — day code + du lieu + checkpoint Pha 1 len mot may vast roi phong INT1.
#
#   bash scripts/provision_int1.sh ntat  "1 2"
#   bash scripts/provision_int1.sh ntat2 "3"
#
# Chia theo FOLD TRON VEN (CLAUDE.md muc 4): baseline va ca hai doi chung cua mot fold
# deu nam tren cung may voi fold do, nen Delta ghep cap trong fold van sach du hai may
# khac GPU (5060 Ti vs 5070 Ti) va khac ban torch voi may o nha.
#
# Chay lai duoc nhieu lan: rsync dong bo, matrix.sh bo qua o da co ket qua.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/home/ntat/.local/bin:$PATH"
source scripts/endpoints.sh
LABEL="${1:?dat nhan may: ntat hoac ntat2}"
FOLDS="${2:?dat danh sach fold, vd \"1 2\"}"
R=/workspace/MultiVD
PY=/venv/main/bin/python

case "$LABEL" in ntat|ntat2) ;; *) echo "!! CHI duoc dung nhan ntat/ntat2 (VAST_RULES)"; exit 1;; esac

VAST_CACHE_TTL=1 read -r H P <<< "$(vast_endpoint "$LABEL" 2>/dev/null)"
[[ -n "${H:-}" && "${P:-None}" != "None" ]] || { echo "!! $LABEL chua san sang"; exit 1; }
SSH="ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=20 -p $P"
echo "== $LABEL  $H:$P  fold [$FOLDS] =="

$SSH root@"$H" "mkdir -p $R/{data,log,results,model/n48/phase1}" || exit 1
$SSH root@"$H" "$PY -c 'import torch,transformers,sklearn,numpy' " || {
  echo "!! moi truong tren $LABEL chua du — dung han, KHONG chay"; exit 2; }

echo "-- day ma nguon --"
for D in src run scripts tools; do
  rsync -az --delete -e "$SSH" "$D/" "root@$H:$R/$D/" || exit 1
done

echo "-- day du lieu --"
rsync -az -e "$SSH" data/phase1_4cwe.jsonl data/phase1_common.jsonl "root@$H:$R/data/" || exit 1
rsync -az -e "$SSH" data/sven_python_folds_norm/ "root@$H:$R/data/sven_python_folds_norm/" || exit 1

echo "-- day checkpoint Pha 1 (876 MB, co the lau) --"
for S in 4cwe com; do
  D="model/n48/phase1/t5p__latent_bottleneck_${S}_l0p05/seed_42"
  $SSH root@"$H" "mkdir -p $R/$D"
  rsync -az -e "$SSH" "$D/best.pt" "root@$H:$R/$D/best.pt" || exit 1
done

echo "-- doi chieu TUNG BYTE (md5, ca du lieu lan checkpoint) --"
{ for f in data/phase1_4cwe.jsonl data/phase1_common.jsonl \
           model/n48/phase1/t5p__latent_bottleneck_4cwe_l0p05/seed_42/best.pt \
           model/n48/phase1/t5p__latent_bottleneck_com_l0p05/seed_42/best.pt; do
    md5sum "$f"; done
  find data/sven_python_folds_norm -type f -printf '%p %s\n'; } | LC_ALL=C sort > /tmp/int1_local.txt
$SSH root@"$H" "cd $R && { for f in data/phase1_4cwe.jsonl data/phase1_common.jsonl \
    model/n48/phase1/t5p__latent_bottleneck_4cwe_l0p05/seed_42/best.pt \
    model/n48/phase1/t5p__latent_bottleneck_com_l0p05/seed_42/best.pt; do md5sum \$f; done
  find data/sven_python_folds_norm -type f -printf '%p %s\n'; } | LC_ALL=C sort" > /tmp/int1_remote.txt
if LC_ALL=C comm -23 /tmp/int1_local.txt /tmp/int1_remote.txt | grep -q .; then
  echo "!! LECH — KHONG phong:"; LC_ALL=C comm -23 /tmp/int1_local.txt /tmp/int1_remote.txt | head; exit 1
fi
echo "   khop $(wc -l < /tmp/int1_local.txt) muc"

echo "-- phong INT1 --"
$SSH root@"$H" "cd $R && FOLD_LIST='$FOLDS' SOURCES_LIST='4cwe com' SEED=42 PYTHON=$PY \
  setsid nohup bash run/int1.sh >> log/int1_${LABEL}.log 2>&1 </dev/null & disown; sleep 6; tail -4 log/int1_${LABEL}.log"
