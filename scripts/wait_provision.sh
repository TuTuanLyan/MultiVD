#!/usr/bin/env bash
# Cho mot instance vast nhan ntat chuyen sang `running`, roi gan key + provision.
# Khong lam gi voi nhan ngoai ho ntat.
#
#   bash scripts/wait_provision.sh ntat2
#
# Vi sao gan lai SSH key moi lan: tai khoan nay la TEAM context nen
# `vastai create ssh-key` bi tu choi ("Team SSH keys are not supported"), phai
# `vastai attach ssh <id> <pubkey>` cho TUNG instance. Instance moi thue = key
# chua gan = Permission denied.
#
# Vi sao cai torch vao python3 he thong chu khong dung /venv/main: image nay chia
# doi bo thu vien — python3 he thong co transformers/sklearn/pandas nhung KHONG
# co torch; /venv/main co torch nhung KHONG co transformers. Cai torch vao
# python3 la duong ngan nhat de co mot moi truong day du. Chi tiet trong
# VAST_TEMPLATE_ERROR.MD.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh

L="${1:?nhan: ntat | ntat2}"
case "$L" in ntat|ntat[0-9]*) ;; *) echo "[$L] khong thuoc ho ntat — tu choi"; exit 2 ;; esac
MAXWAIT="${MAXWAIT:-90}"     # 90 x 20s = 30 phut

echo "[$L] cho instance san sang..."
for ((i=0;i<MAXWAIT;i++)); do
  ST=$(vast_state "$L" 2>/dev/null)
  read -r H P <<< "$(vast_endpoint "$L" 2>/dev/null)"
  if [[ "$ST" == *running* && -n "${H:-}" && -n "${P:-}" && "$P" != "None" ]]; then
    echo "[$L] running tai $H:$P"; break
  fi
  sleep 20
done
read -r H P <<< "$(vast_endpoint "$L")"
[[ -z "${H:-}" || "$P" == "None" ]] && { echo "[$L] chua san sang sau $((MAXWAIT*20))s"; exit 3; }

KEY=$(cat ~/.ssh/id_ed25519.pub)
vastai attach ssh "$(vast_id "$L")" "$KEY" >/dev/null 2>&1 && echo "[$L] da gan SSH key"

SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -p $P"
for ((i=0;i<20;i++)); do
  $SSH "root@$H" 'echo ok' >/dev/null 2>&1 && break
  sleep 10
done
$SSH "root@$H" 'echo ok' >/dev/null 2>&1 || { echo "[$L] SSH van khong vao duoc"; exit 4; }
echo "[$L] SSH thong"

$SSH "root@$H" 'mkdir -p /workspace/MultiVD' || exit 1
echo "[$L] rsync ma nguon + du lieu..."
# Loai tru NEO GOC (`/log`, `/model_*`). Mau khong neo (`log*`, `model*`) se nuot
# ca src/logging_utils.py va src/model.py — da hong dung vay sang 27/08 va moi
# job chet o dong import.
rsync -az -e "$SSH" --exclude '.git' --exclude '__pycache__' \
  --exclude '/results' --exclude '/results_*' --exclude '/model' --exclude '/model_*' \
  --exclude '/log' --exclude '/log_*' --exclude '/records' --exclude '/figures' --exclude '/archive' \
  --exclude 'data/ccpp_primevul_paired_*' --exclude 'data/train_ccpp_js*' \
  ./ "root@$H:/workspace/MultiVD/" || exit 1

echo "[$L] cai torch cu128 (Blackwell sm_120)..."
# Cai TACH ROI phien SSH: goi torch ~2.5 GB, moi timeout ngan cua phia goi deu
# cat ngang chung. Va PHAI KIEM TRA LAI sau khi cai — ban truoc viet
# `import torch || pip install ... | tail -2`, `tail` nuot ma loi cua pip nen
# script bao "XONG PROVISION" trong khi torch chua co. Chi co cong `import torch`
# o dau driver moi bat duoc, sau khi da mat mot luot phong.
if ! $SSH "root@$H" 'python3 -c "import torch"' 2>/dev/null; then
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -f -p "$P" root@"$H" \
    "setsid bash -c 'pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128 > /workspace/pip_torch.log 2>&1' </dev/null >/dev/null 2>&1"
  echo "[$L] dang cai torch nen, cho..."
  for ((k=0;k<60;k++)); do
    $SSH "root@$H" 'python3 -c "import torch"' 2>/dev/null && break
    sleep 30
  done
fi
$SSH "root@$H" 'python3 -c "import torch;print(\"  torch\",torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name(0))"' 2>/dev/null | tail -1

# Bo con lai. KHONG chi kiem torch: anh may thay doi giua cac lan thue. Ngay
# 27/08 python3 he thong co san transformers/sklearn/pandas (thieu moi torch);
# ngay 29/08 mot may khac THIEU CA BON. Kiem TUNG GOI, dung gia dinh.
MISS=$($SSH "root@$H" 'for m in transformers sklearn pandas scipy; do python3 -c "import $m" 2>/dev/null || echo -n "$m "; done' 2>/dev/null)
if [[ -n "${MISS// /}" ]]; then
  echo "[$L] thieu: $MISS — dang cai nen..."
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -f -p "$P" root@"$H" \
    "setsid bash -c 'pip install --no-cache-dir -r /workspace/MultiVD/requirements-pin.txt scikit-learn pandas scipy > /workspace/pip_rest.log 2>&1' </dev/null >/dev/null 2>&1"
  for ((k=0;k<40;k++)); do
    R=$($SSH "root@$H" 'for m in transformers sklearn pandas scipy; do python3 -c "import $m" 2>/dev/null || echo -n "x"; done' 2>/dev/null)
    [[ -z "$R" ]] && break
    sleep 30
  done
fi

# Cong cuoi: MOI goi ma code that su import deu phai vao duoc. Provision bao
# "XONG" ma thieu mot goi la ca luot chay chet o dong import — da xay ra hai lan.
BAD=$($SSH "root@$H" 'for m in torch transformers sklearn pandas numpy; do python3 -c "import $m" 2>/dev/null || echo -n "$m "; done' 2>/dev/null)
[[ -n "${BAD// /}" ]] && { echo "[$L] VAN THIEU: $BAD — provision COI NHU THAT BAI"; exit 6; }

# Cong PHIEN BAN. Du goi thoi chua du: mot may moi keo ve transformers 5.x trong
# khi moi ket qua truoc chay tren 4.57.1, va checkpoint sinh ra se khong so duoc
# voi checkpoint cu cua cung backbone.
TFV=$($SSH "root@$H" 'python3 -c "import transformers;print(transformers.__version__)"' 2>/dev/null | tr -d "[:space:]")
if [[ "$TFV" != "4.57.1" ]]; then
  echo "[$L] transformers=$TFV, can 4.57.1 — dang ghim lai..."
  $SSH "root@$H" 'pip install -q --no-cache-dir "transformers==4.57.1" 2>&1 | tail -1'
  TFV=$($SSH "root@$H" 'python3 -c "import transformers;print(transformers.__version__)"' 2>/dev/null | tr -d "[:space:]")
  [[ "$TFV" == "4.57.1" ]] || { echo "[$L] VAN LA transformers=$TFV — provision THAT BAI"; exit 7; }
fi
echo "[$L] transformers=$TFV (khop moi lan chay truoc)"
for M in logging_utils model train; do
  $SSH "root@$H" "test -f /workspace/MultiVD/src/$M.py" || { echo "[$L] THIEU src/$M.py"; exit 5; }
done
echo "[$L] XONG PROVISION"
