#!/usr/bin/env bash
# Day khoi MW_ASSEMBLE_BABEL sang mot may thue roi PHONG o do.
#
#   bash scripts/provision_babel.sh ntat            # giai dia chi theo NHAN vast
#   bash scripts/provision_babel.sh 1.2.3.4 41234   # hoac host + cong truc tiep
#
# CHI dung may nhan `ntat` (VAST_RULES.md). Khong dung toi instance cua nguoi khac.
# IP cong khai cua vast DOI khi may bi doi may chu, nen giai dia chi ngay luc goi
# thay vi hardcode — doc nham timeout thanh "may chet" la cach de huy nham nhat.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ $# -ge 2 ]; then H_IP="$1"; H_PORT="$2"
else
  LABEL="${1:-ntat}"
  case "$LABEL" in ntat|ntat2) ;; *) echo "!! chi chay tren nhan ntat/ntat2, khong phai '$LABEL'"; exit 2;; esac
  source scripts/endpoints.sh
  read -r H_IP H_PORT <<< "$(vast_endpoint "$LABEL")"
  [ -n "${H_IP:-}" ] || { echo "!! khong giai duoc dia chi cua nhan '$LABEL'"; exit 1; }
fi
R="${REMOTE_ROOT:-/workspace/MultiVD}"
SSH="ssh -p $H_PORT -o StrictHostKeyChecking=no -o ConnectTimeout=25"
HOST="root@$H_IP"
RS="rsync -az -e \"$SSH\""
say(){ echo "== $* =="; }

say "may: $H_IP:$H_PORT | thu muc: $R"
$SSH $HOST "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader" || exit 1

# `python3` cua image vast thuong KHONG co torch — torch nam trong /venv/main.
# Do tu thay vi gia dinh: doan sai o day thi moi cong sau deu bao "thieu thu vien".
say "0/7 tim interpreter co torch"
RPY="${REMOTE_PY:-}"
if [ -z "$RPY" ]; then
  RPY=$($SSH $HOST 'for P in /venv/main/bin/python /opt/conda/bin/python /usr/bin/python3; do
      [ -x "$P" ] || continue
      "$P" -c "import torch,transformers,sklearn" 2>/dev/null && { echo "$P"; break; }
    done' | tr -d "\r" | head -1)
fi
[ -n "$RPY" ] || { echo "!! khong tim thay interpreter nao co torch tren may"; exit 1; }
echo "  dung: $RPY"

say "1/7 moi truong"
$SSH $HOST "$RPY -c 'import torch,transformers,sklearn;print(\"torch\",torch.__version__,\"| tf\",transformers.__version__,\"| sk\",sklearn.__version__);import torch as t;print(\"cuda\",t.cuda.is_available(),t.cuda.get_device_name(0))'" || {
  echo "  cai dat thieu — thu ghim theo requirements-pin.txt"
  rsync -az -e "$SSH" requirements-pin.txt "$HOST:$R/" 2>/dev/null
  $SSH $HOST "$RPY -m pip install -q -r $R/requirements-pin.txt && $RPY -c 'import torch,transformers;print(transformers.__version__)'" || exit 1
}
# transformers PHAI la 4.57.1: moi ket qua cu chay tren ban do; lech major la khong so duoc.
TFV=$($SSH $HOST "$RPY -c 'import transformers;print(transformers.__version__)'" | tr -d '\r')
[ "$TFV" = "4.57.1" ] || echo "  !! CANH BAO transformers=$TFV (ky vong 4.57.1) — ghi vao bao cao"

say "2/7 day ma nguon"
$SSH $HOST "mkdir -p $R/data $R/model/shuf1/phase1 $R/log $R/results"
for D in src run tools scripts; do rsync -az --delete -e "$SSH" "$D/" "$HOST:$R/$D/" || exit 1; done

say "3/7 day du lieu dich (4.5 MB)"
rsync -az -e "$SSH" data/sven_python_folds_norm/ "$HOST:$R/data/sven_python_folds_norm/" || exit 1

say "4/7 day checkpoint Pha 1 (476 MB)"
rsync -az --info=stats1 -e "$SSH" model/shuf1/phase1/codebert__none_com_real/ \
      "$HOST:$R/model/shuf1/phase1/codebert__none_com_real/" || exit 1

say "5/7 doi chieu TUNG BYTE"
{ find data/sven_python_folds_norm -type f -printf '%p %s\n'
  find model/shuf1/phase1/codebert__none_com_real -type f -printf '%p %s\n'
  find src run -name '*.py' -o -name '*.sh' | sed 's|^|./|' | while read -r f; do stat -c '%n %s' "${f#./}"; done
} | LC_ALL=C sort > /tmp/babel_local.txt
$SSH $HOST "cd $R && { find data/sven_python_folds_norm -type f -printf '%p %s\n'
  find model/shuf1/phase1/codebert__none_com_real -type f -printf '%p %s\n'
  find src run -name '*.py' -o -name '*.sh' | sed 's|^./||' | while read -r f; do stat -c '%n %s' \"\$f\"; done; } | LC_ALL=C sort" > /tmp/babel_remote.txt
if LC_ALL=C diff -u /tmp/babel_local.txt /tmp/babel_remote.txt > /tmp/babel_diff.txt; then
  echo "  khop 0 byte lech ($(wc -l < /tmp/babel_local.txt) file)"
else
  echo "  !! LECH — xem /tmp/babel_diff.txt"; head -20 /tmp/babel_diff.txt; exit 1
fi

say "6/7 model offline"
$SSH $HOST "cd $R && HF_HOME=$R/.hf $RPY -c '
from transformers import AutoTokenizer, AutoModel
AutoTokenizer.from_pretrained(\"microsoft/codebert-base\"); AutoModel.from_pretrained(\"microsoft/codebert-base\")
print(\"  tai ve OK\")'" || {
  echo "  tai ve that bai — day cache tu local (~950 MB)"
  rsync -az --info=stats1 -e "$SSH" ~/.cache/huggingface/hub/models--microsoft--codebert-base \
        "$HOST:$R/.hf/hub/" || exit 1
}
$SSH $HOST "cd $R && HF_HOME=$R/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $RPY -c '
import sys; sys.path.insert(0,\"src\")
from transformers import AutoTokenizer
from model import build_backbone
AutoTokenizer.from_pretrained(\"microsoft/codebert-base\"); build_backbone(\"microsoft/codebert-base\")
print(\"  offline OK\")'" || exit 1

say "7/7 THU KHOI mot o nho truoc khi phong ca khoi"
$SSH $HOST "cd $R && HF_HOME=$R/.hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 timeout 900 $RPY -u src/train_babel.py \
  --run_name __smoke__ --method_name babel --fold 1 --seed 42 --max_train_samples 8 \
  --max_lines 32 --max_line_tokens 24 --graph_hidden 64 --batch_size 2 --eval_batch_size 8 \
  --epochs 1 --min_epochs 1 --patience 1 2>&1 | tail -4" || { echo "!! thu khoi HONG — KHONG phong"; exit 1; }
$SSH $HOST "rm -rf $R/results/__smoke__"

say "PHONG khoi that (n=5, 2 nhanh = 10 o)"
$SSH $HOST "cd $R && HF_HOME=$R/.hf setsid nohup env PYTHON=$RPY FOLDS='1 2 3 4 5' \
  bash run/babel.sh </dev/null > log/babel_driver.log 2>&1 & echo phong-roi"
sleep 20
$SSH $HOST "cat $R/log/babel_driver.log"
echo
echo "Theo doi:  ssh -p $H_PORT $HOST 'tail -f $R/log/babel_driver.log'"
echo "Keo ve  :  rsync -az -e \"$SSH\" $HOST:$R/results/mw_assemble_babel/ results/mw_assemble_babel/"
