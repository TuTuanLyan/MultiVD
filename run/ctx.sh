#!/usr/bin/env bash
# CTX — TRUC NGU CANH: 256 / 512 / 1024 token, CHI doi MOT bien la `max_length`.
#
# Cau hoi (nguoi dung neu 15/09): "512 co nen mat dac trung lo hong khi code dai khong?"
# FACTS §60 do duoc: tren nhom `none`, ROC tut 0.087 (codebert) va 0.080 (t5p) tu dai
# 0-128 xuong 256-512 — tuc NGAY TRONG vung KHONG bi cat. Nhung phep do do khong tach duoc
# "pha loang bieu dien" khoi "ham dai thi von kho hon". Khoi nay tach.
#
# CHI t5p. codebert co tran vi tri 514 (RoBERTa) nen khong nang duoc; codet5p dung vi tri
# TUONG DOI (32 bucket) nen doi mot co la xong, 0 dong ma.
#
# --grad_checkpointing BAT cho CA BA nhanh. Neu chi bat cho 1024 thi phep so doi HAI bien.
#
# DOI CHUNG NOI TAI — day moi la cho dat gia:
#   Voi hang test <= 256 token, ca ba nhanh nhan DUNG CUNG dau vao (khong co gi bi cat).
#   Chenh lech tren nhom hang do CHINH LA san nhieu cua phep do nay — do duoc truc tiep,
#   khong phai di muon tu khoi khac. Hieu ung ngu canh that phai LON HON san do, va phai
#   nam o nhom hang DAI, khong nam o nhom hang ngan.
#
# Bac 1 — kiem chung, n=3 fold. Chi baseline (target-only) de khong lan bien Pha 1:
# Pha 1 duoc huan luyen o 512, nen chay Pha 2 o 1024 se lech ngu canh giua hai pha.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${HF_HOME:-}" ] && [ -d /workspace ]; then export HF_HOME=/workspace/.hf_home; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-/venv/main/bin/python}"
FOLDS_LIST="${FOLDS:-1 2 3}"
LENS="${LENS:-512 1024 256}"          # 512 truoc: no la moc, hong som thi biet ngay
BB="t5p=Salesforce/codet5p-220m-bimodal:mean"

exec 4>/tmp/mvd_ctx.lock || exit 1
flock -n 4 || { echo "DA CO ctx dang chay"; exit 3; }
PIDFILE="${PIDFILE:-/data/ntat/ctx.pid}"
echo "$$" > "$PIDFILE" 2>/dev/null || true
trap 'rm -f "$PIDFILE"' EXIT
ts(){ date -u '+%F %T'; }

for f in run/matrix.sh src/train_baseline.py data/sven_python_folds_norm/fold1/test.jsonl; do
  [ -f "$f" ] || { echo "!! THIEU $f — DUNG"; exit 5; }
done
grep -q "BASELINE_ONLY" run/matrix.sh || { echo "!! matrix.sh chua co BASELINE_ONLY"; exit 5; }
grep -q "grad_checkpointing" src/train_baseline.py || { echo "!! train_baseline.py chua co --grad_checkpointing"; exit 5; }
"$PY" -c "import torch,transformers,sklearn;print('env OK',torch.__version__,transformers.__version__)" || exit 5
"$PY" - <<'PYCHECK' || { echo "!! KHONG nap duoc model offline — DUNG"; exit 5; }
from transformers import AutoTokenizer, AutoConfig
m = "Salesforce/codet5p-220m-bimodal"
AutoTokenizer.from_pretrained(m, trust_remote_code=True)
c = AutoConfig.from_pretrained(m, trust_remote_code=True)
assert getattr(c, "max_position_embeddings", None) is None, \
    "codet5p le ra KHONG co tran vi tri tuyet doi — kiem lai truoc khi chay 1024"
print("  t5p OK | vi tri tuong doi, khong co tran")
PYCHECK

echo "########## CTX bat dau $(ts) | $(hostname) | len: $LENS | fold: $FOLDS_LIST ##########"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

for FOLD in $FOLDS_LIST; do
  for L in $LENS; do
    echo "===== $(ts) | FOLD $FOLD | max_length $L ====="
    BASELINE_ONLY=1 SKIP_BASELINE=0 RUN_NAME="ctx$L" SEED=42 FOLDS="$FOLD" \
    BACKBONES="$BB" MODES=none OPTIMIZERS=adamw MAX_LENGTH="$L" \
    BASELINE_EXTRA="--grad_checkpointing" \
    DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
    PYTHON="$PY" bash run/matrix.sh 4>&-
  done
  echo "----- $(ts) | het fold $FOLD | o: $(find results/ctx*_t5p -name 'fold*.json' 2>/dev/null | wc -l)/9 -----"
done
echo "########## CTX xong $(ts) | $(find results/ctx*_t5p -name 'fold*.json' 2>/dev/null | wc -l)/9 o ##########"
