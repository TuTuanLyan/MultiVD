#!/usr/bin/env bash
# GATE1 — DE XUAT 1 (ghep muon hai model) : LAP LAI ket qua 0-GPU + them DOI CHUNG con thieu.
#
# Ket qua 0-GPU da co (tools/late_fusion_gate.py tren cay shuf1, 2 backbone, n=5):
#   ghep(baseline, transfer REAL) − baseline : F1 +0.0208 10/10 p=0.002 | ROC +0.0153 9/10
#   ghep(baseline, transfer XAO ) − baseline : F1 -0.0032  5/10        | ROC -0.0103 0/10 p=0.002
# => loi ich den tu TRI THUC NGUON, khong tu viec ghep hai model noi chung.
#
# CHO CON HO: doi chung `shufall` la mot model KEM (Pha 1 nhan xao). No khong tra loi duoc
# cau "ghep hai model TOT NGANG NHAU thi co tu tang khong?" — tuc doi chung ENSEMBLE THUAN
# ma chinh de xuat doi (SR + S). Khoi nay chay no: mot model target-only THU HAI, khac seed.
#
# Ba nhanh, CUNG cay CUNG may CUNG phien (CLAUDE.md muc 4):
#   1. transfer none com real, seed 42   <- METHOD, chay TRUOC (nguoi dung neu 15/09)
#   2. baseline seed 42                  <- model A (SR)
#   3. baseline seed 7                   <- model S, DOI CHUNG ENSEMBLE THUAN
# Pha 1 DUNG LAI tu model/shuf1/phase1, khong huan luyen lai gi.
#
# Bac 1 — KIEM CHUNG, n=3 fold (nguoi dung neu 15/09: n=3 truoc, tot thi len n=5).
# Fold la vong NGOAI (CLAUDE.md muc 1), nen xong fold 3 la co ngay mot lat cat so duoc.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# HF_HOME: `/workspace` la duong dan cua VAST. Tren server 158 no KHONG ton tai, va cong
# voi HF_HUB_OFFLINE=1 thi moi lan nap model deu chet bang thong bao "check your internet
# connection" — nghe nhu loi mang chu khong nhu loi duong dan. Da mat mot lan phong vi cai nay.
# Ton trong bien da dat san; neu chua, chi dat khi /workspace co that, con lai de HF dung
# mac dinh ~/.cache/huggingface (tren 158 ca hai model da nam san o do).
if [ -z "${HF_HOME:-}" ] && [ -d /workspace ]; then export HF_HOME=/workspace/.hf_home; fi
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-/venv/main/bin/python}"
RN="${RUN_NAME:-gate1}"
FOLDS_LIST="${FOLDS:-1 2 3}"   # bac 1 — kiem chung; len n=5 bang FOLDS="1 2 3 4 5"
BB_LIST="${BB_LIST:-codebert=microsoft/codebert-base:cls t5p=Salesforce/codet5p-220m-bimodal:mean}"
P1STORE="${PHASE1_STORE:-model/shuf1/phase1}"

exec 4>/tmp/mvd_gate1.lock || exit 1
flock -n 4 || { echo "DA CO gate1 dang chay"; exit 3; }
PIDFILE="${PIDFILE:-/workspace/gate1.pid}"
echo "$$" > "$PIDFILE" 2>/dev/null || true
trap 'rm -f "$PIDFILE"' EXIT
ts(){ date -u '+%F %T'; }

# Kiem MOI duong dan se cham vao TRUOC khi chay.
for BB in $BB_LIST; do
  L="${BB%%=*}"
  CK="$P1STORE/${L}__none_com_real/seed_42/best.pt"
  [ -f "$CK" ] || { echo "!! THIEU Pha 1: $CK — DUNG"; exit 5; }
  echo "  Pha 1 co san: $CK ($(stat -c%s "$CK") byte)"
done
for f in run/matrix.sh src/train_transfer.py src/train_baseline.py src/report_fold.py data/sven_python_folds_norm/fold1/test.jsonl; do
  [ -f "$f" ] || { echo "!! THIEU $f — DUNG"; exit 5; }
done
grep -q "BASELINE_ONLY" run/matrix.sh || { echo "!! run/matrix.sh chua co BASELINE_ONLY"; exit 5; }
"$PY" -c "import torch,transformers,sklearn;print('env OK',torch.__version__,transformers.__version__)" || exit 5

# CONG NAP MODEL OFFLINE. Kiem THU THAT su se duoc nap, khong chi kiem import duoc thu vien.
# Cong nay ton ~5 giay va bat duoc dung lop loi da lam hong lan phong truoc (HF_HOME sai).
"$PY" - <<'PYCHECK' || { echo "!! KHONG nap duoc model offline — kiem HF_HOME/cache, DUNG"; exit 5; }
import os
from transformers import AutoTokenizer, AutoConfig
for m in ("microsoft/codebert-base", "Salesforce/codet5p-220m-bimodal"):
    AutoTokenizer.from_pretrained(m, trust_remote_code=True)
    AutoConfig.from_pretrained(m, trust_remote_code=True)
print("  HF offline OK | HF_HOME =", os.environ.get("HF_HOME", "(mac dinh ~/.cache/huggingface)"))
PYCHECK

echo "########## GATE1 bat dau $(ts) | $(hostname) | fold: $FOLDS_LIST ##########"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

common_env() {
  echo "RUN_NAME=$RN MODES=none OPTIMIZERS=adamw CWE_VOCAB=precomputed \
ARM_TAG=_com_real PHASE1_TAG=_com_real PHASE1_STORE=$P1STORE \
PHASE1_DATA_PATH=data/phase1_common.jsonl LAMBDA_CWE=0.05 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
PHASE2_EXTRA=--sam_rho_0 DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python"
}

for FOLD in $FOLDS_LIST; do
  for BB in $BB_LIST; do
    L="${BB%%=*}"
    # --- 1) METHOD truoc (baseline hoan lai)
    echo "===== $(ts) | $L | FOLD $FOLD | 1/3 METHOD transfer real seed 42 ====="
    SKIP_BASELINE=1 RUN_NAME="$RN" SEED=42 FOLDS="$FOLD" BACKBONES="$BB" \
    MODES=none OPTIMIZERS=adamw CWE_VOCAB=precomputed \
    ARM_TAG="_com_real" PHASE1_TAG="_com_real" PHASE1_STORE="$P1STORE" \
    PHASE1_DATA_PATH=data/phase1_common.jsonl LAMBDA_CWE=0.05 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
    PHASE2_EXTRA="--sam_rho 0" DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
    PYTHON="$PY" bash run/matrix.sh 4>&-

    # --- 2) baseline seed 42 (model A)
    echo "===== $(ts) | $L | FOLD $FOLD | 2/3 baseline seed 42 ====="
    BASELINE_ONLY=1 SKIP_BASELINE=0 RUN_NAME="$RN" SEED=42 FOLDS="$FOLD" BACKBONES="$BB" \
    MODES=none OPTIMIZERS=adamw CWE_VOCAB=precomputed \
    ARM_TAG="_com_real" PHASE1_TAG="_com_real" PHASE1_STORE="$P1STORE" \
    PHASE1_DATA_PATH=data/phase1_common.jsonl LAMBDA_CWE=0.05 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
    DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
    PYTHON="$PY" bash run/matrix.sh 4>&-

    # --- 3) baseline seed 7 (model S — doi chung ensemble thuan)
    echo "===== $(ts) | $L | FOLD $FOLD | 3/3 baseline seed 7 (DOI CHUNG) ====="
    BASELINE_ONLY=1 SKIP_BASELINE=0 RUN_NAME="$RN" SEED=7 FOLDS="$FOLD" BACKBONES="$BB" \
    MODES=none OPTIMIZERS=adamw CWE_VOCAB=precomputed \
    ARM_TAG="_com_real" PHASE1_TAG="_com_real" PHASE1_STORE="$P1STORE" \
    PHASE1_DATA_PATH=data/phase1_common.jsonl LAMBDA_CWE=0.05 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
    DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
    PYTHON="$PY" bash run/matrix.sh 4>&-
  done
  echo "----- $(ts) | het fold $FOLD | o: $(find results/${RN}_* -name "fold*.json" 2>/dev/null | wc -l) -----"
done

TOT=$(find results/${RN}_* -name 'fold*.json' 2>/dev/null | wc -l)
echo "########## GATE1 xong $(ts) | $TOT/30 o ##########"
