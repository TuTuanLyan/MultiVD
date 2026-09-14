#!/usr/bin/env bash
# FUSNONE7 — chay lai cau hoi "fusion co can head phu khong" o SEED 7, noi CA HAI Pha 1 deu LANH.
#
# Vi sao phai chay lai: khoi `fusnone` (seed 42) so nhanh CO head voi mot Pha 1 khong-head DA HONG
# (train loss dung im o ln(2) suot 13 epoch). Con so +0.0535 cua no KHONG do duoc gia tri cua head.
# Khoi `p1seed` (FACTS §55.1) cho thay o seed 7 ca hai nhanh deu hoc binh thuong:
#     co head    val 0.5932 (train loss 0.814 -> 0.414)
#     khong head val 0.5787 (train loss 0.708 -> 0.468)
# Khoang cach Pha 1 chi 0.0145 — nen day moi la phep so sach.
#
# Pha 1 DUNG LAI tu p1seed, khong huan luyen lai. Baseline chay moi o cung seed 7.
# --grad_checkpointing BAT cho CA HAI nhanh: t5p + adapter + fusion + fine-tune ca backbone
# vuot tran 16 GB dung 24 MiB (OOM that o 23:43, giong het §47/§48). Checkpointing dung
# use_reentrant=False nen gradient khong doi, chi cham hon. Phai bat cho CA HAI nhanh —
# neu chi bat mot ben thi hai nhanh khac NHIEU HON MOT bien.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PY="${PYTHON:-/venv/main/bin/python}"
RN="${RUN_NAME:-fusnone_t5p}"
SD="${SEED:-42}"
BB="${BB:-t5p=Salesforce/codet5p-220m-bimodal:mean}"
STORE="model/${RN}/phase1"

exec 4>/tmp/mvd_fusnone_t5p.lock || exit 1
flock -n 4 || { echo "DA CO fusnone_t5p dang chay"; exit 3; }
PIDFILE="${PIDFILE:-/workspace/${RN}.pid}"
echo "$$" > "$PIDFILE" 2>/dev/null || true
trap 'rm -f "$PIDFILE"' EXIT
ts(){ date -u '+%F %T'; }

# --- dat Pha 1 tu p1seed vao dung bo cuc matrix.sh mong doi ---
place() {  # $1 = file nguon, $2 = ten thu muc nhanh
  local dst="$STORE/$2/seed_$SD/best.pt"
  [ -f "$1" ] || { echo "!! THIEU nguon $1"; return 1; }
  if [ ! -f "$dst" ]; then mkdir -p "$(dirname "$dst")"; cp "$1" "$dst"; fi
  local a b; a=$(stat -c %s "$1"); b=$(stat -c %s "$dst")
  [ "$a" = "$b" ] || { echo "!! byte lech: $a vs $b"; return 1; }
  echo "  dat $2  ($a byte)"
}
place model/p1seed_t5p/head_ad48_s42.pt t5p__latent_bottleneck_com_l0p05_ad48 || exit 5
place model/p1seed_t5p/none_ad48_s42.pt t5p__none_com_ad48                    || exit 5

# Kiem NOI DUNG checkpoint, khong chi kiem file ton tai
"$PY" - "$STORE" "$SD" <<'PYEOF' || exit 5
import torch,sys,os
store,sd=sys.argv[1],sys.argv[2]
for name,mode in [("t5p__latent_bottleneck_com_l0p05_ad48","latent_bottleneck"),
                  ("t5p__none_com_ad48","none")]:
    p=os.path.join(store,name,f"seed_{sd}","best.pt")
    b=torch.load(p,map_location="cpu",weights_only=False)
    sdict=b.get("model_state_dict",b)
    nad=len([k for k in sdict if ".adapters." in k])
    ok = (b.get("aux_mode")==mode) and nad==48 and b["best_val_macro_f1"]>0.53
    print("  %-42s aux_mode=%-18s adapter_keys=%-3d val=%.4f  %s" % (
        name, b.get("aux_mode"), nad, b["best_val_macro_f1"], "OK" if ok else "!! KHONG DAT"))
    if not ok: raise SystemExit(1)
PYEOF
for f in run/matrix.sh src/train_transfer.py src/train_baseline.py src/adapters.py data/phase1_common.jsonl; do
  [ -f "$f" ] || { echo "!! THIEU $f"; exit 5; }
done

echo "########## FUSNONE_T5P bat dau $(ts) | seed $SD | $(hostname) ##########"
for FOLD in ${FOLDS:-1 2 3 4 5}; do
  echo "===== $(ts) | FOLD $FOLD | A: fusft (CO head) + baseline ====="
  SKIP_BASELINE=0 RUN_NAME="$RN" SEED="$SD" FOLDS="$FOLD" \
  BACKBONES="$BB" MODES=latent_bottleneck OPTIMIZERS=adamw \
  PHASE1_DATA_PATH=data/phase1_common.jsonl CWE_VOCAB=precomputed \
  ARM_TAG="_com_l0p05_ad48_fusft" PHASE1_TAG="_com_l0p05_ad48" PHASE1_STORE="$STORE" \
  LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_PATIENCE=10 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
  PHASE1_EXTRA="--sam_rho 0 --adapter_dim 48 --adapter_lr 1e-4" \
  PHASE2_EXTRA="--sam_rho 0 --phase2_fusion ft --adapter_lr 1e-4 --grad_checkpointing" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PYTHON="$PY" bash run/matrix.sh 4>&-

  echo "===== $(ts) | FOLD $FOLD | B: nonefus (KHONG head) ====="
  SKIP_BASELINE=1 RUN_NAME="$RN" SEED="$SD" FOLDS="$FOLD" \
  BACKBONES="$BB" MODES=none OPTIMIZERS=adamw \
  PHASE1_DATA_PATH=data/phase1_common.jsonl CWE_VOCAB=precomputed \
  ARM_TAG="_com_ad48_fusft" PHASE1_TAG="_com_ad48" PHASE1_STORE="$STORE" \
  LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_PATIENCE=10 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
  PHASE1_EXTRA="--sam_rho 0 --adapter_dim 48 --adapter_lr 1e-4" \
  PHASE2_EXTRA="--sam_rho 0 --phase2_fusion ft --adapter_lr 1e-4 --grad_checkpointing" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PYTHON="$PY" bash run/matrix.sh 4>&-

  echo "----- $(ts) | het FOLD $FOLD | o: $(find results/${RN}_t5p -name 'fold*.json' 2>/dev/null | wc -l)/15 -----"
done
echo "########## FUSNONE_T5P xong $(ts) | $(find results/${RN}_t5p -name 'fold*.json' 2>/dev/null | wc -l)/15 o ##########"
