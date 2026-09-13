#!/usr/bin/env bash
# FUSION_BASE — chay ba nhanh tren CUNG MOT MAY LOCAL de co con so "fusion hon BASELINE".
# Nguoi dung 14/09: TAM THOI chi can baseline vs fusion de thay XU HUONG; phep so voi
# latent o cac optimizer khac (RecAdam, ASAM) de kiem chung SAU, khong chay bay gio.
#     baseline            khong Pha 1
#     latent_bottleneck   phuong phap chot (Pha 1 + head phu, KHONG adapter)
#     fusft               adapter nguon + adapter dich + fusion, fine-tune ca backbone
#
# VI SAO PHAI CHAY LAI CA BA. Hai cay tren vast da mat theo may bi huy. Baseline chay o may
# khac KHONG ghep cap duoc voi fusft cu — muc 4 cam, va san nhieu giua cac loai GPU la 0.028,
# lon hon ca hieu ung dang do (+0.0226).
#
# Pha 1 DUNG LAI ca hai, khong huan luyen lai (da keo ve tu vast):
#   khong adapter: model/n48/phase1/codebert__latent_bottleneck_com_l0p05
#   co adapter   : model/fus2/phase1/codebert__latent_bottleneck_com_l0p05_ad48
#
# MAY DUNG CHUNG: tu cho GPU that su ranh moi chay. KHONG BAO GIO kill tien trinh cua
# nguoi khac (CLAUDE.md UU TIEN 2). Doi hoi VRAM du VA on dinh qua nhieu lan kiem, VA
# khong con tien trinh compute cua user khac — mot nhip dip xuong khong duoc tinh la ranh.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
PY="${PYTHON:-/home/ntat/miniconda3/envs/vdenv/bin/python}"
exec 4>/tmp/mvd_fusion_base.lock || exit 1
flock -n 4 || { echo "DA CO fusion_base dang chay"; exit 3; }
GPU_LOCK="${GPU_LOCK:-/tmp/multivd_opt1.lock}"
NEED_MIB="${NEED_MIB:-11500}"
STABLE="${STABLE:-3}"        # phai ranh lien tiep ngan nay lan moi tin
WAIT_MAX="${WAIT_MAX:-720}"  # toi da 12 gio
ts(){ date -u '+%F %T'; }

ok=0
for i in $(seq 1 "$WAIT_MAX"); do
  free_lock=0; flock -n "$GPU_LOCK" -c true 2>/dev/null && free_lock=1
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  tot=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
  avail=$(( ${tot:-0} - ${used:-0} ))
  # Dem JOB HUAN LUYEN cua nguoi khac, KHONG dem dich vu nen.
  # `ollama` chay thuong truc nhieu ngay va chi giu ~0,7 GB; neu doi "khong co tien trinh nao
  # cua nguoi khac" thi cong se KHONG BAO GIO mo. Dieu kien that la VRAM trong — con day chi
  # de chan dung luc nguoi khac dang huan luyen.
  others=0
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    u=$(ps -o user= -p "$p" 2>/dev/null | tr -d ' ')
    c=$(ps -o comm= -p "$p" 2>/dev/null | tr -d ' ')
    [ -n "$u" ] && [ "$u" != "ntat" ] && [[ "$c" == python* ]] && others=$((others+1))
  done
  if (( free_lock == 1 && avail >= NEED_MIB && others == 0 )); then
    ok=$((ok+1))
    echo "$(ts) | GPU ranh lan $ok/$STABLE (${avail}MiB, 0 tien trinh nguoi khac)"
    (( ok >= STABLE )) && break
  else
    (( ok > 0 )) && echo "$(ts) | ranh gian doan — dem lai tu dau"
    ok=0
    (( i % 10 == 0 )) && echo "$(ts) | cho... lock=$free_lock VRAM=${avail}MiB nguoi_khac=$others"
  fi
  sleep 60
done
(( ok >= STABLE )) || { echo "$(ts) | het $WAIT_MAX phut ma GPU khong ranh on dinh — DUNG"; exit 4; }

echo "########## FUSION_BASE bat dau $(ts) | $(hostname) ##########"
for FOLD in ${FOLDS:-1 2 3 4 5}; do
  echo "===== $(ts) | FOLD $FOLD | baseline + latent_bottleneck ====="
  SKIP_BASELINE=0 RUN_NAME=fus3 SEED=42 FOLDS="$FOLD" \
  BACKBONES="codebert=microsoft/codebert-base:cls" MODES=latent_bottleneck OPTIMIZERS=adamw \
  PHASE1_DATA_PATH=data/phase1_common.jsonl CWE_VOCAB=precomputed \
  ARM_TAG="_com_l0p05" PHASE1_TAG="_com_l0p05" PHASE1_STORE="model/n48/phase1" \
  LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
  PHASE1_EXTRA="--sam_rho 0" PHASE2_EXTRA="--sam_rho 0" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PYTHON="$PY" bash run/matrix.sh 4>&-

  echo "===== $(ts) | FOLD $FOLD | fusft ====="
  SKIP_BASELINE=1 RUN_NAME=fus3 SEED=42 FOLDS="$FOLD" \
  BACKBONES="codebert=microsoft/codebert-base:cls" MODES=latent_bottleneck OPTIMIZERS=adamw \
  PHASE1_DATA_PATH=data/phase1_common.jsonl CWE_VOCAB=precomputed \
  ARM_TAG="_com_l0p05_ad48_fusft" PHASE1_TAG="_com_l0p05_ad48" PHASE1_STORE="model/fus2/phase1" \
  LAMBDA_CWE=0.05 PHASE1_EPOCHS=15 PHASE1_MIN_VAL=0 MIN_EPOCHS=3 \
  PHASE1_EXTRA="--sam_rho 0 --adapter_dim 48 --adapter_lr 1e-4" \
  PHASE2_EXTRA="--sam_rho 0 --phase2_fusion ft --adapter_lr 1e-4" \
  DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
  PYTHON="$PY" bash run/matrix.sh 4>&-
done
N=$(find results/fus3_codebert -name 'fold*.json' 2>/dev/null | wc -l)
echo "########## FUSION_BASE xong $(ts) | $N/15 o ##########"
