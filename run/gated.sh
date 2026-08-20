#!/usr/bin/env bash
# Sàng lọc có cổng chặn: mỗi fold chạy baseline trước, rồi các phương pháp, rồi
# các pretrained cùng phương pháp — tất cả trên MỘT máy.
#
# Vì sao phải cùng máy: hai instance đang dùng hai GPU khác nhau (5070 Ti và
# 4070 Ti SUPER), và chênh lệch phần cứng đo được là 0.028 Macro-F1 — lớn hơn
# chính hiệu ứng đang đo. So một backbone chạy máy này với backbone chạy máy kia
# là so nhầm biến.
#
# Vì sao baseline chạy trước trong từng fold: mọi Δ đều phải quy về baseline của
# CHÍNH backbone đó, CHÍNH fold đó, CHÍNH máy đó. Chạy baseline sau hoặc tái dùng
# từ run khác là mở đường cho so sánh lệch.
#
# Cổng chặn sau fold 3:
#   xu hướng yếu  -> dừng, đi tìm phương án thay thế, không tốn nốt 2 fold
#   xu hướng mạnh -> chạy tiếp fold 4-5, rồi mới tính đến nhiều seed
#
# Ba fold để QUYẾT ĐỊNH DỪNG thì đủ, nhưng không đủ để KẾT LUẬN — tài liệu này có
# bốn lần tín hiệu n=3 đảo dấu ở n=5 (§30.1). Nên cổng này chỉ dùng theo một
# chiều: xu hướng yếu thì bỏ, xu hướng mạnh thì phải chạy đủ 5 fold mới được nói.
#
# Cách gọi:
#   BACKBONES="codebert=microsoft/codebert-base:cls t5p=Salesforce/codet5p-220m:cls" \
#   MODES="cwe latent_bottleneck none" SEED=42 RUN_NAME=gate1 bash run/gated.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source /venv/main/bin/activate 2>/dev/null || true
export HF_HOME="${HF_HOME:-/workspace/hf}"
PYTHON="${PYTHON:-python}"

RUN_NAME="${RUN_NAME:-gate}"
SEED="${SEED:-42}"
MODES="${MODES:-cwe latent_bottleneck none}"
BACKBONES="${BACKBONES:-codebert=microsoft/codebert-base:cls}"
DATA_ROOT="${DATA_ROOT:-data/sven_python_twin}"
TARGET_LANG="${TARGET_LANG:-python}"
PHASE1_DATA_PATH="${PHASE1_DATA_PATH:-data/train_ccpp_js.jsonl}"
GATE_FOLDS="${GATE_FOLDS:-1 2 3}"
REST_FOLDS="${REST_FOLDS:-4 5}"

CWE_VOCAB="${CWE_VOCAB:-fixed4}"
NUM_LATENT="${NUM_LATENT:-8}"
LATENT_TEMPERATURE="${LATENT_TEMPERATURE:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-15}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-30}"
LR="${LR:-2e-5}"
LAMBDA_CWE="${LAMBDA_CWE:-0.2}"

shared_args() {  # $1 = model_name, $2 = pooling
  echo --seed "$SEED" --batch_size "$BATCH_SIZE" --eval_batch_size "$BATCH_SIZE" \
       --max_length "$MAX_LENGTH" --truncation_strategy head_middle_tail \
       --weight_decay 0.01 --patience 5 --min_epochs 3 --max_grad_norm 1.0 --num_workers 0 \
       --data_root "$DATA_ROOT" --target_lang "$TARGET_LANG" \
       --model_name "$1" --pooling "$2"
}

# --- Phase 1: một lần cho mỗi (backbone, mode) -----------------------------
for BB in $BACKBONES; do
  LABEL="${BB%%=*}"; REST="${BB#*=}"; MODEL="${REST%%:*}"; POOL="${REST##*:}"
  for MODE in $MODES; do
    RN="${RUN_NAME}_${LABEL}"; METHOD="transfer_$MODE"
    MODEL_DIR="model/$RN/$METHOD/seed_$SEED"; mkdir -p "$MODEL_DIR/source"
    LOG="log/$RN/$METHOD/seed_$SEED"; mkdir -p "$LOG"
    mkdir -p "results/$RN/$METHOD/seed_$SEED"
    if [[ -f "$MODEL_DIR/source/best.pt" ]]; then
      echo "=== $(date '+%F %T') | phase1 $LABEL/$MODE | da co, bo qua ==="
      continue
    fi
    echo "=== $(date '+%F %T') | phase1 $LABEL/$MODE ==="
    $PYTHON -u src/train_transfer.py --phase phase1 \
      --run_name "$RN" --method_name "$METHOD" --data_path "$PHASE1_DATA_PATH" \
      --aux_mode "$MODE" --cwe_vocab "$CWE_VOCAB" --num_latent "$NUM_LATENT" \
      --latent_temperature "$LATENT_TEMPERATURE" \
      --epochs "$PHASE1_EPOCHS" --learning_rate "$LR" --lambda_cwe "$LAMBDA_CWE" \
      --checkpoint_path "$MODEL_DIR/source/best.pt" \
      $(shared_args "$MODEL" "$POOL") >> "$LOG/phase1.log" 2>&1 \
      || echo "  phase1 $LABEL/$MODE THAT BAI"
  done
done

run_fold() {  # $1 = fold
  local FOLD="$1"
  echo ""
  echo "############ FOLD $FOLD ############"

  # 1) baseline cho TỪNG backbone, chạy trước mọi phương pháp
  for BB in $BACKBONES; do
    local LABEL="${BB%%=*}" REST="${BB#*=}"; local MODEL="${REST%%:*}" POOL="${REST##*:}"
    local RN="${RUN_NAME}_${LABEL}"
    local BM="model/$RN/baseline/seed_$SEED/fold$FOLD"; mkdir -p "$BM"
    local BR="results/$RN/baseline/seed_$SEED"; mkdir -p "$BR"
    local BL="log/$RN/baseline/seed_$SEED"; mkdir -p "$BL"
    if [[ -f "$BR/fold$FOLD.json" ]]; then
      echo "=== $(date '+%F %T') | fold $FOLD | $LABEL baseline | da co ==="
    else
      echo "=== $(date '+%F %T') | fold $FOLD | $LABEL baseline ==="
      $PYTHON -u src/train_baseline.py --phase train \
        --run_name "$RN" --method_name baseline --fold "$FOLD" \
        --epochs "$PHASE2_EPOCHS" --learning_rate "$LR" \
        --checkpoint_path "$BM/best.pt" \
        $(shared_args "$MODEL" "$POOL") >> "$BL/train_fold$FOLD.log" 2>&1 \
        && $PYTHON -u src/train_baseline.py --phase infer \
          --run_name "$RN" --method_name baseline --fold "$FOLD" \
          --checkpoint_path "$BM/best.pt" \
          $(shared_args "$MODEL" "$POOL") >> "$BL/infer_fold$FOLD.log" 2>&1 \
        || echo "  $LABEL baseline fold$FOLD THAT BAI"
      [[ -f "$BR/fold$FOLD.json" ]] && rm -f "$BM/best.pt"
    fi
  done

  # 2) các phương pháp, rồi 3) các pretrained cùng phương pháp
  #    Vòng ngoài là MODE nên với mỗi phương pháp, mọi backbone chạy liền nhau —
  #    đúng thứ tự "phương pháp -> pretrained cùng phương pháp".
  for MODE in $MODES; do
    for BB in $BACKBONES; do
      local LABEL="${BB%%=*}" REST="${BB#*=}"; local MODEL="${REST%%:*}" POOL="${REST##*:}"
      local RN="${RUN_NAME}_${LABEL}" METHOD="transfer_$MODE"
      local MD="model/$RN/$METHOD/seed_$SEED"
      local RS="results/$RN/$METHOD/seed_$SEED"; mkdir -p "$RS"
      local LG="log/$RN/$METHOD/seed_$SEED"; mkdir -p "$LG"
      if [[ -f "$RS/fold$FOLD.json" ]]; then
        echo "=== $(date '+%F %T') | fold $FOLD | $LABEL/$MODE | da co ==="; continue
      fi
      if [[ ! -f "$MD/source/best.pt" ]]; then
        echo "  fold $FOLD $LABEL/$MODE bo qua, thieu checkpoint nguon"; continue
      fi
      echo "=== $(date '+%F %T') | fold $FOLD | $LABEL/$MODE ==="
      mkdir -p "$MD/fold$FOLD"
      $PYTHON -u src/train_transfer.py --phase phase2 \
        --run_name "$RN" --method_name "$METHOD" --fold "$FOLD" \
        --aux_mode "$MODE" --cwe_vocab "$CWE_VOCAB" --num_latent "$NUM_LATENT" \
        --latent_temperature "$LATENT_TEMPERATURE" \
        --epochs "$PHASE2_EPOCHS" --learning_rate "$LR" \
        --source_checkpoint "$MD/source/best.pt" \
        --checkpoint_path "$MD/fold$FOLD/best.pt" \
        --output_dir "results/$RN/$METHOD" \
        $(shared_args "$MODEL" "$POOL") >> "$LG/phase2_fold$FOLD.log" 2>&1 \
        && $PYTHON -u src/train_transfer.py --phase test \
          --run_name "$RN" --method_name "$METHOD" --fold "$FOLD" \
          --aux_mode "$MODE" --cwe_vocab "$CWE_VOCAB" --num_latent "$NUM_LATENT" \
          --checkpoint_path "$MD/fold$FOLD/best.pt" \
          --output_dir "results/$RN/$METHOD" \
          $(shared_args "$MODEL" "$POOL") >> "$LG/test_fold$FOLD.log" 2>&1 \
        || echo "  $LABEL/$MODE fold$FOLD THAT BAI"
      [[ -f "$RS/fold$FOLD.json" ]] && rm -f "$MD/fold$FOLD/best.pt"
    done
  done

  echo "---- bang sau fold $FOLD ----"
  RUN_PREFIX="$RUN_NAME" SEED="$SEED" $PYTHON src/report_gate.py || true
}

for FOLD in $GATE_FOLDS; do run_fold "$FOLD"; done

echo ""
echo "################ CONG CHAN SAU FOLD ${GATE_FOLDS// /,} ################"
RUN_PREFIX="$RUN_NAME" SEED="$SEED" $PYTHON src/report_gate.py --gate
touch "/workspace/${RUN_NAME}_GATE"

if [[ "${STOP_AT_GATE:-0}" == "1" ]]; then
  echo "STOP_AT_GATE=1 — dung lai de nguoi quyet dinh."
  exit 0
fi

for FOLD in $REST_FOLDS; do run_fold "$FOLD"; done

echo ""
echo "################ DU 5 FOLD ################"
RUN_PREFIX="$RUN_NAME" SEED="$SEED" $PYTHON src/report_gate.py --gate
touch "/workspace/${RUN_NAME}_DONE"
