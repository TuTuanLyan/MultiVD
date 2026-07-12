#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# push2vast.sh
# Push project MeiVD lên Vast bằng rsync.
#
# Có push:
#   - src/
#   - run/
#   - data/
#   - notebook/
#   - environment.yml
#   - các file nhỏ ở root
#
# Không push:
#   - model/
#   - result/
#   - log/
#   - __pycache__/
#   - checkpoint / weight nặng
# ============================================================


# =========================
# 1. LOCAL CONFIG
# =========================

# Script này nên đặt ở root MeiVD.
# LOCAL_ROOT tự động là folder chứa file push2vast.sh.
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# =========================
# 2. REMOTE TARGETS
# =========================
# Format:
# "target_name|ssh_user|ip_or_host|ssh_port|remote_project_dir|ssh_key"
#
# Nếu không dùng ssh key riêng, để trống field cuối.
# Ví dụ:
# "vast1|root|123.123.123.123|22|/workspace/MeiVD|"
#
# Với Vast thường dùng root + port riêng.
# Ví dụ:
# "vast1|root|ssh5.vast.ai|12345|/workspace/MeiVD|/home/ntat/.ssh/id_rsa"

TARGETS=(
  "vast1|root|171.235.191.213|10976|/workspace/MultiVD|"
  # "vast2|root|ANOTHER_VAST_IP_OR_HOST|ANOTHER_SSH_PORT|/workspace/MeiVD|/home/ntat/.ssh/id_rsa"
)


# =========================
# 3. OPTIONS
# =========================

# Chạy thử, không push thật:
# DRY_RUN=1 bash push2vast.sh
DRY_RUN="${DRY_RUN:-0}"

# Không bật delete mặc định để tránh xóa nhầm log/result/model trên Vast.
# Nếu muốn remote giống local hơn:
# DELETE_REMOTE=1 bash push2vast.sh
DELETE_REMOTE="${DELETE_REMOTE:-0}"

# Chọn target:
# bash push2vast.sh vast1
# bash push2vast.sh all
TARGET_NAME="${1:-all}"


# =========================
# 4. EXCLUDE RULES
# =========================

EXCLUDES=(
  "--exclude=model/"
  "--exclude=result/"
  "--exclude=log/"

  "--exclude=**/__pycache__/"
  "--exclude=*.pyc"
  "--exclude=.ipynb_checkpoints/"

  "--exclude=.git/"
  "--exclude=.venv/"
  "--exclude=venv/"
  "--exclude=env/"

  "--exclude=wandb/"
  "--exclude=runs/"
  "--exclude=outputs/"
  "--exclude=checkpoints/"

  "--exclude=*.pt"
  "--exclude=*.pth"
  "--exclude=*.ckpt"
  "--exclude=*.bin"
  "--exclude=*.safetensors"
  "--exclude=*.log"
)


# =========================
# 5. PUSH FUNCTION
# =========================

push_one_target() {
  local target="$1"

  IFS='|' read -r name ssh_user host port remote_dir ssh_key <<< "$target"

  if [[ "$TARGET_NAME" != "all" && "$TARGET_NAME" != "$name" ]]; then
    return 0
  fi

  echo "============================================================"
  echo "[TARGET] $name"
  echo "[LOCAL ] $LOCAL_ROOT/"
  echo "[REMOTE] ${ssh_user}@${host}:${remote_dir}/"
  echo "============================================================"

  SSH_OPTS=(
    -p "$port"
    -o StrictHostKeyChecking=accept-new
    -o ServerAliveInterval=30
    -o ServerAliveCountMax=10
  )

  if [[ -n "${ssh_key}" ]]; then
    SSH_OPTS+=(-i "$ssh_key")
  fi

  RSYNC_FLAGS=(
    -azP
    --human-readable
  )

  if [[ "$DRY_RUN" == "1" ]]; then
    RSYNC_FLAGS+=(--dry-run)
    echo "[MODE] DRY RUN: chưa push thật."
  fi

  if [[ "$DELETE_REMOTE" == "1" ]]; then
    RSYNC_FLAGS+=(--delete)
    echo "[MODE] DELETE_REMOTE=1: file remote không còn ở local sẽ bị xóa, trừ các exclude."
  fi

  echo "[INFO] Creating remote directory..."
  ssh "${SSH_OPTS[@]}" "${ssh_user}@${host}" "mkdir -p '${remote_dir}'"

  echo "[INFO] Start rsync..."
  rsync \
    "${RSYNC_FLAGS[@]}" \
    "${EXCLUDES[@]}" \
    -e "ssh ${SSH_OPTS[*]}" \
    "${LOCAL_ROOT}/" \
    "${ssh_user}@${host}:${remote_dir}/"

  echo "[DONE] Pushed to $name"
  echo
}


# =========================
# 6. MAIN
# =========================

echo "[INFO] Local project root: $LOCAL_ROOT"
echo "[INFO] Selected target: $TARGET_NAME"
echo

matched=0

for target in "${TARGETS[@]}"; do
  IFS='|' read -r name _ <<< "$target"

  if [[ "$TARGET_NAME" == "all" || "$TARGET_NAME" == "$name" ]]; then
    matched=1
  fi

  push_one_target "$target"
done

if [[ "$matched" == "0" ]]; then
  echo "[ERROR] Target '$TARGET_NAME' not found."
  echo "Available targets:"
  for target in "${TARGETS[@]}"; do
    IFS='|' read -r name _ <<< "$target"
    echo "  - $name"
  done
  exit 1
fi

echo "[ALL DONE]"