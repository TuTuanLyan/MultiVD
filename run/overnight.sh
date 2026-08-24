#!/usr/bin/env bash
# Hàng đợi qua đêm: nạp sẵn NHIỀU việc hơn một đêm làm được, chạy tuần tự.
#
# Vì sao thiết kế thế này: đã có tiền lệ GPU nằm không cả đêm vì việc "máy còn
# bận hay không" phụ thuộc vào một monitor bên ngoài, và monitor đó không được
# đặt lại. Ở đây GPU KHÔNG phụ thuộc bất cứ thứ gì bên ngoài — danh sách việc nằm
# trọn trên máy, và nó dài hơn một đêm. Hết việc là chuyện không xảy ra được.
#
# Một bước hỏng KHÔNG dừng hàng đợi: ghi lại rồi đi tiếp. Một backbone lỗi không
# được phép làm mất cả đêm của bốn backbone kia.
#
# Mọi bước đều idempotent (matrix.sh bỏ qua Phase 1 và fold đã có, các phép đo ghi
# ra file rồi bỏ qua nếu đã có), nên bật lại giữa chừng chỉ mất đúng job đang dở.
# Đó cũng là điều kiện để watchdog được phép bật lại một cách mù quáng.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

RUN_NAME="${RUN_NAME:?can RUN_NAME}"
BACKBONES="${BACKBONES:?can BACKBONES}"
SEED="${SEED:-42}"
# Seed phụ. MẶC ĐỊNH RỖNG, và đó là chủ ý.
#
# Quy trình sàng lọc của dự án xếp đa seed ở bước CUỐI: seed 42 trước, chứng minh
# phương pháp trên nhiều model đã, rồi mới loại trừ may rủi bằng seed. Lấy seed ra
# lấp thời gian GPU trống là làm sai thứ tự — nó nhân ba chi phí của một câu hỏi
# chưa đến lượt, trong khi câu hỏi đang cần trả lời là "phương pháp có chạy trên
# model khác không".
#
# Muốn thêm việc cho một máy nhanh thì thêm BACKBONE, không thêm seed.
EXTRA_SEEDS="${EXTRA_SEEDS:-}"
FOLDS="${FOLDS:-1 2 3 4 5}"
export PYTHON="${PYTHON:-/venv/main/bin/python}"
export HF_HOME="${HF_HOME:-/workspace/hf}"
STATE="${STATE:-/workspace/overnight_${RUN_NAME}}"
mkdir -p "$STATE"

# Dưới ngưỡng này thì dọn kho Phase 1 của các khối đã đo xong. Đĩa vast là 20–32 GB
# và mỗi checkpoint Phase 1 khoảng 470 MB, nên ba khối seed × 4 nhánh × N backbone
# đủ sức làm đầy đĩa và giết hàng đợi lúc 3 giờ sáng.
DISK_FLOOR_GB="${DISK_FLOOR_GB:-6}"

log() { echo "[$(date -u '+%F %T')] $*"; }

# Chạy một bước có đánh dấu hoàn thành. Bước đã xong thì bỏ qua; bước hỏng thì
# ghi lại và đi tiếp — KHÔNG dừng hàng đợi.
step() {
  local name="$1"; shift
  if [[ -f "$STATE/$name.done" ]]; then
    log "BO QUA $name (da xong)"
    return 0
  fi
  log "=== BAT DAU $name ==="
  if "$@"; then
    touch "$STATE/$name.done"
    log "=== XONG $name ==="
  else
    log "!!! $name HONG (ma loi $?) — di tiep buoc sau"
    echo "$(date -u '+%F %T') $name" >> "$STATE/failed.txt"
  fi
}

free_disk_if_needed() {
  local avail; avail=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  if (( avail < DISK_FLOOR_GB )); then
    log "dia con ${avail}GB < ${DISK_FLOOR_GB}GB — don kho Phase 1 cua khoi da do xong"
    for tag in "$STATE"/measured_*; do
      [[ -f "$tag" ]] || continue
      local dir; dir=$(cat "$tag")
      [[ -d "$dir" ]] && rm -rf "$dir" && log "  da xoa $dir"
    done
    df -h / | tail -1
  fi
}

matrix() {  # $1 = lambda, $2 = arm tag, $3 = modes, $4 = seed, $5 = cờ thêm cho Phase 1
  env RUN_NAME="$RUN_NAME" SEED="$4" FOLDS="$FOLDS" BACKBONES="$BACKBONES" \
      MODES="$3" OPTIMIZERS="recadam adamw" LAMBDA_CWE="$1" ARM_TAG="$2" \
      PHASE1_EXTRA="${5:-}" \
      CWE_VOCAB=fixed4 DATA_ROOT=data/sven_python_folds_norm TARGET_LANG=python \
      PHASE1_DATA_PATH=data/train_ccpp_js.jsonl \
      PYTHON="$PYTHON" HF_HOME="$HF_HOME" \
      bash run/matrix.sh
}

# Độ nhọn + dịch chuyển trọng số cho mọi backbone của một khối. Rẻ (chỉ nạp
# checkpoint, forward/backward, không huấn luyện) và đóng hai ô đang trống trong
# FACTS.md. Chạy NGAY SAU khối sinh ra checkpoint, trước khi kho bị dọn.
measure() {  # $1 = arm tag, $2 = seed
  # Ba dòng riêng, KHÔNG gộp thành một `local`: bash khai triển mọi từ của dòng
  # lệnh trước khi builtin `local` gán, nên `out="...${tag}..."` trên cùng dòng
  # sẽ đọc một `tag` chưa tồn tại và `set -u` giết cả script — đã xảy ra thật,
  # và watchdog khi đó bật lại một hàng đợi chết mãi không thôi.
  local tag="$1"
  local seed="$2"
  local out="$STATE/measure${tag}_seed${seed}.txt"
  : > "$out"
  local pairs=()
  for BB in $BACKBONES; do
    local L="${BB%%=*}" REST="${BB#*=}"; local MODEL="${REST%%:*}" POOL="${REST##*:}"
    local CK="model/$RUN_NAME/phase1/${L}__cwe${tag}/seed_${seed}/best.pt"
    [[ -f "$CK" ]] || { echo "bo qua $L: chua co $CK" >> "$out"; continue; }
    echo "===== do nhon: $L ($MODEL, $POOL) =====" >> "$out"
    $PYTHON src/measure_sharpness.py --checkpoint "$CK" --model_name "$MODEL" \
      --pooling "$POOL" --aux_mode cwe --seed "$seed" --batch_size 16 \
      --rho_mode absolute --rhos 0.01 0.05 0.1 0.2 --n_random 3 >> "$out" 2>&1
    pairs+=("${L}_cwe=${CK}=${MODEL}")
    local CN="model/$RUN_NAME/phase1/${L}__none/seed_${seed}/best.pt"
    [[ -f "$CN" ]] && pairs+=("${L}_none=${CN}=${MODEL}")
  done
  if (( ${#pairs[@]} )); then
    echo "===== dich chuyen trong so =====" >> "$out"
    $PYTHON src/measure_drift.py --pairs "${pairs[@]}" >> "$out" 2>&1
  fi
  echo "model/$RUN_NAME/phase1" > "$STATE/measured${tag}_seed${seed}"
  cat "$out"
}

# Mutex khoi dong bang KHOA FILE PID, khong quet ten tien trinh.
#
# Hai hang doi cung chay tren mot card 16 GB lam CA HAI OOM — do la nguyen nhan
# that su cua 332 job hong o lan chay dau, khong phai loi khoa hoc nao ca.
#
# Ban dau mutex nay quet `ps` tim tien trinh ten `bash run/overnight.sh`, va no
# TU CHAN CHINH MINH: moi lan bash fork cho mot command substitution, tien trinh
# con mang y nguyen dong lenh do nhung PID khac, nen phep loai tru theo `$$`
# khong bat duoc no. Khoa file khong co lop hong nay: PID duoc ghi ra tuong minh,
# va `kill -0` phan biet duoc khoa cu cua mot tien trinh da chet.
LOCK="$STATE/queue.pid"
if [[ -f "$LOCK" ]]; then
  OLD_PID=$(cat "$LOCK" 2>/dev/null)
  if [[ -n "$OLD_PID" && "$OLD_PID" != "$$" ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    log "DA CO hang doi dang chay (PID $OLD_PID) — thoat, khong chay chong len"
    exit 0
  fi
  log "khoa cu cua PID ${OLD_PID:-?} da chet — chiem lai"
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# Don ca hai tang con con sot lai: `bash run/matrix.sh` va tien trinh python.
# Giet moi tang python la chua du — matrix.sh con song se sinh job ke tiep ngay.
for ORPHAN in $(ps -eo pid,args --no-headers \
                | awk '$2=="bash" && $3 ~ /run\/matrix\.sh/ {print $1}'); do
  kill -9 "$ORPHAN" 2>/dev/null && log "don matrix.sh mo coi PID $ORPHAN truoc khi bat dau"
done
for ORPHAN in $(pgrep -f 'src/train_transfer\.py|src/train_baseline\.py' 2>/dev/null); do
  kill -9 "$ORPHAN" 2>/dev/null && log "don job mo coi PID $ORPHAN truoc khi bat dau"
done

log "########## HANG DOI QUA DEM — $RUN_NAME ##########"
log "backbone : $BACKBONES"
log "trang thai: $STATE"

# ---------------------------------------------------------------------------
# Thứ tự = thứ tự ưu tiên. Việc quan trọng nhất chạy trước, nên nếu đêm bị cắt
# ngắn thì thứ mất đi là thứ ít quan trọng nhất.
# ---------------------------------------------------------------------------

# 1. λ=0.2 — trục so được với toàn bộ số liệu cũ.
step "01_lambda020"  matrix 0.2  ""     "none cwe latent_bottleneck latent_proto" "$SEED"

# 2. Đo trên chính checkpoint vừa sinh, trước khi có nguy cơ bị dọn.
step "02_measure020" measure ""   "$SEED"

# 3. λ=0.05 — baseline và `none` dùng lại của bước 1 (cả hai độc lập với λ).
step "03_lambda005"  matrix 0.05 "_l05" "cwe latent_bottleneck latent_proto"      "$SEED"
step "04_measure005" measure "_l05" "$SEED"

# 5. λ HỌC ĐƯỢC — Kendall/Gal/Cipolla, CVPR 2018, arXiv:1705.07115.
#
# Hai vô hướng log-phương sai thay cho hằng số λ. Khởi tạo tại đúng λ=0.2 nên đây
# là mở rộng thật sự của khối 1, không phải một điểm xuất phát khác.
#
# `--aux_weight_lr 1e-2` KHÔNG phải con số tuỳ tiện: với lr chung 2e-5, đo được là
# qua cả một Phase 1 hai vô hướng đó chỉ dịch ~0.024, tức λ_eff đổi ~2% và thí
# nghiệm trả về một kết quả null vô nghĩa. Mô phỏng 1200 bước cho thấy 1e-2 tới
# đúng điểm cân bằng mà 5e-2 cũng tới.
#
# Baseline và `none` dùng lại của khối 1: với aux_mode=none thì không có loss phụ
# để cân, nên cách tính trọng số không đổi được gì.
#
# Đại lượng đáng đọc không phải riêng F1 mà là **λ_eff mà mô hình tự chọn**, ghi
# vào log mỗi epoch và vào checkpoint — nó so trực tiếp được với 0.2 và 0.05.
step "05_lambda_hoc_duoc" matrix 0.2 "_uw" "cwe latent_bottleneck latent_proto" "$SEED" \
     "--aux_weight_mode uncertainty --aux_weight_lr 1e-2"
step "06_measure_uw" measure "_uw" "$SEED"

free_disk_if_needed

# 7+. Seed phụ — CHỈ chạy khi EXTRA_SEEDS được đặt tường minh. Đây là cổng cuối
# của quy trình sàng lọc, không phải thứ dùng để lấp GPU trống.
for S in $EXTRA_SEEDS; do
  step "s${S}_lambda020"  matrix 0.2  ""     "none cwe latent_bottleneck latent_proto" "$S"
  step "s${S}_measure020" measure ""   "$S"
  step "s${S}_lambda005"  matrix 0.05 "_l05" "cwe latent_bottleneck latent_proto"      "$S"
  free_disk_if_needed
done

log "########## HET HANG DOI — moi buoc da chay ##########"
[[ -f "$STATE/failed.txt" ]] && { log "cac buoc hong:"; cat "$STATE/failed.txt"; }
touch "$STATE/ALL_DONE"
$PYTHON src/report_fold.py --prefix "${RUN_NAME}_" --seed "$SEED" || true
