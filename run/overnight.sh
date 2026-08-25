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

# $@ = cac HAU TO kho Phase 1 duoc phep xoa ("_l05", "_uw", ...). Khong truyen
# thi xoa moi kho da do xong.
#
# Vi sao phai co doi so: kho hau to rong ("") la NGUON cua khoi SAM-Phase-2, vi
# khoi do chi doi Phase 2 nen PHASE1_TAG="". Mot lan don dia khong phan biet se
# xoa dung thu khoi sau con dang doc, va hong kieu do khong de lai dau hieu gi.
free_disk_if_needed() {
  local avail; avail=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
  (( avail >= DISK_FLOOR_GB )) && return 0
  local tags=("$@"); (( ${#tags[@]} )) || tags=("*")
  log "dia con ${avail}GB < ${DISK_FLOOR_GB}GB - don kho Phase 1 DA DO xong: ${tags[*]}"
  # Xoa TUNG thu muc da ghi, khong quet sach ca kho.
  #
  # Ban dau file danh dau chi chua "model/$RUN_NAME/phase1", tuc CA KHO, nen mot
  # lan don dia se xoa MOI checkpoint - ke ca cai ma khoi sau con dang can. Dem
  # nay tinh co vo hai vi thu tu buoc, nhung chi can doi thu tu hoac bat them seed
  # la no xoa dung thu dang dung, va hong kieu do khong de lai dau hieu gi.
  local tag marker dir
  for tag in "${tags[@]}"; do
    for marker in "$STATE"/measured${tag}_seed*; do
      [[ -f "$marker" ]] || continue
      while IFS= read -r dir; do
        [[ -n "$dir" && -d "$dir" ]] || continue
        rm -rf "$dir" && log "  da xoa $dir"
      done < "$marker"
    done
  done
  df -h / | tail -1
}

matrix() {  # $1 lambda, $2 arm tag, $3 modes, $4 seed, $5 co Phase1, $6 co Phase2, $7 khoa Phase1
  env RUN_NAME="$RUN_NAME" SEED="$4" FOLDS="$FOLDS" BACKBONES="$BACKBONES" \
      MODES="$3" OPTIMIZERS="recadam adamw" LAMBDA_CWE="$1" ARM_TAG="$2" \
      PHASE1_TAG="${7-$2}" \
      PHASE1_EXTRA="${5:-}" PHASE2_EXTRA="${6:-}" \
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
  # Ghi ra DUNG cac thu muc thuoc khoi nay, moi dong mot cai, de free_disk chi
  # xoa phan da do xong chu khong quet sach ca kho.
  : > "$STATE/measured${tag}_seed${seed}"
  local BB2 L2 m2 d2
  for BB2 in $BACKBONES; do
    L2="${BB2%%=*}"
    for m2 in none cwe latent_bottleneck latent_proto; do
      d2="model/$RUN_NAME/phase1/${L2}__${m2}${tag}"
      [[ -d "$d2" ]] && echo "$d2" >> "$STATE/measured${tag}_seed${seed}"
    done
  done
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
# Danh muc bước. Thứ tự KHÔNG cố định trong file này nữa — nó là biến `STEPS`,
# vì hai máy đang ở hai chỗ khác nhau trong cùng danh mục và ép chung một thứ tự
# thì một trong hai phải bỏ dở việc đang chạy.
#
# Tên bước là khóa của cờ `.done`, nên ĐỔI TÊN một bước = chạy lại bước đó.
# ---------------------------------------------------------------------------
run_step() {
  case "$1" in
    # λ=0.2 — trục so được với toàn bộ số liệu cũ.
    01_lambda020)
      step "$1" matrix 0.2 "" "none cwe latent_bottleneck latent_proto" "$SEED" ;;
    # Đo trên chính checkpoint vừa sinh, trước khi có nguy cơ bị dọn.
    02_measure020)
      step "$1" measure "" "$SEED" ;;
    # λ=0.05 — baseline và `none` dùng lại của khối λ=0.2 (cả hai độc lập với λ).
    03_lambda005)
      step "$1" matrix 0.05 "_l05" "cwe latent_bottleneck latent_proto" "$SEED" ;;
    04_measure005)
      step "$1" measure "_l05" "$SEED" ;;

    # SAM o PHASE 1 — Watts et al., ICML 2026, arXiv:2605.02105.
    #
    # Bai do dat SAM o giai doan PRETRAIN va bao checkpoint thu duoc quen it hon
    # toi 80% khi fine-tune ve sau, tren mo hinh 20M-150M tham so, dung dai cua
    # du an. Moi run SAM cua du an truoc day deu o Phase 2. Bai KHONG so truc
    # tiep hai cho dat, nen day la o trong that.
    #
    # Khac 07_sam o dung mot cho: SAM nam o PHASE1_EXTRA thay vi PHASE2_EXTRA.
    # Va vi Phase 1 doi thi kho checkpoint phai RIENG — PHASE1_TAG mac dinh bang
    # ARM_TAG ("_sam1") nen dieu do tu dung; KHONG duoc truyen "" o day.
    #
    # Doi chung dung la khoi 01 (lambda=0.2, khong SAM o dau ca) tren CHINH may
    # nay: cung lambda, cung Phase 2, chi khac SAM o Phase 1.
    05_sam_phase1)
      step "$1" matrix 0.2 "_sam1" "none cwe latent_bottleneck latent_proto" "$SEED" \
           "--sam_rho 0.05" ;;
    # Do nhon cua chinh checkpoint SAM-Phase-1: SAM co that su cho cuc tieu phang
    # hon khong. Dong thang voi FACTS muc 4 (do nhon KHONG du bao transfer) — neu
    # SAM lam phang that ma transfer khong doi, muc 4 duoc xac nhan lan hai.
    06_measure_sam1)
      step "$1" measure "_sam1" "$SEED" ;;

    # SAM o PHASE 2 - Foret et al., ICLR 2021, ban port o src/sam.py tu ma JAX goc.
    #
    # PHASE1_TAG="" la chi tiet QUAN TRONG NHAT o day: SAM chi dung toi Phase 2,
    # nen Phase 1 cua no phai la DUNG checkpoint cua khoi lambda=0.2, khong phai
    # mot lan rut moi. Khong tach khoa nay thi phep so "chi doi SAM" doi hai bien,
    # va do dung la cach run `same_emb` da hong ma khong ai thay.
    #
    # Chay ca hai optimizer vi docs/SAM_REFERENCE.md canh bao SAM va RecAdam CHONG
    # LAN nhau - ca hai deu sua buoc cap nhat. SAM+AdamW moi la phep thu sach.
    #
    # rho=0.05 tuyet doi, dung quy uoc bai bao. Moi buoc 2 luot forward-backward.
    # KHONG co buoc do rieng: Phase 1 cua no CHINH LA kho lambda=0.2, da do o 02.
    07_sam)
      step "$1" matrix 0.2 "_sam" "none cwe latent_bottleneck latent_proto" "$SEED" \
           "" "--sam_rho 0.05" "" ;;

    # λ HOC DUOC — Kendall/Gal/Cipolla, arXiv:1705.07115. KHONG con trong thu tu
    # mac dinh: da bac o 9/9 nhanh va bac vi ly do CAU TRUC, xem DEAD_ENDS muc 13
    # (cuc tieu cua exp(-s)L + s/2 cho w=0.5/L, nen lambda_eff = L_bin/L_aux —
    # trong so toi uu ti le NGHICH voi loss, va mot head 4 lop tren 1284 dong luon
    # la task loss thap). Giu lai de chay duoc bang STEPS neu can them bang chung.
    08_lambda_hoc_duoc)
      step "$1" matrix 0.2 "_uw" "cwe latent_bottleneck latent_proto" "$SEED" \
           "--aux_weight_mode uncertainty --aux_weight_lr 1e-2" ;;
    09_measure_uw)
      step "$1" measure "_uw" "$SEED" ;;

    # Don dia GIUA cac buoc. Chi duoc xoa kho da do xong VA khong buoc sau nao con
    # doc: "_l05" va "_uw" la ngo cut/da do, con "" thi 07_sam van dang can.
    don_dia_l05_uw)
      free_disk_if_needed "_l05" "_uw" ;;

    *)
      log "!!! buoc khong biet: $1 — bo qua"
      echo "$(date -u '+%F %T') $1 KHONG-BIET" >> "$STATE/failed.txt" ;;
  esac
}

# Thu tu mac dinh. Dat STEPS de doi tren tung may.
STEPS="${STEPS:-01_lambda020 02_measure020 03_lambda005 04_measure005 \
don_dia_l05_uw 05_sam_phase1 06_measure_sam1 07_sam}"

for STEP_NAME in $STEPS; do
  run_step "$STEP_NAME"
done

# Van CHI don hai kho ngo-cut/da-do. Ban truoc goi khong doi so o day, va tren
# mot may co marker cu (dong duy nhat "model/m1/phase1", tuc CA KHO) thi mot lan
# don dia se xoa sach ca checkpoint lambda=0.2 lan checkpoint SAM-Phase-1 truoc
# khi kip keo ve. Kho nao con dang gia thi khong tu dong xoa.
free_disk_if_needed "_l05" "_uw"


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
