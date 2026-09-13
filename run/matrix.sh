#!/usr/bin/env bash
# Ma trận chính: vòng ngoài là FOLD, trong mỗi fold chạy trọn baseline + mọi
# nhánh của MỌI backbone, rồi in bảng ngay.
#
# Vì sao vòng ngoài là fold chứ không phải backbone: mọi Δ đều quy về baseline
# của CHÍNH backbone đó, CHÍNH fold đó, CHÍNH máy đó — chênh lệch phần cứng đo
# được là 0.028 Macro-F1, lớn hơn hiệu ứng đang đo. Chạy trọn một fold rồi mới
# sang fold sau nghĩa là sau fold thứ hai đã có một bảng đọc được, và một nhánh
# hỏng bị phát hiện sau ~2 giờ thay vì sau cả đợt.
#
# Fold khác nhau chạy máy khác nhau thì HỢP LỆ và nhanh gấp đôi: đặt FOLDS="1 3 5"
# ở máy này và FOLDS="2 4" ở máy kia. Điều duy nhất cấm là lấy baseline máy này
# ghép với nhánh máy kia trong cùng một phép so — mà bố cục này không cho phép,
# vì baseline và nhánh của một fold luôn nằm cùng chỗ.
#
# ---------------------------------------------------------------------------
# Phase 1 nằm ở MỘT KHO DÙNG CHUNG, khoá theo (backbone, nhánh, seed):
#
#     model/$RUN/phase1/<backbone>__<mode>/seed_$SEED/best.pt
#
# Không sao chép giữa các thư mục nhánh nữa. Cách cũ — driver `cp` file Phase 1
# từ run này sang run kia — đã hỏng im lặng một lần: `sam-gate.sh` tìm Phase 1 ở
# `model/fam1_emb/...` trong khi run thật tên `emb1_emb`, lệnh cp trượt, và nhánh
# đó tự huấn luyện một Phase 1 KHÁC. Kết quả là phép so "chỉ đổi SAM" thực ra đổi
# hai biến, và với sd giữa các lần rút Phase 1 là 0.0521 thì hiệu ứng ~0.01 ở đó
# không đọc được. Kho dùng chung làm lỗi ấy không xảy ra được: có file thì dùng,
# không có thì huấn luyện, không có đường thứ ba.
#
# Phase 1 KHÔNG phụ thuộc fold (source không chia fold) và KHÔNG phụ thuộc
# optimizer của Phase 2. Nên một lần huấn luyện dùng cho cả 5 fold và cả hai
# optimizer — đó là lý do nó nằm ngoài vòng lặp fold.
# ---------------------------------------------------------------------------
#
# Cách gọi:
#   FOLDS="1 2 3 4 5" bash run/matrix.sh                 # đủ, một máy
#   FOLDS="1 3 5" RUN_NAME=m1 bash run/matrix.sh         # máy A
#   FOLDS="2 4"   RUN_NAME=m1 bash run/matrix.sh         # máy B, cùng RUN_NAME
#   BACKBONES="codebert=microsoft/codebert-base:cls" FOLDS=1 bash run/matrix.sh
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
PYTHON="${PYTHON:-python}"

# CHAN TRUOC, khong doi den luc hong: mot $PYTHON khong import noi torch se lam MOI cong
# tham do that bai, va truoc 08/09/2026 dieu do dan thang toi `rm -f` checkpoint Pha 1 tot.
# Kiem ngay tu dau va DUNG HAN, thay vi de ma tran chay tiep roi pha du thu.
if ! $PYTHON -c "import torch, numpy, sklearn" 2>/dev/null; then
  echo "!! DUNG: PYTHON='$PYTHON' khong import duoc torch/numpy/sklearn." >&2
  echo "   Dat duong tuyet doi cua env du an truoc khi chay, vi du:" >&2
  echo "     PYTHON=/home/ntat/miniconda3/envs/vdenv/bin/python   (161)" >&2
  echo "     PYTHON=/data/ntat/envs/vdenv/bin/python              (158)" >&2
  exit 2
fi

RUN_NAME="${RUN_NAME:-m1}"
SEED="${SEED:-42}"
FOLDS="${FOLDS-1 2 3 4 5}"   # KHONG dung :- ; FOLDS="" phai giu nguyen rong (giai doan chi-Phase-1)

# Năm backbone. t5p = bản BIMODAL, thống nhất trong nhóm.
# Ba checkpoint họ CodeT5+ (220m / 220m-bimodal / 110m-embedding) có cùng
# 84,954,240 tham số ngoài embedding, nên so giữa chúng là so PRETRAIN thuần.
BACKBONES="${BACKBONES:-\
codebert=microsoft/codebert-base:cls \
unixcoder=microsoft/unixcoder-base:cls \
t5=Salesforce/codet5-base:mean \
t5p=Salesforce/codet5p-220m-bimodal:mean \
t5pe=Salesforce/codet5p-110m-embedding:mean}"

MODES="${MODES:-none cwe latent_bottleneck latent_proto}"
# RecAdam và AdamW là HAI NHÁNH cạnh nhau trong cùng fold, không phải hai đợt
# chạy. Đặt cạnh nhau thì hiệu giữa chúng ghép cặp được theo fold và không dính
# chênh lệch phần cứng.
OPTIMIZERS="${OPTIMIZERS:-recadam adamw}"

DATA_ROOT="${DATA_ROOT:-data/sven_python_folds_norm}"
TARGET_LANG="${TARGET_LANG:-python}"
PHASE1_DATA_PATH="${PHASE1_DATA_PATH:-data/train_ccpp_js.jsonl}"
CWE_VOCAB="${CWE_VOCAB:-fixed4}"
LAMBDA_CWE="${LAMBDA_CWE:-0.2}"
NUM_LATENT="${NUM_LATENT:-8}"
LATENT_TEMPERATURE="${LATENT_TEMPERATURE:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PHASE1_EPOCHS="${PHASE1_EPOCHS:-15}"
PHASE2_EPOCHS="${PHASE2_EPOCHS:-30}"
LR="${LR:-2e-5}"
PATIENCE="${PATIENCE:-5}"
# Patience RIENG cho Pha 1. Rong => dung chung PATIENCE nhu cu, khong doi mot byte.
#
# Vi sao can tach: Pha 1 `codebert x com` co adapter nam NGAY RANH GIOI giua "hoc duoc" va
# "doan mot lop". Do duoc 13/09: hai lan chay cung seed 42, cung ma, cung phien ban thu vien,
# bam sat nhau toi epoch 5 (train loss lech < 0.003) roi tach o epoch 6 — lan may man co mot
# nhip val loss giam nen patience reset va no thoat cao nguyen (val 0.5583); lan kia khong co
# nhip do, can patience 5 o epoch 7 va dung lai o 0.3333. Noi patience cho RIENG Pha 1 la
# dung cho no du cho thoat, ma KHONG doi dieu kien dung cua Pha 2 — noi ca hai thi phep so
# giua cac nhanh doi them mot bien.
PHASE1_PATIENCE="${PHASE1_PATIENCE:-}"
MIN_EPOCHS="${MIN_EPOCHS:-3}"
# Hậu tố gắn vào TÊN NHÁNH và vào khoá kho Phase 1. Dùng khi một run cần chứa hai
# giá trị của cùng một siêu tham số — ví dụ λ=0.2 và λ=0.05 — mà vẫn DÙNG CHUNG
# baseline và nhánh `none`.
#
# Vì sao dùng chung được, và vì sao đó không phải lối tắt cẩu thả: baseline không
# đọc dữ liệu source và không có λ; còn `none` thì src/train.py cho aux_loss = None
# nên λ KHÔNG xuất hiện trong hàm loss. Hai thứ đó giống hệt ở mọi λ.
#
# Cách đúng là để chúng nằm CHUNG một thư mục run và được bỏ qua vì đã tồn tại —
# KHÔNG phải `cp` kết quả từ run này sang run kia. Chính lối `cp` đó đã hỏng im
# lặng ở sam-gate.sh và làm một phép so đổi ba biến thay vì một.
ARM_TAG="${ARM_TAG:-}"

# Khoa cua kho Phase 1. MAC DINH bang ARM_TAG, nhung phai TACH DUOC.
#
# Mot can thiep chi dung toi Phase 2 - SAM la vi du - co Phase 1 GIONG HET khoi
# goc. Neu khoa Phase 1 di theo ten nhanh thi nhanh `_sam` se khong tim thay kho,
# tu huan luyen mot Phase 1 KHAC, va phep so "chi doi SAM" thuc ra doi hai bien.
# Do dung la chuyen da xay ra voi run `same_emb`, noi Phase 1 va ca baseline deu
# khac ma khong ai thay. Dat PHASE1_TAG="" cho cac can thiep Phase 2 de chung
# DUNG LAI dung checkpoint cua khoi goc.
PHASE1_TAG="${PHASE1_TAG-$ARM_TAG}"
PHASE1_EXTRA="${PHASE1_EXTRA:-}"
PHASE2_EXTRA="${PHASE2_EXTRA:-}"

PHASE1_STORE="${PHASE1_STORE:-model/$RUN_NAME/phase1}"   # ghi de duoc: khoi chi-doi-Pha-2 dung lai kho cua khoi khac

# Đếm job hỏng và TRẢ VỀ MÃ LỖI KHÁC 0 ở cuối. Không có cái này thì một đợt hỏng
# 332/375 job vẫn thoát 0, và bất cứ lớp điều phối nào ở trên cũng ghi nó là "đã
# xong" rồi không bao giờ chạy lại. Đó là cách một lỗi tranh GPU biến thành một
# khoảng trống dữ liệu im lặng.
FAILED=0

# Nhật ký từng job. Trước đây output đi thẳng /dev/null nên "THAT BAI" không kèm
# lý do, và phải chạy lại mới biết vì sao — mà chạy lại thì thường không tái hiện.
JOBLOG="log/$RUN_NAME"
mkdir -p "$JOBLOG"

# Giảm phân mảnh bộ nhớ GPU; các lần OOM đã gặp đều báo có vùng reserved-nhưng-
# chưa-dùng.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

shared_args() {  # $1 = model_name, $2 = pooling
  echo --seed "$SEED" --batch_size "$BATCH_SIZE" --eval_batch_size "$BATCH_SIZE" \
       --max_length "$MAX_LENGTH" --truncation_strategy head_middle_tail \
       --weight_decay 0.01 --patience "$PATIENCE" --min_epochs "$MIN_EPOCHS" \
       --max_grad_norm 1.0 --num_workers 0 \
       --data_root "$DATA_ROOT" --target_lang "$TARGET_LANG" \
       --model_name "$1" --pooling "$2"
}

# Checkpoint Phase 1 CÓ TỒN TẠI chưa đủ để dùng lại — phải xem nó có gì bên trong.
#
# Một job Phase 1 bị giết giữa chừng ở phiên bản cũ (ghi thẳng vào best.pt) để lại
# một file hợp lệ về mặt cú pháp nhưng dừng ở epoch 1. Đo được thật: bốn file như
# vậy còn sót sau đợt tranh GPU, trong đó `t5pe__none` có val 0.4309 — DƯỚI mức
# ngẫu nhiên. Và `none` là nhánh đối chứng của mọi nhánh khác, nên nó hỏng thì cả
# backbone đó vô nghĩa mà không có dấu hiệu gì để nhận ra từ kết quả.
#
# Hai tiêu chí, cả hai đều bảo thủ để không vứt nhầm một lần rút yếu nhưng thật:
#   best_epoch <= 1  -> gần như chắc chắn là file bị cắt ngang
#   val < 0.55       -> ngang ngẫu nhiên, không dùng được dù vì lý do gì
# Ma thoat cua phep tham do duoi day:
#   0 = dung duoc | 2 = doc duoc nhung SUY BIEN | 3 = FILE hong (torch.load nem)
#   77 = MOI TRUONG hong (khong import noi torch/numpy) — TUYET DOI khong duoc xoa gi
#
# 08/09/2026: chay run/int1.sh ma QUEN dat PYTHON, nen `python` la conda base khong co
# numpy. Phep tham do that bai vi ModuleNotFoundError, cong doc thanh "checkpoint hong"
# va `rm -f "$CKPT"` XOA MAT hai checkpoint Pha 1 tot (4cwe, com). Khoi phuc duoc tu 158,
# md5 khop, nhung neu 158 khong con ban thi mat 2 x 40 phut GPU va ca chuoi so sanh.
# CLAUDE.md muc 3 da noi "cong phai phan biet file hong voi chat luong kem" — thieu ve
# thu ba: MOI TRUONG hong. Ba truong hop, ba xu ly khac nhau.
phase1_usable() {
  [[ -f "$1" ]] || return 1
  $PYTHON - "$1" <<'PYEOF' 2>/dev/null
import sys
try:
    import torch
except Exception:
    sys.exit(77)          # MOI TRUONG hong — nguoi goi khong duoc xoa gi
try:
    ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
except Exception:
    sys.exit(3)           # FILE that su hong
epoch = ck.get("best_epoch") or 0
val = float(ck.get("best_val_macro_f1") or 0.0)
# Cong nay de bat checkpoint SUY BIEN, khong phai de xep hang chat luong.
#
# Nguong cu `val >= 0.55` da tu choi codebert/latent_bottleneck/com o 0.549872 —
# truot 0.000128, tuc nho hon SAN NHIEU do duoc cua chinh du an (0.010, chay lai
# cung seed cung cau hinh khac may) khoang 78 lan. Mot checkpoint huan luyen 13/15
# epoch va hoi tu binh thuong bi vut di vi mot hieu so ma ta da chung minh la
# khong phan biet noi voi viec chay lai dung mot thu. Te hon, no chi loai
# `latent_bottleneck` o nhung nguon nhanh do YEU, nen bang ket qua chi con cho
# no manh — thien lech chon loc.
#
# Cai can chan la checkpoint doan MOT LOP: tren bai nhi phan can bang, macro-F1
# cua nguoi doan bua roi vao ~0.33-0.40. Nguong 0.40 chan dung nhung ca do
# (0.3333 cua codebert__none_sam1r01, 0.3403 cua none/full, 0.3432 cua
# latent_proto/full) ma khong dung toi vung 0.5+ noi ket qua con nghia ly.
import os
MIN_VAL = float(os.environ.get("PHASE1_MIN_VAL", "0.40"))
# PHASE1_MIN_VAL=0 -> chay HET, ke ca checkpoint suy bien. Dung khi muc tieu la
# bang ket qua day du: mot o bi cong chan la mot o TRONG, khong viet duoc gi vao
# bai; con mot o co so kem kem theo val Phase 1 = 0.34 thi doc duoc ngay la
# "Phase 1 sap", va do la mot dong ket qua that.
# Nguong EPOCH cung phai chinh duoc, khong chi nguong val.
#
# `best_epoch <= 1` von de bat file BI CAT NGANG. Nhung 08/09 no loai nham dung thu ma thi
# nghiem POOL1 sinh ra de do: nguon pha loang manh lam Pha 1 hoi tu ngay epoch 1 roi khong
# tot len nua — do la KET QUA, khong phai file hong. File hong thi `torch.load` nem (ma 3);
# file nay doc duoc binh thuong. Hai chuyen khac nhau, phai tach.
#
# Mac dinh 2 = giu nguyen hanh vi cu. PHASE1_MIN_EPOCH=0 de chay het, dung cho khoi nao
# CO Y do checkpoint suy bien (CLAUDE.md muc 3).
# Mac dinh SUY RA TU MIN_VAL: dat PHASE1_MIN_VAL=0 la da noi ro "toi muon giu ca checkpoint
# suy bien", nen nguong epoch phai tu tat theo. Neu de mac dinh cung la 2 thi driver DANG
# CHAY (doc moi truong cu, chi co MIN_VAL=0) van bi loai nham — dung chuyen da xay ra
# 08/09 voi pur25_n930: va xong, dong bo xong, ma no van bi doi thanh .rejected lan hai.
MIN_EPOCH = int(os.environ.get("PHASE1_MIN_EPOCH", "0" if MIN_VAL <= 0 else "2"))
sys.exit(0 if epoch >= MIN_EPOCH and val > MIN_VAL else 2)
PYEOF
}

banner() {
  echo ""
  echo "################################################################"
  echo "  $*"
  echo "################################################################"
}

banner "MA TRAN — run $RUN_NAME · seed $SEED · fold: $FOLDS"
echo "  backbone   : $(echo "$BACKBONES" | tr ' ' '\n' | cut -d= -f1 | tr '\n' ' ')"
echo "  nhanh      : $MODES"
echo "  optimizer  : $OPTIMIZERS"
echo "  source     : $PHASE1_DATA_PATH   lambda $LAMBDA_CWE   vocab $CWE_VOCAB"
[[ -n "$ARM_TAG" ]] && echo "  hau to nhanh: $ARM_TAG"
[[ "$PHASE1_TAG" != "$ARM_TAG" ]] && echo "  kho Phase 1 : hau to [$PHASE1_TAG] - DUNG LAI cua khoi khac"
echo "  target     : $DATA_ROOT ($TARGET_LANG)"
echo "  Phase 1 kho: $PHASE1_STORE"

# ---------------------------------------------------------------------------
# Phase 1 — một lần cho mỗi (backbone, nhánh). Ngoài vòng fold, ngoài vòng
# optimizer, vì nó không phụ thuộc cả hai.
# ---------------------------------------------------------------------------
banner "PHASE 1 — kho dung chung"
for BB in $BACKBONES; do
  LABEL="${BB%%=*}"; REST="${BB#*=}"; MODEL="${REST%%:*}"; POOL="${REST##*:}"
  for MODE in $MODES; do
    CKPT="$PHASE1_STORE/${LABEL}__${MODE}${PHASE1_TAG}/seed_$SEED/best.pt"
    # DA BI TU CHOI MOT LAN -> KHONG huan luyen lai.
    #
    # Cong chat luong doi ten checkpoint hong thanh `.rejected`. Lan goi sau,
    # matrix.sh thay `best.pt` khong ton tai va huan luyen LAI tu dau — voi cung
    # seed, cung du lieu, cung sieu tham so, nen no sap y het roi lai bi tu choi.
    # Vong lap nay chay MOI FOLD. Do duoc ngay 27/08: codebert/none/full va
    # codebert/latent_proto/full moi cai ~30 phut, 2 nhanh x 5 fold = ~5 gio dot
    # vo ich, va no la ly do ntat tut lai sau hai may kia.
    if [[ -f "${CKPT}.rejected" ]]; then
      echo "=== $(date -u '+%F %T') | phase1 $LABEL/$MODE | DA BI TU CHOI truoc do, bo qua (xoa .rejected neu muon thu lai) ==="
      continue
    fi
    if [[ -f "$CKPT" ]]; then
      # BAT MA THOAT TRUC TIEP, khong dung `if ...; fi` roi doc $? — bash tra 0 sau `fi`
      # khi dieu kien sai, nen ma 77 se bien mat va cong lai xoa nham. Da thu hai chieu.
      phase1_usable "$CKPT"; rc=$?
      if (( rc == 0 )); then
        echo "=== $(date -u '+%F %T') | phase1 $LABEL/$MODE | da co, dung lai ==="
        continue
      fi
      if (( rc == 77 )); then
        echo "!! MOI TRUONG HONG: $PYTHON khong import duoc torch. KHONG xoa gi, DUNG HAN."
        echo "   Dung duong tuyet doi cua env du an, vd PYTHON=/home/ntat/miniconda3/envs/vdenv/bin/python"
        exit 2
      fi
      echo "=== $(date -u '+%F %T') | phase1 $LABEL/$MODE | CO NHUNG HONG (ma $rc), huan luyen lai ==="
      rm -f "$CKPT"
    fi
    mkdir -p "$(dirname "$CKPT")"
    echo "=== $(date -u '+%F %T') | phase1 $LABEL/$MODE ==="
    # Ghi vào đường dẫn tạm rồi mới đổi tên: một job Phase 1 bị giết giữa chừng
    # KHÔNG được để lại file mà lần chạy sau coi là "đã có".
    #
    # Đây không phải phòng xa. Lần khởi động lại đầu tiên để lại
    # `t5__latent_bottleneck/best.pt` ở epoch 1 với val 0.3846 — dưới mức ngẫu
    # nhiên — và nếu nó được dùng lại thì cả 10 job của nhánh đó (5 fold × 2
    # optimizer) sẽ thừa hưởng một checkpoint hỏng, và kết quả trông y hệt như
    # "nhánh latent_bottleneck không hợp với backbone này".
    PART="${CKPT}.partial"
    rm -f "$PART"
    $PYTHON -u src/train_transfer.py --phase phase1 \
      --run_name "$RUN_NAME" --method_name "phase1_${LABEL}_${MODE}${PHASE1_TAG}" \
      --data_path "$PHASE1_DATA_PATH" \
      --aux_mode "$MODE" --cwe_vocab "$CWE_VOCAB" --num_latent "$NUM_LATENT" \
      --latent_temperature "$LATENT_TEMPERATURE" \
      --epochs "$PHASE1_EPOCHS" --learning_rate "$LR" --lambda_cwe "$LAMBDA_CWE" \
      --checkpoint_path "$PART" \
      $PHASE1_EXTRA \
      $(shared_args "$MODEL" "$POOL") \
      ${PHASE1_PATIENCE:+--patience "$PHASE1_PATIENCE"} > "$JOBLOG/phase1_${LABEL}_${MODE}${PHASE1_TAG}.log" 2>&1
    tail -3 "$JOBLOG/phase1_${LABEL}_${MODE}${PHASE1_TAG}.log" 2>/dev/null | sed "s/^/    /"
    if [[ -f "$PART" ]]; then
      # Cong chat luong phai chay o CA HAI duong, khong chi duong tai dung.
      #
      # Truoc day `phase1_usable` chi duoc goi khi checkpoint DA CO san tu truoc.
      # Mot lan rut MOI hoan tat binh thuong thi duoc `mv` thang vao cho, du no
      # vo dung. Da xay ra that: `codebert__none_sam1r01` ket o val 0.3333 (doan
      # mot lop), duoc cong bo, va 6 job Phase 2 chay tren no cho ra -0.43 —
      # trong y het mot ket qua that.
      #
      # Doi ten thanh .rejected chu khong xoa: bang chung con lai, nhung glob tim
      # `best.pt` khong bat duoc nua nen khong nhanh nao thua huong no.
      if phase1_usable "$PART"; then
        mv "$PART" "$CKPT"
      else
        mv "$PART" "${CKPT}.rejected"
        FAILED=$((FAILED + 1))
        echo "  !! phase1 $LABEL/$MODE KHONG DAT (best_epoch<=1 hoac val<=0.40 — suy bien) — da doi thanh ${CKPT}.rejected"
        grep -E "New best model" "$JOBLOG/phase1_${LABEL}_${MODE}${PHASE1_TAG}.log" 2>/dev/null | tail -1 | sed "s/^/       /"
      fi
    else
      # Phase 1 hong PHAI tinh la hong, khong duoc im lang.
      #
      # Truoc day no chi in mot dong roi di tiep, va hau qua la: moi nhanh bi bo
      # qua vi thieu checkpoint, khong job Phase 2 nao chay, bo dem hong van bang
      # 0, matrix.sh thoat 0, va lop tren ghi ca khoi la DA XONG trong khi no sinh
      # ra dung 0 ket qua. Da xay ra that: ca 9 Phase 1 cua khoi lambda-hoc-duoc
      # chet vi src/ chua duoc day len may, va khoi do van duoc danh dau hoan thanh.
      FAILED=$((FAILED + 1))
      echo "  !! phase1 $LABEL/$MODE THAT BAI — xem $JOBLOG/phase1_${LABEL}_${MODE}${PHASE1_TAG}.log"
      tail -3 "$JOBLOG/phase1_${LABEL}_${MODE}${PHASE1_TAG}.log" 2>/dev/null | sed "s/^/       /"
    fi
  done
done

# ---------------------------------------------------------------------------
# Phase 2 — vòng ngoài là fold.
# ---------------------------------------------------------------------------
run_fold() {
  local FOLD="$1"
  banner "FOLD $FOLD"

  # 1) baseline của TỪNG backbone, trước mọi nhánh: mọi Δ quy về nó.
  #
  # SKIP_BASELINE=1 — HOAN baseline lai, chay nhanh method truoc (nguoi dung neu 13/09).
  # Dung khi dang THU MOT THU MOI: neu method da thap hon nguong biet truoc thi khoi
  # ton GPU chay baseline, vi method thua roi. MAC DINH 0 => duong chay cu khong doi.
  # Baseline chay bu sau bang chinh lenh nay voi SKIP_BASELINE=0: vong lap tren tu bo
  # qua o da co, nen khong chay lai gi.
  for BB in $BACKBONES; do
    [[ "${SKIP_BASELINE:-0}" == 1 ]] && break
    local LABEL="${BB%%=*}" REST="${BB#*=}"; local MODEL="${REST%%:*}" POOL="${REST##*:}"
    local RN="${RUN_NAME}_${LABEL}"
    local RES="results/$RN/baseline/seed_$SEED"; mkdir -p "$RES"
    if [[ -f "$RES/fold$FOLD.json" ]]; then
      echo "=== fold $FOLD | $LABEL baseline | da co ==="; continue
    fi
    local CK="model/$RN/baseline/seed_$SEED/fold$FOLD"; mkdir -p "$CK"
    echo "=== $(date -u '+%F %T') | fold $FOLD | $LABEL baseline ==="
    # BASELINE_EXTRA: co truyen them CHO RIENG nhanh doi chung, song song voi PHASE2_EXTRA.
    # Mac dinh RONG => duong chay cu khong doi mot byte. Can cho moi khoi ma doi chung phai
    # nhan CUNG mot can thiep voi nhanh chinh — vi du quet co tap train (--max_train_samples):
    # neu chi nhanh chuyen giao bi cat du lieu thi phep so doi HAI bien, khong con doc duoc.
    $PYTHON -u src/train_baseline.py --phase train \
      --run_name "$RN" --method_name baseline --fold "$FOLD" \
      --epochs "$PHASE2_EPOCHS" --learning_rate "$LR" --checkpoint_path "$CK/best.pt" \
      ${BASELINE_EXTRA:-} \
      $(shared_args "$MODEL" "$POOL") >> "$JOBLOG/${LABEL}_baseline_fold${FOLD}.log" 2>&1 \
      && $PYTHON -u src/train_baseline.py --phase infer \
        --run_name "$RN" --method_name baseline --fold "$FOLD" \
        --checkpoint_path "$CK/best.pt" \
        ${BASELINE_SOURCE_EVAL:+--source_eval_data "$BASELINE_SOURCE_EVAL"} \
        $(shared_args "$MODEL" "$POOL") >> "$JOBLOG/${LABEL}_baseline_fold${FOLD}.log" 2>&1
    # KEEP_CKPT=1: giu checkpoint Pha 2 lai. Mac dinh van XOA — model/ da 41GB va
    # dia tung day 98% roi lam cut torch.save (§16). Chi bat cho khoi noi suy
    # trong so, va phai don tay sau.
    [[ "${KEEP_CKPT:-0}" == 1 ]] || rm -rf "$CK"
    if [[ ! -f "$RES/fold$FOLD.json" ]]; then
      FAILED=$((FAILED + 1))
      echo "  !! $LABEL baseline fold$FOLD THAT BAI — xem $JOBLOG/${LABEL}_baseline_fold${FOLD}.log"
      tail -2 "$JOBLOG/${LABEL}_baseline_fold${FOLD}.log" | sed 's/^/       /'
    fi
  done

  # 2) mọi nhánh. Vòng trong là backbone nên với mỗi (nhánh, optimizer) thì các
  #    backbone chạy liền nhau — dễ đọc "phương pháp này có phụ thuộc pretrained
  #    không" ngay trong lúc chạy.
  for MODE in $MODES; do
    for OPT in $OPTIMIZERS; do
      # Hau to theo optimizer. Phai la "khac recadam thi them ten" chu KHONG phai
      # "== adamw thi them _adamw": them optimizer thu ba (spd, 07/09) ma quen cho nay
      # thi nhanh spd ghi de dung len nhanh recadam va mat ca hai.
      local SUFFIX=""; [[ "$OPT" != "recadam" ]] && SUFFIX="_$OPT"
      for BB in $BACKBONES; do
        local LABEL="${BB%%=*}" REST="${BB#*=}"; local MODEL="${REST%%:*}" POOL="${REST##*:}"
        local RN="${RUN_NAME}_${LABEL}" ARM="transfer_${MODE}${ARM_TAG}${SUFFIX}"
        local RES="results/$RN/$ARM/seed_$SEED"; mkdir -p "$RES"
        if [[ -f "$RES/fold$FOLD.json" ]]; then
          echo "=== fold $FOLD | $LABEL/${MODE}${ARM_TAG}/$OPT | da co ==="; continue
        fi
        local SRC="$PHASE1_STORE/${LABEL}__${MODE}${PHASE1_TAG}/seed_$SEED/best.pt"
        if [[ ! -f "$SRC" ]]; then
          FAILED=$((FAILED + 1))
          echo "  fold $FOLD $LABEL/${MODE}${ARM_TAG}/$OPT THAT BAI — thieu Phase 1 $SRC"; continue
        fi
        local CK="model/$RN/$ARM/seed_$SEED/fold$FOLD"; mkdir -p "$CK"
        echo "=== $(date -u '+%F %T') | fold $FOLD | $LABEL/${MODE}${ARM_TAG}/$OPT ==="
        $PYTHON -u src/train_transfer.py --phase phase2 \
          --run_name "$RN" --method_name "$ARM" --fold "$FOLD" \
          --aux_mode "$MODE" --cwe_vocab "$CWE_VOCAB" --num_latent "$NUM_LATENT" \
          --latent_temperature "$LATENT_TEMPERATURE" \
          --epochs "$PHASE2_EPOCHS" --learning_rate "$LR" \
          --phase2_optimizer "$OPT" \
          --source_checkpoint "$SRC" --checkpoint_path "$CK/best.pt" \
          --output_dir "results/$RN/$ARM" \
          $PHASE2_EXTRA \
          $(shared_args "$MODEL" "$POOL") >> "$JOBLOG/${LABEL}_${ARM}_fold${FOLD}.log" 2>&1 \
          && $PYTHON -u src/train_transfer.py --phase test \
            --run_name "$RN" --method_name "$ARM" --fold "$FOLD" \
            --aux_mode "$MODE" --cwe_vocab "$CWE_VOCAB" --num_latent "$NUM_LATENT" \
            --checkpoint_path "$CK/best.pt" --output_dir "results/$RN/$ARM" \
            ${SOURCE_EVAL_DATA:+--source_eval_data "$SOURCE_EVAL_DATA"} \
            ${SOURCE_INTERP_GRID:+--source_interp_grid "$SOURCE_INTERP_GRID"} \
            $(shared_args "$MODEL" "$POOL") >> "$JOBLOG/${LABEL}_${ARM}_fold${FOLD}.log" 2>&1
        # KEEP_CKPT=1: giu checkpoint Pha 2 lai. Mac dinh van XOA — model/ da 41GB
        # va dia tung day 98% roi lam cut torch.save (§16). Chi bat cho khoi noi
        # suy trong so, va phai don tay sau.
        [[ "${KEEP_CKPT:-0}" == 1 ]] || rm -rf "$CK"
        if [[ ! -f "$RES/fold$FOLD.json" ]]; then
          FAILED=$((FAILED + 1))
          echo "  !! $LABEL/${MODE}${ARM_TAG}/$OPT fold$FOLD THAT BAI — xem $JOBLOG/${LABEL}_${ARM}_fold${FOLD}.log"
          tail -2 "$JOBLOG/${LABEL}_${ARM}_fold${FOLD}.log" | sed 's/^/       /'
        fi
      done
    done
  done

  banner "BANG SAU FOLD $FOLD"
  $PYTHON src/report_fold.py --prefix "${RUN_NAME}_" --seed "$SEED" || true
}

for FOLD in $FOLDS; do run_fold "$FOLD"; done

banner "XONG — fold: $FOLDS   (job hong: $FAILED)"
$PYTHON src/report_fold.py --prefix "${RUN_NAME}_" --seed "$SEED" || true
if (( FAILED > 0 )); then
  echo "!! $FAILED job khong sinh ra ket qua — TRA VE MA LOI de lop tren khong ghi la da xong"
  exit 1
fi
touch "${DONE_FLAG:-/tmp/${RUN_NAME}${ARM_TAG}_DONE}"
