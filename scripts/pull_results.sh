#!/usr/bin/env bash
# Kéo kết quả từ các máy vast về local.
#
# Gọi khi MỘT KHỐI vừa xong, không theo nhịp đồng hồ. Máy mượn khi cần trả vẫn
# chạy nốt được tác vụ dở, nên không có lý do gì phải đồng bộ dồn dập; kéo theo
# ranh giới công việc thì mỗi lần kéo tương ứng một đơn vị kết quả đọc được.
#
# Kho Phase 1 nặng (~470 MB mỗi checkpoint) nên KHÔNG kéo mặc định; thêm
# `--with-phase1` trước khi trả máy, để lần chạy sau không phải huấn luyện lại.
#
# Không dùng --delete: đây là gộp từ nhiều máy về một cây, xoá theo nguồn sẽ xoá
# mất phần của máy kia.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WITH_PHASE1=0
[[ "${1:-}" == "--with-phase1" ]] && WITH_PHASE1=1

# Địa chỉ giải theo nhãn tại thời điểm gọi; xem scripts/endpoints.sh.
source "$(dirname "${BASH_SOURCE[0]}")/endpoints.sh"

# nhãn|thư mục từ xa|thư mục local
MACHINES=(
  "ntat2|/workspace/MultiVD|results_m1"
  "ntat|/workspace/MultiVD|results_n1"
)

for M in "${MACHINES[@]}"; do
  IFS='|' read -r NAME RDIR LDIR <<< "$M"
  mkdir -p "$LDIR"
  if ! read -r HOST PORT <<< "$(vast_endpoint "$NAME")" || [[ -z "${HOST:-}" ]]; then
    echo "$(date -u '+%F %T') $NAME: KHONG GIAI DUOC DIA CHI — instance $(vast_state "$NAME")"
    continue
  fi
  SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o BatchMode=yes -p $PORT"

  # Phan biet "may khong voi toi duoc" voi "may song nhung chua co ket qua".
  # Gop hai truong hop nay lam mot se bao dong gia moi lan hang doi con o Phase 1,
  # va khi bao dong that thi khong ai tin nua.
  if ! $SSH "root@$HOST" true 2>/dev/null; then
    echo "$(date -u '+%F %T') $NAME: KHONG KET NOI DUOC — may tat, doi mang, hoac da tra"
    continue
  fi

  BEFORE=$(find "$LDIR" -name 'fold*.json' 2>/dev/null | wc -l)
  if $SSH "root@$HOST" "test -d $RDIR/results" 2>/dev/null; then
    rsync -az -e "$SSH" "root@$HOST:$RDIR/results/" "$LDIR/" 2>/dev/null
    AFTER=$(find "$LDIR" -name 'fold*.json' 2>/dev/null | wc -l)
    echo "$(date -u '+%F %T') $NAME: $AFTER ket qua (+$((AFTER - BEFORE)))"
  else
    echo "$(date -u '+%F %T') $NAME: song, chua co ket qua nao (con o Phase 1)"
  fi

  # Nhật ký từng job — nhẹ, và là thứ duy nhất giải thích được vì sao một job hỏng.
  rsync -az -e "$SSH" "root@$HOST:$RDIR/log/" "log_run_$NAME/" 2>/dev/null || true
  # Bản ghi trạng thái hàng đợi và các phép đo.
  rsync -az -e "$SSH" --include='*/' --include='*.txt' --include='*.done' --exclude='*' \
        "root@$HOST:$RDIR/../overnight_"* "records/overnight_$NAME/" 2>/dev/null || \
  rsync -az -e "$SSH" --include='*/' --include='*.txt' --include='*.done' --exclude='*' \
        "root@$HOST:$RDIR/overnight_"* "records/overnight_$NAME/" 2>/dev/null || true

  if (( WITH_PHASE1 )); then
    echo "  ... keo kho Phase 1 cua $NAME (nang, vai phut)"
    mkdir -p "model_run_$NAME"
    # CHI kho Phase 1. Thu muc model/ con chua checkpoint Phase 2 cua TUNG fold
    # (5 fold x moi nhanh), nang gap nhieu lan va khong dung lai duoc — Phase 2
    # phai chay lai tu dau moi khi doi bat cu thu gi.
    rsync -az --partial -e "$SSH" \
      --include='*/' --include='phase1/**/best.pt' --exclude='*' \
      "root@$HOST:$RDIR/model/" "model_run_$NAME/" 2>/dev/null \
      && echo "  $(find "model_run_$NAME" -path '*phase1*' -name best.pt | wc -l) checkpoint Phase 1, $(du -sh "model_run_$NAME" | cut -f1)"
  fi
done

# Dem TAT CA cay results, khong chi hai cai. Ban truoc bo sot results_n1 va
# results/ (run local), nen con so in ra thap hon thuc te ma khong ai biet.
echo "tong ket qua da ve local: $(find results results_* -name 'fold*.json' 2>/dev/null | wc -l)"
