#!/usr/bin/env bash
# Tải hết về rồi HỦY các instance vast. Chạy được độc lập, không cần phiên Claude.
#
# Vì sao script này tồn tại: hủy là BẮT BUỘC vào giờ nghỉ, nhưng mọi cơ chế tự
# động khác (monitor, cron) đều chết theo phiên. Đây là bản chạy tay một lệnh.
#
# KHÔNG hủy nếu đối chiếu không khớp — thà trả tiền thêm còn hơn mất dữ liệu.
#
# --------------------------------------------------------------------------
# Ba lỗi của bản trước, cả ba đều sẽ hỏng IM LẶNG đúng lúc cần nhất:
#
#  1. Hardcode `id:port:host` của máy hôm trước. IP công khai của vast ĐỔI khi
#     instance được dời máy chủ, và id thì đổi mỗi lần thuê lại. Bản cũ còn giữ
#     id 48356959/48357013 — hai máy không còn tồn tại. Nay giải theo NHÃN.
#  2. Kéo `model/*/source/best.pt`, trong khi kho hiện tại là
#     `model/*/phase1/<backbone>__<nhánh>/seed_*/best.pt`. Nó sẽ tải về 0
#     checkpoint, in `checkpoint nguon tai ve=0`, rồi vẫn HỦY vì phép đối chiếu
#     chỉ nhìn file kết quả.
#  3. Ghi vào `results_$LABEL`/`model_$LABEL`, khác hẳn thư mục mà
#     `pull_results.sh` và `build_records.py` đọc. Dữ liệu về đúng đĩa nhưng
#     không cây phân tích nào thấy.
#
# Nay: dùng thẳng `pull_results.sh --with-phase1` cho phần tải, nên chỉ có MỘT
# định nghĩa về "kéo về đâu", rồi đối chiếu CẢ kết quả LẪN checkpoint Phase 1
# trước khi hủy.
# --------------------------------------------------------------------------
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh

LABELS="${LABELS:-ntat ntat2}"
FORCE="${FORCE:-0}"
ACTION="${ACTION:-destroy}"        # destroy | stop | none

# DUNG HANG DOI TRUOC KHI KEO. Khong co buoc nay thi phep doi chieu KHONG BAO GIO
# khop: may van sinh file moi trong luc rsync chay, nen lan nao cung "thieu" vai
# file va script tu choi huy mai mai. Do dung la thu da xay ra o lan chay thu.
#
# Giet ca ba tang (watchdog -> overnight.sh -> matrix.sh -> python) va loc theo VI
# TRI, khong dung `pkill -f`: dong lenh SSH cua chinh phien nay cung chua cac chuoi
# do, va pkill -f se giet luon phien dang chay lenh.
if [[ "${STOP_QUEUES:-1}" == "1" ]]; then
  echo "########## DUNG HANG DOI — $(date -u '+%F %T') UTC ##########"
  for LABEL in $LABELS; do
    read -r H P <<< "$(vast_endpoint "$LABEL")" || continue
    [[ -z "${H:-}" ]] && continue
    echo "--- $LABEL ---"
    timeout 60 ssh -o StrictHostKeyChecking=no -o BatchMode=yes -p "$P" "root@$H" '
      for pat in watchdog overnight "run/matrix.sh"; do
        for pid in $(ps -eo pid,args --no-headers | awk -v p="$pat" '"'"'$2=="bash" && $3 ~ p {print $1}'"'"'); do
          kill -9 "$pid" 2>/dev/null && echo "  giet [$pat] PID $pid"
        done
      done
      for pid in $(pgrep -f "src/train_transfer\.py|src/train_baseline\.py" 2>/dev/null); do
        kill -9 "$pid" 2>/dev/null && echo "  giet job PID $pid"
      done
      sleep 3; echo "  con lai: $(ps -eo args --no-headers | grep -c "[s]rc/train_") job"
    ' 2>/dev/null | grep -v "^Welcome\|^Have fun\|^AI agents"
  done
fi

echo "########## TAI VE TRUOC KHI HUY — $(date -u '+%F %T') UTC ##########"
bash scripts/pull_results.sh --with-phase1

all_ok=1
for LABEL in $LABELS; do
  echo ""
  echo "=========== $LABEL ==========="
  if ! read -r HOST PORT <<< "$(vast_endpoint "$LABEL")" || [[ -z "${HOST:-}" ]]; then
    echo "  KHONG GIAI DUOC DIA CHI (trang thai: $(vast_state "$LABEL")) — KHONG lam gi"
    all_ok=0; continue
  fi
  SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o BatchMode=yes -p $PORT"
  ID=$(python3 -c "
import json,sys
for i in json.load(open('$VAST_CACHE')):
    if i.get('label')=='$LABEL': print(i.get('id')); break")
  [[ -z "$ID" ]] && { echo "  khong tim duoc id"; all_ok=0; continue; }

  # Thư mục local tương ứng — phải khớp với bảng trong pull_results.sh.
  case "$LABEL" in
    ntat2) LRES=results_m1 ;;
    ntat)  LRES=results_n1 ;;
    *)     LRES="results_$LABEL" ;;
  esac
  LMOD="model_run_$LABEL"

  # LC_ALL=C bắt buộc: `sort` trên máy thuê và `sort` ở đây có thể dùng locale
  # khác nhau, và khi đó `comm` đọc nhầm thứ tự rồi báo thiếu file không hề
  # thiếu. Đã xảy ra thật: ntat2 bị báo "thieu=15" trong khi đối chiếu lại là
  # 223/223 — suýt bỏ lỡ một lần hủy, và nếu tin ngược lại thì đã hủy khi thiếu thật.
  cmp_set() {  # $1 = lệnh find từ xa, $2 = thư mục local, $3 = tên để in
    local rem loc miss n
    rem=$(timeout 120 $SSH "root@$HOST" "$1" 2>/dev/null | LC_ALL=C sort)
    loc=$( eval "$2" 2>/dev/null | LC_ALL=C sort )
    n=$(echo "$rem" | grep -c . || true)
    miss=$(LC_ALL=C comm -23 <(echo "$rem") <(echo "$loc") | grep -c . || true)
    echo "  $3: tren may=$n  thieu o local=$miss"
    [[ "$miss" == "0" && "$n" -gt 0 ]]
  }

  ok=1
  cmp_set \
    'cd /workspace/MultiVD/results 2>/dev/null && find . -name "fold*.json" -printf "%p %s\n"' \
    "cd $LRES && find . -name 'fold*.json' -printf '%p %s\n'" \
    "ket qua" || ok=0
  cmp_set \
    'cd /workspace/MultiVD/model 2>/dev/null && find . -path "*phase1*" -name "best.pt" -printf "%p\n"' \
    "cd $LMOD && find . -path '*phase1*' -name 'best.pt' -printf '%p\n'" \
    "checkpoint Phase 1" || ok=0

  if [[ "$ok" == "1" || "$FORCE" == "1" ]]; then
    case "$ACTION" in
      destroy) echo "  -> HUY instance $ID"; vastai destroy instance "$ID" -y ;;
      stop)    echo "  -> STOP instance $ID"; vastai stop instance "$ID" ;;
      none)    echo "  -> doi chieu KHOP, khong lam gi (ACTION=none)" ;;
    esac
  else
    echo "  -> KHONG $ACTION: doi chieu khong khop. Sua roi chay lai, hoac FORCE=1."
    all_ok=0
  fi
done

echo ""
[[ "$all_ok" == "1" ]] && echo "TAT CA KHOP." || echo "CO MAY CHUA XU LY — doc lai o tren."
exit $(( all_ok == 1 ? 0 : 1 ))
