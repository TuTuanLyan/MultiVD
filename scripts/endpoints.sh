#!/usr/bin/env bash
# Giải địa chỉ SSH của một instance vast theo NHÃN, tại thời điểm gọi.
#
# Vì sao không hardcode: IP công khai của instance ĐỔI khi nó được dời máy chủ.
# Đã xảy ra thật — ntat2 nhảy từ 1.54.247.106 sang 42.112.118.114 giữa buổi, và
# mọi script hardcode IP cũ lập tức timeout trong khi máy vẫn chạy bình thường.
# Đọc nhầm cái timeout đó thành "máy chết" là cách dễ nhất để hủy nhầm một máy
# đang làm việc.
#
# Cổng lấy từ bảng `ports` khoá `22/tcp`, ghép với `public_ipaddr`. KHÔNG ghép
# cổng đó với `ssh_host` (ssh*.vast.ai) — đó là hai đường vào khác nhau, ghép
# chéo thì connection refused.
#
# Cách dùng:
#   source scripts/endpoints.sh
#   read -r HOST PORT <<< "$(vast_endpoint ntat2)"
set -uo pipefail

VAST_CACHE="${VAST_CACHE:-/tmp/vast_endpoints.json}"
VAST_CACHE_TTL="${VAST_CACHE_TTL:-120}"   # giây

_vast_refresh() {
  local age=99999
  if [[ -f "$VAST_CACHE" ]]; then
    age=$(( $(date +%s) - $(stat -c %Y "$VAST_CACHE" 2>/dev/null || echo 0) ))
  fi
  if (( age > VAST_CACHE_TTL )); then
    vastai show instances --raw > "$VAST_CACHE.tmp" 2>/dev/null && mv "$VAST_CACHE.tmp" "$VAST_CACHE"
  fi
}

# In "host port" cho nhãn $1; không in gì nếu không tìm thấy hoặc máy không chạy.
vast_endpoint() {
  _vast_refresh
  [[ -f "$VAST_CACHE" ]] || return 1
  python3 - "$VAST_CACHE" "$1" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
for inst in data:
    if inst.get("label") != sys.argv[2]:
        continue
    if inst.get("cur_state") != "running":
        sys.exit(1)
    ports = (inst.get("ports") or {}).get("22/tcp") or []
    ip = inst.get("public_ipaddr")
    if ip and ports and ports[0].get("HostPort"):
        print(ip, ports[0]["HostPort"])
        sys.exit(0)
sys.exit(1)
PY
}

# In trạng thái thô của một nhãn: running/stopped/khong-ton-tai.
vast_state() {
  _vast_refresh
  python3 - "$VAST_CACHE" "$1" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    print("khong-doc-duoc"); sys.exit(0)
for inst in data:
    if inst.get("label") == sys.argv[2]:
        print(f"{inst.get('cur_state')}/{inst.get('actual_status')}")
        sys.exit(0)
print("khong-ton-tai")
PY
}

# In INSTANCE ID cua mot nhan. Dung truoc moi lenh stop/destroy — KHONG BAO GIO
# hardcode id, vi id doi khi thue may moi ma nhan thi giu nguyen; huy nham bang
# id cu la huy mot may dang lam viec (hoac cua nguoi khac).
# Chi tra ve id cho nhan thuoc ho `ntat`; nhan khac tra ve rong va thoat != 0.
vast_id() {
  case "$1" in
    ntat|ntat[0-9]*) ;;
    *) echo "" ; return 1 ;;
  esac
  _vast_refresh
  python3 - "$VAST_CACHE" "$1" <<'PYID'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(1)
for inst in data:
    if inst.get("label") == sys.argv[2]:
        print(inst.get("id")); sys.exit(0)
sys.exit(1)
PYID
}
