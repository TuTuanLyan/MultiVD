#!/usr/bin/env bash
# DEM THEO TEN CHUONG TRINH (comm), khong theo dong lenh: dong lenh cua chinh minh
# (bash/ssh/git) co chua chuoi 'tools/wblend.py' se bi mau cu dem vao -> bao 'may dang
# ban' GIA, ma huong nguy hiem la no CHE MAT mot may thuc su dang trong. CLAUDE.md muc 8.
# fleet_status.sh — mot man hinh trang thai cho ca ba may. Doc nhanh, khong sua gi.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/endpoints.sh
R=/workspace/MultiVD
echo "===== $(date -u '+%F %T UTC') / $(date '+%H:%M gio VN') ====="
for L in ntat ntat2; do
  VAST_CACHE_TTL=1 read -r H P <<< "$(vast_endpoint "$L" 2>/dev/null)" || true
  if [[ -z "${H:-}" || "${P:-None}" == "None" ]]; then
    # PHAI phan biet "khong giai duoc dia chi" voi "may khong con chay". Gop lam mot
    # thi mot truc trac mang thoang qua doc y HET nhu may da mat — va quyet dinh sai
    # o day la HUY NHAM mot may dang lam viec. Xay ra 08/09/2026 22:15 UTC: ntat2 bao
    # "KHONG giai duoc" trong khi no dang chay GPU 86%.
    ST=$(vast_state "$L" 2>/dev/null || echo "?")
    # vast_state in dang "<cur_state>/<actual_status>", vi du "running/running" —
    # so bang `==` se TRUOT va cong lai bao nham may chet. Khop tien to thay vi bang.
    if [[ "$ST" == running/* ]]; then
      echo "$L: DANG CHAY nhung chua giai duoc dia chi — truc trac tam thoi, TUYET DOI KHONG huy"
    else
      echo "$L: khong giai duoc dia chi | trang thai instance = $ST"
    fi
    continue
  fi
  out=$(timeout 60 ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=15 -p "$P" root@"$H" \
    "cd $R 2>/dev/null || exit 1
     echo \"  job=\$(ps -eo comm=,args= | grep -cE '^python[0-9.]*[[:space:]].*(src/train_|tools/wblend)') worklist=\$(ps -eo args --no-headers|grep -c '[v]ast_worklist.sh') vram=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)MiB\"
     echo \"  muc xong: \$(grep -xF -f <(grep -v '^\s*#\|^\s*\$' log/worklist.txt) log/worklist.done 2>/dev/null | grep -c .)/\$(grep -vc '^\s*#\|^\s*\$' log/worklist.txt 2>/dev/null)  (dem SO GIAO done∩todo — dem tho bi thoi phong boi dong cua danh sach cu)\"
     echo \"  o: int1=\$(ls results/int1_t5p/*/seed_*/fold*.json 2>/dev/null|wc -l) asam1=\$(ls results/asam1_t5p/*/seed_*/fold*.json 2>/dev/null|wc -l)\"
     echo \"  dang chay: \$(grep 'CHAY:' log/worklist.log 2>/dev/null | tail -1 | cut -c1-84)\"
     echo \"  con lai:\"; grep -v '^\s*#\|^\s*\$' log/worklist.txt 2>/dev/null | while read -r l; do
        grep -Fqx \"\$l\" log/worklist.done 2>/dev/null || echo \"    - \$l\"; done" 2>/dev/null) || true
  echo "$L  ($H:$P)"; [[ -n "$out" ]] && echo "$out" || echo "  KHONG SSH DUOC"
done
echo "161 (local)"
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null|head -1)
echo "  job=$(ps -eo comm=,args= | grep -cE '^python[0-9.]*[[:space:]].*(src/train_|tools/wblend)') ft2=$(ps -eo args --no-headers|grep -c '[r]un/ft2.sh') vram=${used}MiB"
tail -1 log/ft2.log 2>/dev/null | sed 's/^/  ft2: /'
echo "158"
ssh -o BatchMode=yes -o ConnectTimeout=8 tranmanhcuong@112.137.129.158 \
  'echo "  job=$(ps -eo comm=,args= | grep -cE \"^python[0-9.]*[[:space:]].*(src/train_|tools/wblend)\") vram=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)MiB"' 2>/dev/null || echo "  khong ssh duoc"
