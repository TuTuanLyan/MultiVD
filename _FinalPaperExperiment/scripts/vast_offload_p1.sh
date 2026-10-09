#!/bin/bash
# Giải phóng đĩa máy vast (người dùng 08/10 17:4x: "trên vast để ý dung lượng chọn checkpoint nhé"): với mỗi checkpoint Pha 1
# (best.pt + best_train_loss.pt, ~1,07 GB / ô) mà Pha 2 cùng fold ĐÃ có kết quả ở LOCAL 161 -> kéo về 161, so md5 hai đầu, khớp mới xoá
# bản trên vast. Bản tổng quát của pn2_offload_p1.sh (chỉ paper_night2), đọc ssh / cổng / thư mục từ runs.json qua remote_env.sh.
# Kết quả Pha 2 tra ở LOCAL 161 vì sync_remote.sh chuyển kết quả bằng --remove-source-files (results/ trên vast rỗng sau mỗi nhịp).
# Chỉ đụng p1_e30_* / p1_mb_* / p1_mbmix_* / p1_w20_* / p1_adanlr1_* (09/10 01:2x thêm cho phiên MultiVD Colab; bán kính nổ giới hạn), từ chối nếu một lượt định kéo > 4 thư mục.
# In dòng "ĐĨA <host>: còn X GB" ở cuối; còn < 3 GB thì in thêm "!! ĐĨA THẤP".
# Dùng: bash scripts/vast_offload_p1.sh <host> [--dry]
set -u
HOST=$1; DRY=${2:-}
BASE=$(cd "$(dirname "$0")/.." && pwd)
. "$BASE/scripts/remote_env.sh" "$HOST"
RM=${OUT%/_FinalPaperExperiment_staging}
LM=/drive1/cuongtm/ntat/MultiVD/model/mwonly5
LIST=$(timeout 120 ssh -n "${SSHO[@]}" "$SSH" "cd $RM/model/mwonly5 2>/dev/null && for d in p1_e30_*/seed_*/fold* p1_mb_*/seed_*/fold* p1_mbmix_*/seed_*/fold* p1_w20_*/seed_*/fold* p1_adanlr1_*/seed_*/fold*; do [ -s \$d/best.pt ] && echo \$d; done" | while read -r d; do
  r=${d%%/*}; f=${d##*/fold}; t=rasam_${r#p1_}   # p1_w20_x -> rasam_w20_x
  [ -s "$BASE/results/$t/fold$f.json" ] && echo "$d"
done)
disk () {
  local free; free=$(timeout 60 ssh -n "${SSHO[@]}" "$SSH" "df -BG --output=avail / | tail -1 | tr -dc 0-9")
  echo "ĐĨA $HOST: còn ${free:-?} GB"
  [ -n "$free" ] && [ "$free" -lt 3 ] && echo "!! ĐĨA THẤP $HOST: còn $free GB"
}
[ -z "$LIST" ] && { echo "vast_offload $HOST: không có checkpoint nào đủ điều kiện"; disk; exit 0; }
N=$(echo "$LIST" | wc -l)
# MAXN: trần bán kính nổ, mặc định 4; nâng lên chỉ sau khi đã liệt kê và kiểm tay danh sách (09/10 11:0x: 8 thư mục tồn sau lúc đăng nhập hết hạn, đĩa paper3 còn 0,8 GB)
MAXN=${MAXN:-4}
[ "$N" -gt "$MAXN" ] && { echo "vast_offload $HOST: $N thư mục (> $MAXN) - từ chối, kiểm tay"; disk; exit 2; }
for d in $LIST; do
  mkdir -p "$LM/$d"
  rsync -a -e "$RSH" "$SSH:$RM/model/mwonly5/$d/" "$LM/$d/" 2>/dev/null || { echo "!! rsync lỗi $d"; continue; }
  A=$(cd "$LM" && md5sum $d/*.pt | LC_ALL=C sort -k2)
  B=$(timeout 300 ssh -n "${SSHO[@]}" "$SSH" "cd $RM/model/mwonly5 && md5sum $d/*.pt | LC_ALL=C sort -k2")
  n=$(echo "$B" | grep -c "\.pt$")
  if [ -n "$A" ] && [ "$A" = "$B" ] && [ "$n" -ge 1 ] && [ "$n" -le 2 ]; then
    if [ "$DRY" = "--dry" ]; then echo "[dry] sẽ xoá $n file $d (md5 khớp)"; else timeout 60 ssh -n "${SSHO[@]}" "$SSH" "rm -f $RM/model/mwonly5/$d/*.pt" && echo "đã kéo + xoá $n file $d (md5 khớp)"; fi
  else
    echo "!! md5 LỆCH hoặc số file lạ ($n) ở $d - KHÔNG xoá"
  fi
done
disk
