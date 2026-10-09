#!/bin/bash
# Giải phóng đĩa paper_night2 (vast 54653423, 23 GB) cho khối 📄E (người dùng 08/10: tab "Kết quả paper (30 epoch pha 1)"):
# với mỗi checkpoint Pha 1 p1_e30_* mà Pha 2 rasam_e30_* cùng fold ĐÃ có kết quả trên paper_night2 -> kéo về 161, so md5 hai đầu,
# khớp mới xoá bản trên vast. Chỉ đụng thư mục p1_e30_* / p1_mb_* / p1_mbmix_* / p1_w20_* (bán kính nổ giới hạn; 08/10 07:3x thêm p1_mb_*, 15:2x thêm p1_w20_* cho tab warmup 0,2), từ chối nếu một lượt định xoá > 8 file.
# Dùng: bash scripts/pn2_offload_p1.sh [--dry]
set -u
DRY=${1:-}
SSH="ssh -o ConnectTimeout=20 -p 36826 root@202.122.49.242"
RM=/workspace/MultiVD; ROUT=$RM/_FinalPaperExperiment_staging
LM=/drive1/cuongtm/ntat/MultiVD/model/mwonly5
# danh sách (run seed fold) có best.pt Pha 1 trên vast; Pha 2 "đã xong" tra ở KẾT QUẢ LOCAL 161, vì sync_remote.sh chuyển kết quả
# bằng --remove-source-files nên trên vast thư mục results rỗng sau mỗi nhịp (lỗi bản đầu 04:3x: không bao giờ thấy ô nào đủ điều kiện)
LB=/drive1/cuongtm/ntat/MultiVD/_FinalPaperExperiment
LIST=$($SSH "cd $RM/model/mwonly5 && for d in p1_e30_*/seed_*/fold* p1_mb_*/seed_*/fold* p1_mbmix_*/seed_*/fold* p1_w20_*/seed_*/fold*; do [ -s \$d/best.pt ] && echo \$d; done" 2>/dev/null | while read -r d; do
  r=${d%%/*}; f=${d##*/fold}; t=rasam_${r#p1_}   # p1_e30_x -> rasam_e30_x, p1_mb_x -> rasam_mb_x
  [ -s "$LB/results/$t/fold$f.json" ] && echo "$d"
done)
[ -z "$LIST" ] && { echo "pn2_offload: không có checkpoint nào đủ điều kiện"; $SSH "df -h /workspace | tail -1" 2>/dev/null; exit 0; }
N=$(echo "$LIST" | wc -l)
[ "$N" -gt 4 ] && { echo "pn2_offload: $N thư mục (> 4) - từ chối, kiểm tay"; exit 2; }
for d in $LIST; do
  mkdir -p "$LM/$d"
  rsync -a -e "ssh -p 36826" "root@202.122.49.242:$RM/model/mwonly5/$d/" "$LM/$d/" 2>/dev/null || { echo "!! rsync lỗi $d"; continue; }
  A=$(cd "$LM" && md5sum $d/*.pt | LC_ALL=C sort -k2)
  B=$($SSH "cd $RM/model/mwonly5 && md5sum $d/*.pt | LC_ALL=C sort -k2" 2>/dev/null)
  n=$(echo "$B" | grep -c "\.pt$")
  if [ -n "$A" ] && [ "$A" = "$B" ] && [ "$n" -ge 1 ] && [ "$n" -le 2 ]; then
    if [ "$DRY" = "--dry" ]; then echo "[dry] sẽ xoá $n file $d (md5 khớp)"; else $SSH "rm -f $RM/model/mwonly5/$d/*.pt" 2>/dev/null && echo "đã kéo + xoá $n file $d (md5 khớp)"; fi
  else
    echo "!! md5 LỆCH hoặc số file lạ ($n) ở $d - KHÔNG xoá"
  fi
done
$SSH "df -h /workspace | tail -1" 2>/dev/null
