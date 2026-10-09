#!/bin/bash
# Một nhịp cập nhật artifact khối mwonly5 (mỗi 30 phút, người dùng 04/10): kéo kết quả xong từ 158 / vast về 161, dựng dòng `folds`
# (xong: fpe.py dbrows; đang chạy: live_all.sh), rồi artifact_push.py plan -> dòng "WRITES [...]" để phiên điều phối gửi bằng ArtifactData batch.
cd "$(dirname "$0")/.." || exit 1
PY=/home/ntat/miniconda3/envs/vdenv/bin/python
# 05/10 13:1x: paper_mw, paper_mw2 đã destroy - chỉ còn đồng bộ 158
# 07/10 21:1x: thêm paper_night2 (54653423); 17:0x: paper_night (54537908) destroy (03:3x thêm); 06/10 23:00: paper_mw4 (54468293) destroy (17:5x thêm); paper_mw3 (54437777) destroy 14:2x
for h in 158 paper_night2 paper3; do bash scripts/sync_remote.sh $h | grep -c "^chuyển" | sed "s/^/$h: chuyển /;s/$/ file/"; done
$PY scripts/fpe.py dbrows 161 2>&1 | tail -1
# 08/10 17:4x người dùng: "trên vast để ý dung lượng chọn checkpoint nhé" - mỗi nhịp kéo checkpoint Pha 1 đã dùng xong về 161 (md5 khớp mới xoá
# trên vast), chạy NỀN (đường truyền chậm), mỗi máy một lượt; dòng ĐĨA của lượt TRƯỚC in ra đây để thấy dung lượng còn lại
for h in paper_night2 paper3; do
  [ -f state/offload_$h.out ] && grep -h "ĐĨA\|!!" state/offload_$h.out | tail -2
  # 09/10 11:1x: phiên "MultiVD Colab" tự kéo checkpoint Adan bằng adan_offload.sh <host> (cùng thư mục đích, cùng md5) - đang chạy thì KHÔNG phóng
  # thêm, hai rsync cùng một thư mục tranh nhau ghi file tạm (đã xảy ra 11:0x trên paper3)
  ps -eo args --no-headers | awk -v h="$h" '$1=="bash" && ($2=="scripts/vast_offload_p1.sh" || $2 ~ /\/adan_offload\.sh$/) && $3==h' | grep -q . \
    || (timeout 3000 bash scripts/vast_offload_p1.sh $h > state/offload_$h.out 2>&1 &)
done
bash scripts/live_all.sh > /dev/null 2>&1
$PY scripts/artifact_push.py plan
