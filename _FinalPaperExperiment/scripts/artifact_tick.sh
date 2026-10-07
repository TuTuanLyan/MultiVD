#!/bin/bash
# Một nhịp cập nhật artifact khối mwonly5 (mỗi 30 phút, người dùng 04/10): kéo kết quả xong từ 158 / vast về 161, dựng dòng `folds`
# (xong: fpe.py dbrows; đang chạy: live_all.sh), rồi artifact_push.py plan -> dòng "WRITES [...]" để phiên điều phối gửi bằng ArtifactData batch.
cd "$(dirname "$0")/.." || exit 1
PY=/home/ntat/miniconda3/envs/vdenv/bin/python
# 05/10 13:1x: paper_mw, paper_mw2 đã destroy - chỉ còn đồng bộ 158
# 07/10 03:3x: thêm paper_night (54537908); 06/10 23:00: paper_mw4 (54468293) destroy (17:5x thêm); paper_mw3 (54437777) destroy 14:2x
for h in 158 paper_night; do bash scripts/sync_remote.sh $h | grep -c "^chuyển" | sed "s/^/$h: chuyển /;s/$/ file/"; done
$PY scripts/fpe.py dbrows 161 2>&1 | tail -1
bash scripts/live_all.sh > /dev/null 2>&1
$PY scripts/artifact_push.py plan
