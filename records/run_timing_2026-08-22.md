# Thời gian chạy đo từ log đợt 22/08 (trước khi xóa log)

Một "job" = một nhánh (hoặc baseline) trên **một** fold: train Phase 2 + inference.
Đo bằng hiệu hai mốc `=== timestamp | fold N | label/mode ===` liền nhau trong cùng file log.
GPU: RTX 5060 Ti và RTX 5070 Ti, batch 16, max_length 512, tối đa 30 epoch, patience 5.

| backbone | n job | trung vị (s) | trung bình (s) | max (s) |
| --- | --- | --- | --- | --- |
| codet5-base | 17 | 317 | 383 | 611 |
| codebert | 39 | 201 | 212 | 400 |
| codet5p-110m-embedding | 13 | 174 | 210 | 435 |
| codet5p-220m | 49 | 157 | 183 | 708 |
| unixcoder | 36 | 129 | 150 | 498 |

Dùng để ước lượng: **~3 phút/job** là con số lập kế hoạch an toàn cho cả năm backbone.

## Sự cố ghi nhận trong log (đã xử lý xong)

| chỗ | gì | trạng thái |
| --- | --- | --- |
| `emb1_emb/transfer_cwe` fold 1 | forward hỏng trước khi có nhánh `.encoder` cho `codet5p-*-embedding` | đã chạy lại, `fold1.json` có mặt |
| `fam1_*` seed 7, máy B | job bị `Terminated` giữa fold 3 | fold 1–3 có kết quả, fold 4–5 chưa chạy |
| `twin_primevul`, `t5p_twin/cwe+lora32` (seed 36) | 15 traceback | thí nghiệm đã đóng, không dùng lại |
| 2 lần | `CUDA out of memory` khi chạy job đo song song job huấn luyện | đã sửa bằng vòng chờ GPU trong `sharpness-all.sh` |
