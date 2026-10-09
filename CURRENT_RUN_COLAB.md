# CURRENT_RUN_COLAB - việc chạy trên Google Colab

Người dùng 09/10/2026: "Sau này tạo CURRENT_RUN_COLAB.md" - mọi khối chạy trên Colab ghi ở đây (mới nhất ở trên), không ghi vào CURRENT_RUN.md
(ở đó chỉ để một dòng trỏ). Colab là GPU khác 161/158/vast ⇒ số Colab KHÔNG ghép cặp với số máy khác. Cách dùng Colab (VNC :1, colab-mcp,
chuyển mã base85 + md5, kéo kết quả `@@`) ở memory `colab-mcp-setup-on-161.md`.

### 🅰️dan 08/10 23:1x - DỪNG + NGẮT RUNTIME 09/10 01:4x (Colab G4, RTX PRO 6000 Blackwell 96 GB; phiên Claude tách nhánh): ADAN Ở PHA 1 - kiểm 3 ô từng sập XONG trước 23:56 08/10, n = 15 (adanlr1) ĐANG CHẠY từ 23:56 08/10 (giờ +07; bản trước ghi 00:3x là sai)

- **Kết quả kiểm 3 ô (n = 1 mỗi ô, Colab, KHÔNG ghép cặp với tab nào):** adanlr1 (5,66e-5) - ROC test full s1234 f5 **0,928** (ô gốc 0,570),
  full s7 f1 **0,907** (tab Pha 1 30 epoch 0,512), 4CWE s1234 f1 **0,877** (RAdam β2 0,99 0,567); Pha 1 rời ln2 cả 3 ô (train loss cuối 0,26 / 0,32 /
  0,38) nhưng checkpoint Pha 1 vẫn chọn sớm (epoch 8 / 8 / 2) vì val chỉ JS giảm khi học (cặp ngược nhãn). adanlr5 (2,83e-4, không warmup) - Pha 1
  kẹt ln2 cả 3 ô suốt 16 epoch; 4CWE s1234 f1 ROC **0,412**, full s1234 f5 **0,461**, full s7 f1 **0,501** (Pha 2 kẹt ln2 cả 3 ô) ⇒ LR 5× không warmup SẬP 3/3. Dự đoán: adanlr1 ĐÚNG (Pha 1
  rời ln2 3/3, ROC ≥ 0,88 ở 2/3); adanlr5 ĐÚNG (≥ 1/3 Pha 1 hỏng - thực tế 3/3).
- **n = 15** (người dùng 09/10 00:1x: "Nếu ổn chạy thử n=15 luôn"; chọn "Chỉ Adan n=15"; "Baseline ... nếu cần thì tôi sẽ báo chạy sau" ⇒ KHÔNG xếp
  baseline): adanlr1, 3 nguồn (4CWE / common mở rộng / full) × seed 42 / 1234 / 7 × 5 fold = 45 ô, Pha 2 = rasam_w20. Mỗi nguồn một chuỗi trên
  Colab, fold vòng ngoài, trong fold seed 42 → 1234 → 7; 3 ô của lượt kiểm dùng lại (cùng cấu hình). Dữ liệu 90 / 90 file trùng md5 161; pool common
  mở rộng dựng lại từ HF bằng bitmask chọn hàng (logic `build_js_common_ext_pool.py`, trùng md5 a852207f / d27542ae). Ước ~7-8 h.
- **Kéo kết quả:** ô "Ô 7" trên Colab in JSON từng ô; output lớn được Claude Code lưu thành file ⇒ `results_colab/adan/save_pull.py <file>` tách
  về `_FinalPaperExperiment/results_colab/adan/<run>/fold<k>.json`; output nhỏ hiện thẳng trong hội thoại thì tách từ transcript phiên
  (`save_pull.py <transcript>.jsonl`). 09/10 00:5x: 15 file (12 của lượt kiểm + 3 Pha 1 fold 1 seed 42). Runtime Colab mất là mất log / checkpoint chưa kéo.
- **DỪNG 09/10 01:4x** (người dùng 09/10 01:0x: chuyển n = 15 sang A4000 - 161 / paper_night2 / paper3; "Khi có dù chỉ 1 máy hãy dừng colab ... đang ở đầu
  phase 2 của run thì hủy luôn"). paper_night2 chạy Adan 01:43 ⇒ giết cả 2 chuỗi (common_ext s7 f2 Pha 1 epoch 15/16, full s1234 f3 Pha 1 epoch 7/16 - cả hai
  còn nguyên Pha 2 nên huỷ), kéo hết kết quả, đối chiếu danh sách file hai đầu **33 / 33 khớp** (0 thiếu, mọi file có test_roc_auc), rồi
  `runtime.unassign()`. Checkpoint trên Colab mất theo runtime (đã báo trước). **Kết quả Colab giữ RIÊNG ở `_FinalPaperExperiment/results_colab/adan/`**
  (không vào results/, không lên artifact; GPU khác nên không ghép cặp). Pha 2 adanlr1 đã xong trên Colab (13 ô, ROC test): 4cwe f1 0,913; 4cwe_s1234 f1 0,877; 4cwe_s7 f1 0,916; common_ext f1 0,938; common_ext_s1234 f1 0,892; common_ext_s1234 f2 0,946; common_ext_s7 f1 0,925; full f1 0,934; full_s1234 f1 0,907; full_s1234 f2 0,929; full_s1234 f5 0,928; full_s7 f1 0,907; full_s7 f2 0,923.
  n = 15 chạy lại TOÀN BỘ trên A4000 (xem `CURRENT_RUN.md` mục 🅰️A4000).
- **ĐỔI HÀNG ĐỢI 09/10 01:00** (người dùng: "Ưu tiên chạy của cái common với full nhé vì có khi không đủ chạy n=15 đâu. Chạy n=10 cũng
  được ưu tiên cái seed dễ sập ấy để xem xu hướng"; Colab còn 118,84 compute unit, ~8,9 unit/giờ ⇒ tối đa ~13 h; "dùng cẩn thận ... Chạy ưu tiên
  trước"). Chuỗi 4CWE bị giết (mất Pha 2 dở của 4CWE f2 s42, Pha 1 của nó giữ lại). Còn **2 chuỗi** (common mở rộng, full), mỗi chuỗi:
  (a) **n = 10 trước**: fold 1→5 × seed 1234 → 7 (hai seed từng sập trên 161: full s1234 f5 0,570, full s7 f1 0,512, 4CWE s1234 f1 0,567);
  (b) seed 42 fold 2-5; (c) 4CWE cùng thứ tự, hai chuỗi chia nhau (nhận ô bằng `mkdir /content/mvd/claims/<ô>`). Mã chuỗi ở
  `/content/mvd/queue/chain_{common_ext,full}_0109.sh` (Ô 9a/9b), log hàng đợi `queue/q_*.log`. Ô dở lúc đổi (common f1 s7 Pha 1, full f1 s1234
  Pha 2) chạy nốt dưới dạng mồ côi, chuỗi mới chờ PID rồi chỉ chạy test. Chạy thử: 37 ô còn lại, 37 test Pha 2 khác nhau, 0 trùng; 240/240 lệnh
  khớp NGUYÊN VĂN argv chuỗi cũ (cấu hình không đổi); nhánh "chỉ test" thử hai chiều đạt. Ước (2 chuỗi): n = 10 common/full xong ~04:45,
  seed 42 ~06:30, 4CWE ~08:30; tổng ~7,5 h ≈ 67 unit. **Xong + kéo đủ về 161 thì ngắt runtime ngay** (để không tốn unit khi nằm không).
- **Tiến độ 09/10 00:37 (+07):** 4CWE 2/15, common mở rộng 1/15, full 3/15 ô xong. Pha 2 fold 1 seed 42 (n = 1 mỗi nguồn, Colab): ROC test 4CWE 0,913,
  common mở rộng 0,938, full 0,934. 4CWE s7 f1 Pha 1 rời ln2 (train loss cuối 0,46) nhưng checkpoint chọn epoch 2 (train loss 0,72). Thời gian
  một ô (3 chuỗi chung GPU): 4CWE ~26 phút, common mở rộng ~33, full ~34 ⇒ **ước xong: 4CWE ~06:00, full ~07:00, common mở rộng ~07:30-08:00
  sáng 09/10** (hai chuỗi sau nhanh lên khi 4CWE xong).

(ghi lúc phóng kiểm, 23:1x:)

- **Yêu cầu** (người dùng 08/10 22:4x): "https://arxiv.org/pdf/2208.06677 https://github.com/sail-sg/adan hãy đọc artifact và xem các phần đang sập hãy
  chạy lại các seed đó thử luôn với colab ... Với nguồn dựa trên thí nghiệm đang cố chống sập hiện tại, cái này thì bỏ qua warmup và áp dụng ở phase 1
  thôi"; LR: chọn "cả hai mức".
- **3 ô** (ROC test < 0,75 ở các tab paper): full s1234 f5 (tab Kết quả paper 0,570, kẹt ln2), full s7 f1 (tab Pha 1 30 epoch 0,512), 4CWE s1234 f1
  (RAdam β2 0,99, 0,567). Tab warmup 0,2 chưa có ô sập (28/45 ô, thấp nhất 0,901).
- **Cấu hình:** Pha 1 = `p1_w20_jspy41jsval_<nguồn>_s<seed>` nhưng optimizer **Adan** (mã nguyên văn sail-sg/Adan@2c65bea, betas 0,98/0,92/0,99,
  eps 1e-8, weight decay 0,02 mặc định repo, no_prox False), **warmup 0** (LR giảm tuyến tính về 0 trong 16 epoch), patience 999, LR **5,66e-5**
  (`adanlr1`, bằng AdamW) và **2,83e-4** (`adanlr5`, 5× theo README Adan). Pha 2 = `rasam_w20_*` nguyên vẹn (RecAdam + ASAM, patience 999, 30 epoch).
  Mã: `src_mwonly_radam` + lựa chọn `--p1_optimizer adan` (mặc định adamw giữ nguyên) - chỉ có trên Colab, CHƯA đưa vào repo; thử khói CPU 161 đạt.
- **Dữ liệu / mô hình trên Colab** dựng lại từ HF `LyanDumpling/MultiDataSource4VD@387d9df` bằng chính `build_hf_pools.py` + `build_p1_mixpy.py`,
  CodeBERT `refs/pr/8`: **24/24 file trùng md5 với 161**. Colab ≠ GPU 161/158/vast ⇒ KHÔNG ghép cặp với tab nào; chỉ đọc "có thoát sập không".
- **Dự đoán (ghi trước khi có epoch đầu):** adanlr1 - Pha 1 rời ln2 (train loss < 0,65 trước epoch 6) ở ≥ 2/3 ô và Pha 2 ROC test ≥ 0,88 ở ≥ 2/3 ô;
  adanlr5 - không warmup với LR 5× nên kém ổn định hơn adanlr1, ít nhất 1/3 ô Pha 1 hỏng (kẹt ln2 hoặc loss tăng vọt).
- Kết quả về: `_FinalPaperExperiment/results_colab/` (KHÔNG vào `results/` để nhịp artifact không đọc nhầm).

