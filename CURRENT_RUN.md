# CURRENT_RUN — sáng 09/09/2026

> Cập nhật **09/09 03:45 UTC (10:45 giờ VN)**. Người dùng đã quay lại lúc 02:00 UTC.
> File cũ (đêm 08→09) đã bị thay: nó nói về `asam4`/`pool_lm` — cả hai đã xong.

## Đang chạy ở đâu — không máy nào nằm không

| máy | mục đang chạy | xong lúc (ước) | còn lại trong worklist |
|---|---|---|---|
| **ntat** (vast, $0.0818/h) | `asam_aw.sh\|4cwe com full\|4 5` — fold 5 | ~03:55 UTC | `ensctl.sh` (~1h), rồi `wblend.sh` (~1,5h) |
| **ntat2** (vast, $0.1222/h) | `pool_cb_lm.sh\|-\|1 2 3` — codebert × lưới `lm*` | ~06:40 UTC | hết — **cần cấp việc hoặc huỷ** |
| **161** (local, GPU chung) | `pool_cb.sh` — codebert × lưới `lm*` | — | — |
| **158** (local, A4000) | có job | — | — |

**Việc phải làm khi ntat2 xong (~06:40 UTC):** hoặc cấp việc đáng chạy, hoặc `vastai destroy
instance 50223345 -y` sau khi kéo hết kết quả về và đối chiếu số file + byte.

## Ba mục đã xếp trên ntat, theo thứ tự

### 1. `asam_aw.sh` fold 4–5 — ĐANG CHẠY, để lên n=15
Kết quả hiện tại ở FACTS §24. Bất đồng hai máy **đã giải**: ntat n=10 cho ROC +0.0244 (9/10,
p=0.021) và PR +0.0352 (10/10, p=0.002); ntat2 n=9 độc lập cho ROC +0.0135 (7/9), PR +0.0216
(7/9). Cộng số fold: ROC 16/19, PR 17/19. **Hiệu ứng ở XẾP HẠNG, không ở F1** — ntat2 cho F1 chỉ
4/9. Chưa chốt cấu hình cuối khi ntat chưa xong fold 4–5.

### 2. `ensctl.sh` — ĐỐI CHỨNG BẮT BUỘC cho §25
Chỉ baseline, seed 7 và 1234, 5 fold, ghi vào **đúng cây `asamaw_t5p`** đã có baseline seed 42
→ trộn baseline⊕baseline ghép cặp cùng máy cùng fold. Trả lời: *"trộn hai mô hình nào cũng lợi"*
hay *"lợi đến từ Pha 1"*. **Nếu nó cũng cho +0.013 ROC thì §25 sập.** ~10 ô, ~1h.

### 3. `wblend.sh` — NỘI SUY TRỌNG SỐ (FACTS §25.5)
`4cwe com full`, fold 1–3, bậc 1, seed 42. α chọn trên **VAL**, báo trên TEST, in kèm trộn xác
suất α=0.5 trên **cùng cặp**. Cơ chế đã kiểm 0 GPU: 201/205 tensor nội suy được. Cần
`KEEP_CKPT=1` (đã vá `run/matrix.sh`, mặc định vẫn XOÁ) và tự dọn checkpoint sau mỗi fold.
Khối này cũng sinh ô có `val_probabilities` → gỡ giới hạn "α chỉ chọn được trên test" của §25.

## Kết quả đêm nay — xem FACTS §25 → §25.5

Tất cả **0 GPU**, tính lại trên xác suất từng mẫu đã lưu sẵn:

| mục | nội dung một dòng |
|---|---|
| §25 | trộn đều baseline ⊕ chuyển giao (α=0.5) **vượt cả hai đầu mút**: ROC +0.0165 so với chuyển giao thuần (59/83 khối, p=0.0002). Lặp trên hai backbone cùng biên độ. |
| §25.1 | đối chứng âm: trộn với nguồn **pha loãng** cho ROC −0.0036 — không lợi bừa. Hiệu ghép cặp nguyên−loãng +0.0145 (12/13, p=0.0034). |
| §25.3 | ở điểm vận hành: CWE-022 recall +0.073, CWE-079 **+0.146**, precision cũng tăng; hai CWE thường **không bị đụng**. Cảnh báo: 079 chỉ ~8 hàng dương/fold. |
| §25.4 | **sống sót phép kiểm rò rỉ**. Trên hàng sạch (`none`, 73%) trộn dương cả ba chỉ số còn chuyển giao thuần **âm** ở cả hai AUC. Hiệu trộn−chuyển giao: `none` +0.0182, `test` +0.0190, **`train` −0.0147**. |
| §25.5 | cơ chế nội suy trọng số đã kiểm; khối đã xếp. |

Trang tổng hợp: https://claude.ai/code/artifact/d18d51d3-ac51-491d-b8b6-90685506a8a6

## Hai điều §25 CHƯA trả lời được

1. **Đối chứng baseline⊕baseline khác seed** — `ensctl` đang xếp. Chưa có thì chưa trích §25.
2. **F1@ngưỡng-val của bản trộn** — ô cũ không lưu xác suất val. Đã vá
   `src/train_{transfer,baseline}.py` ghi thêm `val_probabilities`/`val_labels`, **đã xác minh
   chạy thật** trên ô ntat sinh sau bản vá. Mọi ô từ 09/09 đọc được chỉ số thứ tư.

## Hai lỗi vận hành đêm nay, đã vá

- `vast_worklist.sh` in "xong" khi worklist bị **ghi đè** lúc đang đọc → bỏ sót mục, ntat2 nằm
  không ~2 phút. Đã thêm **vòng ngoài**: hết một lượt thì mở lại file, đối chiếu `todo − done`,
  còn việc thì chạy lượt nữa. Kiểm ba chiều. FACTS §25.2.
- `run/pool1.sh` chưa từng được đẩy lên ntat trong khi worklist gọi wrapper của nó → driver chết.
  Quy tắc: xếp script vào máy xa thì kiểm **mọi file nó gọi tới**.
