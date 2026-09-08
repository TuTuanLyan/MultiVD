# CURRENT_RUN — đêm 08→09/09/2026

> Cập nhật 09/09 03:5x giờ VN. Người dùng quay lại 9h sáng VN (02:00 UTC).

## Đang chạy ở đâu

| máy | mục đang chạy | còn lại trong worklist |
|---|---|---|
| **ntat** (vast, $0.0818/h) | `asam4.sh\|full\|1-5` — ρ=4.0 | 8 mục, kết bằng `asam_aw` |
| **ntat2** (vast, $0.1222/h) | `pool_lm.sh\|-\|4 5` | 6 mục |
| **161** (local, GPU dùng chung) | `night_161.sh` → `pool1.sh` | chuỗi đêm |
| **158** (local, A4000) | `e60_codebert` Phase 2 | `e60_cb`, `pool_lm 1-3`, `pool_pur 4-5`, `asam_aw` |

Cả bốn máy đều có việc. Không máy nào nằm không.

## Ba khối đang mở

### 1. Trục ρ của ASAM ở Phase 2 — ĐÃ CÓ KẾT QUẢ (FACTS §21)

Bậc 2, n=15/mức (5 fold × 3 nguồn), đối chứng ρ=0 cùng máy, ghép cặp từng fold.
**Đỉnh nội tại ở ρ = 1.0–2.0**; ρ=2.0 cho PR-AUC +0.0155 (13/15, p=0.007) và là
mức đầu tiên F1 cũng dương (+0.0124). ρ=4.0 đổ (−0.118 ROC, n=8). **ρ=0.1 mà dự
án dùng từ 31/08 nằm ở đáy đường cong.**

Còn chạy: ρ=4.0 nguồn `full` (đang chạy), ρ=8.0 (`asam5.sh`, đã sửa lỗi nháy lệch
và thêm `r0` để chạy bù đối chứng fold 3/5).

### 2. Lưới nguồn — tách độ tinh khiết khỏi ngôn ngữ (RESEARCH phụ lục B)

Đo thành phần thật thì lưới `pur*` **không cô lập biến nào**: pha loãng độ tinh
khiết kéo tỉ lệ js sập 0.873 → 0.192. Lưới `lm*` ghim js ở 0.87: **16/16 ô đều
âm, đơn điệu**, −0.020…−0.035 ROC-AUC (n=4). Trùng CWE nguồn–đích là biến thật.

Còn chạy: `lm` fold 5 (ntat2), `lm` fold 1-3 trên máy thứ hai (158 và ntat),
`poolcb_codebert` (12/60 ô).

### 3. `asam_aw` — Ô QUYẾT ĐỊNH, mới xếp hàng đêm nay

Trục ρ ở khối 1 đo **trên nền RecAdam**, mà RecAdam đã null ở mọi γ. Phương pháp
chốt sẽ dùng AdamW. **Chưa ai đo ASAM ρ=2.0 trên nền AdamW.**

- CÓ tác dụng ⇒ cấu hình chốt là AdamW + head + ASAM ρ=2.0, ASAM là đóng góp độc lập.
- KHÔNG ⇒ lợi ích của ρ=2.0 là tương tác với neo RecAdam, phải phát biểu khác hẳn.

Bậc 1 (n=3 fold, seed 42), 3 nguồn, hai nhánh `aw_r0` / `aw_r2p0`, cùng máy cùng
phiên. Xếp trên **158** (3 mục phía trước) và **ntat** (cuối hàng, làm bản lặp).

## Lỗi đã bắt và sửa đêm nay

1. **`run/asam5.sh` dấu nháy lệch** nuốt cả vòng `for`; `bash -n` vẫn báo OK; chạy
   thật cho **0 ô** nhưng worklist vẫn ghi "xong". Bắt được trước khi tới lượt.
   Từ nay kiểm runner bằng stub `echo` + **đếm số lần gọi**, cả hai chiều.
2. **Hai cây kết quả song song** (phẳng + lồng) sau hai kiểu `rsync` khác nhau —
   đối chiếu từng byte thấy trùng khít rồi mới xoá bản phẳng, để chỉ còn một
   đường đọc. Nếu không, lần kéo sau sẽ chỉ cập nhật một bên và tôi đọc bản cũ.
3. **`tools/report2.py` im lặng trả 0 cặp** với tên nhánh không có `_l0p05_`
   (toàn bộ khối pool). Đã sửa trong chính công cụ, kiểm hai chiều.
4. Tôi **giết nhầm** một ô của chuỗi `night_161.sh` khi chẩn đoán sai một tiến
   trình là của mình; chuỗi tự phóng lại, mất ~30 giây. Phải truy `ppid` lên tận
   gốc **trước** khi giết, không chỉ khớp dòng lệnh.
