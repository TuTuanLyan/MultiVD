# Ghi dự đoán TRƯỚC KHI ĐO — khối `shuf1`: xáo nhãn nguồn

Viết **14/09/2026, trước khi chạy ô nào**. Nhánh git `fusion`.
Ai đọc lại sau: mọi con số dưới đây được viết khi chưa có một ô kết quả nào.

---

## Câu hỏi

Pha 1 cho +0.0092 (`com`, n=15, §40.5) so với baseline. Bao nhiêu phần của con số đó là
**tri thức lỗ hổng học từ nhãn**, và bao nhiêu là **phơi nhiễm miền** — backbone chỉ đơn giản
được nhìn thêm 3 744 hàm code thật trong 12 epoch?

Bối cảnh khiến câu hỏi này cấp thiết (**FACTS §51**): tác vụ nguồn chỉ học được tới **0.57–0.59**
macro-F1 (ba seed, ba máy), hai đồng nghiệp đo độc lập CleanVul js và cpp cũng **dưới 0.6**,
và CleanVul tự công bố bộ dữ liệu lỗ hổng mang **40–75% nhiễu nhãn**.

## Thiết kế

Ba nhánh Pha 1, **`aux_mode=none`** (không head phụ, để cô lập đúng nhãn nhị phân),
nguồn `com`, **cùng seed 42, cùng thứ tự dữ liệu, cùng split** — chỉ khác cột `label`:

| nhánh | nhãn | giữ gì | phá gì |
|---|---|---|---|
| `real` | thật | — | — |
| `shufall` | hoán vị toàn cục | tỉ lệ 50/50 | toàn bộ liên hệ code↔nhãn **và** cân bằng trong cặp (0.509) |
| `shufpair` | đổi chỗ trong cặp | tỉ lệ 50/50 **và** cân bằng trong cặp (1.000) | **duy nhất**: bản nào là bản trước khi vá |

**Khớp số bước gradient**: cả ba dùng `--save_last_epoch`, chạy **đúng 12 epoch**, tắt dừng sớm,
lấy checkpoint cuối. Không làm thế thì nhánh xáo dừng sớm hơn hẳn và ta đổi **hai** biến.
E=12 chọn theo `best_epoch` của lần chạy thật đã công bố trên `com`/codebert; giá trị cụ thể
không quan trọng bằng việc **ba nhánh dùng cùng một E**.

Pha 2: `adamw`, `--sam_rho 0`, fold 1–3, seed 42, cộng `baseline`.
**Cả hai backbone** (codebert + t5p) theo cổng 2 của `NEXT_CONTRIBUTION.md`.
24 ô: (3 nhánh × 3 fold + 3 baseline) × 2 backbone.

Đây là **bậc 1 — kiểm chứng, n=3**. Không có con số nào ở đây được viết vào bài.

## Dự đoán của tôi, có số

Đặt `G(x) = Δ(x − baseline)` ghép cặp theo `(backbone, fold)`, 6 điểm mỗi nhánh.

1. `G(real) > 0`, khoảng **+0.005 … +0.02** trên F1@0.5. *(nhất quán với §40.5: +0.0092 ở n=15)*
2. `G(shufall)` và `G(shufpair)` **đều dương nhưng nhỏ hơn** `G(real)`.
   Tôi đoán phơi nhiễm chiếm phần **lớn hơn một nửa**: `G(shuf) ≈ 0.4 … 0.8 × G(real)`.
   Lý do đoán vậy: §46 đã cho thấy head phụ chỉ thêm +0.0005 trên 132 ô, và §44 cho thấy
   nút thắt thua PCA-8 — hai dấu hiệu rằng nội dung nhãn đóng góp ít.
3. `G(shufpair) ≈ G(shufall)`, chênh dưới sàn nhiễu 0.010. Nếu chúng khác nhau rõ thì
   thứ mô hình dùng là **cân bằng trong cặp** chứ không phải chiều — một kết quả khác hẳn.

## Ngưỡng quyết định — đặt TRƯỚC, kể cả vùng "không kết luận"

Gọi `D = G(real) − G(shufpair)`, ghép cặp theo `(backbone, fold)`, 6 điểm.

| `D` trên F1@0.5 | số fold cùng dấu | kết luận |
|---|---|---|
| ≥ **+0.020** | ≥ 5/6 | **nhãn mang tri thức chuyển giao được** ⇒ hướng giám sát-tương-đối có chỗ đứng, đi tiếp |
| ≤ **+0.010** | bất kỳ | **không phân biệt được với phơi nhiễm** ⇒ bỏ mọi hướng dựa trên nhãn nguồn; bản thân điều này là phát hiện đáng công bố |
| giữa 0.010 và 0.020 | | **KHÔNG KẾT LUẬN** — leo lên n=5, không được đọc theo hướng mình muốn |
| `G(shuf) > G(real)` rõ | ≥ 5/6 | nhiễu nhãn **chủ động gây hại**, đúng dự đoán của arXiv:2309.17002 cho out-of-domain |

Bắt buộc in **cả bốn chỉ số** (F1@0.5, F1@val, ROC-AUC, PR-AUC) kèm đếm dấu, bằng
`tools/report2.py`. Kết luận chỉ được phát biểu khi **cả bốn cùng hướng**; một chỉ số ngược
thì ghi nguyên trạng chứ không chọn chỉ số có lợi (bài học §2b "ASAM null").

## Điều đã biết có thể làm hỏng phép đo, và cách chặn

- **Pha 1 dao động giữa hai lần chạy cùng cấu hình** (§48.2: hai lần độc lập cho độ tản
  0.0379 và 0.1209). Ở n=3 một lần Pha 1 mỗi nhánh là **mong manh**. Chặn một phần bằng
  cùng seed / cùng split / cùng E; phần còn lại chỉ chặn được bằng seed thứ hai ở bậc 2.
  **Ghi nhận trước, không giấu.**
- **Nhánh xáo có thể sập dưới mức ngẫu nhiên** ở Pha 1. Đó không phải lý do bỏ ô
  (CLAUDE.md mục 3): chạy Pha 2 và ghi kèm val Pha 1. `PHASE1_MIN_VAL=0`.
- Cả ba nhánh và baseline chạy **cùng một máy, cùng một phiên** (mục 4).

## Kết quả

*(để trống — điền sau khi đo, không sửa phần trên)*
