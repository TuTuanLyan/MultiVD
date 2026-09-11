# KHAI BÁO TRƯỚC — lợi ích của transfer có phải là hiệu ứng DỮ LIỆU ÍT không?

**Viết lúc 03:05 UTC 11/09/2026, TRƯỚC KHI chạy một ô nào của khối `tsize`.**
Ghi ra file để phép kiểm không thành "đọc số rồi kể chuyện" — đây là lỗi mà §38.2 đã mắc.

## Quan sát dẫn tới giả thuyết (đã có, n=15)

Số hàng train của tập đích theo từng CWE, đặt cạnh Δ ROC-AUC của cấu hình chốt so với baseline:

| CWE | hàng train / fold | ΔROC-AUC | fold cùng dấu |
|---|---|---|---|
| 022 | ~40 | **+0.37** | 45/45 |
| 079 | ~47 | **+0.35** | 45/45 |
| 078 | ~124 | +0.02 | 31/45 |
| 089 | ~244 | −0.01 | 16/45 |

Và §36 đo được: trên codebert, đặc trưng Pha 1 cải thiện ở **cả bốn** lớp — mạnh nhất lại chính là
CWE-078 (+0.199 ROC, 5/5 fold). Nghĩa là thông tin **có sẵn** cho 078, mô hình chỉ **không dùng**.

## Giả thuyết

Lợi ích của Pha 1 **không** phải thuộc tính của cặp ngôn ngữ hay của hai CWE cụ thể. Nó là
**hiệu ứng dữ liệu-ít-theo-lớp**: khi đích có đủ ví dụ cho một lớp, mô hình tự học lớp đó và bỏ
qua đặc trưng nguồn; khi không đủ, nó dựa vào đặc trưng nguồn. Ngưỡng quan sát được nằm đâu đó
**giữa 47 và 124 hàng train**.

## Dự đoán CỤ THỂ, có thể sai

Cắt ngẫu nhiên tập train Python xuống còn **N = 228, 152, 76** hàng (giữ nguyên val và test đầy
đủ), chạy cả nhánh chuyển giao lẫn baseline ở **cùng** N với **cùng** seed nên **cùng một tập
con**. N = 456 đã có sẵn ở khối `bridge3`.

Số hàng train kỳ vọng theo CWE (lấy mẫu đều nên tỉ lệ giữ nguyên):

| N | 022 | 078 | 079 | 089 |
|---|---|---|---|---|
| 456 | 40 | 124 | 47 | 245 |
| 228 | 20 | 62 | 24 | 122 |
| **152** | 13 | **41** | 16 | 82 |
| **76** | 7 | 21 | 8 | **41** |

**Dự đoán 1 — đường cong tổng.** Δ (chuyển giao − baseline) trên **F1@0.5** phải **tăng đơn điệu**
khi N giảm, trên **cả hai** backbone. Cụ thể: Δ ở N=76 phải **lớn hơn** Δ ở N=456.

**Dự đoán 2 — chỗ dễ sai nhất, và là phần đáng giá.** CWE-078 phải **bắt đầu hưởng lợi ở N=152**
(khi nó rơi xuống ~41 hàng, đúng vùng mà 022/079 đang ở tại N=456), và CWE-089 phải bắt đầu hưởng
lợi ở **N=76**. Cụ thể: ΔROC-AUC của CWE-078 tại N=152 phải **≥ +0.10** và cùng dấu ở ≥ 2/3 fold.

**Dự đoán 3 — đối chứng âm nội tại.** CWE-022 và CWE-079 phải **giữ** lợi ích ở mọi N (chúng vốn
đã ở vùng dữ liệu ít), không được biến mất.

## Điều gì sẽ BÁC giả thuyết

* Δ tổng **không** tăng khi N giảm, hoặc tăng trên một backbone mà giảm trên backbone kia.
* CWE-078 ở N=152 vẫn null (ΔROC < +0.05) ⇒ **bác thẳng**: hiệu ứng là đặc thù của hai CWE đó,
  không phải của cỡ dữ liệu. Đây là kết cục có ích không kém, vì nó đóng một cách giải thích.
* Cả hai nhánh **sập** ở N nhỏ (baseline đoán một lớp, F1 ≈ 0.333) ⇒ ô đó **không đọc được**, phải
  loại chứ không được tính là "Δ lớn". Ngưỡng loại khai báo trước: **baseline F1@0.5 < 0.40**.

## Ràng buộc, khai báo trước

* Ngưỡng ở Dự đoán 2 là **+0.10** và ngưỡng loại ô là **0.40** — **không sửa** sau khi thấy số.
* Bậc 1: **n = 3 fold**, seed 42. Chỉ để sàng lọc.
* Chỉ cắt **train**. Val và test giữ nguyên 152 hàng — nếu cắt test thì Δ đổi cả thước đo.
* Cả hai nhánh dùng **cùng seed** nên nhận **cùng tập con** (`limit_records` lấy mẫu ngẫu nhiên có
  seed, không thay thế). Nếu không thì phép so đổi hai biến.

---

## KẾT QUẢ (điền 04:35 UTC 11/09, sau khi đủ 4 mức × 2 backbone × 3 fold)

| | codebert | t5p |
|---|---|---|
| **Dự đoán 1** đơn điệu | **ĐÚNG**, chặt trên cả bốn chỉ số | **SAI** (F1 tụt ở N=152, PR âm ở N=228) |
| dạng hai đầu mút (N=76 > N=456) | ĐÚNG, cả bốn, 3/3 fold | ĐÚNG, cả bốn, 3/3 fold |
| **Dự đoán 2** (078 ≥ +0.10 tại N=152) | **BÁC** (+0.025) | **BÁC** (−0.065, 0/3) |
| **Dự đoán 3** (022/079 giữ lợi ích) | ĐÚNG | ĐÚNG |

Ngưỡng +0.10 và cách phân mức **giữ nguyên**, không sửa sau khi thấy số.

**Bài học về cổng đã đặt sai**: ngưỡng loại ô `baseline F1@0.5 < 0.40` **quá lỏng** — macro-F1 của
bộ phân loại ngẫu nhiên trên tập cân bằng là ~0.49, nên ô `t5p`/`N=76` (baseline ROC 0.4815, đúng
mức ngẫu nhiên) **không bị bắt**. Lần sau đặt cổng trên **ROC-AUC < 0.55**. Không sửa ngưỡng cũ
hồi tố; chỉ ghi rõ cách đọc ô đó.

Chi tiết đầy đủ ở FACTS §40.
