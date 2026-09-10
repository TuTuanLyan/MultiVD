# KHAI BÁO TRƯỚC — probe §36 có dự báo được nên GIỮ hay nên THAY không?

**Viết lúc 22:40 UTC 10/09/2026 (05:40 VN 11/09), TRƯỚC KHI chạy probe trên `com`/`full`.**
Ghi lại để phép kiểm không thành "đọc số rồi kể chuyện" — §38.2 hiện đang mắc đúng lỗi đó.

## Giả thuyết (§38.2, đang ở mức GIẢ THUYẾT, n=3)

Δ của linear probe (đặc trưng Pha 1 − đặc trưng pretrained) nói cho ta biết can thiệp Pha 2 nào
sẽ thắng:

* **Δ probe LỚN** ⇒ đặc trưng nguồn thật sự có ích ⇒ can thiệp loại **GIỮ / KHAI THÁC** thắng
  (`lp3` = fit head trên backbone đóng băng; `fd1` = neo đặc trưng nhẹ).
* **Δ probe NHỎ hoặc ÂM** ⇒ đặc trưng nguồn không có ích ⇒ can thiệp loại **THÊM / THAY** thắng
  (`rp50` = replay dữ liệu nguồn; `rhlp3` = khởi tạo lại head rồi probe).

Đã quan sát trên nguồn `4cwe`: codebert Δ=+0.1125 (5/5) → `lp3`,`fd1` thắng · t5p Δ=+0.0202 (3/5)
→ `rp50`,`rhlp3` thắng. Nhưng bảng đó được đọc **sau khi** nhìn số, nên chưa phải bằng chứng.

## Dự đoán CỤ THỂ, có thể sai (đây là phần quan trọng)

Đo Δ ROC-AUC của probe trên `com` và `full`, cả hai backbone. Sau đó:

1. **Ngưỡng phân loại khai báo trước: Δ ROC ≥ +0.06** ⇒ xếp vào nhóm "GIỮ thắng";
   **Δ ROC < +0.06** ⇒ nhóm "THÊM/THAY thắng". (Chọn 0.06 vì nó nằm giữa hai giá trị đã biết
   0.1125 và 0.0202, và cách sàn nhiễu 0.010 đủ xa.)
2. **Dự đoán về chính Δ probe** (chưa biết, sẽ biết trong ~1 giờ):
   - `com` và `full` là nguồn **khó hơn** `4cwe` (94 và 123 CWE, lệch mạnh về ccpp, val Pha 1
     thấp hơn: codebert 0.5598/0.5636 so với 0.6532). **Dự đoán: Δ probe của `com`/`full` THẤP
     HƠN `4cwe` trên cùng backbone.**
   - Nếu Δ probe của codebert trên `com`/`full` **tụt xuống dưới 0.06**, thì giả thuyết nói
     `lp3` sẽ **mất lợi thế** trên hai nguồn đó — dù trên `4cwe` nó là nhánh mạnh nhất.
3. **Phép kiểm GPU sẽ xác nhận hay bác** (CẦN NGƯỜI DÙNG DUYỆT, chưa chạy): chạy `lp3` và `rp50`
   trên nguồn có Δ probe thấp nhất, n=3, cùng máy cùng fold. Giả thuyết đúng ⇒ `rp50` hơn `lp3`
   ở đó. Giả thuyết sai ⇒ `lp3` vẫn thắng bất kể Δ probe, và §38.2 bị **bác**.

## Điều gì sẽ BÁC giả thuyết

* Δ probe của `com`/`full` **không** thấp hơn `4cwe` ⇒ dự đoán (2) sai, giả thuyết yếu đi.
* Δ probe thấp mà `lp3` vẫn thắng ⇒ giả thuyết **bị bác thẳng**.
* Δ probe cao mà `rp50` thắng ⇒ cũng bị bác.

Không được sửa ngưỡng 0.06 hay đổi cách phân nhóm sau khi thấy số.
