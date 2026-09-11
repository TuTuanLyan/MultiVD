# KHAI BÁO TRƯỚC — nút thắt 8 chiều có mang được gì cho nhiệm vụ ĐÍCH không?

**Viết lúc 09:20 UTC 11/09/2026, TRƯỚC KHI chạy một dòng nào của phép đo dưới đây.**
Ghi ra file để phép kiểm không thành "đọc số rồi kể chuyện" — lỗi mà §38.2 đã mắc.

## Trước hết: CỔNG ĐÃ KHAI BÁO TRƯỚC ĐÓ ĐÃ TRƯỢT. Ghi rõ ở đây.

Cổng mảnh 1, khai báo trong `scripts/vast_auxb_run.sh`: *"head phụ phải VƯỢT sàn đoán-lớp-đa-số,
và phải dùng >1 lớp"*. Đo trên codebert, val 96 hàng của `4cwe`, seed 42:

| | độ chính xác | macro-F1 | số lớp head dùng |
|---|---|---|---|
| head phụ **cân bằng lớp** (`auxb`) | **0.2500** | **0.1917** | **4/4** |
| head phụ **không cân bằng** (khối cũ) | 0.6667 | 0.2000 | **1/4** |
| sàn đoán-lớp-đa-số | 0.6667 | 0.2000 | 1 |
| sàn đoán ngẫu nhiên | 0.4870 | — | — |

**Kết luận trung thực: TRƯỢT.** Trọng số nghịch tần suất làm được đúng một nửa việc — head thôi
đoán một lớp và dùng cả bốn — nhưng nó **không** vượt sàn ở cả hai chỉ số. Nó còn tụt xuống
**dưới** sàn ngẫu nhiên về độ chính xác.

Đọc theo lớp thì thấy vì sao: lớp 0 (CWE-022) **8/10 = 80%** đúng, lớp 3 (CWE-089) 33%, lớp 2
(CWE-079, lớp đa số, 64 hàng) chỉ 21.88%, lớp 1 (CWE-078) **0/16**. Head học được lớp hiếm và
đánh mất lớp đa số — đúng cái mà trọng số nghịch tần suất phải làm, chỉ là quá tay.

## Nhưng cổng đó đo SAI thứ cho mảnh 2 — và phải nói rõ tại sao trước khi đo lại

Mảnh 2 **không dùng `cwe_head`**. Nó đóng băng `latent_proj` (768→8) rồi đặt một head nhị phân
**MỚI** lên đầu ra 8 chiều đó, huấn luyện trên nhãn **lỗ hổng của đích**. Cho nên câu hỏi quyết
định không phải *"`cwe_head` đoán CWE có giỏi không"* mà là:

> **Ảnh 8 chiều `P(f)` có còn tách được LỖ HỔNG trên tập đích tuyến tính không, và có tách tốt
> hơn một phép chiếu 8 chiều NGẪU NHIÊN của cùng đặc trưng đó không?**

Vế sau là vế phải có. Nếu `latent_proj` không hơn một ma trận ngẫu nhiên thì "neo vào bảng phân
loại của nguồn" là một câu chuyện rỗng: cái chạy được chỉ là "giảm chiều xuống 8", ai làm cũng
được, và cơ chế mất phần mới. Đây đúng là dạng đối chứng Hewitt & Liang đã dùng ở §36.1.

## Phép đo

Đặc trưng đóng băng, không fine-tune gì. Trên **cả 5 fold** của `data/sven_python_folds_norm`,
mỗi fold fit logistic regression trên train, chọn `C` trên val theo ROC-AUC, chấm test.

Bốn bộ đặc trưng, **cùng một checkpoint Pha 1**, ghép cặp theo fold:

| ký hiệu | đặc trưng | vai trò |
|---|---|---|
| `p768` | pooled 768 chiều của Pha 1 | trần trên — đã đo ở §36 |
| `lat8` | `latent_proj(pooled)`, 8 chiều, **đã học** | cái mảnh 2 dùng thật |
| `rnd8` | phép chiếu ngẫu nhiên 768→8 của **cùng** pooled | **đối chứng** |
| `pca8` | 8 thành phần chính đầu của **cùng** pooled, fit trên train | đối chứng thứ hai, mạnh hơn |

`rnd8` dùng ma trận Gauss, cùng seed cho mọi fold. `pca8` fit **chỉ trên train** của từng fold.

## Dự đoán CỤ THỂ, có thể sai

**Dự đoán A — nút thắt giữ được phần lớn tín hiệu.** `lat8` phải đạt ROC-AUC **≥ 0.60** và nằm
trong **0.05** của `p768`. Nếu `lat8` tụt quá xa thì nhánh thứ hai của mảnh 2 không có gì để nói
và cổng `g` sẽ bị dìm về 0 vì lý do tầm thường.

**Dự đoán B — đây là vế đáng giá, và là vế dễ sai nhất.** `lat8` phải hơn **cả** `rnd8` **và**
`pca8` về ROC-AUC, **cùng dấu ở ≥ 4/5 fold** với cả hai đối chứng. Ngưỡng biên độ: Δ ≥ **+0.02**
(trên sàn nhiễu cùng-GPU 0.010, dưới sàn liên-GPU 0.028 nên phải đọc kèm số fold).

**Dự đoán C — đối chứng âm nội tại.** `rnd8` phải **kém rõ rệt** `p768`. Nếu chiếu ngẫu nhiên
xuống 8 chiều mà vẫn bằng 768 chiều thì bài toán này quá dễ và mọi phép so ở trên mất nghĩa.

## Điều gì sẽ BÁC mảnh 2

* `lat8` **không** hơn `rnd8` (Δ < +0.02 hoặc < 4/5 fold) ⇒ **bác phần "neo vào nguồn"**. Nút
  thắt khi đó chỉ là giảm chiều, không mang hình dạng bảng phân loại nào cả. Mảnh 2 vẫn có thể
  làm điểm số đẹp lên, nhưng **không được viết là cơ chế mới** — phải viết là "một nhánh ít tham
  số làm chính quy hoá".
* `lat8` ROC-AUC < 0.55 ⇒ **bác thẳng**: nhánh thứ hai không có tín hiệu, cổng `g` sẽ về 0.

## Ràng buộc, khai báo trước

* Ngưỡng **+0.02** và **4/5 fold** ở Dự đoán B **không sửa** sau khi thấy số.
* Đo trên **codebert** trước. Kết quả chỉ được coi là mẫu hình khi **lặp lại trên t5p** — luật
  CLAUDE.md mục 2b: thắng ở một backbone thua ở backbone kia đã xảy ra ở **mọi** nhánh của §38.2.
* Đây là **0 GPU**, chạy CPU, nên không đụng vào hàng đợi đang chạy.

---

## KẾT QUẢ

(chưa điền)
