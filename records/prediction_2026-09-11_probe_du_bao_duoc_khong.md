# KHAI BÁO TRƯỚC — probe trên đặc trưng ĐÓNG BĂNG có dự báo được kết quả sau FINE-TUNE không?

**Viết lúc 10:35 UTC 11/09/2026.** Lúc viết, khối đối chứng `auxb` mới xong **3/6 ô** và tôi
**chưa đọc một ô nào** của nó. Ghi ra file để phép kiểm không thành "đọc số rồi kể chuyện".

## Vì sao câu hỏi này đáng hỏi

Mỗi lần đổi một thứ ở Pha 1, hiện tại phải **chạy lại toàn bộ Pha 2** mới biết nó tốt hay xấu:
2 backbone × 3 fold × 2 nhánh = 12 ô GPU cho mỗi câu hỏi. Trong khi `tools/latent_probe.py` đo
đặc trưng **đóng băng** trên CPU, **0 GPU**, xong trong ~10 phút.

Nếu probe **dự báo được dấu** của hiệu ứng sau fine-tune thì mọi thay đổi Pha 1 về sau đều sàng
được bằng CPU trước khi đụng tới GPU. Nếu **không** dự báo được thì phải nói rõ điều đó, và mọi
kết luận rút từ probe (kể cả §36 và §44) chỉ được phát biểu **về đặc trưng đóng băng**, không
được suy sang mô hình đã fine-tune.

## Số đã có (§44), đo TRƯỚC khi có bất kỳ ô Pha 2 nào của đối chứng

Probe ROC-AUC trên đặc trưng 768 chiều đóng băng, 5 fold, đích Python:

| backbone | Pha 1 **không cân bằng** | Pha 1 **cân bằng** | hiệu (bal − unbal) |
|---|---|---|---|
| codebert | 0.7700 | 0.7827 | **+0.0127** |
| t5p | 0.6405 | 0.6118 | **−0.0287** |

Probe nói: cân bằng lớp **giúp codebert**, **hại t5p**.

## Dự đoán CỤ THỂ, có thể sai

Khối đối chứng `auxb` chạy nhánh Pha 1 **không cân bằng** trên **cùng máy, cùng fold, cùng phiên**
với nhánh cân bằng đã có. Gọi `D_bal = Δ(cân bằng − baseline)` và `D_unbal = Δ(không cân bằng −
baseline)`, ghép cặp theo (cây, seed, fold), n = 3 fold, seed 42.

**Dự đoán 1 — DẤU khớp trên cả hai backbone.** `D_bal − D_unbal` trên **ROC-AUC** phải **dương
trên codebert** và **âm trên t5p**. Đây là phần đáng giá và cũng là phần dễ sai nhất, vì mọi can
thiệp trong dự án này tới giờ đều tách theo backbone theo một kiểu **khác** nhau mỗi lần.

**Dự đoán 2 — biên độ cùng cỡ.** `|D_bal − D_unbal|` trên t5p phải **lớn hơn** trên codebert
(probe nói 0.0287 so với 0.0127). Yếu hơn dự đoán 1, vì n=3.

## Điều gì sẽ BÁC

* Dấu **sai ở một backbone** ⇒ probe **không** dự báo được dấu. Kết luận: probe chỉ nói về đặc
  trưng đóng băng, không suy được sang mô hình fine-tune. §36 và §44 vẫn đúng như đã phát biểu
  (chúng vốn phát biểu về đặc trưng đóng băng), nhưng **không được dùng để sàng lọc thay GPU**.
* Dấu đúng cả hai ⇒ **giả thuyết sống sót một lần**, chưa phải kết luận. Cần lặp ở một thay đổi
  Pha 1 **khác** trước khi tin.

## Ràng buộc, khai báo trước

* Chỉ số chính là **ROC-AUC** (probe cũng đo ROC-AUC — phải so cùng thước).
* Bậc 1, **n = 3 fold**, seed 42. Chỉ SÀNG LỌC.
* codebert `D_bal − D_unbal` dự báo chỉ **+0.0127**, tức **ngay trên sàn nhiễu cùng-GPU 0.010**.
  Nên với codebert, **số fold cùng dấu** mới là phần chắc, không phải biên độ. Ghi trước để khỏi
  đọc quá tay.
* Không sửa ngưỡng hay cách đọc sau khi thấy số.

---

## KẾT QUẢ

(chưa điền — đối chứng mới 3/6 ô lúc viết)
