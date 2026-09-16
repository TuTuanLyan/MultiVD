# Khai báo trước — GATE2: ghép hai model BIẾT THỨ KHÁC NHAU vs hai model BIẾT THỨ GIỐNG NHAU

Viết **16/09/2026, trước khi đọc bất kỳ ô nào của `gate2`.** Không sửa phần dự đoán sau khi thấy số.

## Câu hỏi

`gate1` cho thấy phần lớn lợi ích ghép muộn là **ensemble**. Nhưng ensemble có hai nguồn gốc
khác nhau, và chúng dẫn tới hai kết luận khác nhau:

1. **Giảm phương sai** — hai bản sao cùng một loại tri thức, trung bình lại thì bớt nhiễu.
   Có ở *mọi* ensemble, không phải đóng góp.
2. **Bù trừ** — hai model **biết thứ khác nhau** nên sai ở chỗ khác nhau. Đây mới là thứ
   Đề xuất 1 tuyên bố.

Tách bằng cách so hai kiểu ghép, **cùng một model neo** là `transfer seed 42`:

```
X = ghép(transfer₄₂, baseline₄₂)   — model thứ hai KHÔNG có tri thức nguồn  → bù trừ + giảm phương sai
Y = ghép(transfer₄₂, transfer₇)    — model thứ hai CÓ CÙNG tri thức nguồn   → chỉ giảm phương sai
```

`transfer₇` dùng **đúng file checkpoint Pha 1 của seed 42** (symlink, đã đối chiếu:
`source_best_val_macro_f1` trùng tới chữ số cuối). Hai model transfer vì thế khác nhau
**chỉ ở ngẫu nhiên Pha 2** — đúng phép đối ứng với `baseline₄₂` vs `baseline₇`.

## Biến quyết định và ngưỡng

`X − Y` trên **ROC-AUC**, ghép cặp theo `(backbone, fold)`, 6 điểm. Cổng `logreg`.

| kết quả | kết luận đã chốt trước |
|---|---|
| `X − Y ≥ +0.010` **và** ≥ **4/6** | **Bù trừ là THẬT.** Model thứ hai mang tri thức khác thì đáng giá hơn một bản sao. Đây là phần cứu được của Đề xuất 1 |
| `\|X − Y\| < 0.005` **hoặc** ≤ **3/6** | **Chỉ là giảm phương sai.** Đề xuất 1 đóng: ghép muộn ở đây không liên quan tới transfer |
| còn lại | **KHÔNG KẾT LUẬN** |

## Dự đoán của tôi

**Ô giữa hoặc ô cuối — nghiêng về "không kết luận", và tách theo backbone sẽ ngược nhau.**

Lý do: `gate1` đã cho thấy ghép ăn **chỉ trên codebert**, nơi hai model mạnh ngang nhau
(baseline 0.8640 vs transfer 0.8453) và ghép hơn cả hai (+0.0258, 3/3). Trên t5p transfer trội
hẳn (0.9130 vs 0.8722) và ghép không thêm gì. Nên tôi đoán:

- **codebert**: `X − Y` **dương rõ** (≥ +0.02) — đúng chỗ bù trừ có thật.
- **t5p**: `X − Y` **âm hoặc ~0** — ghép với baseline yếu hơn còn hại, nên ghép với một
  transfer thứ hai sẽ tốt bằng hoặc hơn.
- **gộp 6 điểm**: triệt tiêu một phần, rơi vào vùng không kết luận.

Nếu đúng thì phát biểu cuối của Đề xuất 1 **không phải** *"transfer làm ghép tốt hơn"* mà là
*"ghép ăn khi hai model mạnh ngang nhau và sai khác nhau — transfer chỉ là một cách tạo ra
tình huống đó"*. Đó là phát biểu hẹp hơn nhưng bảo vệ được.

## Ràng buộc

- n=3, một máy (158), một hạt giống thay thế duy nhất (7). Bậc 1.
- `Y` chỉ có **một** cặp seed. Muốn chắc thì cần nhiều cặp seed hơn.
- Fold 1 đã cho thấy hai lần chạy cùng checkpoint Pha 1 khác hạt giống lệch **0.061 ROC**
  (0.8898 vs 0.8287) — phương sai theo hạt giống ở đây lớn, nên 6 điểm là ít.
