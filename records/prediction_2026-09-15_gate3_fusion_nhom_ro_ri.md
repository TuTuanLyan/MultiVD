# Khai báo trước — CỔNG 3 cho khối §56: lợi ích của fusion nằm ở nhóm rò rỉ nào?

Viết **15/09/2026, TRƯỚC khi chạy `tools/leak_groups_pair.py` lên hai cây `fusnone7_codebert`
và `fusnone_t5p_t5p`.** Không sửa phần dự đoán sau khi thấy số.

## Vì sao có phép đo này

§56 kết luận head phụ **không đóng góp gì**: `fusft − nonefus` = **+0.0053, 5/10** F1@0.5 và
**−0.0017, 3/10** ROC. Nhưng **điểm tổng trộn ba nhóm rò rỉ**, nên một con số null có thể là
hai hiệu ứng ngược dấu triệt tiêu nhau:

- nhóm **`train`** (hàng test có bản đối nghịch gần trùng, **ngược nhãn**, nằm trong train) —
  đoán đúng ở đây nghĩa là phân biệt được **đặc trưng lỗ hổng**, không phải khớp mẫu văn bản.
  Đây chính là tiêu chí reviewer nêu khi yêu cầu giữ split ngẫu nhiên.
- nhóm **`none`** (73% số hàng, không có bản đối nghịch) — **khái quát hoá thường**.

§50 đã chứng minh phép tách này phân biệt được hai cơ chế khác nhau: `fusft − fusftrnd` tổng
là +0.0115 nhưng tách ra thì `train` = **−0.0595** còn `none` = **+0.0273**.

**0 GPU, 0 huấn luyện** — đọc lại `test_probabilities` đã ghi trong 30 ô của §56.

## Câu hỏi 1 — null của head là ĐỀU hay là TRIỆT TIÊU?

`fusft − nonefus` trên nhóm `train`, gộp hai backbone (10 ô ghép cặp):

| kết quả | kết luận đã chốt trước |
|---|---|
| Δ ≥ **+0.020** **và** ≥ **7/10** | Head **CÓ** đóng góp cho phân biệt lỗ hổng. §56 phải nêu thêm điều kiện: null ở điểm tổng che mất một hiệu ứng thật ở nhóm khó |
| **\|Δ\| < 0.020** hoặc đếm dấu **4–6/10** | Null là **đều** — §56 đứng vững và mạnh thêm: head không đóng góp ở bất kỳ nhóm nào |
| Δ ≤ **−0.020** **và** ≤ **3/10** | Head **làm hại** phân biệt lỗ hổng — cùng mẫu hình với adapter đã học ở §50 |
| còn lại | **KHÔNG KẾT LUẬN** |

**Dự đoán của tôi: ô giữa (null đều).** Lý do: §55.3 đo được nhánh không-head khi học được
đạt Pha 1 **0.5868** so với 0.5790 của nhánh có head — head không tạo ra biểu diễn tốt hơn,
nên nó không có lý do gì để ăn riêng ở nhóm khó.

## Câu hỏi 2 — lợi ích +0.05 so với baseline trông giống HỌC hay giống KHÁI QUÁT HOÁ?

`fusft − baseline` và `nonefus − baseline`, tách theo nhóm, **từng backbone riêng**.

Giả thuyết **H1** đang kiểm (sức chứa là **đường vòng**, không phải tri thức) dự đoán:

- **cả hai** nhánh dương ở **cả hai** nhóm, và nhóm `none` gánh **ít nhất ngang** nhóm `train`:
  `Δ_none ≥ Δ_train − 0.020`.
- Nếu ngược lại — `Δ_train` lớn hơn `Δ_none` **ít nhất 0.030** ở **cả hai** nhánh và **cả hai**
  backbone — thì đó là bằng chứng **chống H1**: lợi ích tập trung đúng chỗ đo phân biệt
  gần-trùng-lặp, tức nó trông giống tri thức lỗ hổng thật.

**Dự đoán của tôi: H1 sống** — hai nhóm gần nhau, `none` không thua `train` quá 0.02.

## Ba ràng buộc phải nêu khi đọc kết quả

1. **Nhóm `train` chỉ ~16–27 hàng/fold.** Phương sai lớn hơn hẳn nhóm `none` (107–113 hàng).
   Mọi Δ ở nhóm này phải đọc kèm đếm dấu, không đọc trung bình một mình. §50 đã ghi đúng
   cảnh báo này.
2. **Công cụ chỉ in macro-F1@0.5** — một chỉ số, trái với luật bốn chỉ số của CLAUDE.md §2b.
   Nên kết quả này **chỉ được dùng để bổ nghĩa** cho §56, không thay thế nó.
3. **n=10 thì sàn kiểm định dấu là p=0.002 (10/10)**; 7/10 cho p=0.344 — tức ngưỡng ≥7/10 ở
   trên là ngưỡng **mô tả**, không phải ngưỡng có ý nghĩa thống kê. Đã ghi rõ ở đây để không
   diễn giải nới ra sau.
