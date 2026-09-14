# Hướng tiếp — TÌM MỘT ĐÓNG GÓP HỌC THUẬT MỚI

> **Đọc file này đầu tiên khi bắt đầu phiên mới.** Nó thay cho toàn bộ ngữ cảnh đã compact.
> Viết 14/09/2026.

---

## 0. TINH THẦN — đọc kỹ, đây là chỗ dễ đi sai nhất

Mục tiêu **KHÔNG** phải chứng minh phản biện sai, **KHÔNG** phải "pass review".

Người dùng **và** reviewer cùng muốn: **một đóng góp thực tiễn đủ mới** — một **cách huấn
luyện** hoặc **kiến trúc** mới trong lĩnh vực phát hiện lỗ hổng — và nó phải **tốt bởi chính
nó**, **áp dụng được tổng quát**, chứ không phải hiệu ứng của **backbone** hay của **seed**.
Reviewer không làm khó; họ cũng đang tìm một đóng góp tốt.

**Hệ quả khi đề xuất việc: ưu tiên THỬ CƠ CHẾ MỚI.** Đừng khuyên "dừng thêm module". Đừng đề
xuất thí nghiệm phòng thủ.

Phần hiện có — học biểu diễn ở Pha 1 với head phụ nhận diện CWE — mới **một phần**, **chưa đủ
mới** theo chuẩn hiện tại. Đó là lý do cần hướng khác.

**Hai điều đã chốt, đừng đề xuất lại:**

1. **Split ngẫu nhiên (`sven_python_folds_norm`) là CHỦ Ý của reviewer, không phải điểm yếu.**
   Nó cố ý chứa ca gần-trùng-lặp *và* ca gần giống mẫu train nhưng **ngược nhãn** ở test —
   đoán đúng ở đó mới chứng tỏ học đặc trưng lỗ hổng. `twin` **vẫn chạy, như SIDE RESULT, để
   sau** (chính reviewer đề xuất), nhưng **không** phải "câu trả lời cho phản biện rò rỉ".
2. **ccpp-only / js-only / trộn đã thử** (kết quả thất lạc một phần): `ccpp` một mình quá ít
   dữ liệu nên yếu; `js` nhiều CWE-079 nên bổ trợ 079 nhưng vẫn yếu hơn gộp.

---

## 1. Trạng thái — cái gì ĐÃ CHẮC

| | bằng chứng |
|---|---|
| **Phương pháp chốt khái quát hoá THẬT, không sống nhờ rò rỉ** | §50: trên **73% hàng sạch** vẫn **+0.0459, 14/15 fold, p=0.001** so với baseline |
| **Lợi ích transfer tăng đơn điệu khi tập đích co lại** | §40.5: n=15, 120 ô, **cả hai backbone**. Ở N=456 hiệu ứng thứ hạng **bằng 0** (8/15, p=1.000); ở N=76 là **+0.1849 / +0.1188 (14/14)** |
| **ASAM cải thiện THỨ HẠNG, không cải thiện ngưỡng 0.5** | §2b, lặp 5 lần |

## 2. Trạng thái — cái gì ĐÃ BỊ BÁC (đừng làm lại)

| hướng | vì sao chết | mục |
|---|---|---|
| Head phụ chất lượng cao hơn | head giỏi gấp ba không đổi gì sau fine-tune | §45 |
| Nút thắt 8 chiều là "biểu diễn" | thua PCA-8 ở cả bốn checkpoint | §44 |
| Head phụ hơn "finetune 2 lần thuần" | +0.0005 trên 132 ô; toàn bộ trung bình đến từ 5 ô Pha 1 sập | §46 |
| **AdapterFusion** | một nửa lợi ích tái tạo được bằng adapter **nhiễu**; không lặp trên t5p | §47, §48 |
| ↳ và trên nhóm khó, adapter **đã học** THUA adapter **ngẫu nhiên** | −0.0595, 1/5 | §50 |
| Neo đặc trưng / LP-FT | mọi nhánh thắng backbone này đều thua backbone kia | §38, §38.2 |
| `RecAdam + ASAM ρ=2.0` làm cấu hình chốt | trung bình **thấp hơn baseline**, âm ở 5/11 điều kiện | §49 |

## 3. BA CỔNG mà mọi đề xuất mới phải qua

Rút từ chính các thất bại trên. Thiết kế thí nghiệm **trước khi chạy**, không thêm sau.

1. **Đối chứng KHỚP SỨC CHỨA.** Thêm tham số thì lợi ích tăng theo sức chứa chứ không theo tri
   thức — đã đo hai lần (§46, §48). Mọi module mới phải có nhánh "cùng số tham số, nội dung
   ngẫu nhiên/xáo". *Cách né sạch nhất: đóng góp ở **hàm mục tiêu**, không thêm tham số lúc suy
   luận — khi đó phản biện "chỉ là sức chứa" không áp dụng được về mặt cấu trúc.*
2. **Lặp trên CẢ HAI backbone** (codebert + t5p) ở n=3 trước khi leo bậc. §38.2: mọi cơ chế đã
   thử đều tách theo backbone.
3. **Tách theo NHÓM RÒ RỈ** (`tools/leak_groups_pair.py`). Thắng ở nhóm `none` = khái quát hoá
   thật; thắng chỉ ở nhóm `train` = học vẹt. §21.1 và §50 cho thấy hai cơ chế khác nhau rơi vào
   hai nhóm khác nhau — điểm tổng che mất điều đó.

Và: **n=1 không đủ để mô tả một cơ chế** — §48.2 đo được biến thiên **giữa hai lần chạy cùng
cấu hình** còn lớn hơn hiệu ứng cần đo trên một fold.

## 4. ĐỀ XUẤT A đã BỊ BÁC bằng chính số đo — 14/09, FACTS §53

**Đề xuất A** (căn hướng-vá xuyên ngôn ngữ theo CWE) **đóng**. Lý do, đo bằng
`tools/patch_direction_probe.py` trên CPU, không tốn giây GPU huấn luyện nào:

| | CodeBERT **gốc** | sau **Pha 1** |
|---|---|---|
| cos hướng-vá cùng CWE khác ngôn ngữ | +0.0081 | +0.2850 |
| hiệu (cùng CWE − khác CWE) | −0.0049, z=−1.76 | +0.0383, **z=+4.24** |
| `\|trung bình toàn cục\|/\|d\|` | 0.152 | **0.779** |
| AUC trên PYTHON từ μ ước ở ccpp+js | **0.5242** | **0.6477** |

Backbone gốc **không có** trục lỗ hổng — mọi AUC 0.50–0.53, ngang hướng ngẫu nhiên, hướng-vá
**trực giao**. **Pha 1 tạo ra toàn bộ hiệu ứng.** Nên `d = h(vul) − h(fixed)` chính là trục
quyết định của bộ phân loại Pha 1, và một loss ép căn nó chỉ **dựng lại tường minh** cái mà
huấn luyện nhị phân đã dựng ngầm. Không phải cơ chế mới.

Và phần "theo CWE" tuy chắc về thống kê thì **nhỏ**: dùng μ của **sai** CWE chỉ mất ~0.03 AUC.

### Giả thuyết THAY THẾ, rơi ra từ chính số đo — NÉN SỤP CHIỀU

Pha 1 kéo không gian hướng-vá từ **trực giao** (0.152) về **gần một chiều** (0.779). Sau huấn
luyện, bản vá SQL-injection và bản vá path-traversal trỏ gần cùng hướng: thứ phân biệt loại lỗ
hổng **bị ném đi**. Đó đúng là cơ chế **arXiv:2309.17002** (ICLR'24) nêu là nguyên nhân nhiễu
nhãn tiền-huấn-luyện **luôn làm hại out-of-domain** — và xuyên ngôn ngữ **là** out-of-domain.

> **Vấn đề không phải hướng-vá chưa đủ căn. Pha 1 căn QUÁ TAY.**
> Hàm mục tiêu **chống nén sụp** trong không gian hướng-vá sẽ giữ lại phần phân biệt CWE,
> và phần đó mới là thứ chuyển giao sang một ngôn ngữ có phân bố CWE khác hẳn.

Ba điểm khiến nó đáng theo: **rơi ra từ số đo** chứ không từ suy đoán; vẫn là **thay đổi hàm
mục tiêu, 0 tham số lúc suy luận** ⇒ qua cổng 1 về mặt cấu trúc; và có **đại lượng cơ chế đo
trực tiếp được** (`|trung bình|/|d|`, hiện 0.779) để kiểm cơ chế hoạt động **tách khỏi** việc
điểm số có lên hay không — thứ mà mọi hướng đã thất bại trước đây đều thiếu.

**Chưa được chạy.** Phải đợi `shuf1` trả lời trước: nếu nhãn nguồn không mang tri thức gì thì
hướng này cũng vô nghĩa.

### Đề xuất B vẫn còn — mất mát BIÊN trong cặp

`score(vul) ≥ score(fixed) + m` trong từng cặp, thay phân loại nhị phân độc lập từng dòng.
**0 tham số.** Lập luận chống nhiễu (§51): nhãn *tuyệt đối* sai 40–75%, nhưng quan hệ *tương
đối* "bản này trước bản vá" **đúng theo cấu tạo**; một cặp mà commit không liên quan bảo mật
chỉ cho gradient **yếu** thay vì gradient **sai**. Chưa bị bác, chưa được chạy.

## 5. Hạ tầng đã sẵn

| | |
|---|---|
| `tools/leak_groups_pair.py` | so hai nhánh bất kỳ theo nhóm rò rỉ (cổng 3) |
| `tools/report2.py` | bắt buộc in cả bốn chỉ số + đếm dấu |
| `tools/head_vs_none.py` | giá trị riêng của head phụ |
| `src/adapters.py` + `tests/test_adapters.py` | adapter/fusion, 18 phép kiểm hai chiều |
| `--fusion_src_random` | mẫu cho đối chứng khớp sức chứa |
| `data/*.jsonl` có `pair_id` | cấu trúc cặp cho đề xuất A/B |
| Pha 1 đã có ở local | `model/n48/phase1/` (không adapter), `model/fus2/phase1/` (có adapter) |

## 6. Đang chạy / đang chờ — cập nhật 14/09 04:10 UTC

- **`shuf1` trên vast `ntat`** (5060 Ti), 24 ô, bậc 1. Đối chứng **xáo nhãn nguồn**: lợi ích
  Pha 1 là **tri thức lỗ hổng** hay chỉ **phơi nhiễm miền**? Dự đoán ghi trước khi đo ở
  `records/prediction_2026-09-14_xao_nhan_phoi_nhiem_hay_tri_thuc.md`. **Câu trả lời của nó
  quyết định mọi hướng phía sau có đáng làm không.**
- **`fus3` local** (A4000), 15 ô — bản độc lập của `fus5060`, cây tên khác nên không gộp.
- **ĐÃ XONG**: `fus5060` 15/15 ô (**FACTS §52**), probe hướng-vá (**FACTS §53**).
