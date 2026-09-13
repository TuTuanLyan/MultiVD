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

## 4. ĐỀ XUẤT đã chuẩn bị — dùng cấu trúc cặp đang bị vứt bỏ

**Phát hiện then chốt (14/09):** dữ liệu Pha 1 có trường `pair_id` — mỗi cặp là **cùng một hàm,
trước và sau khi vá, ngược nhãn**. Hiện Pha 1 **xáo trộn rồi ném bỏ** thông tin này.

| nguồn | cặp đầy đủ | ccpp / js | % dòng nằm trong cặp |
|---|---|---|---|
| `com` | **1755** | 1063 / 692 | 93,8% |
| `full` | 3463 | 2685 / 778 | 91,2% |
| `4cwe` | 455 | 49 / 406 | 97,8% |

Mỗi cặp là **âm khó nhất có thể** — đúng thứ nhóm `train` ở §50 đo.

### Đề xuất A — căn HƯỚNG-VÁ xuyên ngôn ngữ *(mạnh nhất)*

Với mỗi cặp, `d = h(vul) − h(fixed)` là **vector hướng lỗ hổng**. Giả thuyết: với **cùng một
CWE**, hướng đó phải **giống nhau giữa các ngôn ngữ**. Ba thành phần loss:

1. **trong cặp** — tách `vul` khỏi chính bản vá của nó
2. **xuyên ngôn ngữ** — `d` của cặp ccpp cùng hướng với `d` của cặp js **cùng CWE**
3. **tương phản theo CWE** — và khác hướng với `d` của CWE khác

Vì sao đáng: **0 tham số lúc suy luận** ⇒ qua cổng 1 về mặt cấu trúc; **chỉ có nghĩa khi ≥2
ngôn ngữ** ⇒ là phát biểu cross-language thật, không phải "khởi tạo tốt"; **dùng thông tin đang
bị vứt**.

Đối chứng đã thiết kế: (a) **xáo nhãn CWE** khi ghép cặp dương xuyên ngôn ngữ — giữ độ lớn
loss, phá nội dung; (b) căn **chỉ trong cùng ngôn ngữ** — kiểm phần "xuyên ngôn ngữ" có làm gì
không.

> **CHƯA tra tài liệu.** Contrastive/patch-based cho vuln detection thì đã có; *căn hướng-vá
> giữa các ngôn ngữ theo CWE* thì chưa thấy, nhưng đó là trí nhớ chứ không phải phép tra cứu.
> **Việc đầu tiên sáng mai: tra tài liệu trước khi viết code.**

### Đề xuất B — mất mát BIÊN trong cặp *(rẻ, bậc thang dưới của A)*

Thay phân loại nhị phân độc lập từng dòng bằng `score(vul) ≥ score(fixed) + m` **trong từng
cặp**. Cũng **0 tham số**. Nếu riêng nó đã ăn thì phần "xuyên ngôn ngữ" của A mới là thứ phải
chứng minh thêm.

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

## 6. Đang chạy / đang chờ

- **local `fus3`**, 15 ô (5 baseline + 5 chốt + 5 `fusft`, codebert, n=5), `run/fusion_base.sh`.
  Đang **chờ GPU rảnh** — cổng đòi VRAM ≥ 11,5 GB ổn định 3 lần liên tiếp và không có job
  python của người khác. Cho con số "fusion vs baseline" đo **cùng máy** — hiện chưa có.
- **vast**: không còn instance nào. Chỉ thuê khi thật sự cần gấp.
