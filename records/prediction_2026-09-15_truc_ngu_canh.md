# Khai báo trước — TRỤC NGỮ CẢNH: 256 / 512 / 1024 token trên t5p

Viết **15/09/2026, TRƯỚC khi chạy ô nào.** Không sửa phần dự đoán sau khi thấy số.

## Câu hỏi

Người dùng 15/09: *"512 đôi khi nó sẽ nén mất đặc trưng lỗ hổng khi bị dài?"* và đề xuất
multi-window (chunk → embed → mean). Trước khi viết chunking, đo **độ dốc của trục ngữ cảnh**.

Chỉ **t5p**: codebert có trần vị trí 514 (RoBERTa) nên không nới được; codet5p dùng vị trí
**tương đối** (32 bucket) nên đổi một cờ là xong, 0 dòng mã. Chỉ **baseline** (target-only) —
Pha 1 được huấn luyện ở 512 nên chạy Pha 2 ở 1024 sẽ lệch ngữ cảnh giữa hai pha.

## ĐỐI CHỨNG NỘI TẠI — chỗ đắt giá nhất của thiết kế này

Hàng test **≤ 256 token** nhận **đúng cùng một đầu vào** ở cả ba nhánh: không token nào bị cắt
ở bất kỳ độ dài nào. Nên Δ đo trên nhóm hàng đó **chính là sàn nhiễu của phép đo này** — đo
trực tiếp, không phải đi mượn con số 0.010 từ khối khác.

> Hiệu ứng ngữ cảnh thật phải **lớn hơn** sàn đó, và phải nằm ở nhóm hàng **DÀI**, không nằm ở
> nhóm hàng ngắn. Nếu Δ ở hai nhóm bằng nhau thì thứ ta đo được là **nhiễu huấn luyện lại**,
> không phải ngữ cảnh.

## Biến quyết định và ngưỡng, chốt TRƯỚC

Chính: `Δ = ROC(ctx1024) − ROC(ctx512)` trên **hàng test > 512 token** (nơi 1024 thực sự thêm
nội dung), ghép cặp theo fold, n=3. Ký hiệu `S` = |Δ| trên hàng **≤ 256 token** (sàn nhiễu).

| kết quả | kết luận đã chốt trước |
|---|---|
| `Δ ≥ +0.020` **và** ≥ **2/3 fold** **và** `Δ > 2·S` | **Ngữ cảnh có giá** — đáng bỏ công dựng chunking cho codebert |
| `\|Δ\| < 0.010` | **Trục ngữ cảnh phẳng** — multi-window sẽ không ăn ở đây; §60 là khó nội tại chứ không phải pha loãng |
| `Δ ≤ −0.020` **và** ≤ **1/3 fold** | **Nới ngữ cảnh làm HẠI** — xem dự đoán chính bên dưới |
| còn lại | **KHÔNG KẾT LUẬN** |

## Dự đoán của tôi: ô thứ BA — 1024 sẽ **KÉM HƠN** 512

Ba lý do, xếp theo sức nặng:

1. **t5p gộp bằng `mean` trên toàn chuỗi.** Ở 1024 token, trung bình trải trên gấp đôi số token,
   nên một vùng lỗi 5–10 token bị pha loãng **gấp đôi**. Nới ngữ cảnh mà không đổi cách gộp là
   làm mẫu số to ra.
2. **§59.1: cắt không xoá tín hiệu.** `head_middle_tail` giữ 170+170+170, và **0/117** hàng nhóm
   `train` trở nên trùng với bản đối nghịch. Thứ 1024 thêm vào phần lớn là **ngữ cảnh**, không
   phải **phần phân biệt** — mà ngữ cảnh thì đã có đại diện qua cửa sổ giữa và cuối.
3. **28.9% hàng đích vượt 512**, nên chỉ ~1/4 tập test được hưởng gì từ 1024, trong khi **toàn
   bộ** tập chịu chi phí pha loãng.

**Nếu dự đoán này đúng thì nó là lập luận mạnh nhất cho `max`/`log-sum-exp` thay vì `mean`
trong multi-window** — nó cho thấy vấn đề nằm ở **cách gộp**, không ở **lượng ngữ cảnh**.

**Nếu tôi sai** (1024 hơn hẳn) thì multi-window đáng làm ngay, và phải làm cho cả codebert
bằng chunking vì codebert không nới được.

## Dự đoán phụ

- `Δ(512 − 256)` sẽ **dương** và lớn hơn `Δ(1024 − 512)`: đoạn 256→512 còn cứu được phần
  phân biệt bị cắt, đoạn 512→1024 thì chủ yếu thêm ngữ cảnh.
- Sàn nhiễu `S` đo trên hàng ≤256 token sẽ vào khoảng **0.01–0.03** — cùng cỡ với sàn
  cross-GPU 0.028 đã biết, vì đây cũng là "huấn luyện lại cùng cấu hình".

## Thiết lập

9 ô: 3 độ dài × 3 fold, t5p, seed 42, baseline (target-only), `--grad_checkpointing` bật cho
**cả ba** nhánh (chỉ bật cho 1024 thì phép so đổi hai biến). Cùng máy cùng phiên.
Thứ tự `512 1024 256` — 512 là mốc, hỏng sớm thì biết ngay.

## Ràng buộc

- n=3, một máy, một seed. Bậc 1 — chỉ để **dừng hoặc leo bậc**, không viết vào bài.
- Nhóm hàng > 512 token chỉ khoảng **44 hàng/fold** (28.9% của 152). Phương sai lớn; đọc đếm dấu.
- Chạy sau khi `gate1` xong — 158 chỉ còn ~4.5 GB trống khi gate1 đang chạy.
