# KHAI BÁO TRƯỚC — `+0.0259` của codebert là TRANSFER hay chỉ là THÊM SỨC CHỨA?

**Viết lúc 11:2x UTC 13/09/2026, TRƯỚC khi chạy một ô nào của khối đối chứng.** Số của
nhánh thật (FACTS §47) đã có; số của đối chứng thì **chưa tồn tại**.

## Câu hỏi

FACTS §47: `fusft` hơn đối chứng `latent_bottleneck` trên codebert **+0.0259 F1@0.5 (5/5)**
và **+0.0224 ROC-AUC (5/5)**, n=5, chạm sàn Wilcoxon p=0.0625.

Nhưng `fusft` **thêm ~22M tham số fusion + 1,8M tham số adapter** so với đối chứng. Hai
cách giải thích **không phân biệt được** bằng số hiện có:

* **(T) transfer** — lợi ích đến từ tri thức mà adapter nguồn học được ở Pha 1 trên ccpp+js.
* **(C) sức chứa** — lợi ích đến từ việc có thêm tham số và thêm một đường đi, nội dung của
  adapter nguồn không quan trọng.

Nếu là **(C)** thì hướng này không có giá trị khoa học: bất kỳ module thừa nào cũng làm được,
và bài báo không đứng được.

## Can thiệp

Giữ **nguyên mọi thứ** của nhánh `fusft` — cùng kiến trúc, cùng số tham số, cùng optimizer,
cùng Pha 1 checkpoint, cùng fold, cùng máy — **chỉ thay** trọng số adapter **nguồn** bằng
nhiễu Gauss **cùng thang độ** (khớp std từng tensor), rồi **đóng băng y hệt**. Gọi nhánh này
là `fusftrnd`.

Khớp thang độ là **bắt buộc**: nếu dùng khởi tạo gốc (`up = 0`) thì adapter nguồn thành ánh
xạ đồng nhất, và phép đo chỉ trả lời "bỏ hẳn adapter nguồn thì sao" — chưa loại được (C).

## Dự đoán CỤ THỂ, có thể sai

Gọi `D_that = Δ(fusft − đối chứng)` và `D_rnd = Δ(fusftrnd − đối chứng)`, ghép cặp theo fold,
codebert, n=5, seed 42. Đã biết `D_that = +0.0259` (F1@0.5) và `+0.0224` (ROC-AUC).

**Dự đoán: (T) đúng.** Cụ thể, trên **F1@0.5**:

* `D_rnd < +0.0130` (tức **dưới một nửa** `D_that`), **và**
* số fold dương của `D_rnd` **≤ 3/5**.

## Điều gì BÁC — khai báo trước, không sửa sau khi thấy số

| kết quả của `D_rnd` (F1@0.5) | đọc là |
|---|---|
| `< +0.0130` **và** ≤ 3/5 fold | **(T)** — lợi ích cần tới tri thức Pha 1. Dự đoán sống sót. |
| `≥ +0.0130` **và** ≥ 4/5 fold | **(C)** — chỉ là sức chứa. **Hướng adapter-fusion BỊ BÁC**, ghi DEAD_ENDS và dừng. |
| nằm giữa hai vùng trên | **KHÔNG kết luận.** n=5 không đủ tách; phải lên n=15 hoặc bỏ. Không được đọc theo hướng có lợi. |

## Ràng buộc

* Chỉ số chính **F1@0.5** (vì `D_that` mạnh nhất ở đó); ROC-AUC báo cáo kèm nhưng **không**
  dùng để đổi kết luận — chọn chỉ số sau khi thấy số là cách tự lừa mình.
* **Chỉ codebert.** t5p không có hiệu ứng để giải thích (§47).
* Bậc 2, n=5 fold, seed 42, **chỉ codebert**.

### SỬA 11:5x UTC 13/09 — TRƯỚC khi có bất kỳ ô nào của khối này

Bản đầu viết *"dùng lại đối chứng và Pha 1 đã có ở `results/fus1_codebert`"*. **Không làm được**:
máy vast cũ (50857599) đã huỷ, máy mới (50882617) là **phần cứng khác**, mà mục 4 cấm so nhánh
trên máy này với đối chứng trên máy kia — sàn nhiễu giữa các loại GPU là **0.028**, lớn hơn cả
hiệu ứng đang đo (+0.0259).

Nên khối này chạy **cả ba nhánh trên cùng máy mới**, cây `results/fus2_codebert`:
`latent_bottleneck` (đối chứng) · `fusft` · `fusftrnd`, mỗi nhánh 5 fold = **15 ô**.
Pha 1 có adapter **phải huấn luyện lại** (checkpoint cũ mất theo máy bị huỷ).

Sửa này **không** đụng tới dự đoán, ngưỡng, hay chỉ số chính ở trên — chúng giữ nguyên từng chữ.
Nó chỉ đổi chỗ lấy số, và đổi vì một ràng buộc vật lý, không phải vì đã thấy số.
* Seed nhiễu đổi theo fold (`seed*1000 + fold`) để kết luận không cược vào **một** lần bốc bài.
* **Không** sửa ngưỡng, không đổi chỉ số chính, không thêm fold sau khi thấy số.

### SỬA LẦN 2 — 12:4x UTC 13/09, vẫn TRƯỚC khi có ô Pha 2 nào dùng được

Pha 1 huấn luyện lại trên máy mới **sập**: val macro-F1 kẹt **0.3333** (đoán một lớp) suốt 7
epoch rồi cạn patience. Máy cũ cùng seed 42, cùng mã, cùng phiên bản thư viện thì đạt **0.5583**.

Đối chiếu log từng epoch: hai lần chạy bám sát nhau tới epoch 5 (train loss lệch < 0.003) rồi
**tách ở epoch 6** — lần cũ có một nhịp val loss giảm nên patience reset và nó thoát cao nguyên;
lần mới không có nhịp đó. Tức **Pha 1 này nằm ngay ranh giới**, và khác biệt là phi tất định
của GPU, không phải lỗi môi trường.

**Xử lý, khai báo trước:** nới `PHASE1_PATIENCE=10` cho **riêng Pha 1** (Pha 2 giữ nguyên 5, để
không đổi thêm một biến nào của phép so giữa các nhánh).

**LUẬT CHẤP NHẬN Pha 1 — chốt trước, không sửa sau:**

* Chạy **ĐÚNG MỘT LẦN** với patience 10. Nhận val bằng bao nhiêu thì nhận.
* `val ≥ 0.50` ⇒ chấp nhận, chạy tiếp 15 ô.
* `val < 0.50` ⇒ **DỪNG**, báo "Pha 1 `codebert × com` + adapter không ổn định, câu hỏi
  transfer-hay-sức-chứa **không trả lời được** ở cấu hình này", và **không** bốc lại seed.

Ghi rõ để chặn đúng cái bẫy "chạy lại tới khi ra kết quả mình muốn": Pha 1 là **đầu vào** của
thí nghiệm chứ không phải thứ được đo, nhưng chọn đầu vào theo kết quả vẫn là thiên lệch.

---

## KẾT QUẢ (điền sau khi đủ 5 ô — ĐỂ TRỐNG)
