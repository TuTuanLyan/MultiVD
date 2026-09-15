# Khai báo trước — GATE1: lặp lại ghép muộn + đối chứng ENSEMBLE THUẦN

Viết **15/09/2026, TRƯỚC khi chạy ô nào trên vast.** Không sửa phần dự đoán sau khi thấy số.

## Nền: kết quả 0-GPU đã có

`tools/late_fusion_gate.py` trên cây `shuf1` (2 backbone, n=5, 10 điểm ghép cặp), cổng `logreg`
(4 tham số, học trên val của chính fold đó):

| | F1@0.5 | ROC-AUC | PR-AUC |
|---|---|---|---|
| ghép(baseline, transfer **REAL**) − baseline | **+0.0208 10/10 p=0.002** | +0.0153 9/10 p=0.021 | +0.0147 9/10 p=0.021 |
| ghép(baseline, transfer **XÁO**) − baseline | −0.0032 5/10 | **−0.0103 0/10 p=0.002** | −0.0144 2/10 |

Ba cổng của `NEXT_CONTRIBUTION §3` đều qua: khớp sức chứa (đối chứng cùng kiến trúc, cùng
quy trình, chỉ khác nhãn nguồn thật/xáo), lặp trên **cả hai** backbone, và tách theo nhóm rò rỉ
(lợi ích tập trung ở `train`: +0.0454 so với `none` +0.0095).

## Chỗ CÒN HỞ mà khối này bịt

Đối chứng `shufall` là một model **kém** (ROC 0.775 so với baseline 0.871). Nó **không** trả lời
được: *"ghép hai model TỐT NGANG NHAU thì có tự tăng không?"* — tức đối chứng **ensemble thuần**
mà chính Đề xuất 1 đòi (SR + S). Khối này chạy model target-only **thứ hai**, khác seed (7), để
tính `ghép(baseline₄₂, baseline₇)`.

## Dự đoán, và ngưỡng chốt TRƯỚC

**Biến quyết định:** `Δ_nguồn = [ghép(base₄₂, transfer) − base₄₂] − [ghép(base₄₂, base₇) − base₄₂]`
trên **ROC-AUC**, ghép cặp theo `(backbone, fold)`, 10 điểm.

| kết quả | kết luận đã chốt trước |
|---|---|
| `Δ_nguồn ≥ +0.010` **và** ≥ **8/10** | Đề xuất 1 **đứng**: lợi ích đến từ **tri thức nguồn**, không từ ensemble. Đủ điều kiện lên bậc 3 |
| `Δ_nguồn ≤ +0.003` **hoặc** ≤ **5/10** | Đề xuất 1 **hỏng**: ensemble hai model target-only cũng cho ngần ấy ⇒ không phải đóng góp về transfer |
| còn lại | **KHÔNG KẾT LUẬN** — phải thêm seed |

**Dự đoán của tôi: ô thứ nhất.** Lý do: hai model target-only khác seed sai ở **cùng** những
hàng (cùng dữ liệu, cùng quy trình), nên ghép chúng chủ yếu làm **giảm phương sai**, không
**bổ khuyết**. Còn §58 đã đo được transfer mạnh hơn baseline **+0.12 ROC** đúng ở nhóm `train`
trong khi baseline mạnh hơn ở nơi khác — tức hai model sai ở **chỗ khác nhau**, điều kiện để
ghép ăn.

**Dự đoán phụ (ghi để đếm cả khi sai):**
1. `ghép(base₄₂, base₇) − base₄₂` sẽ **dương nhưng nhỏ**: **+0.003…+0.010** ROC — ensemble luôn
   giúp một chút. Nếu nó ≥ +0.015 thì phần lớn hiệu ứng ở bảng nền là ensemble, không phải nguồn.
2. Khối này **lặp lại** được bảng nền: `ghép(base, transfer) − base` ≥ +0.010 ROC, ≥ 7/10.
   Đây là phép lặp trên **phiên khác** (máy đã tắt rồi bật lại) — nếu nó không lặp thì bảng nền
   chịu chung số phận của §52 (5/5 fold biến mất khi đổi phần cứng).

## Thiết lập

30 ô: 2 backbone × 5 fold × {transfer real seed 42, baseline seed 42, baseline seed 7}.
Pha 1 **dùng lại** `model/shuf1/phase1/*__none_com_real/seed_42/best.pt`, không huấn luyện lại.
Fold là vòng ngoài. Method chạy **trước** baseline (người dùng nêu 15/09).
Cùng cây, cùng máy, cùng phiên. `--sam_rho 0`, AdamW, `max_length 512`, `head_middle_tail`.

## Ràng buộc phải nêu khi đọc

- Cổng `logreg` học trên **152 hàng val**. Đã có đối chứng nhiễu (ghép với xác suất ngẫu nhiên
  cho **−0.0062 ROC, 0/5**), nên nó không tự bịa ra lợi ích — nhưng n=152 vẫn nhỏ.
- n=5 mỗi backbone. Sàn kiểm định dấu ở 10 điểm là p=0.002.
- **Một máy.** Theo luật §52.1, kết quả này vẫn cần một lần lặp trên phần cứng khác trước khi viết.
