# Ghi dự đoán TRƯỚC KHI ĐO — khối `pairB`: mất mát BIÊN TRONG CẶP ở Pha 1

Viết **15/09/2026, trước khi chạy ô nào**. Nhánh git `fusion`. Đây là **Đề xuất B** của
`NEXT_CONTRIBUTION.md` §4.

---

## Lập luận

**FACTS §51**: CleanVul tự công bố bộ dữ liệu lỗ hổng mang **40–75% nhiễu nhãn**. Số của dự án
khớp: tác vụ nguồn chỉ học được tới **0.57–0.59** trên ba seed ở ba máy.

**FACTS §54.1** đo được biên độ: tiền-huấn-luyện bằng nhãn **sai** hại **−0.064**; bằng nhãn
**thật** được **+0.024**. Khoảng **0.06** đang bị tiêu vào việc gỡ lại thiệt hại do chính việc
huấn luyện trên bộ dữ liệu này gây ra.

**Cơ chế đề xuất.** Nhãn *tuyệt đối* sai 40–75%, nhưng quan hệ *tương đối* — "bản này trước bản
vá, bản kia sau" — **đúng theo cấu tạo** (đến từ thứ tự commit, không từ phán đoán bảo mật).
Với một cặp mà commit **không** liên quan bảo mật, hai hàm gần như đồng nghĩa:

| | cặp không liên quan bảo mật |
|---|---|
| BCE từng dòng | ép khẳng định **chắc chắn** bản trước "có lỗ hổng" ⇒ gradient **SAI** |
| biên trong cặp | ràng buộc thoả bằng một khe **nhỏ** ⇒ gradient **YẾU** |

Nhiễu bị hạ cấp từ **giám sát sai** xuống **giám sát yếu**. **0 tham số thêm lúc suy luận** ⇒
qua cổng 1 của `NEXT_CONTRIBUTION.md` về mặt cấu trúc.

## Thiết kế

`L = CE(vul_logits, label) + β · mean_pairs softplus(m − (s_vul − s_fixed))`, với
`s = logit[1] − logit[0]` (biên quyết định, không phải xác suất — xác suất bão hoà làm gradient
tắt đúng chỗ mô hình đã tự tin).

| | |
|---|---|
| **β = 0.5, m = 1.0** | chọn trước; ở thử khói, loss cặp ≈ 1.27 so với CE ≈ 0.69, nên β=0.5 làm hai số hạng cùng cỡ |
| sampler | `PairBatchSampler` — xáo theo **cặp**, hai nửa luôn cùng batch. Không có nó thì xác suất cặp lọt cùng batch 16 trên 3370 dòng chỉ ~0.4% |
| nguồn | `com` — **1755 cặp đầy đủ**, 93.8% số dòng |
| ba nhánh | `baseline` (không Pha 1) · `bce` (Pha 1 BCE thuần) · `pairB` (BCE + biên cặp) |
| Pha 2 | **thuần**, không adapter/fusion, `aux_mode=none`, adamw, `--sam_rho 0` — để cô lập **đúng hàm mục tiêu Pha 1** |
| quy mô | **bậc 1, n=3 fold**, seed 42, **cả hai backbone** (cổng 2) |

Bỏ head phụ vì **§56** vừa cho thấy nó không đóng góp gì (+0.0053, 5/10).

## Dự đoán của tôi, có số

1. `pairB` Pha 1 val **thấp hơn** `bce` (0.57–0.59) — vì biên cặp không tối ưu trực tiếp cho
   độ chính xác từng dòng. Đoán **0.54–0.58**. **Val Pha 1 thấp hơn KHÔNG phải tin xấu** ở đây.
2. Trên đích: `pairB − bce` **dương**, khoảng **+0.01…+0.03** F1@0.5.
3. Hiệu ứng **lớn hơn trên `full`** (nguồn bẩn nhất) so với `com` — nhưng khối này chỉ chạy
   `com`, nên đây là dự đoán cho khối sau nếu có.

## Ngưỡng quyết định — đặt TRƯỚC, kèm vùng "không kết luận"

Biến quyết định: `D = Δ(pairB − bce)` trên **F1@0.5**, ghép cặp theo `(backbone, fold)`, 6 điểm.

| `D` | số điểm cùng dấu | kết luận |
|---|---|---|
| ≥ **+0.020** | ≥ **5/6** | **đáng leo lên n=5**; và phải kiểm tiếp bằng đối chứng xáo-chiều-cặp |
| ≤ **+0.005** | bất kỳ | **không phân biệt được với nhiễu** ⇒ đóng Đề xuất B ở β này |
| giữa 0.005 và 0.020 | | **KHÔNG KẾT LUẬN** — thử β khác trước khi bỏ |
| **âm** rõ | ≥ 5/6 ngược dấu | biên cặp **gây hại** — ghi lại và đóng |

In **cả bốn** chỉ số kèm đếm dấu. Kết luận chỉ phát biểu khi **cả bốn cùng hướng** và **cả hai
backbone cùng dấu** — §56 vừa cho thấy hai backbone đổi dấu là chuyện thường.

## Điều đã biết có thể làm hỏng phép đo

- **Một giá trị β duy nhất.** Kết quả null ở β=0.5 **không** bác bỏ ý tưởng, chỉ bác bỏ β đó.
  Phải ghi rõ khi báo cáo.
- **Pha 1 dao động giữa các lần rút** (§48.2, §55.3: `none`+adapter hỏng 3/6 seed). Một lần rút
  mỗi nhánh là mong manh; nếu kết quả nằm ở vùng giữa thì phải thêm seed chứ không đọc theo
  hướng mình muốn.
- **Sampler đổi thứ tự batch** so với nhánh `bce`. Đó là thay đổi **bắt buộc** để loss kích hoạt,
  nhưng nó có nghĩa hai nhánh khác nhau **hai** thứ: hàm mục tiêu **và** thứ tự dữ liệu. Nếu
  `pairB` thắng, phải chạy thêm một nhánh `bce + sampler theo cặp nhưng β=0` để tách. **Đã ghi
  trước, không phải nghĩ ra sau.**

## Kết quả

*(để trống — điền sau khi đo, không sửa phần trên)*
