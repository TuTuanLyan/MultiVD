# Dự đoán ghi trước khối λ học được — 24/08/2026

Viết **trước khi** bước `05_lambda_hoc_duoc` chạy (nó đang xếp sau khối λ=0.05 trên
`ntat2`). Ghi ra đây để không thể sửa lại sau khi nhìn kết quả.

## Cơ sở

Hai quan sát từ chính đợt này, seed 42, bộ fold gốc:

1. **λ=0.2, đủ 5 fold.** Trên ba backbone họ CodeT5, hai head phụ **dùng nhãn CWE**
   (`cwe`, `latent_bottleneck`) âm ở **12/12 ô**; head phụ **không dùng nhãn**
   (`latent_proto`) dương ở **5/6 ô**.
2. **λ=0.05, mới 2 fold.** Giảm λ cải thiện `cwe` ở 5/6 ô và `latent_bottleneck` ở
   5/6 ô, nhưng chỉ 3/6 với `latent_proto` — và **âm ở đúng hai ô `latent_proto`
   đang thắng** (t5p +0.0297 → +0.0231; t5pe +0.0299 → +0.0069).

Đọc gộp: tín hiệu **có nhãn** gây hại trên họ T5 nên hạ trọng số thì đỡ hại; tín
hiệu **không nhãn** có lợi nên hạ trọng số thì mất lợi. Một hằng số λ duy nhất
không phục vụ được cả hai vì chúng cần đi **ngược chiều nhau**.

## Dự đoán

Trên ba backbone họ CodeT5, khi λ được học (Kendall/Gal/Cipolla, khởi tạo tại 0.2):

> **λ_eff học được của `cwe` và `latent_bottleneck` sẽ NHỎ HƠN λ_eff học được của
> `latent_proto`.**

Ngưỡng cụ thể để khỏi cãi nhau sau: dự đoán đúng nếu
`mean(λ_eff của cwe, latent_bottleneck) < mean(λ_eff của latent_proto)` ở **ít nhất
2 trong 3 backbone**.

## Điều dự đoán này KHÔNG nói

σ được học từ **loss huấn luyện Phase 1**, không nhìn thấy Macro-F1 trên target sau
Phase 2. Nên nếu dự đoán **đúng**, nó chỉ chứng minh hai tín hiệu phụ khác nhau về
độ khó/độ nhiễu theo hướng trùng với hướng có lợi cho transfer — một sự trùng khớp
đáng chú ý, **không phải** bằng chứng rằng cân bằng loss là cách chọn λ đúng.

Nếu dự đoán **sai** — λ_eff gần như nhau giữa các nhánh, hoặc ngược chiều — thì kết
luận mạnh hơn: λ tự cân bằng theo loss **không** bắt được thứ quyết định transfer, và
hướng đúng là bilevel căn theo downstream (Karpukhin & Savchenko arXiv:2605.07756,
BiSSL arXiv:2410.02387; xem RESEARCH §4.4).

## Cách kiểm

`λ_eff = exp(s_bin − s_aux)` được ghi vào log mỗi epoch và vào `training_args` của
checkpoint dưới khoá `learned_lambda_eff`. Đọc bằng:

```
grep "lambda hoc duoc KET THUC" <log>
```

---

# KẾT QUẢ — 25/08/2026, nhánh đầu tiên (`t5/cwe_uw`)

## Dự đoán BỊ BÁC, và bác theo hướng ngược hẳn

Dự đoán: `cwe` sẽ học được λ_eff **nhỏ**. Thực tế:

| epoch | 1 | 4 | 7 | 10 | 13 | 15 |
| --- | --- | --- | --- | --- | --- | --- |
| λ_eff | 0.55 | 2.18 | 13.2 | 92.5 | 136.2 | **101.2** |

Khởi tạo tại 0.2, kết thúc tại **101.19** — **gấp 500 lần**, đơn điệu tăng suốt 13 epoch.

## Cơ chế, kiểm bằng giải tích rồi đối chiếu số đo

Cực tiểu `exp(−s)·L + s/2` theo `s` cho `w = 0.5 / L`. Từ trọng số đo được:

| | trọng số học được | ⟹ loss của task |
| --- | --- | --- |
| nhị phân | 0.7831 | **0.6385** |
| phụ (CWE 4 lớp) | 79.24 | **0.00631** |

`λ_eff = 79.24 / 0.7831 = 101.19` — khớp đúng con số ghi trong log.

Nghĩa là: **head CWE 4 lớp trên 1284 dòng hạ loss về ~0.006, tức gần như thuộc lòng.**
Uncertainty weighting đọc loss thấp là "task ít nhiễu, đáng tin" và dồn trọng số cho nó.
Task nhị phân kẹt ở 0.64 nên bị bỏ rơi.

Hệ quả đo được: **val Macro-F1 của Phase 1 tụt còn 0.5652**, so với **0.6350** của cùng
nhánh ở λ cố định. Phase 1 gần như ngừng học chính task cần học.

## Điều này chứng minh

Đúng giới hạn đã ghi trước khi chạy: **σ cân bằng loss huấn luyện, không nhìn thấy lợi
ích transfer.** Và ở đây hai thứ đó không chỉ khác nhau mà **ngược nhau**:

- Số liệu λ=0.05 đủ 5 fold nói `cwe` cần trọng số **THẤP HƠN** (giảm λ bốn lần cải thiện
  6/6 ô, trung bình +0.0383).
- Uncertainty weighting đẩy trọng số của đúng nhánh đó lên **CAO HƠN 500 lần**.

Không phải lỗi cài đặt — đó là hành vi đúng của công thức, và công thức đang tối ưu sai
đại lượng. Một task phụ **dễ thuộc lòng** luôn được ưu ái, bất kể nó có chuyển giao được
gì hay không.

## Hệ quả cho hướng đi

Loại bỏ uncertainty weighting cho bối cảnh này. Hướng còn lại là bilevel căn theo mục
tiêu **downstream** — Karpukhin & Savchenko arXiv:2605.07756 (~30% chi phí thêm) hoặc
BiSSL arXiv:2410.02387 (xem RESEARCH §4.4). Kết quả này cũng cho một dự đoán kiểm được:
nếu λ_eff ≈ 101 thì Phase 2 của nhánh `cwe_uw` phải **tệ hơn** cả λ=0.2, vì nó nằm xa
hơn nữa về phía trọng số cao trên một trục mà số liệu đã cho thấy càng thấp càng tốt.
