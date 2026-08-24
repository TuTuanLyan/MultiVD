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

---

# KẾT QUẢ ĐỦ 9 NHÁNH — 25/08/2026

## λ_eff học được, tách đôi hoàn hảo theo loại tín hiệu

| backbone | `cwe` | `latent_bottleneck` | `latent_proto` |
| --- | --- | --- | --- |
| t5 | 101.19 | 33.21 | **1.06** |
| t5p | 89.15 | 105.47 | **1.04** |
| t5pe | 87.58 | 60.59 | **1.07** |

**Dự đoán 1 (nhánh có nhãn học λ nhỏ hơn): BỊ BÁC ở 3/3 backbone, ngược chiều, cách nhau
30–100 lần.**

## Vì sao λ_eff của `latent_proto` bằng ~1.05 ở cả ba

Từ `w = 0.5/L`: `λ_eff = w_aux/w_bin = L_bin / L_aux`. Nghĩa là uncertainty weighting
**chỉ làm đúng một việc — cân bằng hai loss** — và λ nó "học được" hoàn toàn bị quyết
định bởi tỉ số hai loss lúc hội tụ. Nó không mang thông tin gì về transfer.

Head `latent_proto` dùng Sinkhorn với gán nhãn cân bằng theo thiết kế, nên loss của nó
nằm cùng thang với cross-entropy nhị phân ⟹ tỉ số ≈ 1. Hai head dùng nhãn CWE thì
thuộc lòng được task 4 lớp trên 1284 dòng, loss về ~0.006 ⟹ tỉ số ~100.

## Dự đoán 2 (λ_eff cao phải tệ hơn λ=0.2): ĐƯỢC XÁC NHẬN, 11/12 ô

Trục λ đủ ba điểm, `none` cùng optimizer làm đối chứng (fold 1–2):

| CodeT5-base, RecAdam | λ=0.05 | λ=0.2 | λ≈101 |
| --- | --- | --- | --- |
| `cwe` | −0.0262 | −0.0723 | **−0.3693** |
| `latent_bottleneck` | −0.0196 | −0.0365 | **−0.3817** (λ≈33) |

## Điều quan trọng nhất: hai loại tín hiệu có DẠNG ĐƯỜNG CONG λ KHÁC NHAU

- **Tín hiệu có nhãn** (`cwe`, `latent_bottleneck`): **đơn điệu giảm theo λ** trên cả ba
  điểm đã đo. Càng đặt nhiều trọng số càng hại. Không có cực trị trong khoảng đã quét —
  giá trị tốt nhất là giá trị nhỏ nhất từng thử.
- **Tín hiệu không nhãn** (`latent_proto`): **có cực đại trong khoảng**, quanh λ≈0.2.
  t5pe/RecAdam: +0.0069 (λ=0.05) → **+0.0299** (λ=0.2) → +0.0034 (λ≈1.07).
  t5p/RecAdam: +0.0231 → **+0.0297** → −0.0006.

Đây là khác biệt **về chất**, không phải về lượng. Một tín hiệu có cực đại nội tại là
tín hiệu thật sự có ích ở đúng trọng số; một tín hiệu đơn điệu giảm là tín hiệu mà cách
tốt nhất là bớt đi. Đây là lần đầu dự án thấy một đường cong λ có đỉnh.
