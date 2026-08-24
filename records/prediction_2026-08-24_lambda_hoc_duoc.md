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
