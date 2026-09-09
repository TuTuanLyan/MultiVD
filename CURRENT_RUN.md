# CURRENT_RUN — 09/09/2026, 07:36 UTC (14:36 giờ VN)

> Bản trước viết 03:45, trước khi có §25.8 → §28. Đã thay hẳn.

## ĐANG CHẠY — ưu tiên mới 09/09 16:20 VN: cấu hình chốt trên HAI backbone

Người dùng: *"chốt gần như là latent_bottleneck + ASAM 2.0 + RecAdam 0.05 trên t5p nhưng codebert
thì tôi chưa rõ. Hyper param có thể điều chỉnh theo từng backbone được."* — đúng: bảng tổng hợp
cho thấy **codebert CHƯA HỀ được chạy với ASAM ρ=2.0**; mức tốt nhất hiện có của nó là `r0p1`
(+0.0125 ROC, 10/10) và `plain` (+0.0112, 11/16), đều ở n nhỏ.

**Khối `run/chot2bb.sh`** trên **161**, phóng 09:23 UTC. Thiết kế đúng như người dùng nêu —
2 backbone × 2 điều kiện × 5 fold:

| | điều kiện | Pha 2 |
|---|---|---|
| **A** | `r2p0` | **RecAdam + ASAM ρ=2.0** — đầy đủ optimizer của phương pháp |
| **B** | `plain` | **AdamW, không SAM/ASAM** — tắt cả hai cùng lúc |

Backbone: `t5p` (codet5p-220m-bimodal, mean) và `codebert` (codebert-base, cls).
Nguồn **4cwe**, λ **0.05**, seed 42, fold 1–5. Đối chứng `baseline` chạy cùng máy cùng fold, dùng
chung cho cả hai điều kiện. **30 ô** (10 baseline + 20 Pha 2).

**Vì sao λ=0.05 chứ không phải 0.01**: dòng λ=0.01 mà người dùng thấy (`r2p0`, ROC +0.0553) chỉ
có ở **n=3** — bậc 1, nơi p=0.250 là **sàn**, không phân biệt được với may mắn. Bản λ=0.05 có
**n=18, ROC +0.0399 (17/18)**. Thêm nữa λ nằm ở Pha 1 nên đổi λ là phải huấn luyện lại Pha 1 cả
hai backbone (CLAUDE.md mục 5).

Pha 1: t5p đã có ở `model/n48/phase1`; **codebert phải huấn luyện lại** (checkpoint cũ nằm trên
ntat2 đã huỷ) — script tự làm, ~20 phút.

**Giám sát**: `scripts/watch_chot2bb.sh` chạy cron 10 phút, phóng lại nếu driver chết mà chưa đủ
30 ô. Nó hỏi **lock** chứ không đếm tiến trình (`ps|grep` bắt luôn dòng lệnh của chính nó — lần
đầu báo driver=4 trong khi chỉ có một).

**Đọc kết quả khi xong**:
```
python3 tools/report2.py --a r2p0  --b plain results/chot_t5p results/chot_codebert
python3 tools/build_summary.py results/chot_t5p results/chot_codebert   # tổng hợp riêng
```

## Máy — CẢ HAI MÁY VAST ĐÃ HUỶ, không còn gì tính tiền

| máy | trạng thái |
|---|---|
| ~~ntat~~ (id 50223254) | **huỷ 08:07 UTC** — 221/221 file khớp byte, 204 file log đã kéo về |
| ~~ntat2~~ (id 50223345) | **huỷ 09:18 UTC** — 240/240 file khớp byte, 258 file log đã kéo về |
| 161 · 158 (local) | trống — được phép, không tốn tiền |

Cả hai lần đều theo đủ trình tự: kéo **mọi** cây kết quả + log → đối chiếu **số file và kích
thước byte** → `destroy` (không phải `stop`) → xác nhận bằng `vastai show instances`.
Máy của người khác (`dung`, `cuongtm4070s`) **không bị đụng tới**.

## Việc CÒN LẠI nếu thuê máy tiếp

1. **`wblend_cb` nguồn `full`** — 3 ô Pha 2 đã chạy nhưng bước nội suy bị bỏ vì `wblend.sh` dọn
   checkpoint baseline sau mỗi fold (§27.2). Muốn có n=9 cho bản lặp codebert thì phải **xoá file
   kết quả JSON** của baseline + nhánh `full` để `matrix.sh` huấn luyện lại, rồi chạy
   `run/wblend_cb.sh|full|1 2 3`. ~40 phút.
2. **Sửa `run/wblend.sh`** trước khi chạy lại: giữ checkpoint baseline đến hết khối, hoặc xoá JSON
   khi chạy bù. Và cổng đếm hiện vật nên đếm **file nội suy**, không phải `fold*.json`.
3. **`twin`** — người dùng đã nêu để sau.

## Kết quả đêm nay — đọc theo thứ tự này

### 1. §28 — con số đầu bài (cấu hình chốt, bậc 3, KHÔNG cần chạy thêm)

`latent_bottleneck` + λ0.05 + **AdamW + ASAM ρ=2.0** vs **baseline**, ntat n=15:
F1 **+0.0535** (12/15) · F1@val **+0.0685** (12/15) · ROC **+0.0337** (14/15) · PR **+0.0300** (13/15).
ntat2 độc lập n=9 cùng dấu cả bốn. Tắt ASAM thì ROC chỉ +0.0137 và PR **−0.0018**.

Theo CWE (ntat n=15): **022 +0.2238 ROC (13/15)** · **079 +0.3638 ROC (15/15)** ·
078 −0.0097 (ns) · 089 +0.0079 (ns). ntat2: **079 +0.2614 với 9/9 fold**.
→ Toàn bộ mức tăng nằm ở hai lớp hiếm; hai lớp thường chiếm 121/150 hàng và đúng bằng không.

### 2. §25.x — phép trộn, sau khi đã trừ đối chứng

- Đối chứng âm (trộn hai baseline **khác seed**, n=48): ROC +0.0079, PR +0.0111. Nên
  *"trộn hơn baseline"* **một mình nó không phải bằng chứng**.
- **Ghép cặp trực tiếp** (cùng fold, cùng baseline) t5p n=48: ROC **+0.0121 (42/48)**,
  F1 +0.0200, PR +0.0116. Trên sàn nhiễu.
- **codebert n=50**: biên độ tổng **KHÔNG lặp** (ROC +0.0052, p=0.48) nhưng **per-CWE thì lặp**:
  022 +0.0858 (38/50) · 079 **+0.1127 (45/50)**; CWE-089 (54% hàng) **âm có ý nghĩa** → tổng triệt tiêu.
- ⇒ **Phát biểu không phụ thuộc backbone là phát biểu per-CWE**, không phải biên độ tổng.

### 3. §27 — chi phí suy luận (t5p n=9, đang lên n=15)

Nội suy **trọng số** với α chọn trên val: ROC +0.0198 (8/9), PR +0.0152 (**9/9**) — **một mô hình**.
Ngang trộn xác suất ở AUC (ROC −0.0004, PR +0.0025, đều ns), thua ở F1 (−0.0077, 1/9).
α=0.5 **cố định** trong không gian trọng số thì **không dùng được** (F1 −0.0127).

### 4. §26 — nguồn, lên bậc 2 trên codebert

`lm12` âm **0/5 fold trên cả bốn chỉ số** (p=0.0625 mỗi cái, sàn ở n=5). Bất đồng PR-AUC ở bậc 1
(n=3, khi đó PR **+0.0177**) đã **biến mất** — ví dụ sạch cho quy tắc *n=3 chỉ đủ để dừng*.

## Ba lần phải rút kết luận trong đêm — đọc để không lặp

1. **Hiệu của hai trung bình**: lấy +0.0129 (87 khối) trừ +0.0079 (48 cặp) → "+0.0050, dưới sàn
   nhiễu, §25 sập". Ghép cặp đúng cho **+0.0121, 42/48, p<1e-4**. Dấu hiệu: hai số sắp trừ nhau có
   **n khác nhau**.
2. **Kết luận từ n=1**: ô đầu của `wblend` cho α=1.0 → tôi ghi "nội suy trọng số là ngõ cụt, có
   hàng rào giữa hai mô hình", kèm cơ chế. Ở n=9 thì α **nội tại ở 8/9 ô**.
3. **Biên độ tổng ≠ tính chất chung**: §25.9 trên t5p đẹp, trên codebert **không lặp** — chỉ
   per-CWE mới lặp.

## Trang cho người hướng dẫn

- Sổ kết quả (1 162 phép so sánh, lọc + sắp xếp, 2 chế độ bảng):
  https://claude.ai/code/artifact/35c5f832-c306-42e4-959d-9d51adfdcef0
- Phép trộn, chi tiết + đối chứng: https://claude.ai/code/artifact/d18d51d3-ac51-491d-b8b6-90685506a8a6
