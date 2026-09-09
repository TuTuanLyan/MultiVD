# CURRENT_RUN — 09/09/2026, 07:36 UTC (14:36 giờ VN)

> Bản trước viết 03:45, trước khi có §25.8 → §28. Đã thay hẳn.

## Máy — không máy vast nào nằm không

| máy | đang chạy | còn trong hàng đợi | xong (ước) |
|---|---|---|---|
| **ntat** ($0.0818/h) | `wblend.sh\|4cwe com full\|4 5` — nâng §27 lên n=15 | hết | ~08:00 UTC |
| **ntat2** ($0.1222/h) | `wblend_cb.sh\|4cwe com full\|1 2 3` — bản lặp backbone của §27 | `p1fill_cb\|full`, `wblend_cb\|full` | ~10:00 UTC |
| 161 (local) | trống — **được phép**, người dùng nêu để trống chờ quyết hướng | | |
| 158 (local) | không ssh được; local nên không tốn tiền | | |

**Khi ntat xong (~08:00)**: đọc `tools/wblend_report.py results_wblend_ntat` để chốt §27 ở n=15.
Nếu hết việc đáng chạy thì kéo hết về, đối chiếu **số file + byte**, rồi
`vastai destroy instance 50223254 -y`. Đã đối chiếu ntat2 lúc 06:57: **222/222 file khớp byte**.

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
