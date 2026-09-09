# CURRENT_RUN — 09/09/2026, 07:36 UTC (14:36 giờ VN)

> Bản trước viết 03:45, trước khi có §25.8 → §28. Đã thay hẳn.

## ĐANG CHẠY — cấu hình chốt trên HAI backbone, chia hai máy

Ưu tiên 09/09 16:20 VN. Người dùng: *"chốt gần như là latent_bottleneck + ASAM 2.0 + RecAdam trên
t5p nhưng codebert thì tôi chưa rõ"* — đúng: bảng tổng hợp cho thấy **codebert CHƯA HỀ chạy với
ASAM ρ=2.0**; tốt nhất hiện có là `r0p1` (+0.0125 ROC, 10/10) và `plain` (+0.0112, 11/16), n nhỏ.

**Chia theo backbone** (CLAUDE.md mục 4: một backbone trọn một máy — Δ nội bộ sạch):

| máy | backbone | fold | cây kết quả | trạng thái |
|---|---|---|---|---|
| **vast `ntat`** RTX **5060 Ti** 16 GB, id 50132360 | **codebert** — ưu tiên xong trước | 1–5 | `results/chot_codebert` | chạy từ 09:42 UTC |
| ↳ rồi **t5p** | | **4–5** | `results/chotv_t5p` | nhận sau khi codebert xong |
| **161** (A4000 16 GB, dùng chung) | **t5p** | **1–3** | `results/chot_t5p` | chạy từ 09:44 UTC |

### Chia fold sang vast (09/09 17:25 VN)

Người dùng: *"Tận 4 task chạy cho 2 backbone × 2 setting hơn nữa còn 2 source và 5 folds thì cứ
chia bớt mà chạy đừng phí vast là được."*

Đo thật (phút/fold): vast+codebert **19,1** (GD1) và **16,2** (GD2); 161+t5p **61,9** và **52,8**.
161 là nút cổ chai — ôm cả 5 fold t5p thì chạy đến **02:18 VN** trong khi vast rảnh từ **19:55 VN**,
tức **6,4 giờ vast nằm không**. Chia fold 4–5 sang vast thì cả hai cùng xong **~22:00–22:30 VN**.

**Chỉ dùng một cách chia** mà CLAUDE.md mục 4 cho phép: **theo FOLD TRỌN VẸN**. Baseline và cả hai
nhánh của một fold nằm cùng một máy, nên Δ ghép cặp trong fold đó vẫn sạch; không bao giờ lấy hiệu
giữa hai máy. Cây kết quả của vast đặt tên **khác** (`chotv_t5p`) và `tools/chot_report.py` khoá ô
theo `(cây, backbone, seed, fold)` nên không thể bắc cầu qua máy.

**Phải ghi vào báo cáo**: fold 1–3 của t5p chạy trên A4000, fold 4–5 trên 5060 Ti. Δ từng fold
sạch, nhưng độ tản **giữa các fold** có thêm phần của phần cứng.

Máy vast là máy đồng nghiệp bàn giao (đã backup, đổi nhãn thành `ntat`). Môi trường sẵn
`torch 2.11.0+cu128`, 23 GB trống, **GPU đã nằm không ~1 tiếng** trước khi tôi nhận (lần chạy cuối
của họ ghi lúc 08:39, GPU 2 MiB/0%/6 W). Thư mục `MAML` của họ **giữ nguyên** — 469 MB, đĩa còn
thừa nên không cần xoá.

### Hai giai đoạn, mỗi máy tự chuyển

| | nguồn | ô/máy | ghi chú |
|---|---|---|---|
| **GD1** | `4cwe` + `com` | 25 | 5 fold × (1 baseline + 2 nguồn × 2 cấu hình) |
| **GD2** | `full` | 10 | chỉ chạy **sau khi GD1 của máy đó xong**; Pha 1 **đã có sẵn**, dùng lại |

Người dùng: *"sau khi xong toàn bộ khối thì chạy với full từ Phase 1, vì tôi muốn thấy đủ kết quả
cwe và common trước."* `scripts/watch_chot.sh` (cron 10 phút) tự chuyển sang GD2 cho **từng máy**
khi máy đó đủ 25 ô — hai máy xong lệch nhau nên không dùng script chuỗi chung.

### Pha 1 — dùng lại cả sáu, không huấn luyện lại cái nào

Người dùng 09/09: *"Phase 1 dùng lại được thì nên dùng nhé cứ seed 42 đã n=5."*

| backbone | 4cwe | com | full | nguồn |
|---|---|---|---|---|
| codebert | ✅ | ✅ | ✅ | chép từ `model/s42/phase1/`, md5 khớp cả hai phía |
| t5p | ✅ | ✅ | ✅ | `model/n48/phase1/` |

Log xác nhận ở **cả hai máy**: `phase1 <bb>/latent_bottleneck | da co, dung lai`.

Đã lỡ huấn luyện lại **~27 phút GPU** trước khi phát hiện (codebert `4cwe` 11 ph trên 161,
`com` 16 ph trên vast, phải giết giữa chừng) vì tra checkpoint **theo tên thư mục**: khối s42
đặt tên **không có hậu tố λ** khi λ=0.05 và chỉ thêm `_l02` khi λ=0.02 — hai quy ước tên cho
cùng một λ. `training_args` của bản s42 trùng khít mọi trường với bản đang huấn luyện lại.
FACTS §29.

### Head phụ trên `com`/`full` là **10 pillar**, không phải 94/123 CWE

Phát hiện khi đối chiếu ba checkpoint trên: `com` và `full` **cùng đúng 498 700 197 byte** trong
khi lẽ ra phải lệch ~1 KB. `cwe_head` của cả hai là `(10, 8)` — `cwe_class` trong file dữ liệu là
**pillar CWE-1000**, không phải CWE cụ thể. Ba lớp chỉ có 2–4 dòng; hai lớp chiếm 2/3 dữ liệu;
`full` ném **10.1%** số dòng khỏi loss phụ (`-100`). Không có confound (mọi file dữ liệu còn nguyên
mtime 27/08). **Mô tả phương pháp trong bài không được viết "head phụ 94 lớp".** FACTS §30.

### Hai điều kiện so sánh

| | Pha 2 |
|---|---|
| **A** `r2p0` | RecAdam + **ASAM ρ=2.0** — đầy đủ optimizer của phương pháp |
| **B** `plain` | **AdamW, không SAM/ASAM** — tắt cả hai cùng lúc |

λ **0.05**, seed 42, fold 1–5, đối chứng `baseline` cùng máy cùng fold dùng chung cho cả hai.

**Vì sao λ=0.05 chứ không phải 0.01**: dòng λ=0.01 (`r2p0`, ROC +0.0553) chỉ có ở **n=3** — bậc 1,
p=0.250 là **sàn**. Bản λ=0.05 có **n=18, ROC +0.0399 (17/18)**. λ nằm ở Pha 1 nên đổi λ là phải
huấn luyện lại Pha 1 cả hai backbone (CLAUDE.md mục 5).

### Ba bẫy đã mắc và sửa trong lúc dựng khối này

1. **Cha và con giành cùng một lock.** `chot2bb.sh` giữ `/tmp/multivd_opt1.lock` — đúng cái
   `opt1.sh` cần — nên mọi lần gọi con đều in *"DA CO driver opt1 dang chay"* và khối chạy hết
   5 fold × 2 backbone mà sinh **đúng 0 ô**. Cổng đếm hiện vật bắt được. Đã cho nó lock riêng, và
   đổi luôn **tên biến** (`CHOT_LOCK`) vì cả hai cùng đọc `MVD_LOCK`.
2. **Giết driver không giết tiến trình train.** Nó thành mồ côi (ppid=1) và chạy tiếp, giữ lock,
   chặn khối mới. Phải truy `ps` rồi giết theo PID.
3. **Đếm tiến trình tự khớp.** `ps|grep 'chot2bb.sh'` báo `driver=4` trong khi chỉ có **một** —
   nó bắt cả dòng lệnh của chính watchdog. Đổi sang **hỏi lock** (CLAUDE.md mục 8).

### Đọc kết quả

```
python3 tools/report2.py --a r2p0 --b plain results/chot_t5p results/chot_codebert
python3 tools/build_summary.py results/chot_t5p results/chot_codebert    # tổng hợp RIÊNG
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
