# HANDOFF — trạng thái bàn giao cho phiên làm việc mới

**Cập nhật 09/09/2026** (mục 0 ở dưới là phần mới nhất). Đọc file này trước, rồi `CLAUDE.md` (nạp tự động),
`SERVER.md`, `FACTS.md`.

**Cách đọc:** đây chỉ là *cấu hình đã đặt*, *số đã đo*, *việc chưa chạy* và *ràng
buộc vận hành*. Không diễn giải, không xếp hạng kết quả nào quan trọng hơn kết quả
nào. Hãy tự đọc số và hình thành cách hiểu riêng trước, rồi mới đọc phần lập luận
trong `FACTS.md` và `README.md`.

Lý do làm vậy: dự án này đã có vài cách giải thích được nêu rồi rút lại khi có thêm
dữ liệu ("họ backbone giải thích được", "cực tiểu nhọn giải thích được", "hạ λ sẽ
cứu T5"). Nạp sẵn cách hiểu hiện tại vào phiên sau nhiều khả năng là nạp một cách
hiểu sai.

---

## 0. Bổ sung 09/09/2026 — PHÁT HIỆN CHÍNH của đêm và việc đang chạy

> Đọc mục này trước. Bên dưới (§0.1 trở đi) là bản ghi lúc 00:40 UTC, giữ lại để tra cứu
> nhưng **đã lạc hậu**: các khối nêu ở đó đã xong. Trạng thái máy hiện tại ở `CURRENT_RUN.md`.

### 0.-2 KHỐI `chot` ĐÃ XONG 09/09 16:00 UTC — 70/70 ô (FACTS §34)

2 backbone × 2 điều kiện × 3 nguồn × 5 fold. **A** = RecAdam + ASAM ở ρ tốt nhất của chính
backbone (t5p 2.0, codebert 0.1); **B** = AdamW tắt hết; đối chứng `baseline` cùng fold.
Đọc bằng `python3 tools/chot_report.py results/chot_t5p results/chotv_t5p results/chot_codebert`.

| | ΔF1@0.5 (gộp 15) | ΔROC-AUC | **A − B ghép cặp** |
|---|---|---|---|
| codebert B | +0.0540 15/15 | +0.0264 13/15 | |
| codebert A | +0.0479 14/15 | +0.0228 12/15 | **−0.0061 6/15** (null) |
| t5p B | +0.0337 12/15 | +0.0198 12/15 | |
| t5p A | +0.0319 13/15 | +0.0111 13/15 | **−0.0018 10/15** (null) |

Theo CWE, **cả bốn dòng** (2 backbone × 2 nhánh): CWE-022 và CWE-079 dương **14–15/15 fold**;
CWE-078 và CWE-089 null. Nhưng CWE-022 chỉ **8 hàng test** và CWE-079 **19** trên 152.

**Máy**: vast `ntat` id 50132360 đã **huỷ 15:29 UTC** sau khi đối chiếu 50/50 file khớp byte,
md5 7/7, và có dòng kết thúc do chính driver in. 161 rảnh từ 16:00 UTC.

### Việc chưa làm (ghi để không mất)

- **Artifact riêng cho khối `chot`** — đã hỏi người dùng, chưa có trả lời.
- **§30.1** — chưa từng đo head phụ có học được gì không. Sàn lớp-đa-số: `4cwe` 0.744, `com` 0.377,
  `full` 0.529. Một lượt forward CPU trên checkpoint đã có là đủ, không tốn GPU.
- **Khối λ trên codebert** (λ 0.05/0.2/0.5 tại ρ=0.1) — §32 chỉ đo λ cao ở ρ=0 nên chưa kết luận
  được. Đổi λ phải huấn luyện lại Pha 1; **cần người dùng duyệt**.
- **`cwe_mapping`** trong checkpoint `com`/`full` là rác (ghi bộ 4 CWE trong khi `num_cwes=10`).
  Chỉ được ghi, chưa bao giờ được đọc — sửa `train_transfer.py:462` nếu muốn siêu dữ liệu đúng.
- **`run/wblend.sh`** chưa chạy lại được (dọn checkpoint theo fold nên lần chạy bù không có gì
  để nội suy).
- **Hai file `.rejected` 0 byte** ở `model/s42/phase1/codebert__{none,latent_proto}_full/` —
  ghi cụt từ đợt đĩa đầy 29/08, **không phải checkpoint tốt bị loại nhầm**. Đã đối chiếu: ô kết
  quả tương ứng đều có (10 ô `none`, 5 ô `latent_proto`) và còn checkpoint khác dùng được. Không
  đáng chạy lại — `latent_proto` đã bị loại có bằng chứng (§7), `none` ở λ0.05 đã đủ 5 fold ở
  khối `ft2`. **Đừng điều tra lại.**

### 0.-1 ĐANG CHẠY 09/09 chiều — khối `chot`, hai backbone × hai điều kiện, n=5

Cấu hình: `latent_bottleneck` λ=0.05, seed 42, đích `sven_python_folds_norm`.
**A** = `r2p0` (RecAdam + ASAM ρ=2.0). **B** = `plain` (AdamW, không SAM). Đối chứng `baseline`
(không Pha 1) dùng chung cho cả hai, cùng máy cùng fold.

| máy | backbone | fold | nguồn | cây kết quả | mốc |
|---|---|---|---|---|---|
| vast `ntat` id 50132360 (5060 Ti) | codebert | 1–5 | `4cwe`,`com` → `full` | `results/chot_codebert` | 35 ô |
| ↳ rồi t5p | t5p | **4–5** | như trên | `results/chotv_t5p` | 14 ô |
| 161 (A4000, dùng chung) | t5p | **1–3** | như trên | `results/chot_t5p` | 21 ô |

Chia theo **fold trọn vẹn** (CLAUDE.md mục 4) vì 161 chậm ~3,3× nên vast sẽ nằm không 6,4 giờ.
`scripts/watch_chot.sh` (cron 10 phút) tự chuyển nấc cho từng máy. Đọc bằng
`python3 tools/chot_report.py` (mặc định gộp cả ba cây; nó khoá ô theo `(cây, backbone, seed,
fold)` nên không bắc cầu qua máy).

**Pha 1 dùng lại cả sáu**, không huấn luyện lại cái nào (`val` đọc từ chính checkpoint):

| | 4cwe | com | full |
|---|---|---|---|
| codebert | 0.6532 | 0.5598 | 0.5636 |
| t5p | 0.6976 | 0.5897 | 0.5648 |

**Ba việc chưa làm, ghi để không mất:**
- §30.1 — chưa từng đo head phụ có học được gì không. Sàn lớp-đa-số: `4cwe` 0.744, `com` 0.377,
  `full` 0.529. Đo được bằng một lượt forward CPU trên checkpoint đã có.
- §30 — `cwe_mapping` trong checkpoint `com`/`full` ghi bộ 4 CWE trong khi `num_cwes=10`. Chỉ
  được ghi, chưa bao giờ được đọc; sửa `train_transfer.py:462` nếu muốn siêu dữ liệu đúng.
- `run/wblend.sh` chưa chạy lại được (dọn checkpoint theo fold nên lần chạy bù không có gì để nội suy).

### 0.0 KẾT QUẢ ĐẦU BÀI — §28, cấu hình chốt ở bậc 3

`latent_bottleneck` (nút thắt 8 chiều) + λ=0.05 + Pha 2 dùng **AdamW + ASAM ρ=2.0**, so với
**baseline** (không có Pha 1) cùng máy cùng fold. **Số đã có sẵn, không cần chạy thêm.**

| máy | n | ΔF1@0.5 | ΔF1@val | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|---|
| **ntat** | **15** | **+0.0535 (12/15)** | **+0.0685 (12/15)** | **+0.0337 (14/15)** | **+0.0300 (13/15)** |
| ntat2 (độc lập) | 9 | +0.0319 | +0.0280 | +0.0264 (7/9) | +0.0204 (7/9) |

Tắt ASAM (cùng head, cùng λ): ROC chỉ +0.0137 và PR **−0.0018**. Bật ρ=2.0 làm ROC hơn **2,5×**.

**Theo CWE — đây là chỗ đáng viết** (ntat n=15):

| CWE | hàng test | ΔF1@0.5 | ΔROC-AUC | ntat2 (n=9) |
|---|---|---|---|---|
| **022** | 13 | **+0.2050 (11/15)** | **+0.2238 (13/15, p=0.002)** | +0.1619 (6/9) |
| **079** | 16 | **+0.2978 (14/15)** | **+0.3638 (15/15, p<0.001)** | **+0.2614 (9/9, p=0.004)** |
| 078 | 40 | −0.0082 (ns) | −0.0097 (ns) | +0.0173 |
| 089 | 81 | +0.0142 (ns) | +0.0079 (ns) | −0.0011 |

Toàn bộ mức tăng nằm ở **hai lớp hiếm**; hai lớp thường chiếm **121/150** hàng test và **đúng bằng
không**. Con số tổng nhỏ vì lớp đa số áp đảo, không phải vì hiệu ứng yếu.
**Cảnh báo phải in kèm**: CWE-079 chỉ 16 hàng test/fold, CWE-022 13 hàng — thứ đáng tin là
**15/15 và 9/9 fold cùng dấu trên hai máy độc lập**, không phải biên độ.

### 0.0b §25 — phép trộn, SAU khi trừ đối chứng

Đối chứng âm (trộn hai baseline **khác seed**, n=48) cũng cho ROC +0.0079 / PR +0.0111, nên
*"trộn hơn baseline"* **một mình nó không phải bằng chứng cho chuyển giao**. Con số dùng được là
**ghép cặp trực tiếp** trong cùng fold, cùng baseline:

| A − B (A=trộn với chuyển giao, B=trộn với bản chạy lại) | F1@0.5 | ROC-AUC | PR-AUC |
|---|---|---|---|
| t5p (n=48) | +0.0200 (37/48) | **+0.0121 (42/48, p<1e-4)** | +0.0116 (43/48) |
| codebert (n=50) | +0.0139 (36/50) | +0.0052 (28/50, **p=0.48**) | +0.0042 (**p=1.00**) |

**Biên độ tổng KHÔNG lặp qua backbone.** Nhưng per-CWE thì lặp và mạnh hơn ở đếm dấu:
codebert 022 +0.0858 (38/50) · 079 **+0.1127 (45/50)**. Cơ chế nhìn thấy trực tiếp — trên codebert
**CWE-089 (54% hàng test) đi âm có ý nghĩa** (13/50, p=0.0009) nên kéo triệt tiêu biên độ tổng.

> **Phát biểu không phụ thuộc backbone là phát biểu PER-CWE, không phải biên độ tổng.**

### 0.0c §27 — chi phí suy luận (t5p n=9, đang lên n=15; bản codebert đang chạy)

Nội suy **trọng số** với α chọn trên **val**: ROC +0.0198 (8/9), PR +0.0152 (**9/9**) — và chỉ tốn
**một mô hình** khi suy luận. Ngang trộn xác suất ở AUC (ROC −0.0004, PR +0.0025, đều ns), thua ở
F1 (−0.0077, 1/9). **α=0.5 cố định trong không gian trọng số thì không dùng được** (F1 −0.0127) —
trọng số bắt buộc hiệu chỉnh α trên val; xác suất thì không cần tham số nào.

### 0.0d Ba lần phải rút kết luận trong đêm 08→09/09

1. **Hiệu của hai trung bình** — lấy +0.0129 (87 khối) trừ +0.0079 (48 cặp) rồi kết luận "dưới sàn
   nhiễu, §25 sập". Ghép cặp đúng cho **+0.0121, 42/48, p<1e-4**. Dấu hiệu nhận ra: hai số sắp trừ
   nhau có **n khác nhau** ⇒ tập khác nhau.
2. **Kết luận từ n=1** — ô đầu của `wblend` cho α=1.0, tôi ghi "nội suy trọng số là ngõ cụt, hai mô
   hình có hàng rào", kèm cơ chế. Ở n=9 thì α **nội tại ở 8/9 ô**.
3. **Biên độ tổng ≠ tính chất chung** — đẹp trên t5p, không lặp trên codebert.

### 0.0e Thay đổi mã đêm nay

- `src/train_{transfer,baseline}.py` ghi thêm **`val_probabilities`/`val_labels`** — không có nó thì
  mọi siêu tham số hậu kiểm chỉ chọn được trên TEST, tức rò rỉ.
- `run/matrix.sh` thêm cờ **`KEEP_CKPT=1`** (mặc định vẫn XOÁ).
- `scripts/vast_worklist.sh`: **vòng ngoài** — hết một lượt thì mở lại worklist, đối chiếu
  `todo − done`, còn việc thì chạy lượt nữa.
- `scripts/endpoints.sh`: file tạm **riêng theo `$$`** và **kiểm cache lúc ĐỌC**. Trước đó hai tiến
  trình ghi chung một file tạm làm cache hỏng → báo "máy không giải được địa chỉ" trong khi máy
  đang chạy.
- Ba script đếm job theo **tên chương trình** (`comm`), không theo dòng lệnh.
- `run/ensctl.sh` · `ensctl_cb.sh` · `wblend.sh` · `wblend_cb.sh` · `p1fill_cb.sh` — mới.

### 0.1 Đang chạy lúc 00:40 UTC 09/09 — ĐÃ LẠC HẬU, giữ để tra cứu

| máy | mục | ghi vào |
|---|---|---|
| ntat (vast) | `asam_aw.sh` — ASAM ρ=2.0 trên nền **AdamW**, 3 nguồn × 3 fold | `results/asamaw_t5p/` |
| ntat2 (vast) | `asam_aw.sh` — bản lặp máy thứ hai | `results/asamaw_t5p/` |
| 161 | `night_161.sh` → lưới `lm*` fold 1–3 rồi 4–5 | `results/pool1_t5p/` |
| 158 | `pool_lm.sh` fold 1–3 | `/data/ntat/MultiVD/results/pool1_t5p/` |

Hàng đợi còn: `pool_cb_lm.sh` (lưới ngôn ngữ trên **codebert** — chưa máy nào chạy),
`pool_lm.sh` trên ntat, `e60_cb.sh`.

### 0.2 Khối đã xong, số ở đâu

| khối | quy mô | ghi ở |
|---|---|---|
| Trục ρ của ASAM ở Phase 2, 7 mức | **n=15 mỗi mức**, đối chứng ρ=0 cùng máy | `FACTS.md` §21, §21.1–§21.3 |
| Lưới nguồn `pur*` / `lm*` | n=5, hai máy, hai backbone | `RESEARCH_2026-09-08_transfer.md` phụ lục B |
| Đọc lại head vs `none` bằng bốn chỉ số | 213 cặp có sẵn trên đĩa | `FACTS.md` §22 |
| Fine-tune hai lần thuần | giữ nguyên, không chạy lại | `FACTS.md` §20 |

Trang tổng hợp: <https://claude.ai/code/artifact/b6ba617d-eef2-4bee-ad8b-e8db92f7eadf>

### 0.3 Bẫy thứ TƯ trong dữ liệu — bắt buộc tách nhóm rò rỉ trước khi gọi là phát hiện

Bổ sung cho mục 5. Bộ `sven_python_folds_norm` chia theo **từng dòng**, nên ~16% hàng
test có bản đối nghịch gần trùng trong TRAIN. Trong đêm 08→09/09, phép tách nhóm
(`tools/leak_groups.py`, đọc `test_probabilities` có sẵn — **không cần chạy lại**) đã
đổi **hai** phát biểu tiêu đề:

- Δ tổng của ASAM ρ=2.0 là +0.0124 F1; trên 73% hàng sạch chỉ còn +0.0071 (dưới sàn nhiễu).
- Δ tổng "697 dòng đúng CWE + 233 dòng lệch thua 232 dòng đúng" là −0.0174 F1; trên
  hàng sạch là **+0.0038** — phát biểu đã rút.

**Quy tắc:** mọi Δ tổng trên bộ `norm` phải kèm phép tách này trước khi được gọi là
phát hiện.

### 0.4 Bốn cổng đã sửa — cả bốn đều hỏng IM LẶNG

| cổng | hỏng thế nào | sửa |
|---|---|---|
| `run/asam5.sh` | dấu nháy lệch nuốt cả vòng `for`; `bash -n` báo OK; chạy thật cho **0 ô** | viết lại; kiểm bằng stub `echo` + **đếm số lần gọi** |
| `scripts/vast_worklist.sh` | `while read … done < file` cho tiến trình con thừa kế stdin và nuốt phần còn lại → driver thoát sớm, **máy vast nằm không** | đọc worklist trên **fd 3**, gọi con với `3<&- </dev/null` |
| cổng "xong" của driver | mục thoát 0 mà sinh 0 ô vẫn được ghi `done` → **bỏ qua vĩnh viễn** | **đếm hiện vật** trước/sau; sinh 0 ô thì ghi `log/worklist.noop` + cảnh báo |
| `watch_vast.sh` / `fleet_status.sh` | đếm **số dòng thô** của `worklist.done`; dòng của danh sách cũ làm nó khớp `done>=todo` → **không phóng lại** | đếm **số giao** `done ∩ todo` |

Và `scripts/endpoints.sh`: `_vast_refresh` từng cài cache **rỗng/rác** khi `vastai`
thoát 0 với output hỏng, biến một trục trặc mạng thành hỏng vĩnh viễn. Nay kiểm JSON
trước khi thay; hỏng thì giữ cache cũ. `fleet_status.sh` phân biệt rõ **"đang chạy
nhưng chưa giải được địa chỉ"** với **"máy không còn chạy"** — gộp hai cái là cách
huỷ nhầm một máy đang làm việc.

### 0.5 Ngưỡng VRAM

Một ô `pool1_t5p` Phase 2 **đo được** 13468 MiB (161) và 14002 MiB (ntat2). `NEED=13000`
trong `run/pool1.sh` là đúng, **không hạ**. Trên máy dùng chung, user khác giữ 3–4 GB
là đủ để cổng chặn — đó là hành vi đúng, không phải lỗi.

---

## 1. Nơi làm việc

| | |
|---|---|
| **Thư mục chính** | **`/drive1/cuongtm/ntat/MultiVD`** — chuyển xong 06/09, đối chiếu 6 694 file lệch 0 byte |
| Bản tra cứu | `/home/ntat/workspace/MultiVD` — giữ lại, **không chạy thí nghiệm mới ở đây**; `model/*.pt` đã xoá khỏi đây |
| Server 161 | `112.137.129.161`, RTX A4000 16 GB, chính là máy local |
| Server 158 | `ssh tranmanhcuong@112.137.129.158`, RTX A4000 16 GB, thư mục `/data/ntat/MultiVD`, env `/data/ntat/envs/vdenv` |
| Python 161 | `/home/ntat/miniconda3/envs/vdenv/bin/python` |
| Phiên bản (khớp hai máy) | python 3.11.14 · torch 2.9.1+cu128 · transformers 4.57.1 · sklearn 1.7.2 · numpy 2.3.4 |

Mở phiên mới: `cd /drive1/cuongtm/ntat/MultiVD && claude`

**Memory của Claude Code đánh khoá theo đường dẫn thư mục làm việc**, không theo git
repo. Memory đã được chép sang khoá `-drive1-cuongtm-ntat-MultiVD` (19 file, đối chiếu
khớp với khoá cũ). Sửa memory ở một khoá thì khoá kia không thấy.

Đĩa: `/` còn 80 GB (96% đầy), `/drive1` còn 346 GB (81%). Mọi thứ nặng đặt ở `/drive1`.

---

## 2. Cấu hình đã chốt (`CLAUDE.md` mục 7)

```
latent_bottleneck   Linear(H→8) → Linear(8→C), num_latent=8
λ = 0.05            SAM/ASAM tắt ở cả hai pha   (--sam_rho 0)
optimizer           báo cáo cả AdamW và RecAdam
đối chứng           baseline (không Phase 1) + none (Phase 1 không head)
tập đích            data/sven_python_folds_norm, fold 1–5
```

---

## 3. Dữ liệu kết quả nằm ở đâu

| thư mục | nội dung | số ô |
|---|---|---|
| `results/` | các khối cũ (seed 42, 784 ô) **và** seed 7 fold 1–2 của NIGHT48 | 864 |
| `results_night48/` | NIGHT48 seed 42 | 35 |
| `results_night48b/` | NIGHT48 seed 7 fold 3–5 | 21 |
| `results_night48_158/` | NIGHT48 seed 1234 | 35 |
| `results_n48_161_partial/` | **cách ly** — ô lẻ của seed 7 fold 3/5 chạy trên 161 trước khi OOM. **Không đọc chung** với các thư mục trên, xem mục 5 | 6 |
| `results_confirm47_com/` | đợt xác nhận n=15 ở ρ ∈ {0, 0.2, 0.5}, nguồn `com`, torch 2.11.0 | 60 |
| `results_vast47_*` | log/kết quả trung gian của đợt confirm47 | 63 |

Checkpoint: `/drive1/.../model/` — 75 file `.pt`. **Không còn ở `/home`.**

---

## 4. Đã chạy xong

### 4.1 Năm khối seed 42 — 784 ô (đến 31/08)

| khối | λ | SAM/ASAM Pha 2 | optimizer | ngày |
|---|---|---|---|---|
| A | 0.05 | không | RecAdam | 27/08 |
| B | 0.05 | không | AdamW | 28/08 |
| C | 0.05 | không | RecAdam | 30/08 |
| D | 0.05 | ASAM ρ=0.1 | RecAdam | 30/08 |
| E | 0.05 | ASAM ρ=0.5 | RecAdam | 30/08 |
| F | 0.02 | ASAM ρ=0.1 | RecAdam | 31/08 |

Hai khối cũ hơn (20/08) dùng `codet5p-embedding`, λ=0.2, SAM ρ=0.05 đặt ở **Pha 1**:
`_sam1` (RecAdam) và `_sam1_adamw` (AdamW).

Backbone: `codebert` · `unixcoder` · `codet5p-220m-bimodal` (pooling mean).
Số liệu: `FACTS.md` §15, §16, §17.

### 4.2 Quét λ×ρ — 126 ô (04/09)

`latent_bottleneck` · t5p · RecAdam · seed 42 · fold 1–3 ·
λ ∈ {0.01, 0.05, 0.2} × ASAM ρ ∈ {0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0}, hai nguồn.
Ở `results/sw_t5p`. Tổng hợp: `bash scripts/finalize_sweep46.sh`.

### 4.3 Xác nhận n=15 ở ρ ∈ {0, 0.2, 0.5} — 45 ô + 15 baseline (05/09)

Nguồn `com`, λ=0.05, seed 42/7/1234, torch 2.11.0 trên vast. Ở `results_confirm47_com/`.

### 4.4 NIGHT48 — 90 ô + 15 baseline (06/09) ← mới nhất

`latent_bottleneck` · t5p · RecAdam · λ=0.05 · **ρ ∈ {0, 0.1}** · **cả ba nguồn** ·
seed 42/7/1234 · fold 1–5. `PHASE1_MIN_VAL=0`. **90/90 ô, 15/15 baseline, kiểm toán
không thiếu không chồng lấn.**

Chạy trên bốn máy, chia theo **fold trọn vẹn** nên mọi Δ ghép cặp nằm gọn trong một máy:
seed 42 trên vast (torch 2.9.1+cu130) · seed 7 fold 1–2 trên 161 · seed 7 fold 3–5 trên
vast (2.9.1+cu128, **dùng lại đúng ba checkpoint Pha 1 của 161**) · seed 1234 trên 158.
Mọi máy vast đã huỷ.

Số liệu đầy đủ: `FACTS.md` §18. Dựng lại bảng: `python3 tools/n48_report.py`.

---

## 5. Ba cái bẫy trong dữ liệu, phải biết trước khi đọc số

**`results_n48_161_partial/` không được trộn vào.** Ngày 06/09 `cuongtm` chiếm 5,6 GB
VRAM trên 161, còn trống ~10 GB < 12,6 GB job t5p cần, nên 16 ô của seed 7 OOM. Fold 3
và 5 lúc đó đã kịp sinh vài ô. Chúng bị **cách ly** vì fold 3–5 sau đó chạy lại trọn vẹn
trên máy khác; trộn vào thì Δ ghép cặp của fold đó vắt qua hai máy. `tools/n48_report.py`
không đọc thư mục này, và có kiểm tra chồng lấn báo lỗi nếu một ô xuất hiện hai lần.

**Kết quả trước 06/09 không có trường `runtime`.** Bản vá ghi giờ + phần cứng chỉ áp
sau khi NIGHT48 xong (mục 7). `tools/runtime_report.py` đếm riêng những ô đó và báo rõ.

**Hai nhánh ρ của NIGHT48 dùng chung một checkpoint Pha 1.** Đó là điều kiện để hiệu
giữa chúng đổi đúng một biến. Nếu chạy thêm ρ mới thì phải tách `PHASE1_TAG` khỏi
`ARM_TAG` như `run/night48.sh` đang làm, nếu không nhánh mới tự huấn luyện Pha 1 khác
và phép so đổi hai biến.

---

## 6. Công cụ

| lệnh | làm gì |
|---|---|
| `python3 tools/n48_report.py` | kiểm toán + Δ ghép cặp của khối NIGHT48, cả F1 lẫn ROC |
| `python3 tools/runtime_report.py <thư mục...>` | tổng hợp giờ chạy + phần cứng cho mục setup của bài |
| `python3 tools/asam_effect.py <nguồn> <gốc>` | hiệu ASAM của đợt quét λ×ρ |
| `python3 tools/bylam.py <nguồn> <gốc>` | bảng theo λ của đợt quét |
| `bash scripts/finalize_sweep46.sh` | tổng hợp đợt quét λ×ρ trên cả hai máy |

Driver: `run/matrix.sh` (lõi) · `run/night48.sh` (khối mới nhất, có `SEEDS`,
`SOURCES`, `FOLD_LIST`, `RHOS`, `MVD_LOCK`) · `run/confirm47.sh` · `run/sweep46_p*.sh`.

---

## 7. Ghi giờ chạy + phần cứng — đã áp 06/09

`src/runtime_env.py` (mới) và bản vá vào `src/train.py`, `src/train_transfer.py`,
`src/train_baseline.py` qua `scripts/apply_runtime_logging.py`.

Mỗi lần chạy ghi `<checkpoint>.runtime.json` (ghi nguyên tử: file tạm rồi `os.replace`),
gồm giây/epoch, số epoch, số mẫu, ms/mẫu, `sam_rho`/`sam_variant`, và dấu vân phần cứng
(GPU, driver, CPU, RAM, torch, transformers, sklearn). Kết quả JSON có thêm trường
`runtime` gộp giờ Pha 1, Pha 2 và suy luận.

Đã thử thật cả hai nhánh (baseline và transfer-với-ASAM) trước khi coi là xong. Đã đồng
bộ sang 158.

---

## 8. Việc chưa chạy

- **Đa seed cho `none`.** NIGHT48 đã cho `latent_bottleneck` ở 3 seed, nhưng nhánh đối
  chứng `none` (Pha 1 không head) vẫn chỉ có seed 42. Không so được "head có ăn không"
  ở n=15.
- **Đa seed với AdamW.** Toàn bộ NIGHT48 là RecAdam. Ô mạnh nhất trong lưới cũ lại là
  `latent_bottleneck` + **AdamW** (+0.0111, 33/43, p=0.0006).
- Tập đích `data/sven_python_twin` (chia theo cụm gần trùng, không rò rỉ). Chưa chạy ô
  nào. `sven_python_folds_norm` chia theo từng dòng nên ~40% hàng test có bản sao gần
  giống trong train.
- Lưới λ×ρ mới chỉ fold 1–3, chỉ `codet5p-220m-bimodal`, chưa có `full`, chưa có AdamW.
  Chưa biết ρ có chuyển được giữa các backbone không — SAM ở cấu hình cũ thì không:
  cùng ρ=0.05, `codet5p-embedding` chạy bình thường còn `codebert` kẹt train loss ở
  ln 2 suốt 13 epoch.
- `cwe` chỉ chạy được trên nguồn `4cwe` vì trong CleanVul chỉ phần JavaScript có nhãn CWE.
- Pha 1 cho các cặp (backbone × nguồn) ngoài lưới hiện tại.

---

## 9. Ràng buộc vận hành

- **161 dùng chung với người khác.** Ngày 06/09 `cuongtm` chiếm GPU làm hỏng 16 ô. Job
  t5p cần **~12,6 GB VRAM**; kiểm VRAM trống trước khi phóng, và **nhường** nếu là
  `cuongtm`/`tranmanhcuong` vào trước. Có `ollama` giữ ~684 MB — không đụng.
  Không chạy job CPU nặng trên 161.
- Trên 158, chỉ `/data/ntat/` là toàn quyền.
- vast.ai: chỉ nhãn thuộc họ `ntat`; **huỷ chứ không dừng**; hỏi trước khi thuê;
  quy tắc ở `VAST_RULES.md`. Khoá SSH phải `vastai attach ssh <id>` cho **từng máy mới**
  — `vastai create ssh-key` hỏng ở ngữ cảnh team.
- `/` trên 161 đã từng đầy và làm `torch.save` ghi cụt, hỏng checkpoint mà cổng chất
  lượng dán nhãn nhầm là "phương pháp kém". Kiểm đĩa trước khi tin một nhánh hỏng.
- Cổng chất lượng Pha 1 ở `run/matrix.sh:phase1_usable`, ngưỡng đặt bằng
  `PHASE1_MIN_VAL` (mặc định 0.40). Đặt `PHASE1_MIN_VAL=0` để chạy cả checkpoint suy biến.
- **Cron hiện đang TRỐNG.** Script chuyển chỗ đã gỡ mọi mục giám sát; bản lưu ở
  `/tmp/mvd_crontab.bak`. Không có job nào chạy nên chưa bật lại. Khi bật phải **sửa
  đường dẫn trong đó từ `/home/...` sang `/drive1/cuongtm/ntat/MultiVD`**.
- Watchdog có sẵn: `scripts/watch48_local.sh` (161+158, không có quyền huỷ/giết),
  `scripts/watch48b.sh` (vast, `ARM=destroy`, cổng ba tầng).

---

## 10. Git

Nhánh `latent`, commit cuối `77488cf` (25/08). **86 mục chưa commit**, gồm mọi file
`.md` hiện hành, `tools/`, `src/runtime_env.py`, các script mới và toàn bộ kết quả
NIGHT48. Chúng tồn tại ở hai ổ vật lý (`/` và `/drive1`) nhưng không có lịch sử git.
