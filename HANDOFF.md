# HANDOFF — trạng thái bàn giao cho phiên làm việc mới

**Cập nhật 06/09/2026.** Đọc file này trước, rồi `CLAUDE.md` (nạp tự động),
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
