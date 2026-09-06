# Đang chạy — OPT1 (tối ưu RecAdam, khối 1)

> **Trạng thái: ĐANG CHẠY từ 06/09/2026 ~11:20 UTC.** Nhánh git `optimize-v1`.
> Kế hoạch và lý do: `RESEARCH_2026-09-06_recadam.md` §5.5. Báo cáo: `python3 tools/opt1_report.py`.

## Câu hỏi của khối này

RecAdam ở cấu hình mặc định (γ=5000, t0 = 5% bước ≈ 43 bước, neo cả head) hiện **thua AdamW**
ở hầu hết ô (RESEARCH §5.1). Khối này hỏi: có chế độ neo nào — **yếu hơn** (γ 500/50), **bền hơn**
(t0 = 50%), **không neo head**, hoặc **neo về pretrained** — làm RecAdam ≥ AdamW không; và phần
RecAdam đang mua được là **neo** hay chỉ là **warmup ngầm** (đối chứng `warm`).

## Cấu hình — chỉ đổi Pha 2, dùng lại Pha 1 của NIGHT48

| | |
|---|---|
| backbone | `codet5p-220m-bimodal`, pooling mean (`t5p`) |
| nhánh | `latent_bottleneck`, λ=0.05 — **checkpoint Pha 1 dùng lại** `model/n48/phase1/t5p__latent_bottleneck_{4cwe,com}_l0p05/seed_42/best.pt` (val 0.6976 / 0.5897) |
| nguồn | **giai đoạn 1**: `4cwe`, `com` · **giai đoạn 2** (người dùng 06/09: "có thêm nhưng chạy cuối cùng xem ảnh hưởng"): `full`, chạy sau khi giai đoạn 1 xong **trên cùng máy, cùng fold** |
| seed | 42 (người dùng 06/09: "chạy kiểm tra bằng seed 42 và 3 folds là được") |
| fold | **1, 2, 3** — 161 giữ fold 1–2, 158 giữ fold 3. Fold 4–5 của giai đoạn 1 đã chạy một phần trên 158 trước khi thu hẹp, giữ lại nhưng không tính vào bảng chính |
| SAM/ASAM | tắt (`--sam_rho 0`) |
| tiêu chí chọn checkpoint | **val F1@0.5, không đổi**; metric chính F1@0.5, kèm ROC-AUC và F1@ngưỡng-val |
| sổ sách mới | JSON ghi λ thật, val Pha 1, anchor, γ, k, t0 (bước), và **lịch sử val theo epoch** (`runtime.phase2.val_history`) |

**12 cấu hình Pha 2** cho mỗi (fold, nguồn), hai đối chứng chạy trước.
Danh sách sửa lúc 15:5x UTC 06/09 sau khi phát hiện bẫy λ (xem mục "Bẫy đã mắc" bên dưới):

| tag | optimizer | cờ Pha 2 | vai trò |
|---|---|---|---|
| `c5000_t0p05` | RecAdam | mặc định | **đối chứng 1** (RecAdam hiện tại, cùng phiên) |
| `plain` | AdamW | mặc định | **đối chứng 2** (fine-tune hai lần, cùng phiên) |
| `c500_t0p05`, `c50_t0p05`, `c5_t0p05`, `c0p5_t0p05` | RecAdam | `--pretrain_cof 500/50/5/0.5` | **trục γ** ở lịch mặc định. Hai giá trị cuối thêm 06/09 vì γ càng nhỏ Δ càng cao |
| `c5000_t0p5` | RecAdam | `--anneal_t0_ratio 0.5` | **bằng chứng đóng băng**, giữ đúng một nhánh |
| `c5000_t0p2k02`, `c50_t0p2k02` | RecAdam | `--anneal_t0_ratio 0.2 --anneal_k 0.02` | neo **bền thật sự**: λ = 0.05/0.50/0.97 ở epoch 1/6/12 |
| `nohead_c5000_t0p05` | RecAdam | `--recadam_anchor_head none` | không neo `vul_head` (như bài gốc) |
| `pre_c20_t0p2k02` | RecAdam | `--recadam_anchor pretrained --pretrain_cof 20 --anneal_t0_ratio 0.2 --anneal_k 0.02` | neo về pretrained (archive §40.5), ở lịch đúng |
| `warm` | AdamW | `--adamw_anneal_lr` | AdamW + đúng λ(t) của RecAdam trên lr, **không neo** |

### Bẫy đã mắc — λ có hai tham số KHÔNG độc lập

Bản đầu của khối đặt `t0_ratio=0.5` (435 bước) mà giữ `k=0.05`, cho k·t₀ = 21.7 nên
**λ(1) ≈ 4e−10** và sau 7 epoch mới lên 9e−06. Bước cập nhật task bị nhân với ~0, còn lực
kéo neo cũng ~0 vì θ vẫn bằng θ*. Mô hình **đứng nguyên tại checkpoint Pha 1**: cả ba γ
(5000/500/50) cho **trùng khít** test F1 0.4738, val F1 0.5131 đứng im 7 epoch.

Đó là phép đo "Pha 1 áp thẳng lên Python", không phải "neo bền". Ba ô trùng nhau đã chạy ở
fold 1 (161) và fold 3 (158) — **giữ lại làm bằng chứng**, không xoá, báo cáo ghi rõ n.

**Quy tắc rút ra: k phải tỉ lệ nghịch với t₀, giữ k·t₀ ≈ 3–4.**

| lịch | t₀ | λ ep1 | λ ep6 | λ ep12 |
|---|---|---|---|---|
| `t0p05` (mặc định) | 43 | 0.332 | 0.999 | 1.000 |
| `t0p2k02` (mới) | 174 | 0.052 | 0.500 | 0.970 |
| `t0p5 k=0.05` (đóng băng) | 435 | 0.000 | 0.000 | 0.013 |

**Kỳ vọng sau khi thu hẹp**: 3 nguồn × **3 fold** × 12 = **108 ô Pha 2 + 3 baseline**.

## Máy — chia theo FOLD TRỌN VẸN

| máy | fold | lock | log driver | kết quả |
|---|---|---|---|---|
| **161** | 1, 2 | `/tmp/multivd_opt1.lock` | `log/opt1_161.log` | `results/opt1_t5p/` (tại chỗ) |
| **158** | **3** | `/tmp/multivd_opt1.lock` | `log/opt1_158.log` | `/data/ntat/MultiVD/results/opt1_t5p/` → kéo về `results_opt1_158/` |

158 đã được thu về **chỉ fold 3** lúc 15:5x UTC 06/09 (driver bị dừng bằng TERM đúng PID, job
Pha 2 đang chạy để nguyên cho xong). Fold 4–5 giai đoạn 1 giữ lại, không tính vào bảng chính.

Mỗi máy tự chạy baseline cho fold của mình nên mọi Δ ghép cặp nằm gọn trong một máy. Hai
checkpoint Pha 1 seed 42 đã đẩy sang 158 và đối chiếu byte + `torch.load` trước khi phóng.

Phóng gđ1: `FOLD_LIST="1 2" PYTHON=<vdenv> setsid nohup bash run/opt1.sh > log/opt1_161.log 2>&1 < /dev/null &`
Mỗi lần driver chạy xong in `########## OPT1 xong ... ##########` (gđ2 in lần thứ hai).

**Xếp hàng hai giai đoạn do `scripts/watch_opt1.sh` (cron 10 phút trên 161) đảm nhiệm**: khi không còn
driver nào giữ lock, nó **đếm hiện vật** theo nguồn — thiếu 4cwe/com ⇒ phóng lại gđ1; đủ gđ1 mà thiếu
`full` ⇒ phóng `SOURCES=full` cho đúng fold của máy đó; đủ cả ⇒ không làm gì. Mỗi giai đoạn tối đa 3 lần
phóng (`log/opt1_<máy>_passes_<gđ>`). Không giết, không xoá. 161 dùng chung GPU: chỉ phóng khi VRAM trống
≥ 13 GB và không còn job train của khối này. 158 chỉ phóng `full` khi checkpoint Pha 1 full seed 42 đủ
438 519 277 B. Thử: `bash scripts/watch_opt1.sh --test-stage`.

## Mã mới trên `optimize-v1`

- `src/anneal_adamw.py` — AdamW nhân lr với λ(t) của RecAdam (đối chứng warmup).
- `src/train_transfer.py` — `--recadam_anchor_head`, `--adamw_anneal_lr`, sổ sách Pha 1 → JSON.
- `src/train.py` — `val_history` theo epoch vào sidecar runtime.
- `run/matrix.sh` — `PHASE1_STORE` ghi đè được. `run/opt1.sh` — driver. `tools/opt1_report.py` — báo cáo.

## Sau khi xong

1. `python3 tools/opt1_report.py results/opt1_t5p results_opt1_158` — bảng 1 (vs baseline), bảng 2 (vs hai đối chứng cùng phiên), bảng 3 (epoch theo F1 vs theo AUC).
2. Cấu hình thắng ⇒ lặp ở seed 7 (161) và 1234 (158), rồi codebert (checkpoint `model/s42/phase1/codebert__latent_bottleneck_{4cwe,com}`).
3. Cập nhật đầu file này: ĐÃ XONG + giờ.

---
---

# Đang chạy — NIGHT48

> **Trạng thái: ĐÃ XONG — 05/09/2026 15:55 UTC.** 90/90 ô Pha 2 + 15/15 baseline,
> không ô nào thiếu, không ô nào chồng lấn. Mọi máy vast đã huỷ. Số liệu và kết
> luận ở `FACTS.md` §18; chạy lại báo cáo bằng `python3 tools/n48_report.py`.
>
> Còn treo một việc: áp `scripts/apply_runtime_logging.py` (xem mục cuối file).

## Câu hỏi của khối này

ASAM ở **ρ=0.1** — bán kính nhỏ hơn hẳn hai giá trị đã xác nhận (0.2 và 0.5) — có
giữ được gì không, và **có giữ đều trên cả ba nguồn** không. Đợt xác nhận trước
chỉ chạy nguồn `common`; lần này đủ `4cwe`, `com`, `full` nên đọc được cả chiều
"nguồn nào chịu được ASAM".

## Cấu hình — một dòng duy nhất, không quét

| | |
|---|---|
| backbone | `codet5p-220m-bimodal`, pooling mean (`t5p`) |
| nhánh | `latent_bottleneck` (Linear(H→8) → Linear(8→C), `num_latent=8`) |
| optimizer | **RecAdam** |
| λ | **0.05** |
| Pha 1 | `--sam_rho 0` (SAM tắt), 15 epoch |
| Pha 2 | **ρ=0 (đối chứng)** và **ASAM ρ=0.1**, `--asam_eta 0.01` |
| nguồn | `4cwe` · `com` · `full` — **cả ba** |
| seed | 42 · 7 · 1234 |
| fold | 1–5, tập đích `data/sven_python_folds_norm` |
| cổng chất lượng | `PHASE1_MIN_VAL=0` — **chạy hết**, kể cả Pha 1 sập |

**90 ô Pha 2** (3 nguồn × 3 seed × 5 fold × **2 nhánh ρ**) **+ 15 ô baseline**
(3 seed × 5 fold, baseline không phụ thuộc nguồn nên `matrix.sh` chạy một lần rồi
bỏ qua).

Hai nhánh ρ **dùng chung đúng một checkpoint Pha 1** (`PHASE1_TAG` tách khỏi
`ARM_TAG`), nên hiệu giữa chúng đổi **một biến duy nhất** — đó là điều kiện để nói
được ASAM có mua gì không, chứ không chỉ "cấu hình này tốt".

**9 checkpoint Pha 1** (3 nguồn × 3 seed). λ không đổi trong khối nên mỗi
(nguồn, seed) chỉ cần một checkpoint, dùng chung cho cả 5 fold.

## Thứ tự chạy

```
for seed in 42 7 1234:          ← vòng ngoài cùng
    huấn luyện Pha 1 cho cả 3 nguồn (fold không đụng tới Pha 1)
    for fold in 1..5:               ← vòng ngoài
        for src in 4cwe com full:   ← vòng trong
            for rho in 0, 0.1:      ← trong cùng, đối chứng TRƯỚC
                baseline (chỉ lần đầu của seed) + ô Pha 2
```

ρ nằm trong cùng để hai nhánh của cùng (fold, nguồn) chạy **cạnh nhau**: hiệu giữa
chúng ghép cặp đúng theo fold đó, và nếu khối đứt giữa chừng thì cái đã xong luôn
là những **cặp trọn vẹn**, không phải một nửa nhánh chính.

Hết fold 1 của một seed là đã có lát cắt so được: baseline + cả ba nguồn, cùng
fold cùng seed cùng máy. Hết seed 42 là n=5, thêm seed 7 lên n=10, seed 1234 lên
n=15.

## Máy — chia ba từ 03:15 UTC 05/09

| máy | seed | GPU | torch / transformers | lock | kết quả |
|---|---|---|---|---|---|
| ~~vast `ntat` 49893040~~ | **42** ✅ | A4000 | 2.9.1+cu130 / 4.57.1 | — | `results_night48/` — **đã huỷ máy 04:4x UTC 05/09** |
| **161** (local) | **7**, fold 1–2 ✅ | A4000 | 2.9.1+cu128 / 4.57.1 | `/tmp/multivd_night48.lock` | tại chỗ `results/n48_t5p/` |
| vast `ntat` 49952744 | **7**, fold 3–5 | A4000 | 2.9.1+cu128 / 4.57.1 | `/tmp/multivd_night48.lock` | kéo về `results_night48b/` |
| **158** | **1234** | A4000 | 2.9.1+cu128 / 4.57.1 | `/tmp/multivd_night48_ntat.lock` | kéo về `results_night48_158/` |

Chia theo **seed trọn vẹn**: mỗi seed tự có 3 checkpoint Pha 1 và 5 baseline riêng, nên
mọi Δ ghép cặp — cả Δ vs baseline lẫn Δ ρ=0.1 vs ρ=0 — nằm gọn trong một máy. Ba máy
cùng A4000, cùng transformers 4.57.1, torch chỉ khác bản CUDA. Phép gộp 15 điểm cuối
cùng có mang chênh lệch máy, ghi rõ khi báo cáo.

### Seed 42 đã xong — máy vast đã huỷ

30/30 ô Pha 2 + 5/5 baseline, driver tự in dòng kết thúc lúc **04:22:35 UTC 05/09**.
Trước khi huỷ đã đối chiếu: **35 file JSON lệch 0**, và 3 checkpoint Pha 1 khớp từng
byte (438 519 085 / 438 519 277 / 438 519 277) **và đọc lại được** bằng torch với đúng
`val` như trên máy — kích thước khớp thôi chưa đủ. Checkpoint nằm ở
`model/n48/phase1/*/seed_42/best.pt`, giữ lại phòng khi cần thêm một nhánh ρ cho seed 42
mà không phải huấn luyện lại 2 h.

Cron của watchdog vast đã gỡ; chỉ còn `watch48_local.sh`.

### Chi tiết máy vast (lịch sử)

**vast, nhãn `ntat`, id 49893040** — RTX A4000, 29 GB đĩa, $0.068/h,
IP 202.122.49.242 cổng 38433, image `vastai/pytorch:2.9.1-cu130-cuda-13.2-mini-py314`.
Driver bắt đầu **20:01 UTC 04/09**. Từ 03:15 UTC 05/09 chỉ còn chạy **seed 42**.

Máy đầu (49890960) pull image hỏng nên bỏ; máy này thuê lại cùng IP, tài khoản lúc đó
không có khoá SSH nào đăng ký nên phải `vastai attach ssh` trước khi vào được.
Image là bản `mini` nên thiếu thư viện: đã cài `transformers==4.57.1` (ghim theo
`requirements-pin.txt`), `scikit-learn`, `scipy`, `evaluate` vào `/venv/main`.
Python 3.14.7 · torch 2.9.1+cu130 · tokenizers 0.22.2.

Cả khối nằm **trọn một máy** nên mọi Δ ghép cặp trong khối đều sạch — đó là điều
duy nhất cần cho mọi kết luận của khối này. Chỉ khi so **sang** đợt xác nhận n=15
trước (`results_confirm47_com/`, torch 2.11.0, máy khác) mới phải ghi chú: hai đợt
so tương đối được, so từng ô thì không.

Lúc phóng (20:01 UTC 04/09) cả 161 và 158 đều bận, VRAM trống ~10 GB < 13 GB job t5p
cần, nên vast phải gánh cả ba seed. Đến 03:10 UTC 05/09 hai máy đã rảnh (161 chỉ còn
ollama giữ 684 MB, 158 trống hẳn) nên chia lại như bảng trên.

## Điều khiển

| việc | lệnh |
|---|---|
| driver | `run/night48.sh` — biến `SEEDS` chọn seed, `MVD_LOCK` chọn lock |
| đẩy code + phóng | `bash scripts/provision48.sh` |
| watchdog vast | `scripts/watch48.sh`, cron 10 phút, `ARM=destroy` |
| watchdog local | `scripts/watch48_local.sh`, cron 10 phút, **không bao giờ huỷ/giết** |
| log driver | vast `log/night48.log` · 161 `log/night48_161.log` · 158 `log/night48_158.log` |
| log watchdog (local) | `log/watch48.log`, `log/cron48.log` |
| kết quả | `results_night48/` (vast) · `results/n48_t5p/` (161) · `results_night48_158/` (158) |

**Điều kiện tự huỷ máy**: watchdog chỉ huỷ khi (a) driver đã in đúng dòng
`########## NIGHT48 xong ...` do chính nó phát ra, (b) lock đã nhả, và (c)
`comm` dưới `LC_ALL=C` xác nhận **không lệch file nào kể cả kích thước byte**.
Không bao giờ chốt bằng "đếm đủ 45 ô". Tháo ngòi: sửa cron bỏ `ARM=destroy`.

Driver chết mà chưa in dòng kết thúc ⇒ watchdog **phóng lại**, không huỷ. Ở hai máy
local, nếu lúc đó VRAM trống < 13 GB thì **nhường** người khác, không phóng đè.

Bàn giao vast: driver được phóng ban đầu với cả ba seed, nên khi thấy dòng
`NIGHT48 seed 42 xong` mà driver vẫn chạy, watchdog **dừng nó** (tìm pgid qua người
giữ lock, không grep `ps` — lần grep trước bắt nhầm pgid 1552 thay vì 1555) rồi
phóng lại với `SEEDS=42`; lần chạy mới thấy 30/30 ô đã có, bỏ qua hết, và tự in
dòng kết thúc **thật** của chính nó. Không bao giờ tự tay ghi dòng đó vào log.

Driver in dòng kết thúc **nhưng thiếu ô** (ví dụ vài checkpoint Pha 1 hỏng) ⇒ cũng
phóng lại, tối đa **2 lần** (`log/n48_passes`), vì `matrix.sh` bỏ qua ô đã có nên
lần chạy lại chỉ làm phần thiếu. Hết 2 lần thì chấp nhận, đối chiếu rồi huỷ — để
không rơi vào bẫy ngưỡng không bao giờ đạt.

## Ước tính

~6,5 h/seed (Pha 1 ba nguồn ~2,1 h + **30 ô Pha 2 ~4 h** + 5 baseline ~0,4 h)
Nhịp thực đo trên vast: **10,7 phút/job** (GPU chạy 93 °C, nhiều khả năng hạ xung).

Một máy chạy cả ba seed sẽ mất ~24 h. Chia ba từ 03:15 UTC nên mỗi máy chỉ lo một
seed: vast xong seed 42 quanh **04:20 UTC**, hai máy local xong quanh **11:30 UTC
05/09** (≈ 18:30 giờ VN). Chi phí vast ~$0,6.

## Sau khi xong

Δ ghép cặp theo `(nguồn, seed, fold)` vs baseline cùng seed cùng fold. Baseline
lệch tới 0.0175 giữa các seed nên **bắt buộc ghép theo seed**, không lấy trung
bình. Sàn kiểm dấu ở n=15 là p=0.0001 nếu cùng dấu cả 15.

Hai phép so, phải nêu **cả hai**:

- **Δ vs baseline** — ghép theo (nguồn, seed, fold), cho từng nhánh ρ.
- **Δ ρ=0.1 vs ρ=0** — ghép theo (nguồn, seed, fold), dùng chung checkpoint Pha 1
  nên đây mới là phép đo riêng của ASAM.

---

## Seed 7 bị cắt đôi — vì sao, và cắt thế nào cho sạch

Lúc 09:2x UTC 05/09 `cuongtm` chiếm 5,6 GB VRAM trên 161, còn trống ~10 GB < 12,6 GB
job t5p cần, nên 16 ô của seed 7 OOM. Theo quy tắc đã chốt thì nhường, không giành.

Chia lại **theo fold trọn vẹn**, đúng §4:

- **fold 1–2** ở lại 161, mỗi fold đủ 6 ô + 1 baseline.
- **fold 3, 4, 5** chạy lại **trọn vẹn** trên vast 49952744. Fold 3 và 5 trước đó có
  vài ô lẻ trên 161; chúng đã được **cách ly** sang `results_n48_161_partial/` (không
  xoá) vì nếu để lẫn thì Δ ghép cặp của fold đó vắt qua hai máy.
- 161 đã bị chặn không lấp lại seed 7 (`log/n48_161_passes` = 12/12), tránh làm trùng.

**Máy vast phải dùng lại đúng ba checkpoint Pha 1 của seed 7 từ 161**, không huấn luyện
mới — nếu không thì fold 3–5 xuất phát từ nguồn khác fold 1–2 và seed 7 tự mâu thuẫn.
Đã đẩy 1,25 GB lên, đối chiếu 26 file lệch 0 byte, và **load lại bằng torch trên chính
máy đó** ra đúng val 0.6684 / 0.5907 / 0.5834 như trên 161. Driver xác nhận
`da co checkpoint Pha 1 — bo qua` cho cả ba nguồn.

`scripts/watch48b.sh` có thêm **chốt**: không phóng driver khi chưa đủ cả ba checkpoint
trên máy. Chốt này đã chặn thật một lần lúc 10:33, khi cron định phóng lúc rsync mới
đẩy được 1/3 — nếu lọt thì driver đã tự huấn luyện Pha 1 mới cho hai nguồn còn thiếu.

---

## Việc đã chuẩn bị, ÁP SAU KHI KHỐI NÀY XONG

Ghi nhật ký **phần cứng + thời gian chạy** cho mỗi lần chạy (một nguồn × một fold ×
một setting), để có sẵn số cho mục "training setup" của bài.

| file | trạng thái |
|---|---|
| `src/runtime_env.py` | **đã tạo** — an toàn, chưa file nào import nên không đụng job đang chạy |
| `tools/runtime_report.py` | **đã tạo** — công cụ tổng hợp |
| `scripts/apply_runtime_logging.py` | **đã tạo, CHƯA CHẠY** — vá 3 file `src/` |

**Chưa áp** vì `matrix.sh` phóng một python mới cho mỗi ô; sửa `src/train*.py` giữa
chừng thì ô kế tiếp có thể vớ phải file ghi dở. Đã thử vá trên bản sao: cả 7 đoạn khớp
đúng một lần, ba file biên dịch được.

Khi 161 và 158 báo xong (`NIGHT48 xong` trong `log/night48_161.log` và
`log/night48_158.log`), chạy:

```bash
python3 scripts/apply_runtime_logging.py --check   # xác nhận còn khớp
python3 scripts/apply_runtime_logging.py           # áp thật
rsync -az src/ <158>:/data/ntat/MultiVD/src/       # đồng bộ sang 158
```

Sau đó mỗi lần chạy sẽ tự ghi `<checkpoint>.runtime.json` (ghi nguyên tử: file tạm rồi
`os.replace`, không bao giờ để lại JSON cụt), và mỗi kết quả có thêm trường `runtime`
gồm giờ Pha 1, giờ Pha 2, giờ suy luận và dấu vân phần cứng.

Đọc bằng `python3 tools/runtime_report.py results_night48 results results_night48_158`.
Công cụ **khử trùng** lần chạy Pha 1 dùng chung giữa các nhánh ρ — không thì tổng giờ
GPU bị tính gấp đôi. Ô chạy trước khi có bản vá được đếm riêng và báo rõ, không nuốt im.

Kết quả 90 ô của khối này **sẽ không có** trường `runtime` (chạy trước bản vá); số giờ
của nó lấy từ dòng `Elapsed` trong `log/jobs/*.log` nếu cần.
