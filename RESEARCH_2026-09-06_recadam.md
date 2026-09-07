# Nghiên cứu — 06/09/2026 — Tối ưu RecAdam và các hướng lân cận

Nhánh `optimize-v1`. Mục đích: tìm cách làm RecAdam (phần không thể bỏ của phương pháp
transfer hai pha) **ổn định hơn**, hướng tới Δ dương đều trên `4cwe` và `com`, và xử lý
hiện tượng **F1 tăng nhưng ROC-AUC tăng yếu** trên `codet5p-220m-bimodal`.

Mọi mục trong §2 đã fetch trực tiếp (arXiv / ACL / NeurIPS / OpenReview). Bổ sung cho
`RESEARCH_2026-08-20_0959.md` §3 (RecAdam và đối thủ), không lặp lại phần đó.

---

## 1. RecAdam đang chạy thế nào — số đo thật, không phải giả định

Mọi ô từ trước tới nay (1 896 dòng trong `records/results_all.jsonl`) dùng **đúng một cấu
hình RecAdam mặc định**, chưa từng quét:

| tham số | giá trị | nguồn |
|---|---|---|
| `anneal_fun` | sigmoid | `train_transfer.py:1118` |
| `anneal_k` | 0.05 | `:1120` |
| `anneal_t0_ratio` | 0.05 → **t0 = 43 bước** | `:1121`; log `RecAdam schedule` |
| `anneal_w` | 1.0 | `:1123` |
| `pretrain_cof` γ | 5000 → lr·γ = **0.1** | `:1154`, LR 2e-5 |
| `recadam_anchor` | `source` (θ Pha 1) | `:1150`; **`pretrained` chưa từng chạy** |
| tổng bước Pha 2 | 870 (30 epoch × 29 bước, batch 16, 456 mẫu) | log |

λ(t) thực tế: bước 1 → 0.109 · hết epoch 1 → 0.332 · bước 43 → 0.5 · bước ~90 (epoch 3) → ~0.9.
Số epoch Pha 2 thực chạy (đếm dòng `RecAdam target-task weight at epoch end` trong 300 log
Pha 2): **phần lớn 10–20 epoch**, đỉnh 14–18. Nên neo về nguồn chỉ còn tác dụng trong
**~3 epoch đầu trên 10–20 epoch**, tức 15–25 % quá trình. Phần còn lại là AdamW thuần
(λ≈1 ⇒ hệ số kéo (1−λ)γ ≈ 0).

Đối chiếu với bài gốc (Chen et al., EMNLP 2020, §4.1, đọc trực tiếp PDF):
- γ = 5000 **cố định**, không quét. t0 chọn trong {100, 250, 500, 1000} **bước**, k trong
  {0.05, 0.1, 0.2, 0.5, 1}, theo dev set từng task.
- Số bước huấn luyện: RTE 7 800, MRPC 11 500, CoLA 13 400 ⇒ t0 chiếm **1,3 %–13 %** quá trình,
  nhưng **tuyệt đối là 100–1 000 bước** neo, còn ta chỉ có 43.
- Lớp đầu ra (head) được tối ưu bằng **Adam thường, không neo**. Ta đang neo cả `vul_head`
  vào Pha 1 (`build_recadam_anchor`, head "stay anchored on Phase 1").
- Bảng 2: khởi tạo **ngẫu nhiên + neo về pretrained (RI)** 78.7 > khởi tạo pretrained + neo (PI)
  78.3 > fine-tune thường 77.0. Tác giả lý giải RI có không gian tìm kiếm rộng hơn.
- Lợi ích dồn vào task nhỏ (<10k mẫu): +1,7 trung bình; task lớn ≈ 0.

Nhận xét đã có từ 20/08 (§3 file cũ, phân tích của mình, không phải trích dẫn): vì λ(t) nhân
vào bước Adam đã chuẩn hoá, RecAdam ở đây ≈ **warmup ngầm 43 bước + neo ngắn**. Baseline
AdamW không có warmup. Chưa có đối chứng tách hai thứ đó.

Hai kết luận cũ cần đọc lại trong bối cảnh mới:
- `archive/RESULT_2026-08-23.md` §40.4: "giữ neo lâu hơn là giữ lại thiệt hại" — viết khi
  Pha 1 **làm hại** CodeT5+ (`cwe`, folds twin). Nay `latent_bottleneck` trên t5p cho Δ dương,
  nên lập luận đó không còn áp dụng thẳng; phải đo lại.
- §40.5 đã đề xuất **neo về pretrained, yếu và bền** (`pretrain_cof=20`, `t0_ratio=0.5`,
  `k=0.005`) và thêm cờ `--recadam_anchor pretrained`, nhưng **chưa từng chạy** (không có
  trong `results_all.jsonl`, không có log).

---

## 2. Tài liệu đã xác minh (mới so với 20/08)

### 2.1 Neo có trọng số theo tham số / theo lớp

| Bài | Nguồn | Điểm dùng được |
|---|---|---|
| Kirkpatrick et al. **EWC** | PNAS 2017, DOI 10.1073/pnas.1611835114 | Phạt bậc hai **nhân Fisher chéo** F_i(θ_i−θ*_i)². Fisher tính một lần trên dữ liệu task cũ (ở ta: dữ liệu Pha 1 tại checkpoint Pha 1). |
| Xu et al. **Child-Tuning** | EMNLP 2021, arXiv:2109.05687 | Biến thể **D** dùng Fisher để chọn mạng con rồi chỉ cập nhật phần đó. Trong benchmark Match-Tuning (IJCAI 2022) thắng RecAdam 79.92 vs 79.08. |
| Tian et al. **Selective Projection Decay (SPD)** | **NeurIPS 2024**, arXiv:2411.01713, mã GT-RIPL/Selective-Projection-Decay | L2-SP **chọn lọc theo lớp**: chỉ kéo về θ₀ khi điều kiện c_t = −g_tᵀ(θ_{t−1}−θ₀) < 0 (hướng đi hiện tại không còn khớp gradient), cường độ = λ · r_t với r_t = max(0, γ_t−γ_{t−1})/γ_t (tỉ lệ độ lệch tăng thêm). Khuyên bắt đầu λ=1. Không cần Fisher, một dòng điều kiện trong optimizer. Ghi rõ L2-SP đồng nhất mọi lớp là nguyên nhân under-fit hoặc thiếu regularize. |
| Song et al. **Hierarchical Layer-Wise and Element-Wise Regularization** | arXiv:2501.13669 (01/2025) | Độ quan trọng từng phần tử = **tích phân đường của gradient** trong quá trình học task cũ, ω_i = −∫ g θ' dt, chuẩn hoá Ω_i = Σω_i/((Δ_i)²+ξ); hệ số **theo lớp** = softmax(‖Ω_l‖₂). Loss = L_task + φ·ΣΩ_i(θ_i−θ*_i)², φ=e⁻³. Ablation: bỏ layer-wise −0.3 acc, bỏ cả hai −1.0. Nhanh hơn EWC-LoRA 20×. Trên LLaMA-3/GPT-J, không có PLM nhỏ. |
| Somayajula et al. **Attention-guided weight mixup (AGWM)** | **NAACL 2024**, arXiv:2403.12918, mã Sai-Ashish/Attention_guided_weight_mixup_BLO | Mỗi trọng số = α·θ_task + (1−α)·θ_pretrained, α học bằng **bilevel** trên hai split. BERT-large, 4 task GLUE nhỏ, 10 seed: vanilla 78.88±1.64, Mixout 79.54, R3F 79.23, Child-Tuning-D 79.62, Re-init 79.75, DPS 80.03, **AGWM 80.42±0.93**. Chi phí 1,8–4× vanilla. **Bảng không có RecAdam.** Ở 300/500/1000 mẫu: 68.97 vs vanilla 62.54 (300 mẫu). |
| Jin et al. **Rotation-Preserving SFT** | arXiv:2605.10973 (05/2026) | Phạt thay đổi trong khối top-k vector kỳ dị của mỗi ma trận pretrained; lập luận Fisher quá đắt ở quy mô LLM. Không áp thẳng cho 220M nhưng là bằng chứng hướng "neo có cấu trúc" vẫn sống năm 2026. |

### 2.2 Ổn định giữa seed/fold — trung bình trọng số

| Bài | Nguồn | Điểm dùng được |
|---|---|---|
| Lu et al. **SWA cho PLM** | Findings EMNLP 2022, arXiv:2212.05956 | SWA lúc fine-tune PLM nhỏ, không tốn thêm tính toán, cực tiểu phẳng hơn. |
| Pecher et al. **DENI** | Findings EMNLP 2024, arXiv:2406.12471 | Ensemble trễ + nội suy nhiễu; giảm sd giữa seed, thắng Mixout/SWA/ensemble với chi phí thấp hơn ensemble. 3 model × 7 bộ phân loại. |
| Sadrtdinov et al. **"To Stay or Not to Stay in the Pre-train Basin"** | NeurIPS 2023, arXiv:2303.03374 | Ra khỏi lòng chảo pretrain thì **mất lợi ích transfer**; averaging/ensemble nên nằm trong lòng chảo (StarSSE). Ủng hộ neo + trung bình trọng số cùng lúc. |
| Sherborne et al. **TRAM** | **ICLR 2024 spotlight**, arXiv:2310.03646 | SAM với vùng nhiễu loạn định bởi trust region trên **biểu diễn** (function space), nhắm transfer OOD/cross-lingual. Chi phí 2× như SAM. |

### 2.3 F1 tăng nhưng AUC không

| Bài | Nguồn | Điểm dùng được |
|---|---|---|
| He, Chen, Zhu **Preserving Pre-trained Features Helps Calibrate** | ICLR 2023, arXiv:2305.19249 | Fine-tune phá hiệu chuẩn; giữ đặc trưng pretrained (họ L2-SP/anchor) **cải thiện hiệu chuẩn** nhất là dưới dịch chuyển miền. |
| Guo et al. temperature scaling | ICML 2017 (đã có ở file 20/08) | Hiệu chuẩn đơn điệu **không đổi AUC** ⇒ nếu F1@0.5 tăng mà AUC không, đó là **dịch ngưỡng**, không phải xếp hạng tốt hơn. Kiểm bằng metric ở ngưỡng hiệu chuẩn theo val (pipeline đã tính, `METHOD.md` §8). |

### 2.4 Đã thử và bác trong dự án — không đề xuất lại

- LP-FT (Kumar et al. ICLR 2022), nội suy α giữa θ_Pha1 và θ_pretrained (WiSE-FT kiểu Pha 1),
  LoRA r=8 ở Pha 1 — `DEAD_ENDS.md` #3, bối cảnh `cwe` trên CodeT5+/twin.
- ASAM ρ ∈ {0.1, 0.2, 0.5} ở Pha 2 trên t5p: không vượt sàn nhiễu ở n=15 (`FACTS.md` §18).
- Uncertainty weighting cho λ — `DEAD_ENDS.md` #13, lỗi cấu trúc.

---

## 3. Hướng đề xuất, xếp theo ưu tiên thử

Tiêu chí xếp: (giá trị cho luận điểm "RecAdam là phần cốt lõi và đã được tối ưu") ×
(chi phí — mọi hướng dưới đây **chỉ đổi Pha 2**, dùng lại 9 checkpoint Pha 1 `n48` đã có:
seed 42/7 ở 161, seed 1234 ở 158; ~10,7 phút/ô t5p trên A4000).

### P1 — Quét lịch neo RecAdam (t0, k, γ) — chưa từng làm, rẻ nhất
Trục một chiều trước, không lưới đầy: (a) **độ bền**: `t0_ratio` 0.05→0.2→0.5 (neo tới
epoch ~6 và ~15); (b) **độ mềm**: `pretrain_cof` 5000→500→50 ở t0_ratio 0.5; (c) độ dốc `k`
0.05→0.01. Bài gốc neo 100–1 000 bước, ta 43 — đây là biến duy nhất chưa ai trong dự án chạm.
Đo trên t5p × {4cwe, com} × 5 fold × seed 42 trước (~6 cấu hình × 10 ô = 60 ô ≈ 11 h một máy),
thắng thì lên 3 seed.

### P2 — Neo về `pretrained` (và neo đôi) — cờ đã có, chưa chạy
`--recadam_anchor pretrained` với cấu hình §40.5 (cof 20, t0_ratio 0.5, k 0.005). Câu hỏi
học thuật: nhớ **pretrained** hay nhớ **Pha 1**? Bài gốc (RI > PI) gợi ý neo về pretrained
từ điểm xuất phát Pha 1 có thể là điểm ngọt. Biến thể "neo đôi" (cả hai, hai γ) cần ~20 dòng.

### P3 — RecAdam-Fisher: hệ số kéo theo độ quan trọng tham số
Thay γ đồng nhất bằng γ·F̂_i với F̂ = Fisher chéo (hoặc tích phân đường gradient như
arXiv:2501.13669) tính **một lần** trên dữ liệu Pha 1 tại checkpoint Pha 1 (vài phút/checkpoint).
Chuẩn hoá để trung bình F̂ = 1, giữ γ cũ ⇒ so được cạnh P1. Đây là ghép EWC vào lịch anneal
của RecAdam; tôi không tìm thấy bài nào làm đúng tổ hợp này (phải ghi là đề xuất của mình).
Kể chuyện được: "chỉ nhớ phần Pha 1 thực sự học được, còn lại để Python tự do". Mã: ~60 dòng
(`RecAdam.py` nhận `pretrain_cof` dạng tensor theo nhóm, script tính Fisher).

### P4 — Neo chọn lọc theo lớp (SPD) hoặc giảm dần theo độ sâu
(a) SPD: thêm điều kiện c_t < 0 theo lớp và cường độ r_t vào RecAdam — không cần Fisher,
NeurIPS 2024, có mã tham chiếu. (b) Rẻ hơn nữa: γ_layer = γ·d^(L−l) (lớp dưới neo mạnh, lớp
trên tự do — cùng tinh thần LLRD/gradual unfreezing). Có thể chạy như một điểm phụ trong P1.

### P5 — Trung bình trọng số ở Pha 2 (SWA/EMA) — nhắm **ổn định** và AUC
Áp cho **cả baseline lẫn transfer** để Δ vẫn công bằng; kỳ vọng giảm sd giữa fold, tăng số
fold cùng dấu và cải thiện xếp hạng (AUC) hơn F1@0.5. Là cải tiến pipeline, không phải của
RecAdam — nhưng là cách rẻ nhất để đạt "ổn định trên 4cwe và com".

### P6 — Đối chứng tách warmup khỏi neo (hỏi trước — là nhánh đối chứng thêm)
AdamW + warmup tuyến tính đúng 43 bước (bằng λ(t)) **không neo**. Nếu bằng RecAdam thì phần
"neo" hiện tại không mua gì và P1–P3 càng cần; nếu kém thì có bằng chứng neo có tác dụng dù
ngắn. Reviewer chắc chắn hỏi. 15 ô/seed.

### P7 — TRAM (SAM + trust region trên biểu diễn)
ASAM đã cho null nên xếp cuối; khác ASAM ở chỗ ràng buộc **function space**, đúng bài toán
transfer, nhưng 2× chi phí và cần cài mới.

### Chẩn đoán đi kèm mọi hướng (không tốn GPU)
- Báo Δ ở **cả** F1@0.5, F1@ngưỡng-val, ROC-AUC, PR-AUC. Nếu Δ F1@ngưỡng-val ≈ 0 trong khi
  Δ F1@0.5 > 0 thì hiệu ứng là dịch ngưỡng.
- Cân nhắc chọn checkpoint Pha 2 theo **val ROC-AUC** (hoặc F1@ngưỡng-val) thay vì F1@0.5 —
  đổi tiêu chí chọn, không đổi mô hình; áp cho cả baseline.

---

## 5. Đối chiếu với số liệu thật (1 015 fold JSON, ghép cặp cùng run/seed/fold, 06/09)

Bảng đầy đủ 180 nhóm: scratchpad `out.txt` của phiên này; tóm những điểm đổi thứ tự ưu tiên.

### 5.1 AdamW (fine-tune hai lần) hiện **thắng** RecAdam ở hầu hết ô — cả F1 lẫn AUC

Cùng checkpoint Pha 1, `latent_bottleneck`, λ=0.05, seed 42, Δ vs baseline @0.5 (ΔAUC trong ngoặc):

| backbone | nguồn | AdamW (khối B) | RecAdam (khối A) |
|---|---|---|---|
| codebert | 4cwe | +0.0617 5/5 (+0.0291) | +0.0388 4/5 (+0.0173) |
| codebert | com | +0.0549 5/5 (+0.0264) | +0.0217 4/5 (+0.0029) |
| codebert | full | +0.0647 4/5 (+0.0327) | +0.0577 5/5 (+0.0301) |
| unixcoder | 4cwe | +0.0318 4/5 (+0.0107) | +0.0291 3/5 (+0.0122) |
| unixcoder | com | +0.0142 3/5 (−0.0020) | +0.0119 3/5 (−0.0010) |
| unixcoder | full | +0.0182 3/5 (+0.0066) | **−0.0093 2/5 (−0.0122)** |
| t5p | 4cwe | +0.0322 4/5 (+0.0056) | +0.0322 5/5 (+0.0047) |
| t5p | com | +0.0278 4/5 (+0.0014) | +0.0147 4/5 (−0.0090) |
| t5p | full | +0.0202 5/5 (+0.0046) | +0.0308 4/5 (+0.0102) |

Giá trị của head (LB − `none`, cùng fold): AdamW **+0.0119, 31/40, p=0.0007**; RecAdam +0.0021,
19/40, p=0.87. Ô canonical âm duy nhất vs baseline: `unixcoder × full × RecAdam`. Head âm so
với `none` dưới RecAdam: unixcoder/full **−0.0318 (0/5)**, codebert/com −0.0063, t5p/com −0.0022.

⇒ RecAdam ở cấu hình hiện tại (neo mạnh, ngắn: lr·γ=0.1, t0=43 bước) đang **trả giá** so với
không neo. Đây là lý do mạnh nhất cho P1: chế độ "neo yếu, dài" (L2-SP cổ điển) chưa từng thử.

### 5.2 t5p: F1 tăng nhưng AUC không — đo được

- r(ΔF1, ΔAUC) trên t5p = 0.77 (446 ô), nhưng **29/≈70 nhóm** có ΔF1 > +0.01 mà ΔAUC < +0.005.
- NIGHT48 ρ=0, n=45: ΔF1@0.5 **+0.0126** (31/45) · ΔF1@ngưỡng-val +0.0082 · **ΔAUC −0.0003 (21/45)**.
  4cwe: +0.0159 / +0.0030 AUC; com: +0.0176 / +0.0056; full: +0.0045 / **−0.0094 (4/15)**.
- Codebert thì ngược lại: r = 0.95, ΔAUC +0.02–0.03 dưới AdamW.
⇒ Trên t5p, transfer đổi **điểm vận hành**, gần như không đổi **xếp hạng**. Bài phải báo cả hai
metric; và tiêu chí chọn checkpoint (val F1@0.5) đang tối ưu đúng thứ AUC không nhìn thấy.

### 5.3 Hiệu ứng NIGHT48 dồn vào **seed 42**, và seed 42 chạy trên máy khác

| seed | máy | ΔF1@0.5 (n=15) | baseline F1@0.5 |
|---|---|---|---|
| 42 | vast A4000 cu130 | **+0.0351 (13/15)** | 0.7879 (fold 3: **0.7036**) |
| 7 | 161 | +0.0019 (10/15) | 0.8087 |
| 1234 | 158 | +0.0010 (8/15) | 0.8116 |

Baseline seed 42 fold 3 trên vast là 0.7036, cùng seed/fold trên ntat2 (khối A) là 0.8217 —
lệch 0.118 chỉ do máy/CUDA. Hai seed còn lại ≈ 0. **Ở n=15, hiệu ứng t5p × LB × RecAdam chưa
"ổn định"; nó là một seed trên một máy.** Theo fold (n=9 mỗi fold): +0.0229 / −0.0063 / +0.0328 /
−0.0021 / +0.0159 — hai fold có baseline cao nhất (2: 0.822, 4: 0.849) không lợi gì.

### 5.4 Hai lỗi sổ sách phải sửa trên `optimize-v1` trước khi chạy khối mới

1. `tools/n48_report.py` dùng `test_macro_f1_at_valcal`, còn FACTS §15/§17 và `src/report_*.py`
   dùng `test_macro_f1_at_0.5`. Cùng NIGHT48: 4cwe ρ0 +0.0133 (p=0.119) theo valcal nhưng
   +0.0159 (12/15, p=0.035) theo @0.5. Phải chốt một metric chính và ghi rõ.
2. JSON Pha 2 ghi `lambda_cwe=0.2` (mặc định CLI) bất kể λ thật; không ghi t0 tuyệt đối, không
   ghi val Pha 1. Khối mới phải ghi đủ: λ thật, `recadam_anchor`, `pretrain_cof`, t0 (bước),
   k, val Pha 1, đường dẫn checkpoint Pha 1.

### 5.5 Thứ tự ưu tiên sau khi đối chiếu

| # | việc | vì sao lên/xuống | chi phí (t5p, seed 42, 4cwe+com) |
|---|---|---|---|
| **1** | **P1 quét γ ∈ {50, 500, 5000} × t0_ratio ∈ {0.05, 0.5}**, giữ k=0.05 | §5.1: AdamW (γ→0) > RecAdam (γ=5000, ngắn); điểm giữa chưa đo | 5 cấu hình mới × 10 = 50 ô ≈ 9 h |
| **2** | **P6 đối chứng AdamW + warmup 43 bước** | tách "warmup ngầm" khỏi "neo"; nếu ≥ RecAdam thì neo hiện tại vô dụng | 10 ô ≈ 2 h |
| **3** | **P2 neo `pretrained`** (cof 20, t0 0.5, k 0.005) + biến thể **không neo head** | bài gốc không neo head; cờ đã có | 20 ô ≈ 4 h |
| 4 | P3 RecAdam-Fisher | chỉ đáng nếu #1 cho thấy *có* chế độ neo thắng AdamW; Fisher tinh chỉnh neo, không cứu được neo vô dụng | sau #1 |
| 5 | chọn checkpoint theo val AUC (hoặc F1@ngưỡng-val), áp cả baseline | §5.2 | chạy lại, hỏi trước |
| 6 | P5 SWA/EMA Pha 2 | §5.3 ổn định giữa seed | sau |
| 7 | P4 SPD / theo lớp · P7 TRAM | | sau |

Sau khối seed 42: lặp cấu hình thắng ở seed 7 và 1234 (checkpoint Pha 1 sẵn: 161 và 158),
rồi codebert (checkpoint `model/s42/phase1/codebert__latent_bottleneck_{4cwe,com}` có sẵn)
vì codebert là nơi khoảng cách AdamW–RecAdam về AUC lớn nhất.

## 7. Đo được trong đêm 06→07/09 — bậc 1, n=3 fold, seed 42

Ba nguồn `4cwe`/`com`/`full`, `latent_bottleneck`, λ=0.05, dùng lại checkpoint Pha 1 của
NIGHT48. **Bậc 1, chỉ để sàng lọc.** Đọc lại bằng `python3 tools/opt1_report.py`.

### 7.1 Early stopping đang cắt mọi nhánh neo bền — câu hỏi chính CHƯA được trả lời

| λ ở epoch 5 | số epoch chạy | ô sập (F1 < 0.6) |
|---|---|---|
| 0.000 | 7.0 | 12/12 |
| 0.359 | 7.0 | 1/1 |
| 0.190 | 18.2 | 1/4 |
| 0.994 | 13–18 | 0/24 |

`patience=5`, `min_epochs=3`. Nhánh có λ lên chậm thì năm epoch đầu val không nhúc nhích,
patience kích hoạt, dừng ở **đúng** epoch 7. Cùng nhánh `c5000_t0p2k02`: fold 1 dừng epoch 7
với F1 0.4807, fold 3 chạy 23 epoch và cho 0.7871. Không phải "neo bền có hại" mà là
"chưa kịp học đã bị cắt". Khối **ME10** (`min_epochs=10` cho cả nhánh chính lẫn đối chứng)
đang chạy để trả lời.

**Đây là tương tác giữa lịch anneal và tiêu chí dừng, không phải tính chất của phương pháp.**
Cùng họ với bẫy λ ở §5.4: hai siêu tham số nhìn thì độc lập, thực ra ràng buộc nhau.

### 7.2 Trục γ: neo yếu thắng neo mặc định

Ghép cặp theo cùng (cây, seed, fold, nguồn), so với **AdamW thuần**:

| γ | Δ F1@0.5 | fold dương | Δ ROC-AUC |
|---|---|---|---|
| 5 | +0.0295 | 3/4 | +0.0086 |
| 0.5 | +0.0178 | 2/4 | +0.0133 |
| 50 | +0.0121 | 3/7 | −0.0008 |
| 500 | +0.0044 | 3/7 | −0.0025 |
| 5000 (mặc định) | −0.0086 | 1/6 | −0.0089 |

n nhỏ và biên độ rộng, **chưa kết luận được**. Nhưng hướng đơn điệu và nhất quán với
phát hiện cũ ở §5.1.

### 7.3 Warmup không phải nguồn giá trị của RecAdam

`warm` (AdamW + đúng λ(t) trên learning rate, **không neo**) so với `plain` (AdamW thuần):
**−0.0008, 3/6 fold**. Vậy phần "warmup ngầm" mà §1 nêu ra không mua gì. Nếu RecAdam có
giá trị thì nó nằm ở **neo**, và câu hỏi trở thành neo *thế nào* chứ không phải neo *bao nhiêu*.

### 7.4 Tiêu chí chọn checkpoint: không đáng đổi

81 lần chạy có `val_history`. Chọn theo val F1@0.5 bỏ lỡ trung bình **0.0083** val ROC-AUC;
chọn theo AUC bỏ lỡ **0.0102** val F1. Hai chiều tương đương nên giữ F1@0.5 là đúng.
Câu hỏi §6.4 đóng lại mà không tốn một ô GPU nào.

### 7.5 Hệ quả: neo THÔNG MINH thay vì neo YẾU

§7.2 dẫn tới một vấn đề cho phương pháp: nếu γ→0 là tốt nhất thì RecAdam tiến về AdamW,
và luận điểm "RecAdam là phần không thể bỏ" mất chỗ dựa. Lối ra là giữ **tổng** lực kéo
và đổi **cách phân bố** nó — chính là P3 (§3). Mã đã viết và kiểm xong đêm nay:
`src/fisher.py`, `src/recadam_fisher.py`, `tests/test_recadam_fisher.py`.

Phép kiểm hai chiều bắt được một ràng buộc thật: lực kéo cập nhật hiện là lặp điểm cố định,
ổn định chỉ khi `0 < lr·γ·F_i < 2`. Nên giá trị kẹp Fisher **bị ràng buộc với γ**
(lr=2e-5: γ=5000 ⇒ F_max < 5). Đã ghi memory.

## 8. Tra cứu tài liệu đêm 06→07/09 — cái gì đã có, cái gì thật sự mới

Mọi mục dưới đây fetch trực tiếp bản gốc (PDF ACL / arXiv), trừ chỗ ghi rõ chỉ thấy snippet.

### 8.1 RecAdam KHÔNG dùng early stopping, và t₀ của nó là số bước TUYỆT ĐỐI

Nguyên văn bài gốc §4.1: *"select the training step (61,360 on MNLI … 11,500 on MRPC,
7,800 on RTE) to improve the fine-tuning stability."* Ở batch 32 những con số đó chia đúng
thành **50–100 epoch** cho hai task nhỏ nhất (RTE 2 490 dòng, MRPC 3 668 dòng). Chuỗi
"early stop" và "patience" **không xuất hiện** trong bài; repo chính chủ chỉ dùng `--max_steps`.

Trong `RecAdam.py`, `t` của λ(t) là `state["step"]` — **bộ đếm bước tuyệt đối**, không phải
tỉ lệ. Ánh xạ lưới t₀ của họ sang run của ta (456 dòng, 29 bước/epoch, tối đa 870 bước):

| t₀ (bước) | = epoch của ta | % quá trình của ta | λ tại epoch 5 |
|---|---|---|---|
| 100 | 3.4 | 11.5 % | **0.989** |
| 250 | 8.6 | 28.7 % | 2.8e−5 |
| 500 | 17.2 | 57.5 % | ~0 |
| 1000 | 34.5 | **114.9 % — quá cả cuối quá trình** | ~0 |

**Bảng này tái hiện đúng phép chia 12/12 sập so với 0/24 sập ở §7.1.** Trong run của họ,
t₀=1000 chỉ chiếm 1,6–12,8 % quá trình. Vấn đề của ta không tinh vi: λ≈0 nghĩa là loss đích
mang trọng số ~0 **theo đúng công thức**, và mô hình đứng yên là điều nó *được thiết kế* để làm.

Không tìm thấy bài nào phát biểu tương tác giữa **trọng số loss có lịch** và **tiêu chí dừng
theo val**. Các mảnh rời rạc thì có: Mosbach et al. ICLR 2021 (arXiv:2006.04884) mô tả run
hỏng có *"practically constant training loss"* và khuyên tăng số vòng lặp; Keras thêm
`EarlyStopping(start_from_epoch=…)` ở 2.11 với lời giải thích đúng hiện tượng này
(*"allows for a warm-up period in which no improvement is expected"*), còn PyTorch Lightning
**không có** tham số tương đương; Rieck et al. ICLR 2019 (arXiv:1812.09764) quét lưới
(burn-in × patience). Ngược chiều: Dodge et al. (arXiv:2002.06305) khuyên **giết sớm** ở
20–30 % quá trình dựa trên giả định tương quan sớm-muộn — kết quả của ta là **phản ví dụ
trực tiếp** cho giả định đó dưới mục tiêu có lịch.

**Tiền lệ cho việc đổi t₀ sang tỉ lệ:** HuggingFace thêm `warmup_ratio` đúng vì
`warmup_steps` tuyệt đối không chuyển được giữa các cỡ dữ liệu.

### 8.2 γ = N·F̄ — LÝ THUYẾT cho phát hiện "γ nhỏ hơn" ở §7.2

Dẫn xuất trong chính bài RecAdam: `Loss_S ≈ ½ N F Σ(θᵢ−θᵢ*)² = ½ γ Σ(θᵢ−θᵢ*)²`, tức
**γ = N·F̄ với N là số quan sát hậu thuẫn cho điểm neo**. γ=5000 là đại diện cho **toàn bộ
corpus pretraining**. Neo của ta là checkpoint Pha 1 huấn luyện trên **930–7 598 dòng**, nên
chính dẫn xuất đó nói γ phải nhỏ đi nhiều bậc. Huszár (PNAS 2018, 115(11):E2496, doi
10.1073/pnas.1717042115) hậu thuẫn: hệ số λ_A *"replaces the sample size N_A"*, và cảnh báo
*"double-counting the data from earlier tasks"*.

**Đây là khung lý thuyết tốt nhất cho kết quả thực nghiệm của ta, và trích dẫn được đầy đủ.**

**Hệ quả kiểm được, chưa đủ dữ liệu:** nếu γ ∝ N thì γ tối ưu phải **tăng** theo nguồn
(4cwe 930 < com 3 744 < full 7 598). Ở n=1/ô hiện tại thứ tự ra 5 / 0.5 / 50 — thuần nhiễu.
**Kiểm lại khi đủ 3 fold**; đây là phép kiểm cơ chế không tốn thêm một ô GPU nào.

### 8.3 lr·γ mới là siêu tham số thật

`RecAdam.py:127` đặt bước neo **ngoài** mẫu số thích nghi của Adam, nên độ dịch bị xoá mỗi
epoch (29 bước, λ≈0) là:

| γ | lr·γ | % độ dịch bị xoá mỗi epoch |
|---|---|---|
| 5000 (mặc định) | 0.10 | **95.3 %** |
| 500 | 0.01 | 25.3 % |
| 50 | 1e−3 | 2.9 % |
| **5 (tốt nhất của ta)** | **1e−4** | **0.29 %** |
| 0.5 | 1e−5 | 0.03 % |

Ở giá trị mặc định, mô hình **không thể di chuyển** trong lúc λ≈0 — cùng một cơ chế với §8.1.

**Xác nhận độc lập rất đẹp:** L2-SP (Li et al., ICML 2018, arXiv:1802.01483) quét α ∈
{0, 1e−3, 1e−2, 1e−1, 1} với lr 0.005–0.02, tức **lr·α ≈ 5e−6 … 2e−2**; ở lr=0.01, α=1e−2
cho lr·α = **1e−4**, **trùng khít** lr·γ = 2e−5 × 5 = 1e−4 của ta. Bài đó cũng ghi
*"the test accuracy varies smoothly according to the regularization strength"* và
β = 0.01 luôn được chọn, còn **α thì không công bố giá trị thắng theo từng bộ** — đừng trích
một con số α cụ thể.

Lưu ý phản chứng: Mixout (Lee et al., ICLR 2020) quét λ ∈ {0.01, 0.04, 0.07, 0.10} trên GLUE
nhỏ và cho kết quả **không đơn điệu** (λ=0.04 tệ nhất ở mọi task). Nên phát biểu "yếu hơn thì
tốt hơn" **trong phạm vi cấu hình của ta**, không phải như một quy luật chung.

### 8.4 CẢNH BÁO cho hướng Fisher: nó đã có, và có kết quả NULL

Chính bài L2-SP định nghĩa **L2-SP-Fisher**: `Ω(w) = (α/2) Σ F̂ⱼⱼ(wⱼ−wⱼ⁰)² + (β/2)‖w_S̄‖²`,
với Fisher ước lượng trên **dữ liệu nguồn tại checkpoint nguồn** — đúng bằng `γ·F_i` của ta,
chỉ thiếu phần anneal. Kết luận của họ, nguyên văn:

> *"We expected L2-SP-Fisher to outperform L2-SP … but there is no significant difference
> between the two options. Since L2-SP is simpler … we recommend the former."*
> *"contrary to lifelong learning, our objective does not favor solutions that retain accuracy
> on the source task."*

Lý do đó áp dụng **nguyên vẹn** cho ta: mục tiêu là macro-F1 trên đích, không phải giữ điểm
nguồn. Phần còn mới là **Fisher đặt trong lịch λ(t)** — không tìm thấy tiền lệ, nhưng phải
viết là "chúng tôi không biết có", không phải "chưa ai làm".

**Rủi ro lớn nhất, phải đo trước khi tốn GPU:** tại một checkpoint đã hội tụ, số hạng
`(p_k − y_k) → 0` nên Fisher **tiêu biến**. Hệ quả cho ta: Pha 1 **tốt** ⇒ F≈0 ⇒ `γ·F` suy
biến thành AdamW; Pha 1 **sập** ⇒ F lớn ⇒ quá cứng. **Cổng rẻ: in histogram F_i và tỉ số
(số hạng phạt)/(loss đích) tại từng checkpoint Pha 1 trước khi chạy bất kỳ ô nào** — đúng
quy tắc `measure-the-mechanism-first`.

Recipe đã xác minh: van de Ven (ICLR 2025 Blogpost, arXiv:2502.11756) đo EXACT 84.91 >
SAMPLE 83.77 > EMPIRICAL 83.28, khuyên *"reduce the number of training samples used to
compute the Fisher"* thay vì cắt góc chỗ khác ⇒ với head 2 lớp, **EXACT ở n≈500–2000**.
Thorne & Vlachos (EACL 2021) dùng BERT-base, N=2000, ~25 s, và **λ tốt nhất = 1e7** —
cách γ=5000 của RecAdam **10⁴ lần**, thuần do thang đo Fisher ⇒ **phải quét lại γ từ đầu**
sau khi thêm F. Chuẩn hoá về trung bình 1 **không** có tiền lệ công bố; phải ghi rõ đó là
lựa chọn thiết kế của mình (và nó làm γ so sánh trực tiếp được với RecAdam thường).

### 8.5 Cái gì thật sự mới

| Phát biểu | Trạng thái |
|---|---|
| Lịch λ(t) và tiêu chí dừng tương tác; cặp lệch nhau loại cấu hình vì lý do sai | **Mới** — cách chữa là folklore, chẩn đoán thì chưa ai viết |
| t₀ của RecAdam là bước tuyệt đối, không chuyển được sang run ngắn | **Mới**, đã định lượng ở §8.1 |
| Quét γ; neo yếu hơn tốt hơn trên đích nhỏ | **Mới** — γ bị cố định 5000, chưa từng ablation |
| lr·γ mới là cường độ thật; mặc định = xoá 95 %/epoch | **Mới** cho RecAdam |
| γ phải nhỏ vì neo là checkpoint Pha 1 mang ít bằng chứng hơn | **Mới**, và **suy ra được từ chính γ = N·F̄** của RecAdam |
| Neo có trọng số Fisher | **Đã có** — L2-SP-Fisher, ICML 2018, kết quả **null** |

## 9. Vì sao fold 3 "khó" — đo được, và nó ĐẢO NGƯỢC cách đọc kết quả (07/09)

### 9.1 Chia fold là công bằng

`data/sven_python_folds_norm` là cross-validation 5 khối chuẩn: 760 dòng chia thành năm
khối 152, mỗi fold lấy một khối làm test, một khối làm val, ba khối làm train, xoay vòng.
Kiểm được bằng dấu vân: test của fold *k* trùng khít val của fold *k−1* ở cả n, tỉ lệ nhãn
lẫn phân vị độ dài. Phân bố CWE gần như đồng đều (CWE-089 79–87, CWE-078 38–44 mỗi khối).
**Không có gì sai ở khâu chia.**

### 9.2 Độ khó của fold = mức RÒ RỈ near-duplicate, không phải gì khác

Đo tỉ lệ hàng test có ít nhất một hàng train trùng ≥ ngưỡng theo Jaccard trên 5-gram token:

| fold | ≥0.9 | ≥0.75 | ≥0.5 | Jaccard tb | baseline F1 | baseline AUC |
|---|---|---|---|---|---|---|
| 1 | 2.6 % | 14.5 % | 40.8 % | 0.384 | 0.7828 | 0.8915 |
| 2 | 3.3 % | 17.1 % | 39.5 % | 0.371 | 0.8018 | 0.8879 |
| **3** | **2.0 %** | **10.5 %** | 36.8 % | 0.352 | **0.7036** | **0.8372** |
| 4 | 3.9 % | 17.1 % | 40.8 % | 0.367 | 0.8486 | 0.9144 |
| 5 | 3.3 % | 17.8 % | 33.6 % | 0.339 | 0.8026 | 0.9045 |

Tương quan Pearson trên đủ **n=5**: rò rỉ ≥0.9 với baseline F1 **+0.963**; ≥0.75 với F1
**+0.907**, với AUC **+0.918**. Nhưng ≥0.5 chỉ +0.325 và Jaccard trung bình chỉ +0.222 —
tức **chỉ bản sao gần khít mới quan trọng**, không phải độ giống chung. Đó là dấu vân của
rò rỉ **cặp**: mọi near-dup ≥0.75 đều mang nhãn ngược lại, tức bản vá của chính hàm test
nằm trong train.

**Dự đoán trước khi biết:** fold 4 rò rỉ cao ⇒ baseline phải cao. Kết quả: fold 4 là
0.8486, cao nhất. Fold 3 rò rỉ thấp nhất ⇒ baseline thấp nhất, đúng 0.7036.

### 9.3 Hệ quả: "fold khó" thật ra là "fold SẠCH", và neo giúp đúng ở đó

Δ của neo so với AdamW thuần, tách theo fold (ngoặc là số ô ghép cặp):

| cấu hình | fold1 (14.5 %) | fold2 (17.1 %) | **fold3 (10.5 %)** | fold4 (17.1 %) | fold5 (17.8 %) | tương quan với rò rỉ |
|---|---|---|---|---|---|---|
| γ=5 | −0.0064 (3) | −0.0088 (3) | **+0.0371** (3) | +0.0000 (1) | +0.0023 (3) | **−0.828** |
| γ=0.5 | −0.0108 (3) | −0.0089 (3) | **+0.0282** (3) | −0.0066 (1) | −0.0044 (3) | **−0.845** |
| γ=50 | −0.0086 (3) | −0.0111 (3) | **+0.0325** (3) | +0.0000 (1) | +0.0242 (3) | −0.471 |
| γ=5000 (mặc định) | −0.0042 (3) | −0.0111 (3) | −0.0084 (3) | −0.0034 (2) | +0.0043 (3) | +0.435 |

Tương quan **âm** nghĩa là neo giúp **nhiều hơn ở fold ít rò rỉ**. Neo yếu (γ=5, γ=0.5) cho
−0.83 và −0.85; neo mặc định γ=5000 thì không có tính chất này (+0.435).

**Cách đọc:** khi test có sẵn bản vá của chính nó trong train, mô hình chỉ cần **nhớ**, và
AdamW thuần làm việc đó tốt hơn — neo lúc này là gánh nặng. Khi test sạch và mô hình buộc
phải **khái quát**, tri thức nguồn mới có giá trị, và neo giữ được nó.

Đây là lập luận mạnh nhất cho phương pháp tính tới nay, vì nó vừa giải thích được vì sao
các kết quả trước "lẫn lộn", vừa nói rằng phần đo đáng tin nhất (fold sạch nhất) chính là
phần neo thắng.

**Giới hạn phải nêu:** n=5 fold, một seed, fold 4 mới có 1–2 ô ghép cặp. Bậc 2 đang chạy
sẽ lấp fold 4 và 5. Không đổi bộ dữ liệu — mức rò rỉ này `README.md` đã ghi nhận và nhóm
đã chấp nhận vì reviewer yêu cầu phân phối ngẫu nhiên. Việc cần làm là **báo cáo chỉ số
rò rỉ theo fold** như một biến giải thích, không phải thay dữ liệu.

## 6. Câu hỏi mở cho người dùng (chưa chạy gì cho tới khi có trả lời)

1. Chạy **khối 1 = #1 + #2 + #3 của §5.5** (≈ 80 ô, t5p, seed 42, 4cwe + com, chia fold trọn vẹn
   giữa 161 và 158, ~8 h nếu song song) hay chỉ #1 trước?
2. #2 (AdamW + warmup) là **nhánh đối chứng thêm** — cần đồng ý riêng.
3. Metric chính của bài: F1@0.5 (FACTS §15/§17) hay F1@ngưỡng-val (`n48_report.py`)? Khối mới
   sẽ báo cả hai kèm AUC, nhưng thứ tự xếp hạng cấu hình cần một metric chính.
4. Có cho đổi tiêu chí chọn checkpoint Pha 2 (val AUC hoặc F1@ngưỡng-val, áp cả baseline) không?
5. Có `full` trong khối 1 không? Người dùng đã giải thích `full` âm là negative transfer đặc thù
   ngôn ngữ; nếu bỏ thì tiết kiệm 1/3 chi phí, nhưng mất ô đối chứng "neo yếu có cứu full không".
6. Máy: 158 **trống hoàn toàn**, 161 còn ~14 GB (job t5p cần ~12,6 GB). Không cần vast cho seed 42.
