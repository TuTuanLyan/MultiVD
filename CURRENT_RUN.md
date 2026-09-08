# Đang chạy — INT1 trên vast (ntat + ntat2), 08/09/2026

> **QUYẾT ĐỊNH 08/09 — đừng đề xuất lại:** không chạy ASAM ρ=1.0 trên codebert.
> Người dùng nêu: ρ tối ưu phụ thuộc backbone là điều **đã chứng minh rồi**, nên khối đó chỉ
> chứng minh lại một thứ đã biết chứ không thêm gì. Khối e60 trên codebert thì **vẫn chạy** —
> đó là đối chứng ngân sách 60 epoch, câu hỏi khác hẳn.
>
> Bằng chứng ρ phụ thuộc backbone, có sẵn: ở ρ=0.1, t5p cho ΔAUC −0.0080 (5/15) còn codebert
> +0.0057 (7/10) — ngược dấu ở cùng một ρ.


> **BẬC 1 — KIỂM CHỨNG, 3 fold, seed 42.** Phóng 02:29 và 02:34 UTC 08/09.
> Ba khối trước (OPT1, RET1, SPD1) **đã xong**, chi tiết ở dưới.

## Câu hỏi

Ba khối trước đều nói cái neo không mua được gì, và §11 của `RESEARCH_2026-09-06_recadam.md`
giải thích được vì sao: toàn bộ tri thức nguồn đi vào bài toán đích qua **đúng một đường** —
điểm xuất phát θ*. INT1 hỏi câu còn lại: **ngoài vai trò điểm xuất phát, mô hình nguồn có
đóng góp trực tiếp gì cho điểm đích không?**

Trộn `θ(α) = α·θ_Pha2 + (1−α)·θ_Pha1` sau khi huấn luyện xong, quét α, chọn trên **val**.

| kết quả | nghĩa là |
|---|---|
| α tối ưu nằm hẳn trong (0, 1) | nguồn **đóng góp trực tiếp** — phát biểu transfer thật sự |
| α tối ưu = 1.0 ở mọi ô | nguồn **chỉ là điểm xuất phát**; giả thuyết bị bác, chi phí ~2 phút/ô |

## Máy — VAST, nhãn ntat và ntat2

| | ntat | ntat2 |
|---|---|---|
| id | 50223254 | 50223345 |
| GPU | RTX 5060 Ti 16 GB | RTX 5070 Ti 16 GB |
| giá | $0.0818/h | $0.1222/h |
| SSH | giải theo **nhãn** bằng `scripts/endpoints.sh`, không hardcode IP | |
| môi trường | `/venv/main` của image, torch 2.11.0+cu128 + transformers 4.57.1 + sklearn 1.7.2 | |
| fold | **1 và 2** — 10 ô | **3** — 5 ô |
| log trên máy | `/workspace/MultiVD/log/int1_ntat.log` | `…/int1_ntat2.log` |

**Vì sao lên vast:** 161 và 158 đang bị người dùng khác chiếm 7,0 và 8,9 GB VRAM, hàng đợi
chỉ nằm chờ. Hai hàng đợi ở nhà **đã dừng** để không chạy trùng.

**Chia theo FOLD TRỌN VẸN** (CLAUDE.md mục 4): baseline và cả hai đối chứng của một fold
nằm cùng máy với fold đó, nên Δ ghép cặp vẫn sạch dù hai máy khác GPU và khác bản torch
(2.11 trên vast vs 2.9.1 ở nhà). **Không được** so ô của vast với ô của máy ở nhà.

## Ma trận

| | |
|---|---|
| cây kết quả | `results/int1_t5p/` trên máy vast → kéo về `results_int1_ntat/`, `results_int1_ntat2/` |
| nguồn | `4cwe`, `com` |
| cấu hình | `plain` (AdamW) và `c50_t0p05` (RecAdam γ=50) |
| lưới α | `0, 0.1, …, 1.0` — 11 điểm |
| ô | 15 = 3 fold × (2 nguồn × 2 cấu hình + 1 baseline) |

Hai cấu hình chứ không một là có chủ ý: nếu α tối ưu **khác nhau** giữa có neo và không neo
thì bản thân điều đó là phát hiện — cái neo dịch được điểm ngọt.

## Vận hành

- `scripts/provision_int1.sh <nhãn> "<fold>"` — đẩy code/dữ liệu/checkpoint, **đối chiếu md5
  từng file trước khi phóng**; lệch thì thoát, không phóng. Đã khớp 20 mục trên cả hai máy.
- `scripts/watch_int1_vast.sh` — cron 10 phút: đếm ô, kéo kết quả về, phóng lại driver chết
  tối đa 3 lần. **Không bao giờ tự huỷ máy.**
- **Chưa huỷ máy nào** cho tới khi người dùng xác nhận, và chỉ sau khi đã kéo hết kết quả
  về và đối chiếu từng byte.

---

# (đã xong) Hàng đợi 07/09 — RET1 + SPD1

> **BẬC 1 — KIỂM CHỨNG, 3 fold, seed 42.** Hai khối, hai máy, chốt lúc 09:35 UTC 07/09.
> Người dùng đặt mốc: **phải xong phần dùng GPU trước 21:00 giờ VN (14:00 UTC)**.

| máy | khối | fold | ô | bắt đầu | dự kiến xong | biên |
|---|---|---|---|---|---|---|
| **158** | RET1 (giữ lại tri thức nguồn) | 1, 2, 3 | 21 | 07:56 UTC | **~11:20 UTC** (18:20 VN) | 2 h 40 |
| **158** | SPD1 fold 3 | 3 | 7 | ngay sau RET1 | **~12:30 UTC** (19:30 VN) | 1 h 30 |
| **161** | bậc 2 fold 4 (khối OPT1) | 4 | 21 | 05:29 UTC | **~10:00 UTC** (17:00 VN) | — |
| **161** | SPD1 fold 1+2 | 1, 2 | 14 | ngay sau bậc 2 | **~12:15 UTC** (19:15 VN) | 1 h 45 |

Nhịp đo được: **161 ≈ 9,6 phút/ô** (7 ô gần nhất: 8.8 8.7 8.2 10.1 8.6 13.1),
**158 ≈ 10 phút/ô** (8.5 11 12 10 9). **Kết luận: kịp, dư 1,5 giờ mỗi máy.**

**Vì sao SPD1 chia đôi máy:** 161 chạy một mình 21 ô mất ~3,8 h, cộng lúc bậc 2 xong
(~10:00) là **vượt mốc 21:00 VN**. Chia theo **fold trọn vẹn** (CLAUDE.md mục 4) — baseline
và cả hai đối chứng của mỗi fold đều nằm cùng máy với fold đó nên Δ ghép cặp vẫn sạch.

**Hàng đợi và giám sát**
- `scripts/queue_spd1.sh` (161) — chờ lock `multivd_opt1` + VRAM ≥ 13 GB rồi chạy fold 1,2
- `scripts/queue_spd1_158.sh` (158) — chờ lock `mvd_ret1` nhả rồi chạy fold 3
- `scripts/watch_ret1.sh` — cron 10′, đếm ô theo tag, kéo về `results_ret1_158/`
- `scripts/watch_opt1.sh` — cron 10′, khối OPT1/bậc 2

---

## RET1 — đo GIỮ LẠI TRI THỨC NGUỒN

Chấm chính mô hình Pha 2 trên **đúng tập val của Pha 1**, tái lập bằng
`split_source_records(records, seed)`. Cổng hai chiều đã qua: checkpoint Pha 1 trên tập tái
lập ra `0.697621` — trùng khít `best_val_macro_f1` ghi trong chính nó; trên nguồn khác ra
`0.567395`. Commit `6d56403`, chỉ **thêm trường** `source_retention`.

| | |
|---|---|
| cây | `results/ret1_t5p/` |
| nguồn | `4cwe`, `com` |
| cấu hình | `plain` (AdamW) · `c5000_t0p05` (RecAdam mặc định) · `c50_t0p05` (γ tốt nhất) |
| baseline | chấm trên **cả ba** nguồn — sàn "chưa hề thấy nguồn" |

## SPD1 — thay LỊCH THỜI GIAN bằng ĐIỀU KIỆN GRADIENT

SPD (Tian et al., NeurIPS 2024, arXiv:2411.01713). `src/spd.py`, `tests/test_spd.py`
(11 cổng hai chiều, tất cả đạt), commit `9990229`.

| | |
|---|---|
| cây | `results/spd1_t5p/` |
| nguồn | `4cwe`, `com` |
| cấu hình | `plain` (AdamW) · `c50_t0p05` (RecAdam γ=50) · `spdl1` (SPD λ=1) |
| kèm miễn phí | mọi ô đều chấm lại trên val Pha 1 → trả lời luôn "SPD có giữ nguồn không" |

> **ĐỌC KẾT QUẢ SPD PHẢI XEM `ti le kich hoat` TRONG LOG.** SPD chỉ phạt khi gradient quay
> đầu mà momentum vẫn đẩy tiếp; trên quỹ đạo trơn tru nó **không bao giờ phạt và bằng đúng
> AdamW**. Ô nào có tỉ lệ = 0 phải đọc là "AdamW", không phải "SPD kém". Smoke thật trên
> dữ liệu Python cho tỉ lệ **0.55**, nên không suy biến.

**Bẫy đã vá cùng lúc:** `run/matrix.sh` đặt hậu tố theo `[[ $OPT == adamw ]]`, nên nhánh `spd`
sẽ có hậu tố rỗng và **ghi đè lên nhánh recadam**, mất cả hai. Đổi sang `!= recadam`, kiểm ba chiều.

## Chưa chạy — đã ghi lại để không quên

`RESEARCH_2026-09-06_recadam.md` §11: neo là **ràng buộc**, không phải kênh truyền tri thức.
Ba đề xuất để tri thức cũ thật sự giúp đích, xếp theo thứ tự: **Đ3′** nội suy θ_Pha1 ⊕ θ_Pha2
sau huấn luyện (gần như miễn phí, và **có thể bác** giả thuyết) → **Đ4** chưng cất trong không
gian hàm → **Đ5** replay/đa nhiệm với dữ liệu nguồn ở Pha 2.

---

# (chi tiết khối RET1)

> **BẬC 1 — KIỂM CHỨNG, n=3 fold, seed 42.** Bắt đầu 07:56 UTC 07/09 trên **158**.
> Driver `run/ret1.sh` · log `log/ret1_158.log` · watchdog `scripts/watch_ret1.sh` (cron 10′)
> · kéo về `results_ret1_158/`. Khoảng 21 ô, ~3,5 h (158 chạy ~10 phút/ô).

## Vì sao có khối này — trục F1 trên tập đích đã hết chỗ

Đo trên **10 ô ghép cặp đầy đủ** (cùng cây, cùng fold, cùng máy, có đủ cả 7 cấu hình):

| cấu hình | F1 tb | F1 min | độ tản |
|---|---|---|---|
| `plain` AdamW thuần | 0.8206 | 0.7363 | 0.0333 |
| γ=0.5 | 0.8225 | 0.7958 | 0.0157 |
| γ=5 | **0.8271** | 0.7828 | 0.0205 |
| γ=50 | 0.8244 | 0.8023 | 0.0144 |
| γ=5000 (mặc định bài gốc) | 0.8141 | 0.7239 | 0.0332 |

Chênh lớn nhất giữa RecAdam và AdamW là **+0.0065 — dưới sàn nhiễu 0.010**, p ≥ 0.2 ở mọi γ.

**Phát biểu "neo nâng sàn" (06/09) ĐÃ RÚT LẠI.** Nó dựa vào đúng một ô (4cwe/fold3).
Bỏ ô đó ra: độ tản chỉ còn 0.0188 → 0.0126. Và trên **codebert nó không lặp lại** —
`plain` có độ tản 0.0147 và sàn 0.7828, đều tốt nhất bảng.

## Câu hỏi của RET1

Chưa lần nào đo thứ RecAdam **sinh ra để làm**: giữ tri thức nguồn. Chấm chính mô hình
Pha 2 trên **đúng tập val của Pha 1**, tái lập bằng `split_source_records(records, seed)`.

- neo giữ được nguồn, AdamW quên → khác biệt **có thật**, và đúng là lý do phương pháp
  cần RecAdam thay vì finetune hai lần
- cả hai quên như nhau → cái neo không làm gì thật, phải đổi hướng khác

**Cổng hai chiều đã chạy 07/09 trên 158**: chấm checkpoint Pha 1 trên tập tái lập ra
`0.697621`, **trùng khít** `best_val_macro_f1` ghi trong chính checkpoint; chấm nó trên
nguồn khác ra `0.567395`. Commit `6d56403` chỉ **thêm trường** `source_retention` vào JSON.

## Ma trận RET1

| | |
|---|---|
| cây kết quả | `results/ret1_t5p/` (riêng, để baseline và đối chứng đều chạy lại **cùng phiên**) |
| nguồn | `4cwe`, `com` |
| fold | 1, 2, 3 (vòng ngoài cùng) |
| cấu hình | `plain` (AdamW) · `c5000_t0p05` (RecAdam mặc định) · `c50_t0p05` (γ tốt nhất đo được) |
| baseline | chấm trên **cả ba** nguồn — sàn "chưa hề thấy nguồn" |
| ô | 3 fold × (2 nguồn × 3 cấu hình + 1 baseline) = **21** |

---

# (khối trước) OPT1 — tối ưu RecAdam, khối 1 — **XONG 07:11 UTC 07/09**

> **BẬC 1 — KIỂM CHỨNG, n=3 fold, seed 42. KHÔNG phải kết quả cuối.**
> Xem `CLAUDE.md` mục 1, tiểu mục "KIỂM CHỨNG và CHẠY KẾT QUẢ là HAI VIỆC KHÁC NHAU".
> Mọi Δ trong khối này chỉ dùng để **sàng lọc** cấu hình nào đáng lên bậc 2 (n=5),
> không được trích vào bài. n=3 đã bốn lần đổi dấu ở n=5 trong dự án này.
>
> **Trạng thái: XONG.** Bậc 1 (fold 1–3) xong 06/09; bậc 2 xong fold 5 trên 158 lúc
> 07:11 UTC 07/09, fold 4 trên 161 còn ~12 ô. Kết luận: xem đầu file (RET1).
> Nhánh git `optimize-v1`. Đêm 06→07/09 chạy không người
> trông, người dùng trao toàn quyền và quay lại ~02:00 UTC (9h sáng giờ VN).
> Kế hoạch và lý do: `RESEARCH_2026-09-06_recadam.md` §5.5. Báo cáo: `python3 tools/opt1_report.py`.

## Đường leo bậc của hướng RecAdam

| bậc | quy mô | phạm vi | điều kiện lên bậc |
|---|---|---|---|
| **1. Kiểm chứng** ← **đang ở đây** | 3 fold × 12 cấu hình × 3 nguồn, seed 42 | t5p, λ=0.05 | cấu hình nào vượt rõ hai đối chứng cùng phiên |
| 2. Xác nhận | 5 fold, seed 42 | chỉ cấu hình sống sót bậc 1 | hiệu ứng còn giữ khi thêm fold 4–5 |
| 3. Chạy kết quả | 5 fold × 3 seed = 15 | chỉ khi đã **chốt** | — |

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

## Hàng đợi đêm 06→07/09 — chạy không người trông

Người dùng nghỉ từ ~18:30 UTC 06/09 đến ~02:00 UTC 07/09 và **không trả lời được câu hỏi**,
đã trao toàn quyền quyết định. Hai hàng đợi tuần tự, mỗi cái giữ **lock riêng** để watchdog
biết nó còn sống:

| máy | lock hàng đợi | việc, đúng thứ tự |
|---|---|---|
| **161** (fold 1, 2) | `/tmp/mvd_queue161.lock` | OPT1 `4cwe`+`com` → OPT1 `full` → **ME10** fold 1,2 |
| **158** (fold 3) | `/tmp/mvd_queue158.lock` | OPT1 `full` → **ME10** fold 3 → **CODEBERT** trục γ, fold 1–3 |

`scripts/queue161.sh` và `scripts/queue158.sh`. Mỗi việc chờ lock driver nhả trước khi
chạy; 161 còn chờ VRAM trống ≥ 13 GB và **nhường** nếu người khác đang dùng GPU.
`scripts/watch_opt1.sh` (cron 10 phút) giám sát cả driver lẫn hàng đợi, phóng lại tối đa
3 lần, không bao giờ giết gì.

### Vì sao có khối ME10

Đo được từ `val_history` ngày 06/09: **early stopping đang cắt mọi nhánh neo bền trước khi
λ kịp lên**, nên câu hỏi "neo bền có tốt không" chưa hề được trả lời.

| λ ở epoch 5 | số epoch chạy | số ô sập (F1 < 0.6) |
|---|---|---|
| 0.000 | 7.0 | 12/12 |
| 0.359 | 7.0 | 1/1 |
| 0.190 | 18.2 | 1/4 |
| 0.994 | 13–18 | 0/24 |

Với `patience=5`, `min_epochs=3`, nhánh có λ lên chậm thì năm epoch đầu val không nhúc
nhích và bị dừng ở đúng epoch 7. ME10 nâng `min_epochs=10` cho **cả nhánh chính lẫn đối
chứng** (`c5000_t0p05_me10`) nên phép so vẫn đổi đúng một biến.

### Vì sao có khối CODEBERT

Trục γ trên t5p cho neo **yếu** thắng neo mặc định (ghép cặp cùng fold, so với AdamW thuần:
γ=5 → +0.0295 3/4; γ=0.5 → +0.0178 2/4; γ=5000 → −0.0086 1/6). Câu hỏi kế tiếp là đặc tính
đó có **chuyển được sang backbone khác** không. codebert có sẵn checkpoint Pha 1 λ=0.05 ở
kho `s42` (4cwe val 0.6532, com val 0.5598) nên chỉ tốn Pha 2. Kết quả ở cây riêng
`results/opt1_codebert` → kéo về `results_opt1cb_158/`.

## BẬC 2 đã xếp hàng — 07/09, fold 4 và 5

Người dùng duyệt 07/09: *"Cứ setup cho chạy n=5 5folds luôn cho nó queue luôn vào."*

| máy | fold mới | ô | chờ gì trước |
|---|---|---|---|
| **161** | **4** | 21 ô + 1 baseline | xong nguồn `full` fold 1–2 |
| **158** | **5** | 21 ô + 1 baseline | xong khối 60 epoch fold 3 |

`scripts/queue_bac2.sh`, biến `FOLD_NEW` chọn fold. Lock riêng `/tmp/mvd_queue_bac2.lock`,
watchdog giám sát và phóng lại tối đa 3 lần. Fold trọn vẹn trên một máy, baseline của
chính fold đó chạy cùng máy.

**Bảy cấu hình**, chạy hai lượt vì `min_epochs` phải khớp đúng lúc chạy bậc 1 — nếu không
thì fold 4–5 khác fold 1–3 và phép gộp không hợp lệ:

- lượt A, `min_epochs=3`: `c5000_t0p05` (đối chứng), `plain` (AdamW, đối chứng), `c50_t0p05`,
  `c5_t0p05`, `c0p5_t0p05`, `warm`
- lượt B, `min_epochs=10`: `c5000_t0p2k02_me10`

**Câu hỏi của bậc 2**: câu chuyện "neo nâng sàn" ở bậc 1 dựa vào **đúng một fold khó**
(fold 3, baseline 0.7036). Trong hai fold mới có fold nào khó không, và nếu có thì neo có
lại nâng sàn ở đúng chỗ đó không. Đây là điểm yếu chí mạng của kết luận hiện tại.

## Việc cho sáng 07/09 — xếp theo giá trị, dựa trên số đo VÀ tài liệu

1. **Đọc kết quả ME10** (`python3 tools/opt1_report.py results/opt1_t5p results_opt1_158`).
   Câu hỏi: bỏ trần early stopping ra thì neo bền có thắng neo ngắn không? Nếu có, đó là
   phát hiện chính của khối, và tài liệu đứng sau nó — RecAdam gốc chạy **50–100 epoch**
   trên task nhỏ, **không hề early-stop** (RESEARCH §8.1).

2. **Kiểm giả thuyết γ ∝ N — miễn phí, không tốn một ô GPU.** Dẫn xuất của chính RecAdam
   cho `γ = N·F̄` với N là số quan sát hậu thuẫn điểm neo. Neo của ta là checkpoint Pha 1
   trên 930 / 3 744 / 7 598 dòng, nên γ tối ưu phải **tăng** theo nguồn. Ở n=1/ô tối qua
   thứ tự ra 5 / 0.5 / 50 (thuần nhiễu). Chạy lại khi đủ 3 fold. Nếu đúng, đây là liên hệ
   **lý thuyết → thực nghiệm** mạnh nhất của cả hướng (RESEARCH §8.2).

3. **Fisher: chạy CỔNG KIỂM trước, đừng chạy thí nghiệm trước.**
   ```bash
   python src/fisher.py --source_checkpoint model/n48/phase1/t5p__latent_bottleneck_4cwe_l0p05/seed_42/best.pt \
       --data_path data/phase1_4cwe.jsonl --cwe_vocab fixed4 --max_batches 64 \
       --model_name Salesforce/codet5p-220m-bimodal --pooling mean --aux_mode latent_bottleneck
   ```
   `assess_fisher()` báo **SUY BIEN** nếu >90 % tham số có F < 0.01 — khi đó `γ·F ≈ 0` cho
   gần hết mạng và khối sẽ chỉ đo lại AdamW. Rủi ro này là thật: tại checkpoint đã hội tụ,
   Fisher tiêu biến. **Và L2-SP-Fisher (ICML 2018) đã cho kết quả null trên đúng dạng phạt
   này**, với lý do áp dụng nguyên vẹn cho ta (RESEARCH §8.4). Chỉ chạy tiếp nếu cổng xanh.

4. **Nếu ME10 dương: thử đúng chế độ của bài gốc** — `PHASE2_EPOCHS=60 PATIENCE=10`.
   Bài gốc chạy 100 epoch trên RTE/MRPC không early-stop; ta chạy tối đa 30 với patience 5.
   Đây là biến chưa từng chạm và có lý do tài liệu rõ ràng. Tốn ~2× thời gian mỗi ô.

5. Cấu hình nào sống sót bậc 1 thì **lên bậc 2** (5 fold, seed 42) — `CLAUDE.md` mục 1.

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

## Hàng đợi đêm 06→07/09 — chạy không người trông

Người dùng nghỉ từ ~18:30 UTC 06/09 đến ~02:00 UTC 07/09 và **không trả lời được câu hỏi**,
đã trao toàn quyền quyết định. Hai hàng đợi tuần tự, mỗi cái giữ **lock riêng** để watchdog
biết nó còn sống:

| máy | lock hàng đợi | việc, đúng thứ tự |
|---|---|---|
| **161** (fold 1, 2) | `/tmp/mvd_queue161.lock` | OPT1 `4cwe`+`com` → OPT1 `full` → **ME10** fold 1,2 |
| **158** (fold 3) | `/tmp/mvd_queue158.lock` | OPT1 `full` → **ME10** fold 3 → **CODEBERT** trục γ, fold 1–3 |

`scripts/queue161.sh` và `scripts/queue158.sh`. Mỗi việc chờ lock driver nhả trước khi
chạy; 161 còn chờ VRAM trống ≥ 13 GB và **nhường** nếu người khác đang dùng GPU.
`scripts/watch_opt1.sh` (cron 10 phút) giám sát cả driver lẫn hàng đợi, phóng lại tối đa
3 lần, không bao giờ giết gì.

### Vì sao có khối ME10

Đo được từ `val_history` ngày 06/09: **early stopping đang cắt mọi nhánh neo bền trước khi
λ kịp lên**, nên câu hỏi "neo bền có tốt không" chưa hề được trả lời.

| λ ở epoch 5 | số epoch chạy | số ô sập (F1 < 0.6) |
|---|---|---|
| 0.000 | 7.0 | 12/12 |
| 0.359 | 7.0 | 1/1 |
| 0.190 | 18.2 | 1/4 |
| 0.994 | 13–18 | 0/24 |

Với `patience=5`, `min_epochs=3`, nhánh có λ lên chậm thì năm epoch đầu val không nhúc
nhích và bị dừng ở đúng epoch 7. ME10 nâng `min_epochs=10` cho **cả nhánh chính lẫn đối
chứng** (`c5000_t0p05_me10`) nên phép so vẫn đổi đúng một biến.

### Vì sao có khối CODEBERT

Trục γ trên t5p cho neo **yếu** thắng neo mặc định (ghép cặp cùng fold, so với AdamW thuần:
γ=5 → +0.0295 3/4; γ=0.5 → +0.0178 2/4; γ=5000 → −0.0086 1/6). Câu hỏi kế tiếp là đặc tính
đó có **chuyển được sang backbone khác** không. codebert có sẵn checkpoint Pha 1 λ=0.05 ở
kho `s42` (4cwe val 0.6532, com val 0.5598) nên chỉ tốn Pha 2. Kết quả ở cây riêng
`results/opt1_codebert` → kéo về `results_opt1cb_158/`.

## BẬC 2 đã xếp hàng — 07/09, fold 4 và 5

Người dùng duyệt 07/09: *"Cứ setup cho chạy n=5 5folds luôn cho nó queue luôn vào."*

| máy | fold mới | ô | chờ gì trước |
|---|---|---|---|
| **161** | **4** | 21 ô + 1 baseline | xong nguồn `full` fold 1–2 |
| **158** | **5** | 21 ô + 1 baseline | xong khối 60 epoch fold 3 |

`scripts/queue_bac2.sh`, biến `FOLD_NEW` chọn fold. Lock riêng `/tmp/mvd_queue_bac2.lock`,
watchdog giám sát và phóng lại tối đa 3 lần. Fold trọn vẹn trên một máy, baseline của
chính fold đó chạy cùng máy.

**Bảy cấu hình**, chạy hai lượt vì `min_epochs` phải khớp đúng lúc chạy bậc 1 — nếu không
thì fold 4–5 khác fold 1–3 và phép gộp không hợp lệ:

- lượt A, `min_epochs=3`: `c5000_t0p05` (đối chứng), `plain` (AdamW, đối chứng), `c50_t0p05`,
  `c5_t0p05`, `c0p5_t0p05`, `warm`
- lượt B, `min_epochs=10`: `c5000_t0p2k02_me10`

**Câu hỏi của bậc 2**: câu chuyện "neo nâng sàn" ở bậc 1 dựa vào **đúng một fold khó**
(fold 3, baseline 0.7036). Trong hai fold mới có fold nào khó không, và nếu có thì neo có
lại nâng sàn ở đúng chỗ đó không. Đây là điểm yếu chí mạng của kết luận hiện tại.

## Việc cho sáng 07/09 — xếp theo giá trị, dựa trên số đo VÀ tài liệu

1. **Đọc kết quả ME10** (`python3 tools/opt1_report.py results/opt1_t5p results_opt1_158`).
   Câu hỏi: bỏ trần early stopping ra thì neo bền có thắng neo ngắn không? Nếu có, đó là
   phát hiện chính của khối, và tài liệu đứng sau nó — RecAdam gốc chạy **50–100 epoch**
   trên task nhỏ, **không hề early-stop** (RESEARCH §8.1).

2. **Kiểm giả thuyết γ ∝ N — miễn phí, không tốn một ô GPU.** Dẫn xuất của chính RecAdam
   cho `γ = N·F̄` với N là số quan sát hậu thuẫn điểm neo. Neo của ta là checkpoint Pha 1
   trên 930 / 3 744 / 7 598 dòng, nên γ tối ưu phải **tăng** theo nguồn. Ở n=1/ô tối qua
   thứ tự ra 5 / 0.5 / 50 (thuần nhiễu). Chạy lại khi đủ 3 fold. Nếu đúng, đây là liên hệ
   **lý thuyết → thực nghiệm** mạnh nhất của cả hướng (RESEARCH §8.2).

3. **Fisher: chạy CỔNG KIỂM trước, đừng chạy thí nghiệm trước.**
   ```bash
   python src/fisher.py --source_checkpoint model/n48/phase1/t5p__latent_bottleneck_4cwe_l0p05/seed_42/best.pt \
       --data_path data/phase1_4cwe.jsonl --cwe_vocab fixed4 --max_batches 64 \
       --model_name Salesforce/codet5p-220m-bimodal --pooling mean --aux_mode latent_bottleneck
   ```
   `assess_fisher()` báo **SUY BIEN** nếu >90 % tham số có F < 0.01 — khi đó `γ·F ≈ 0` cho
   gần hết mạng và khối sẽ chỉ đo lại AdamW. Rủi ro này là thật: tại checkpoint đã hội tụ,
   Fisher tiêu biến. **Và L2-SP-Fisher (ICML 2018) đã cho kết quả null trên đúng dạng phạt
   này**, với lý do áp dụng nguyên vẹn cho ta (RESEARCH §8.4). Chỉ chạy tiếp nếu cổng xanh.

4. **Nếu ME10 dương: thử đúng chế độ của bài gốc** — `PHASE2_EPOCHS=60 PATIENCE=10`.
   Bài gốc chạy 100 epoch trên RTE/MRPC không early-stop; ta chạy tối đa 30 với patience 5.
   Đây là biến chưa từng chạm và có lý do tài liệu rõ ràng. Tốn ~2× thời gian mỗi ô.

5. Cấu hình nào sống sót bậc 1 thì **lên bậc 2** (5 fold, seed 42) — `CLAUDE.md` mục 1.

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
