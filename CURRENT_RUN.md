# CURRENT_RUN — SÁNG 15/09: vast `ntat` ĐANG TRỐNG, hết việc đã xếp

> **Đêm 14→15/09 chạy xong ba khối, tổng 30 ô Pha 2 + 12 Pha 1, không ô nào thiếu.**
> Máy `ntat` id **50965796** (RTX 5060 Ti, $0,0818/h) đã chạy **~17 giờ ≈ $1,40**, hiện **trống**.
> Mọi kết quả và checkpoint đã kéo về local, đối chiếu byte khớp tuyệt đối — **huỷ máy lúc nào
> cũng an toàn**. Chưa huỷ vì chờ quyết định: chạy hướng mới hay dừng.

## Đêm 14→15/09 đã trả lời xong ba câu

| khối | câu hỏi | kết quả |
|---|---|---|
| `p1seed` (6 seed codebert) | bỏ head thì Pha 1 có luôn sập không? | **không** — có head 6/6, không head **3/6**; kết cục **lưỡng cực** (**§55.3**) |
| `p1seed_t5p` (3 seed) | bất ổn đó có đặc thù backbone không? | **có** — t5p cả hai nhánh **3/3**, **trượt cổng 2** (**§55.4**) |
| `fusnone_t5p` (15 ô) | bỏ head mất gì khi Pha 1 lành? | **không mất gì** — `fusft − nonefus` = **+0.0053, 5/10** (**§56**) |

### Kết luận gộp — §56

Ghép cặp theo (backbone, fold), **10 điểm**, cả hai Pha 1 lành:

| | F1@0.5 | ROC-AUC | PR-AUC |
|---|---|---|---|
| `fusft` − `nonefus` | **+0.0053 5/10** | **−0.0017 3/10** | +0.0009 5/10 |
| `fusft` − baseline | **+0.0560 9/10 p=0.021** | **+0.0309 9/10 p=0.021** | **+0.0282 9/10 p=0.021** |
| `nonefus` − baseline | **+0.0507 9/10 p=0.021** | +0.0326 8/10 | +0.0273 8/10 |

> **Head phụ `latent_bottleneck` có thể BỎ HẲN.** Thứ tạo ra lợi ích là **Pha 1 + adapter +
> fusion** (~+0.05 F1, ~+0.03 ROC, 9/10 fold, p=0.021, **cả hai backbone**). Rủi ro duy nhất khi
> bỏ là Pha 1 sập — và rủi ro đó **chỉ có ở codebert**.

## Sự cố đêm và cách xử lý — để lần sau khỏi mắc lại

1. **OOM trên t5p**, thiếu đúng **24 MiB**, y hệt §47/§48. Đã **dừng khối ngay** thay vì để chạy
   tiếp, vì nhánh còn lại đang chạy **không** có `--grad_checkpointing` ⇒ hai nhánh sẽ khác nhiều
   hơn một biến. Bật cờ cho **cả hai**, xoá cây, chạy lại từ đầu.
2. **Nối chuỗi bằng file PID của khối KHÁC** (`shuf1.pid` trong khi khối đang chạy là `shuf2`)
   ⇒ cổng kết luận "đã xong" ⇒ **hai chuỗi một GPU**. Cổng nối chuỗi giờ **hai lớp**: argv khớp
   chính xác **và** lock đang bị giữ. Mỗi driver tự ghi PID của chính nó.
3. **`awk '/train_transfer.py/ && /fusnone_t5p/'` trên `ps`** khớp luôn argv của shell từ xa
   ⇒ ssh tự giết mình. Cách chữa: viết script ra file rồi đẩy lên chạy; lọc theo `comm=python`
   kèm **loại trừ chính tiến trình và toàn bộ tổ tiên**.

## Việc còn để ngỏ

- `NEXT_CONTRIBUTION.md` mục 4: **Đề xuất B** (mất mát biên trong cặp) — chưa bị bác, chưa chạy.
- Giả thuyết **chống nén sụp chiều** (§53) — chưa chạy.
- `twin` như **side result** — chính reviewer đề xuất, xếp sau.

---

## Khối `shuf1` — lợi ích Pha 1 là TRI THỨC hay PHƠI NHIỄM?

Ba nhánh Pha 1, đều `aux_mode=none`, nguồn `com`, **cùng seed / cùng split / cùng 12 epoch**
(`--save_last_epoch`, tắt dừng sớm) — chỉ khác cột `label`:

| nhánh | dữ liệu | giữ | phá |
|---|---|---|---|
| `real` | `phase1_common.jsonl` | — | — |
| `shufall` | `phase1_common_shufall.jsonl` | tỉ lệ 50/50 | liên hệ code↔nhãn **và** cân bằng trong cặp (0.509) |
| `shufpair` | `phase1_common_shufpair.jsonl` | tỉ lệ 50/50 **và** cân bằng cặp (1.000) | **chỉ**: bản nào trước khi vá |

Pha 2 `adamw`, `--sam_rho 0`, fold 1–3, seed 42, cộng baseline. **Cả hai backbone**
(codebert + t5p). Cây: `results/shuf1_codebert`, `results/shuf1_t5p`.

Đã kiểm trước khi phóng: hai bản xáo audit **hai chiều** (bản gốc so chính nó cho 1.000);
`--save_last_epoch` thử **hai chiều** (có cờ → 5 epoch, `best_epoch=5`, 0 dừng sớm; không cờ →
dừng ở epoch 2, `best_epoch=1`); t5p nạp được offline qua **đúng `src/model.py:build_backbone`**;
mọi file đối chiếu từng byte; runner chạy khô đếm đúng **18 lần gọi `matrix.sh` + 6 baseline**.

---

## ĐÃ XONG 14/09 03:50 UTC — khối `fus5060`, 15/15 ô (**FACTS §52**)

Kết quả đã kéo về `results_fus5060_ntat/`, đối chiếu 15/15 file khớp tuyệt đối.

| so sánh (codebert, `com`, n=5) | F1@0.5 | ROC-AUC |
|---|---|---|
| `fusft` − baseline | **+0.0637 5/5** | **+0.0445 5/5** |
| chốt − baseline | +0.0240 4/5 | +0.0150 4/5 |
| `fusft` − chốt | **+0.0397 5/5** | **+0.0294 5/5** |

Không hồi sinh AdapterFusion — xem cảnh báo ở §52.

### Chi tiết cấu hình khối `fus5060`

## Khối `fus5060` — baseline vs chốt vs fusft, 15 ô, CÙNG MỘT MÁY

| | |
|---|---|
| bậc | **2 — xác nhận** (5 fold, seed 42) |
| backbone | codebert (`microsoft/codebert-base`, pooling `cls`) |
| nguồn Pha 1 | `com` (`data/phase1_common.jsonl`), `cwe_vocab=precomputed`, λ=0.05, SAM tắt |
| đích | `data/sven_python_folds_norm`, python, fold 1–5 |
| optimizer Pha 2 | **chỉ `adamw`** (fusion từ chối `recadam`/`spd`) |
| cây kết quả | `results/fus5060_codebert` — **KHÔNG** trùng tên cây local `results/fus3_codebert` |

Ba nhánh, mỗi fold chạy đủ ba rồi mới sang fold sau (mục 1):

1. `baseline` — không Pha 1
2. `latent_bottleneck` **cấu hình chốt** — Pha 1 + head phụ, KHÔNG adapter, Pha 1 dùng lại
   `model/n48/phase1/codebert__latent_bottleneck_com_l0p05`
3. `fusft` — adapter nguồn + adapter đích + fusion, fine-tune cả backbone, Pha 1 dùng lại
   `model/fus2/phase1/codebert__latent_bottleneck_com_l0p05_ad48`

**Vì sao chạy lại cả ba.** Hai cây `fus1`/`fus2` cũ nằm trên các máy đã huỷ, và baseline ở máy
khác **không ghép cặp được** với `fusft` (mục 4; sàn nhiễu giữa loại GPU 0.028 > hiệu ứng 0.0226).
Đây là con số "fusion so với baseline, đo cùng một máy" còn thiếu — nó **không đổi** kết luận
§47/§48 đã có, chỉ bịt lỗ hổng đo lường.

**Máy local A4000 vẫn đang chờ** với `run/fusion_base.sh` (PID 3746584, cây `results/fus3_codebert`,
cổng: VRAM ≥ 11500 MiB ổn định 3 lần liên tiếp VÀ không tiến trình của user khác). Nó đã chờ
9h43 chưa chạy được ô nào vì `cuongtm` chiếm GPU. Hai cây tên khác nhau nên **không có nguy cơ
trộn máy**; nếu local chạy được thì đó là một bộ độc lập, cộng thêm chứ không gộp.

### Môi trường trên `ntat` — đã đối chiếu

| | local (vdenv) | ntat (`/venv/main`) |
|---|---|---|
| transformers | 4.57.1 | **4.57.1** ✓ (cài theo `requirements-pin.txt`) |
| torch | 2.9.1+cu128 | 2.11.0+cu128 (lệch sẵn giữa máy, ghi nhận) |
| sklearn | 1.7.2 | 1.9.1 (chỉ dùng cho metric, không vào đường huấn luyện) |
| numpy | 2.3.4 | 2.5.3 |

GPU compute capability **(12, 0)** — Blackwell sm_120; torch 2.11+cu128 chạy được, đã thử.

Thử khói **cả hai chiều** trước khi phóng: có cache ⇒ dựng được tokenizer+backbone trên GPU;
`HF_HOME` rỗng ⇒ **hỏng** đúng như phải thế (cổng offline có tác dụng thật). Rồi chạy một ô
`fusft` thật rút gọn (32 mẫu train): mã thoát 0, checkpoint có **48 khoá adapter nguồn + 48 khoá
adapter đích + 60 khoá fusion** — đúng bằng số đo được trên máy vast trước.

Đối chiếu **từng byte** sau khi đẩy: hai checkpoint Pha 1 (498 700 197 và 502 302 277 B),
`data/phase1_common.jsonl` (8 458 506 B), `run/matrix.sh`, `src/*.py` — khớp tuyệt đối.

---

## Đã xong trước đó — 13/09/2026, cả hai vast ĐÃ HUỶ

> `ntat` 50857599 huỷ ~10:55 UTC, `ntat` 50882617 huỷ ~15:15 UTC. Dữ liệu đối chiếu **từng byte,
> cả hai chiều** trước khi huỷ; lần thứ hai kéo **cả checkpoint Pha 1** về (bài học từ lần đầu).
> Hai máy ~7 giờ tổng ≈ $0,55.

## AdapterFusion (arXiv:2005.00247) — ba khối, kết luận: **KHÔNG đáng đi tiếp**

| khối | ô | kết quả |
|---|---|---|
| `fus1` | 26 | **§47** — `fusft` chắc trên codebert (5/5 cả F1 lẫn ROC), **không lặp trên t5p** |
| `fus2` | 15 | **§48** — **một nửa** lợi ích là **sức chứa**; §47 không lặp trọn trên máy hai |
| `fus2w` | 10 | **§48.1/§48.2** — trọng số fusion: chỉ **hình dạng theo lớp** khác, **độ lớn ngược chiều trực giác** |

### Năm lý do cộng dồn để dừng (chi tiết cuối §48)

1. `fusft` không lặp trên t5p ở n=5 (+0.0036, 3/5).
2. Adapter **ngẫu nhiên** tái tạo ~một nửa lợi ích (+0.0111 / +0.0226, cùng 4/5 fold).
3. Phần "tri thức Pha 1" còn lại: +0.0115 **3/5 fold**, **âm ở PR-AUC** — không tách khỏi nhiễu.
4. ROC-AUC của §47 không lặp trên máy thứ hai (+0.0224 5/5 → +0.0113 3/5).
5. Pha 1 `codebert × com` + adapter **nằm ngay ranh giới**: cùng seed, cùng mã, cùng thư viện,
   huấn luyện lại thì **sập** (val 0.3333). Phi tất định GPU đủ để quyết định nó học hay không.

### Phát hiện phương pháp luận đáng giữ (§48.2)

Cùng fold 1, hai checkpoint huấn luyện **độc lập** cùng seed cho độ tản 0.0379 và 0.1209 —
**biến thiên giữa hai lần chạy lớn hơn hiệu ứng cần đo trên một fold**. Trong một ngày, fold 1
đã **ba lần** vẽ ra bức tranh sạch hơn thực tế ở khối này.

## Việc còn để ngỏ — đều là thí nghiệm MỚI, cần duyệt

| việc | ghi chú |
|---|---|
| vì sao codebert ăn mà t5p không | câu hỏi đáng giá nhất còn lại của hướng này |
| Pha 1 nhiều seed cho cấu hình bấp bênh | hệ quả trực tiếp của §48; ảnh hưởng cả §47 |
| tập `twin` | treo từ 11/09 (mục 6: phụ, chỉ khi được yêu cầu) |
| đối chứng `none` ở cấu hình chốt | đặc tả sẵn ở cuối **FACTS §46**, 42 ô |

---

<!-- ===== LỊCH SỬ ===== -->

# CURRENT_RUN — KHÔNG CÓ GÌ ĐANG CHẠY (cập nhật 12/09/2026)

> **12/09 — không phóng khối nào.** Người dùng hỏi đối chứng "finetune hai lần thuần";
> trả lời được **hoàn toàn bằng dữ liệu đã có, 0 GPU** → **FACTS §46**, công cụ
> `tools/head_vs_none.py`. Cả 161 lẫn 158 đều đang chạy việc của người dùng khác
> (`cuongtm` / `tranmanhcuong`) nên **nhường**, và vast không thuê được.

## XẾP HÀNG — chưa chạy, chờ GPU rảnh: đối chứng `none` ở CẤU HÌNH CHỐT

Lỗ hổng FACTS §46 nêu: ở cấu hình chốt (λ=0.05, SAM tắt cả hai pha) **chưa từng có nhánh
`none` nào** chạy cùng cây cùng phiên với head. Đặc tả đầy đủ ở cuối §46.

| | |
|---|---|
| nhánh | `baseline`, `none`, `latent_bottleneck` — **cùng một cây** |
| Pha 1 | **dùng lại, không huấn luyện lại** (mục 5: `none` dùng được ở mọi λ) |
| Pha 2 | AdamW, `--sam_rho 0`, λ=0.05 |
| nguồn | `4cwe`, `com`, `full` · đích `data/sven_python_folds_norm` |
| quy mô | **bậc 1: 3 fold, seed 42** — 42 ô, codebert ở 161, t5p ở 158 |

Chạy khi GPU rảnh **và người dùng duyệt** — mục 9: không tự thêm thí nghiệm vào hàng đợi.

---

<!-- ===== LỊCH SỬ ===== -->

# CURRENT_RUN — ĐÃ XONG HẾT 11/09/2026 15:43 UTC

> **Không còn gì đang chạy.** Cả ba máy đã xong; vast đã huỷ lúc 13:22.

| máy | trạng thái |
|---|---|
| **161** codebert | xong `SEED15`, **100 ô**, 0 job hỏng |
| **158** t5p | xong `SEED15`, **98 ô**, 0 job hỏng |
| **vast 50570168** | **ĐÃ HUỶ** 13:22 UTC sau khi đối chiếu byte; chạy ~5 giờ ~$0.41 |

## Kết quả chính — BẬC 3 (n = 15 = 5 fold × 3 seed), 120 ô ghép cặp

ROC-AUC, Δ ghép cặp trong cùng ô. **24/24 ô đủ 5 fold.**

| N | codebert | t5p |
|---|---|---|
| 456 (đầy đủ) | +0.0127 **8/15** p=1.000 | +0.0094 **8/15** p=1.000 |
| 228 | +0.0354 15/15 | +0.0504 13/15 |
| 152 | +0.0922 13/15 | +0.1069 12/14 |
| 76 | **+0.1849 14/14** p=0.000 | **+0.1188 14/14** p=0.000 |

Đơn điệu chặt cả bốn mức trên cả hai backbone. Ở dữ liệu đầy đủ hiệu ứng thứ hạng **bằng không**
(PR-AUC codebert +0.0002, 5/15) trong khi F1@0.5 là 15/15 p=0.000 — tức chỉ dời **ngưỡng**,
không đổi **thứ hạng**. Chi tiết FACTS §40.5, bản đọc `RESEARCH_2026-09-10_dactrung.md` Phần 9.

3 ô bị loại đúng theo ngưỡng khai báo trước (`baseline F1@0.5 < 0.40`), đều là baseline sập ở N nhỏ.

## Các khối khác đã xong hôm nay

| khối | kết quả |
|---|---|
| đảo chiều Python→JS, n=3 | **12/12 ô dương** cả bốn chỉ số cả hai backbone; ASAM hơn `plain` trên cả hai chỉ số thứ hạng (FACTS §41.3) |
| `auxb` + đối chứng, 18 ô | mạch head phụ **ĐÓNG** (FACTS §43, §44, §45) |

## Việc còn để ngỏ, CHỜ ANH QUYẾT

| việc | vì sao chưa làm |
|---|---|
| tập **`twin`** (chống phản biện rò rỉ gần-trùng-lặp) | CLAUDE.md mục 6: phụ, **chỉ chạy khi được yêu cầu**. Đây là câu hỏi đáng giá nhất còn lại vì nó tấn công thẳng §40.5 |
| đảo chiều lên **n=5** | `data/js_4cwe_folds` chỉ có 3 fold — phải dựng thêm fold 4–5 |
| đảo chiều `com`/`full` lên n=3 | không tái hiện được fold 1 từng byte ⇒ fold mới sẽ hỏng phép ghép cặp ngầm |

---

<!-- ===== LỊCH SỬ ===== -->

# CURRENT_RUN — ĐÃ XONG 11/09/2026 03:39 VN (20:39 UTC 10/09)

> **Không còn gì đang chạy.** Đêm 10→11/09 chạy xong `bridge3` + `feat3` + `lpft3` = **66 ô GPU**
> (33 mỗi cây, 9 nhánh × 2 backbone × 3 fold) và **6 ô probe** (2 backbone × 3 nguồn, 0 GPU).
> **Không thuê vast, 0 chi phí.** Cron watchdog trong phiên Claude vẫn chạy 20 phút/lần.
>
> **Kết quả:** FACTS §36.1 (control task), §38 (lp3 mạnh nhất trên codebert), §38.1 (vì sao neo
> đặc trưng hại t5p), §38.2 (mọi nhánh thắng ở backbone này đều thua ở kia), §38.3 (phép kiểm khai
> báo trước — probe tách theo BACKBONE không theo NGUỒN, §38.2 yếu đi).
> Bản đọc cho người: `RESEARCH_2026-09-10_dactrung.md`.
>
> **KHÔNG nhánh nào đủ điều kiện leo n=5** (luật: dương cả bốn chỉ số VÀ lặp trên cả hai backbone).
> Đề xuất bước tiếp ghi ở FACTS §38.3, **chưa chạy, chờ duyệt**.

---

## (đã xong) `feat3`: neo KHÔNG GIAN ĐẶC TRƯNG + khởi tạo lại head (17:01 → 20:39 UTC)

Suy trực tiếp từ **FACTS §36** (đặc trưng chuyển giao +0.1125 ROC 5/5 trên codebert; hàm quyết
định thì không, 0.537 F1 zero-shot). Ghi vào **cùng cây** `results/bridge3_<bb>` để dùng lại
`plain` + `baseline` cùng máy cùng fold cùng ngày.

| nhánh | cờ Pha 2 (thêm vào `adamw --sam_rho 0`) |
|---|---|
| `fd1` | `--feat_distill_beta 1.0` |
| `fd10` | `--feat_distill_beta 10.0` |
| `rh` | `--phase2_reinit_head` |
| `fd10rh` | cả hai |

4 nhánh × 3 fold × 2 backbone = 24 ô. codebert trên 161, t5p trên 158. Log `log/feat3_<bb>.log`.

## XẾP HÀNG — `lpft3`: nhánh đối chứng LP-FT (tự phóng sau `feat3`)

`scripts/chain_lpft.sh` chờ driver `feat3` trên **cùng máy** thoát hẳn (trần 4 giờ) rồi phóng
`run/lpft3.sh`. Hai nhánh: `lp3` (`--lp_epochs 3`, giữ head Pha 1) và `rhlp3` (`--lp_epochs 3
--phase2_reinit_head`, LP-FT sách giáo khoa). 2 × 3 × 2 = 12 ô.

**Vì sao bắt buộc có**: Kumar et al. ICLR 2022 kê đơn NGƯỢC với nhánh `rh` — xem
`RESEARCH_2026-09-10_dactrung.md` §5.2a.

## ĐANG CHẠY CPU (0 GPU) — control task cho probe

`tools/feature_probe.py --control --cache` trên cả hai backbone: nhãn xáo trộn cố định, in **độ
chọn lọc**. Chặn phản biện "probe tự học tác vụ". Ghi `results/probe/*_ctl.json`.

## Luật vast đêm nay (người dùng xác nhận lại 11/09)

Local bị chiếm mà còn việc đáng chạy ⇒ thuê **đúng một** vast, trần **$0.080/h**, tìm offer tại
thời điểm cần (`scripts/rent_one_vast.sh`). Tối đa 3 GPU. Local bị chiếm thì **nhượng**, không
bao giờ kill user khác. Cron tự invoke 20 phút/lần đang chạy trong phiên Claude.

**Trạng thái 00:35 VN**: 161 và 158 đều đang chạy `feat3`, **không có vast nào** (0 chi phí).
Lúc 00:16–00:26 VN GPU của 158 bị chiếm bởi `pv_plain.py` của chính chủ tài khoản; cổng
`wait_vram` đã tự chờ và tự chạy tiếp lúc 00:25:50 — **không cần thuê vast**.

---

## ĐÃ XONG 10/09 16:50 UTC — khối `bridge3` (cầu CWE, bậc 1)

> Khối trước (n=15 `chot`, 210 ô) **đã xong 10/09 17:10 VN**, FACTS §35.2. Mục cũ giữ ở dưới để tra cứu.

## ĐANG CHẠY — `bridge3`: đưa dữ liệu NGUỒN vào Pha 2 qua cầu CWE (kiểm chứng, **n=3 fold**, seed 42)

Người dùng 10/09 tối: *"thay vì áp dụng chuẩn theo cái đã có bạn có thể tự do sáng tạo 1 cái vì hiện chỉ
cần chứng minh thêm 1 cái để transfer... Khi tìm ra có thể thử ngay với n=3 trước nếu rảnh. cần vast báo
tôi hoặc hỏi lại."* — chạy trên hai máy local, **không thuê vast**.

**Vì sao hướng này** (RESEARCH_2026-09-06 §11–12, FACTS §35): bốn khối OPT1/RET1/SPD1/INT1 cho thấy tri
thức nguồn vào đích CHỈ qua điểm khởi tạo; mọi neo trọng số (RecAdam/SPD/Fisher/WiSE-FT/LP-FT/LoRA) đều
null vì neo là ràng buộc, không phải kênh truyền. Đường còn lại: cho **gradient của dữ liệu nguồn** nặn
trực tiếp nghiệm đích (Đ5), và dùng **CWE làm cầu**: head phụ 4 lớp của Pha 1 học tiếp trên CẢ HAI ngôn
ngữ (đích Python và nguồn 4cwe dùng cùng bảng `CWE_MAPPING` 022/078/079/089).

Thiết kế **2×2**, cùng fold, cùng máy, cùng phiên với đối chứng; `baseline` (không Pha 1) cùng fold:

| tag | cờ Pha 2 (`--phase2_optimizer adamw --sam_rho 0` + …) | đo cái gì |
|---|---|---|
| `plain` | — | **đối chứng**: fine-tune hai lần thuần |
| `cwe05` | `--phase2_lambda_cwe 0.05` | chỉ nửa "đích" của cầu: head CWE học tiếp trên nhãn CWE Python |
| `rp50` | `--replay_data data/phase1_4cwe.jsonl --replay_mu 0.5 --replay_epochs 6 --replay_stratify` | chỉ replay nguồn: μ(e)=0.5→0 tuyến tính sau 6 epoch, cân tầng (label, CWE) |
| `rpc` | cả hai + `--replay_lambda_cwe 0.05` | **cầu CWE đầy đủ**: replay + head CWE học trên nguồn VÀ đích |

| máy | backbone | Pha 1 (dùng lại, không huấn luyện lại) | cây kết quả | mốc |
|---|---|---|---|---|
| **161** A4000 (dùng chung — `cuongtm` đang chạy 4,9 GB cùng lúc) | codebert | `model/n48/phase1/codebert__latent_bottleneck_4cwe_l0p05/seed_42` val 0.6532 | `results/bridge3_codebert` | 12 ô + 3 baseline |
| **158** A4000 | t5p | `…/t5p__latent_bottleneck_4cwe_l0p05/seed_42` val 0.6976 | `results/bridge3_t5p` | 12 ô + 3 baseline |

Runner `run/bridge3.sh` → `run/opt1.sh` → `run/matrix.sh`; log `log/bridge3_<bb>.log` trên từng máy.
Mã mới: `src/replay.py`, cờ `--replay_*`/`--phase2_lambda_cwe` trong `train_transfer.py`, vòng Pha 2
trong `train.py` (hai backward nối tiếp để không tràn VRAM; SAM tính lại đúng mục tiêu). Kiểm:
`tests/test_replay.py` (4 phép, hai chiều) + smoke GPU cả 4 nhánh optimizer. Mặc định mọi cờ = tắt ⇒
đường cũ không đổi một byte.

**Đọc** (LUÔN cả bốn chỉ số, ghép cặp theo fold):
`python3 tools/report2.py --a transfer_latent_bottleneck_4cwe_l0p05_rpc_adamw --b transfer_latent_bottleneck_4cwe_l0p05_plain_adamw results/bridge3_codebert`
(thay `rpc` bằng `rp50`/`cwe05`; thay cây cho t5p). Bậc 1 ⇒ chỉ được **sàng lọc**, không kết luận.

Giám sát: Monitor trong phiên Claude (5 phút) — không cron. ETA: 158 ~2,5 h; 161 chậm hơn vì dùng chung GPU.

---

## ĐÃ XONG 10/09/2026 17:10 VN (10:10 UTC) — khối n=15 `chot`

> **Khối n=15 hoàn tất: 210 ô.** Cron watchdog đã gỡ.
> Kết quả ở FACTS §35.2; trang: <https://claude.ai/code/artifact/1ced3c61-bf8a-48b1-ab17-6ad575a0123b>

---

## ĐANG CHẠY — bổ sung n=15 cho khối `chot` (seed 7 + 1234)

Người dùng 10/09: *"chạy thêm 2 seed nữa cho đủ n=15 cho cả 3 source và setting... chạy thêm
n=10 là được, cái chạy rồi không cần chạy lại"* và *"nhớ chạy đúng khối đấy gồm cả có và không
recadam + asam theo đúng h-param tối ưu cho 2 backbone"*.

| máy | backbone | ρ | seed | cây kết quả | mốc |
|---|---|---|---|---|---|
| **161** A4000 | codebert | **0.1** | 7, 1234 | `results/chot161_codebert` | 70 ô |
| **158** A4000 | t5p | **2.0** | 7, 1234 | `results/chot158_t5p` | 70 ô |

Mỗi máy 70 ô = 2 seed × 5 fold × (1 baseline + 3 nguồn × 2 cấu hình).
**A** = `recadam` + ASAM ở ρ của chính backbone · **B** = `adamw`, `--sam_rho 0`.
Seed 42 (70 ô, FACTS §34) **giữ nguyên, không chạy lại**.

**Pha 1 huấn luyện lại cho từng seed**, không dùng lại bản seed 42. Phát biểu của khối là "lợi
ích đến từ Pha 1 + head"; giữ nguyên một Pha 1 mà chỉ đổi seed Pha 2 thì không phân biệt được
tính chất của **phương pháp** với tính chất của **đúng một checkpoint**. Seed còn quyết định cách
chia train/val của Pha 1 (`train_transfer.py:334`).

Chạy bằng `run/chot15.sh`. Watchdog cron 10 phút: `scripts/watch_chot15.sh`.

### Pha 1: cái nào đã có, cái nào phải chạy (quét bằng NỘI DUNG, 00:55 VN)

| | seed 7 | seed 1234 |
|---|---|---|
| **codebert** (161) | `4cwe` ✓ (train 00:13) — còn `com`, `full` | **chưa có gì**, cần cả 3 |
| **t5p** (158) | `4cwe` ✓ (train 00:12) — còn `com`, `full` | **đủ cả 3, có sẵn 05/09** ✓ |

t5p seed 1234 đã đọc nội dung xác nhận dùng lại được: `num_cwes` 4/10/10 đúng theo nguồn,
`cwe_vocab` `fixed4`/`precomputed`, λ=0.05, `training_args.seed=1234`. Tiết kiệm ~3 giờ cho 158.

> Chênh 192 byte giữa các file `4cwe` ở các seed **không phải** khác kiến trúc — chỉ là độ dài
> chuỗi trong `training_args`. `num_cwes=4` ở cả ba. Đã kiểm vì §29/§30 dạy không tin kích thước.

### CÂN LẠI KHI 161 XONG (việc phải làm, đừng quên)

Ước tính: 161 ~9 giờ (xong ~10:00 VN), 158 ~16,6 giờ (~17:30 VN). Tổng 26 giờ / 2 máy ⇒ nếu cân
thì cả khối xong ~**13:30 VN**.

Khi 161 đủ 70 ô codebert: nhìn xem 158 đã làm tới đâu, lấy **những fold cao nhất của seed 1234
chưa chạy** (khoảng 2 fold ≈ 14 ô ≈ 3 giờ) giao cho 161, và **khởi động lại 158 với danh sách
fold đã trừ đi phần đó** — nếu không hai máy làm trùng, và vì hai cây kết quả khác nhau nên
không bên nào bỏ qua bên nào. Vẫn chia theo **fold trọn vẹn**.

### Luật đêm 10/09 về việc thuê vast

| tình huống | xử lý |
|---|---|
| 158 mất kết nối / tự khởi động lại / reboot | **CHỜ**, hạn **20 phút**. Lên lại thì chạy tiếp. **Không thuê vast.** |
| Bị tranh mất GPU lúc chuyển pha/fold | `wait_vram` tự chờ. Quá **20 phút** không có tiến trình huấn luyện mà VRAM < 13 GB ⇒ coi là **VỠ**. |
| Một máy VỠ | được thuê **đúng MỘT** vast A4000 |
| **Cả hai** máy VỠ | vẫn chỉ **MỘT** vast, không hơn |
| Không máy nào vỡ | **không thuê** |

Vast phải canh kỹ — dùng bỏ phí là bị phạt. Watchdog **không tự thuê**: nó chỉ dựng cờ `VO` trong
`log/chot15_state`, monitor báo về để quyết bằng tay (phải kiểm giá và cài đặt).

Giá A4000 đã tra 00:15 VN: **$0.068–0.092/h**. Không có offer nào gần $0.008 — hiểu ngưỡng người
dùng nêu là **$0.08**. Đã chọn sẵn `id 39530408`, **$0.0756/h**, tin cậy **0.999**, 63 G RAM,
610 G đĩa, 6151 Mb/s — đắt hơn offer rẻ nhất $0.0075/h nhưng tin cậy 0.999 vs 0.965 và mạng nhanh
gấp 7.

### Khi xong

1. Kéo `results/chot158_t5p` từ 158 về, đối chiếu **số file + byte**.
2. Đọc: `python3 tools/chot_report.py results/chot_t5p results/chotv_t5p results/chot_codebert
   results/chot161_codebert results/chot158_t5p`
3. **Cập nhật artifact khối `chot`** (người dùng yêu cầu):
   <https://claude.ai/code/artifact/1ced3c61-bf8a-48b1-ab17-6ad575a0123b>
   Dựng lại dữ liệu bằng `tools/chot_export.py` rồi xuất bản lại **cùng đường dẫn file** để giữ URL.

---

## Đã xong trước đó

- **Khối `chot` seed 42** — 70 ô, FACTS §34/§34.1. Vast `ntat` id 50132360 đã huỷ 15:29 UTC 09/09.
- **§30.2** — đã đo head phụ: gần như không học được gì ở mọi nguồn và cả hai backbone.
