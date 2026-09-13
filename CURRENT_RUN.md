# CURRENT_RUN — 13/09/2026, nhánh git `fusion`. Máy vast `ntat` 50882617 CÒN SỐNG

> **Đang chạy**: hai ô gỡ lỗi để đo **trọng số lớp fusion** (`tools/fusion_weights.py`).
> Xong là hết việc đáng chạy ⇒ **huỷ máy**.

## Đã xong hôm nay — hai khối, hai kết luận

| khối | máy | kết quả |
|---|---|---|
| `fus1` (26 ô) | 50857599, **đã huỷ** | **FACTS §47** — `fusft` chắc trên codebert (5/5 cả F1 lẫn ROC) nhưng **không lặp trên t5p** ở n=5 |
| `fus2` (15 ô) | 50882617 | **FACTS §48** — **một nửa** lợi ích là **sức chứa**; §47 không lặp trọn vẹn trên máy thứ hai |

### Đọc nhanh §48 (codebert, n=5, cùng máy cùng phiên)

| nhánh | F1@0.5 | ROC-AUC | PR-AUC |
|---|---|---|---|
| `fusft` (adapter **đã học**) | +0.0226 4/5 | +0.0113 3/5 | +0.0056 4/5 |
| `fusftrnd` (adapter **ngẫu nhiên**) | +0.0111 4/5 | +0.0069 3/5 | +0.0117 2/5 |
| hiệu trực tiếp | +0.0115 **3/5** | +0.0044 3/5 | **−0.0061** 3/5 |

Phán quyết theo luật chốt trước: **KHÔNG KẾT LUẬN** (rơi vùng giữa đã khai báo).
Nhưng đọc được: phần "tri thức Pha 1" **không tách được khỏi nhiễu** ở n=5.

### Khuyến nghị

**Không lên n=15.** Năm lý do cộng dồn ghi ở cuối §48.

## Việc còn để ngỏ

| việc | ghi chú |
|---|---|
| trọng số fusion | đang đo, sẽ bổ sung vào §48 |
| vì sao codebert ăn mà t5p không | thí nghiệm mới, cần duyệt |
| Pha 1 `codebert × com` bấp bênh | phát hiện phụ của §48; nếu còn dùng cấu hình này thì nên chạy nhiều seed Pha 1 |
| tập `twin` | treo từ 11/09 |

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
