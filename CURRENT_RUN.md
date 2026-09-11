# CURRENT_RUN — ĐANG CHẠY 11/09/2026 (cập nhật 09:55 UTC)

## Đang chạy: leo §40 lên **bậc 3 (n = 5 fold × 3 seed = 15)**

| máy | việc | trạng thái |
|---|---|---|
| **161** codebert | seed 7 (N=228,152) rồi seed 1234 (cả bốn N) | **ĐANG CHỜ VRAM** — `cuongtm` chiếm ~8 GB, NHƯỜNG đúng luật |
| **158** t5p | seed 7 fold 1–3, rồi seed 1234 cả 5 fold (tự huấn luyện Pha 1) | đang chạy `sz456` fold 1 |
| **vast 50570168** | `auxb` Pha 2 (8/24) rồi t5p seed 7 fold 4–5 | đang chạy, chuỗi nối đã đặt |

**Vì sao khối này**: luật leo bậc đòi dương trên **cả bốn** chỉ số VÀ lặp trên **cả hai** backbone.
Tính đến 11/09, **chỉ §40 qua được**. Và seed 7 đã lặp lại trên codebert: ROC Δ +0.0139 (N=456) →
**+0.1777 (N=76), 5/5 fold**. Mọi can thiệp cơ chế khác đều tách theo backbone.

**Nhuỵ phải nêu khi đọc** (thấy ở cả seed 42 lẫn seed 7): ở N=456 PR-AUC của codebert **âm**
(−0.0087 và −0.0081). "Dương trên cả bốn chỉ số" tự nó cũng là hiện tượng dữ liệu-ít.

## MẢNH 2 (cổng 8 chiều) — **DỪNG**, đã bác trên cả hai backbone

`records/prediction_2026-09-11_nut_that_8_chieu.md` + FACTS §44. Nút thắt 8 chiều **thua PCA ở cả
bốn checkpoint**; trên t5p nó ở mức ngẫu nhiên (0.5210, dưới ngưỡng bác thẳng 0.55). Và làm cho
head phụ học được (t5p macro-F1 0.2000 → 0.5996) **không** làm nút thắt hữu ích hơn (0.5057 →
0.5210). `run/gate3.sh` **không phóng**. Mã `--phase2_gate` + 12 phép kiểm giữ lại, mặc định TẮT.

## Ô TRỐNG cần lấp

`results/rev1full_codebert/.../r0p1/fold1` — **OOM** lúc 09:39 vì `cuongtm` nở VRAM giữa chừng
(GPU còn 3 MiB trống). Nhánh ASAM của đích `js_full` trên codebert vì thế còn thiếu. Không chặn gì;
lấp khi có máy rảnh.

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
