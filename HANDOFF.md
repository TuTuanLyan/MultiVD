# HANDOFF

Trạng thái dự án tính đến 2026-08-23. Tài liệu này chỉ ghi **việc đã làm, kết quả đo được, và
việc còn dở dang**. Không chứa nhận định, giải thích cơ chế, hay đề xuất hướng đi — phần đó nằm ở
`RESULT.md` và nên đọc sau khi đã tự khảo sát dữ liệu.

---

## 1. Bài toán và pipeline

Phát hiện lỗ hổng bảo mật ở mức hàm, nhị phân (có lỗ hổng / không).

Hai giai đoạn:

- **Phase 1** — huấn luyện đa nhiệm trên dữ liệu **source** (ngôn ngữ khác target): head nhị phân
  cộng `λ_cwe ×` head phụ. Sinh ra một checkpoint gọi là "checkpoint nguồn".
- **Phase 2** — fine-tune trên dữ liệu **target** (Python), khởi tạo từ checkpoint nguồn, tối ưu
  bằng RecAdam (mặc định) hoặc AdamW.
- **baseline** — fine-tune thẳng trên target, không qua Phase 1, không đọc dữ liệu source.

Bốn chế độ head phụ (`--aux_mode`):

| chế độ | mô tả | cần nhãn CWE |
| --- | --- | --- |
| `cwe` | phân loại CWE tường minh, C lớp | có |
| `latent_bottleneck` | `Linear(H,K) → Linear(K,C)` | có |
| `latent_proto` | K prototype + Sinkhorn, tự gán nhãn cân bằng | **không** |
| `none` | không có head phụ, λ không xuất hiện trong hàm loss | — |

Đại lượng thường dùng khi so sánh: **Δ so với baseline** của từng nhánh, và **Δ(nhánh) − Δ(`none`)**
(ghi là "head phụ cộng thêm").

---

## 2. Dữ liệu

### Target (Python, từ SVEN)

| thư mục | cỡ test mỗi fold | tình trạng |
| --- | --- | --- |
| `data/sven_python_folds_norm` | 152 ở cả 5 fold | **bộ đang dùng để báo cáo** |
| `data/sven_python_twin` | 154/153/152/151/150 | dùng trong các thí nghiệm trước 21/08 |
| `data/sven_python_random` | 146/153/157/153/151 | dựng ở §38, **chưa có run nào** |

Bộ đang dùng chia theo từng dòng. Đo được: 0 trùng lặp y hệt giữa train/val/test ở cả 5 fold và cả
3 cặp split; **43% hàm test có bản gần-giống (SequenceMatcher > 0.75) nằm trong train**, do corpus
dựng từ cặp vulnerable/fixed.

File kết quả **không ghi bộ fold nào đã dùng**. Suy ra được từ cỡ tập test hàm ý bởi
`test_accuracy_at_0.5`; `src/audit_coverage.py` và `src/report_all.py` làm việc này tự động.

### Source (Phase 1)

| file | dòng | số CWE | % dòng thuộc 4 lớp {022,078,079,089} |
| --- | --- | --- | --- |
| `data/train_ccpp_js.jsonl` | 1284 | 4 | 100% |
| `data/train_js_filtered.jsonl` | 1138 | 4 | 100% |
| `data/train_ccpp_filtered.jsonl` | 146 | 4 | 100% |
| `data/ccpp_primevul_paired_4cwe.jsonl` | 178 | 4 | 100% |
| `data/ccpp_primevul_paired_common.jsonl` | 2975 | 73 | 6% |
| `data/ccpp_primevul_paired_full.jsonl` | 9408 | 121 | 2% |
| `data/ccpp_common_parent.jsonl` | 2975 | 9 pillar | — (nhãn pillar, dùng `--cwe_vocab precomputed`) |
| `data/train_ccpp_js_editsize.jsonl` | 1284 | 4 nhóm kích thước sửa | — |

`train_ccpp_js` = `train_ccpp_filtered` + `train_js_filtered` (146 + 1138).

Ràng buộc: head phụ 4 lớp tường minh (`--cwe_vocab fixed4`) chỉ dùng được với source có đúng 4 CWE
đó. Với source nhiều CWE thì dùng `--cwe_vocab source` (head rộng bằng số CWE trong source) hoặc
`--cwe_vocab precomputed` (đọc thẳng trường `cwe_class` có sẵn trong file).

**Không sửa hay xử lý lại các file `ccpp_primevul_paired_*`** — chúng có quy tắc lọc CWE riêng của
người dùng.

Đo trên phần JS: `train_js_filtered` trùng **1138/1138** hàm với target `js_twin`;
`ccpp_primevul_paired_common` trùng **0/1138**.

---

## 3. Backbone đã dùng

Thông số đọc trực tiếp từ checkpoint bằng `src/inspect_backbone.py`:

| model | lớp nạp về | hidden/lớp | vocab | tham số ngoài embedding | pooling đang dùng |
| --- | --- | --- | --- | --- | --- |
| `microsoft/codebert-base` | RobertaModel | 768/12 | 50,265 | 86,042,112 | `cls` |
| `microsoft/unixcoder-base` | RobertaModel | 768/12 | 51,416 | 86,442,240 | `cls` |
| `Salesforce/codet5-base` | T5EncoderModel | 768/12 | 32,100 | 84,954,240 | `mean` |
| `Salesforce/codet5p-220m` | T5EncoderModel | 768/12 | 32,100 | 84,954,240 | `mean` |
| `Salesforce/codet5p-110m-embedding` | T5Stack (`.encoder`) | 768/12 | 32,103 | 84,954,240 | `mean` |

`codet5p-110m-embedding` cần xử lý riêng: `AutoModel` của nó trả về vector 256 chiều đã chuẩn hoá
chứ không phải chuỗi hidden states. `build_backbone` có một nhánh khoá theo tên model lấy `.encoder`;
mọi backbone khác đi đường cũ.

Đo bằng `/tmp/cmp_enc.py` (không commit): encoder của `110m-embedding` so với encoder của
`codet5p-220m` — 98/99 tensor trùng tên và shape, **0/98 tensor giống hệt**, độ lệch Frobenius toàn
bộ **1.392**.

---

## 4. Thí nghiệm ngày 22/08 — cấu hình

Chung cho toàn bộ: source `train_ccpp_js.jsonl` · target Python · bộ fold gốc · **seed 42** · 5 fold
· `--cwe_vocab fixed4`.

Chạy trên hai instance vast (RTX 5060 Ti và RTX 5070 Ti), mỗi máy giữ một backbone RoBERTa và một
backbone T5 để so sánh liên-họ nằm trong cùng phần cứng.

Script: `run/family-matrix.sh` (MAY=A/B, LUOT=1/2), `run/norecadam-matrix.sh`,
`run/embed-special.sh`, `run/sam-gate.sh`, `run/sharpness-all.sh`. Điều phối bằng
`run/queue_runner.sh` với mốc chốt epoch (chỉ chặn khởi động job mới, không giết job đang chạy).

---

## 5. Kết quả ngày 22/08

`*` = fold có Macro-F1 < 0.55, loại khỏi trung bình. "head phụ" = Δ(nhánh) − Δ(`none`).

### 5.1 CodeBERT · cls

| optimizer | nhánh | F1 | ΔF1 | +/n | AUC | ΔAUC | +/n | head phụ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RecAdam | baseline | 0.7668 | — | — | 0.8672 | — | — | — |
| RecAdam | `none` | 0.7956 | +0.0288 | 5/5 | 0.8645 | −0.0027 | 3/5 | — |
| RecAdam | `cwe` | 0.8070 | +0.0402 | 5/5 | 0.8920 | +0.0248 | 4/5 | +0.0114 |
| RecAdam | `latent_bottleneck` | 0.8295 | +0.0627 | 5/5 | 0.9073 | +0.0401 | 5/5 | +0.0339 |
| RecAdam | `latent_proto` | 0.7650 | −0.0018 | 2/5 | 0.8571 | −0.0101 | 2/5 | −0.0306 |
| AdamW | `none` | 0.8127 | +0.0458 | 5/5 | 0.8728 | +0.0056 | 4/5 | — |
| AdamW | `cwe` | 0.8085 | +0.0417 | 4/5 | 0.8984 | +0.0312 | 5/5 | −0.0042 |
| AdamW | `latent_bottleneck` | 0.8272 | +0.0604 | 5/5 | 0.9055 | +0.0383 | 5/5 | +0.0145 |
| AdamW | `latent_proto` | 0.7700 | +0.0032 | 3/5 | 0.8517 | −0.0155 | 1/5 | −0.0427 |

### 5.2 CodeT5-base · mean

| optimizer | nhánh | F1 | ΔF1 | +/n | AUC | ΔAUC | +/n | head phụ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RecAdam | baseline | 0.7995 | — | — | 0.8941 | — | — | — |
| RecAdam | `none` | 0.8566 | +0.0570 | 5/5 | 0.9134 | +0.0193 | 4/5 | — |
| RecAdam | `cwe` | 0.7758 | −0.0237 | 0/5 | 0.8638 | −0.0303 | 0/5 | −0.0807 |
| RecAdam | `latent_bottleneck` | 0.7853 | −0.0142 | 1/5 | 0.8795 | −0.0146 | 0/5 | −0.0712 |
| RecAdam | `latent_proto` | 0.6901 | −0.1156 | 1/2 | 0.7599 | −0.1563 | 1/2 | −0.1651 (**sập f2,f3,f5**) |
| AdamW | `none` | 0.8535 | +0.0540 | 5/5 | 0.9137 | +0.0196 | 4/5 | — |
| AdamW | `cwe` | 0.7939 | −0.0057 | 1/5 | 0.8853 | −0.0088 | 1/5 | −0.0597 |
| AdamW | `latent_bottleneck` | 0.8033 | +0.0038 | 3/5 | 0.8841 | −0.0100 | 2/5 | −0.0502 |
| AdamW | `latent_proto` | 0.8353 | +0.0358 | 5/5 | 0.9225 | +0.0284 | 5/5 | −0.0182 (không sập) |

### 5.3 UniXcoder · cls

| cấu hình | nhánh | F1 | ΔF1 | +/n | AUC | ΔAUC | +/n | head phụ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| λ=0.2 | baseline | 0.8511 | — | — | 0.9306 | — | — | — |
| λ=0.2 | `none` | 0.8802 | +0.0291 | 3/5 | 0.9420 | +0.0114 | 3/5 | — |
| λ=0.2 | `cwe` | 0.8828 | +0.0318 | 4/5 | 0.9481 | +0.0175 | 4/5 | +0.0027 |
| λ=0.2 | `latent_bottleneck` | 0.8763 | +0.0252 | 4/5 | 0.9420 | +0.0114 | 3/5 | −0.0039 |
| λ=0.2 | `latent_proto` | 0.8788 | +0.0278 | 4/5 | 0.9433 | +0.0127 | 5/5 | −0.0013 |
| λ=0.05 | `cwe` | 0.8722 | +0.0211 | 4/5 | 0.9385 | +0.0079 | 4/5 | −0.0080 |
| λ=0.05 | `latent_bottleneck` | 0.8709 | +0.0198 | 4/5 | 0.9335 | +0.0029 | 3/5 | −0.0093 |
| λ=0.05 | `latent_proto` | 0.8681 | +0.0170 | 3/5 | 0.9354 | +0.0048 | 3/5 | −0.0121 |
| λ=0.2 + SAM ρ=0.05 | `none` | 0.8945 | +0.0435 | 5/5 | 0.9550 | +0.0244 | 5/5 | — |
| λ=0.2 + SAM ρ=0.05 | `cwe` | 0.8853 | +0.0342 | 4/5 | 0.9541 | +0.0235 | 5/5 | −0.0093 |

### 5.4 CodeT5+ 220m · mean

| cấu hình | nhánh | F1 | ΔF1 | +/n | AUC | ΔAUC | +/n | head phụ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| λ=0.2 | baseline | 0.8547 | — | — | 0.9293 | — | — | — |
| λ=0.2 | `none` | 0.8504 | −0.0043 | 3/5 | 0.9277 | −0.0016 | 2/5 | — |
| λ=0.2 | `cwe` | 0.8296 | −0.0252 | 1/5 | 0.9115 | −0.0178 | 0/5 | −0.0209 |
| λ=0.2 | `latent_bottleneck` | 0.8163 | −0.0384 | 0/5 | 0.8967 | −0.0327 | 0/5 | −0.0341 |
| λ=0.2 | `latent_proto` | 0.8392 | −0.0156 | 2/5 | 0.9135 | −0.0158 | 1/5 | −0.0113 |
| λ=0.05 | `cwe` | 0.8484 | −0.0064 | 2/5 | 0.9247 | −0.0046 | 1/5 | −0.0021 |
| λ=0.05 | `latent_bottleneck` | 0.8319 | −0.0228 | 1/5 | 0.9185 | −0.0108 | 0/5 | −0.0185 |
| λ=0.05 | `latent_proto` | 0.8296 | −0.0251 | 1/5 | 0.9162 | −0.0131 | 2/5 | −0.0208 |
| λ=0.2 + SAM ρ=0.05 | `none` | 0.8590 | +0.0042 | 2/5 | 0.9390 | +0.0097 | 5/5 | — |
| λ=0.2 + SAM ρ=0.05 | `cwe` | 0.8177 | −0.0370 | 0/5 | 0.9144 | −0.0149 | 1/5 | −0.0412 |

### 5.5 CodeT5+ 110m-embedding · mean

| cấu hình | nhánh | F1 | ΔF1 | +/n | AUC | ΔAUC | +/n | head phụ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| λ=0.2 | baseline | 0.8005 | — | — | 0.8901 | — | — | — |
| λ=0.2 | `none` | 0.8354 | +0.0349 | 5/5 | 0.9125 | +0.0224 | 4/5 | — |
| λ=0.2 | `cwe` | 0.7997 | −0.0008 | 2/5 | 0.8866 | −0.0035 | 3/5 | −0.0357 |
| λ=0.2 | `latent_bottleneck` | 0.7860 | −0.0145 | 1/5 | 0.8764 | −0.0137 | 1/5 | −0.0494 |
| λ=0.2 | `latent_proto` | 0.8234 | +0.0229 | 3/5 | 0.9085 | +0.0185 | 3/5 | −0.0120 |
| λ=0.05 | `cwe` | 0.8394 | +0.0389 | 3/5 | 0.9043 | +0.0142 | 2/5 | +0.0040 |
| λ=0.05 | `latent_bottleneck` | 0.8445 | +0.0440 | 4/5 | 0.9100 | +0.0199 | 5/5 | +0.0091 |
| λ=0.05 | `latent_proto` | 0.8288 | +0.0283 | 4/5 | 0.9059 | +0.0159 | 4/5 | −0.0066 |
| λ=0.2 + SAM ρ=0.05 | `none` | 0.8403 | +0.0398 | 5/5 | 0.9127 | +0.0226 | 5/5 | — |
| λ=0.2 + SAM ρ=0.05 | `cwe` | 0.8143 | +0.0138 | 2/5 | 0.8933 | +0.0033 | 2/5 | −0.0260 |

### 5.6 Chất lượng Phase 1 (val Macro-F1 trên tập val của source)

| nhánh | CodeBERT | UniXcoder | CodeT5-base | CodeT5+ 220m |
| --- | --- | --- | --- | --- |
| `none` | 0.6575 | 0.6912 | 0.6828 | 0.6771 |
| `cwe` | 0.6586 | 0.6668 | 0.6449 | 0.7078 |
| `latent_bottleneck` | 0.6476 | 0.7075 | 0.6356 | 0.7078 |
| `latent_proto` | 0.5346 | 0.6778 | 0.5357 | 0.6709 |

### 5.7 Độ nhọn cực tiểu Phase 1 (nhánh `cwe`, seed 42)

`‖ε‖ = ρ` tuyệt đối, chuẩn L2 toàn cục, hướng đối kháng `ρ·g/‖g‖`. Cột cuối chuẩn hoá theo loss nền.

| backbone | loss nền | Δ ρ=0.01 | Δ ρ=0.05 | Δ ρ=0.1 | Δ ρ=0.2 | Δ(ρ=0.2)/loss nền |
| --- | --- | --- | --- | --- | --- | --- |
| CodeT5+ 220m | 0.6438 | 0.0282 | 0.1439 | 0.2945 | 0.6522 | 101% |
| CodeT5-base | 0.6463 | 0.0142 | 0.1010 | 0.2800 | 0.7526 | 116% |
| CodeBERT | 0.6830 | 0.0418 | 0.2892 | 0.6728 | 0.9675 | 142% |
| UniXcoder | 0.7425 | 0.0871 | 0.6168 | 1.5757 | 4.1487 | 559% |

Nhiễu loạn theo hướng ngẫu nhiên ở mọi backbone: 1e-6 đến 4.5e-5.

Chưa đo cho `codet5p-110m-embedding`.

---

## 6. Kết quả các ngày trước (tóm tắt số liệu)

### 6.1 Bộ fold gốc, target Python

| backbone | pooling | λ | seed | Δ`none` | head phụ | ghi chú |
| --- | --- | --- | --- | --- | --- | --- |
| CodeBERT | cls | 0.2 | 36 | +0.0019 | +0.0270 (`cwe`) | `latent_proto` +0.0635 vs baseline |
| CodeBERT | cls | 0.05 | 42 | +0.0230 | +0.0169 (`cwe`) | |
| CodeT5-base | mean | 0.05 | 42 | +0.0079 | +0.0105 | |
| CodeT5-base | mean | 0.05 | 7 | +0.0344 | −0.0156 | |
| CodeT5-base | mean | 0.05 | 12 | +0.0285 | −0.0013 | |
| CodeT5+ 220m | cls | 0.2 | 42 | +0.0186 | −0.0027 | |
| CodeT5+ 220m | mean | 0.2 | 42 | +0.0194 | −0.0373 | |
| CodeT5+ · 73 lớp CWE | mean | 0.2 | 42 | +0.0027 | −0.0099 | source `primevul_common` |
| CodeT5+ · 9 pillar | mean | 0.2 | 42 | +0.0027 | −0.0212 | source `ccpp_common_parent` |

CodeT5-base λ=0.05 gộp 3 seed: **−0.0021**, sd giữa seed **0.0131**, dương 1/3 seed.

### 6.2 Bộ fold twin (thí nghiệm phụ, không dùng để báo cáo)

CodeBERT gộp 5 seed (n=25): `cwe` +0.0362 (sd giữa seed 0.0117), `latent_bottleneck` +0.0258
(sd 0.0043), `none` +0.0119 (sd 0.0110).

CodeT5+ gộp 3 seed: `cwe` +0.0062 (sd giữa seed **0.0206**), `latent_bottleneck` −0.0086,
`none` −0.0032.

### 6.3 Target JavaScript

Hai run duy nhất (`headroom_js_t5p`, `jstarget_pvcommon`). Ở cả hai, **mọi nhánh có ROC-AUC < 0.66**;
nhánh `none` của `headroom_js_t5p` có AUC 0.513–0.539 ở cả 5 fold. Không có run nào trên target JS
đạt mức phân biệt được vul/non-vul.

### 6.4 Dịch chuyển trọng số sau Phase 1

`‖θ_phase1 − θ_pretrained‖ / ‖θ_pretrained‖`, trung bình có trọng số theo tham số, seed 42, source
`train_ccpp_js`:

| backbone | `none` | `cwe` |
| --- | --- | --- |
| CodeBERT | 0.011876 | 0.010556 |
| CodeT5+ 220m | 0.022105 | 0.020844 |

---

## 7. Công cụ

| file | dùng để |
| --- | --- |
| `src/report_full_table.py` | bảng từng fold + mean + Δ, có cờ `--metric` |
| `src/report_all.py` | gộp mọi run, tự tách theo bộ fold |
| `src/audit_coverage.py` | kiểm kê run nào đủ/thiếu nhánh, thiếu fold, có fold sập |
| `src/report_seeds.py` | gộp một cấu hình qua nhiều seed, có trạng thái "không kết luận được" |
| `src/check_phase1.py` | phát hiện checkpoint nguồn hỏng (`best_epoch ≤ 1`) và cảnh báo val thấp |
| `src/measure_sharpness.py` | đo độ nhọn cực tiểu, cờ `--rho_mode {absolute,relative}` |
| `src/measure_drift.py` | đo dịch chuyển trọng số so với checkpoint pretrained gốc |
| `src/inspect_backbone.py` | thông số backbone, tách tham số embedding và ngoài embedding |
| `src/sam.py` | SAM, port từ mã JAX của Google; bật bằng `--sam_rho > 0` |
| `scripts/shutdown_vast.sh` | tải results + log + checkpoint nguồn, đối chiếu từng file, chỉ hủy khi khớp |
| `run/queue_runner.sh` | chạy hàng đợi tuần tự có mốc chốt; chốt chỉ chặn job mới |

Tài liệu tham khảo: `docs/SAM_REFERENCE.md` (mã gốc SAM nguyên văn + bốn chi tiết dễ port sai).

---

## 8. Dữ liệu đã tải về

| thư mục | nội dung |
| --- | --- |
| `results_ntat/`, `results_ntat2/` | 373 file kết quả từ đợt 22/08 |
| `log_ntat/`, `log_ntat2/` | 668 file log |
| `model_ntat/`, `model_ntat2/` | **57 checkpoint nguồn Phase 1 (26 GB)** — không vào git, nằm trong `.gitignore` |
| `results_vast/`, `results_vast2/` | kết quả các đợt trước |
| `results/` | các run đời đầu chạy tại local |

Có checkpoint nguồn nghĩa là mọi phân tích không gian trọng số (độ nhọn, dịch chuyển, so sánh
encoder) chạy được offline, không cần thuê GPU.

---

## 9. Trạng thái hạ tầng

**Không còn instance nào trên vast.** Hai instance `ntat` và `ntat2` đã tải hết dữ liệu về và hủy
lúc 2026-08-22 ~17:00 UTC.

Quy tắc thuê máy nằm ở `VAST_RULES.md`. Tóm tắt điểm hay vướng:

- Mọi máy thuê mới là máy trắng: phải upload code + data, cài `transformers==4.57.1` và
  `scikit-learn` (image `vastai/pytorch` chỉ có torch).
- `vastai recycle instance` đổi cổng SSH; `vastai ssh-url` có thể trả về cổng cũ. Nguồn đúng là
  bảng `ports` trong `vastai show instance --raw`, khoá `22/tcp`.
- `vastai execute` chỉ chạy được trên instance đã stop.
- GPU 16 GB không đủ cho hai job cùng lúc: một job huấn luyện chiếm ~12.5 GB. Job đo phụ phải chờ
  GPU trống, và job CPU nặng cũng làm nghẽn khâu nạp dữ liệu (đo được: GPU tụt xuống 15%).

---

## 10. Việc còn dở dang

### 10.1 Chưa chạy

- **Đa seed cho ma trận 22/08.** Toàn bộ mục 5 là **seed 42, một seed**. Người dùng đã chốt quy
  trình: một seed trước, đa seed chỉ khi kết quả ổn định.
- **Ma trận không-RecAdam trên UniXcoder và CodeT5+ 220m** — mới chạy trên CodeBERT và CodeT5-base.
- **SAM trên CodeBERT và CodeT5-base** — mới chạy trên UniXcoder, CodeT5+ 220m, 110m-embedding.
  SAM cũng mới chỉ chạy nhánh `none` và `cwe`, chưa chạy hai nhánh latent.
- **Độ nhọn của `codet5p-110m-embedding`** — bốn backbone kia đã đo.
- **Quét ρ của SAM** — mới chỉ thử ρ=0.05.
- **Bộ fold `sven_python_random`** — chưa có run nào.
- **Nguồn dữ liệu bổ sung / thêm case JS** — người dùng có nhắc, chưa xác định nguồn cụ thể.

### 10.2 Cần người dùng quyết

- **Vai trò của RecAdam.** Số liệu ở 5.1 và 5.2 (một seed, hai backbone) cho thấy bỏ RecAdam thay
  đổi kết quả ở 7/8 nhánh. RecAdam là thành phần trung tâm của phương pháp trong mọi kết quả đã ghi
  từ trước. Việc có kiểm chứng thêm và có đổi phương pháp hay không là quyết định của người dùng.
- **Pooling cho CodeT5+ 220m.** Người dùng chọn `mean` theo quy ước học thuật cho T5. Đo trên bộ
  fold gốc: `cls` cho head phụ −0.0027, `mean` cho −0.0373.

### 10.3 Ràng buộc vận hành người dùng đã nêu

- GPU local dùng chung với đồng nghiệp, chỉ dành cho smoke test.
- Chỉ dùng instance vast có label `ntat`/`ntat2`; instance của đồng nghiệp không đụng vào.
- Cuối buổi làm việc: tải hết về rồi **hủy** instance (không stop — máy stop hay dính scheduling và
  không thuê lại ngay được).
- Mốc dừng cuối ngày là mềm: chặn job mới, không giết job đang chạy.
