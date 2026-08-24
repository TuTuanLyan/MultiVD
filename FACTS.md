# FACTS — cấu hình và số liệu

Tài liệu này chỉ ghi **cấu hình đã chạy và số đo được**. Không có nhận định, không có giải thích
cơ chế, không có đề xuất hướng đi. Ngõ cụt đã kiểm nằm ở `DEAD_ENDS.md`. Bản nhật ký dài kèm mọi
lập luận (gồm cả những lần đã rút lại) nằm ở `archive/RESULT_2026-08-23.md` — dùng để truy nguồn
một con số, không dùng khi thiết kế thí nghiệm mới.

Mọi con số dưới đây **dựng lại được** từ `records/results_all.jsonl` bằng `src/report_paired.py`.
Không con số nào được chép tay.

---

## 1. Pipeline

Phát hiện lỗ hổng ở mức hàm, nhị phân.

- **Phase 1** — huấn luyện đa nhiệm trên **source** (ngôn ngữ khác target): `L = L_binary + λ × L_aux`.
  Sinh ra một "checkpoint nguồn". Không đọc dữ liệu target.
- **Phase 2** — fine-tune trên **target** (Python) từ checkpoint nguồn, bằng RecAdam hoặc AdamW.
  Head phụ **đóng băng** và **không dùng lúc inference** — toàn bộ tác dụng của nó nằm ở chỗ nó
  định hình biểu diễn của Phase 1.
- **baseline** — fine-tune thẳng trên target bằng AdamW, không qua Phase 1.

Bốn chế độ head phụ (`--aux_mode`): `cwe` (C lớp tường minh), `latent_bottleneck`
(`Linear(H,K)→Linear(K,C)`), `latent_proto` (K prototype + Sinkhorn, **không cần nhãn**), `none`.

Đại lượng báo cáo: **Δ so với baseline**, và **head phụ = Δ(nhánh) − Δ(`none`)** — phần head phụ
thêm vào ngoài việc chỉ pretrain thêm. Cả hai luôn **ghép cặp theo fold**.

Chọn checkpoint: val Macro-F1@0.5 của riêng task nhị phân. Ngưỡng test hiệu chỉnh trên val rồi
đóng băng; test inference đúng một lần.

## 2. Dữ liệu

**Target** — `data/sven_python_folds_norm`: SVEN Python, 760 mẫu, 5 fold, 456/152/152.
Bộ đang dùng để báo cáo. **0 dòng trùng khít** giữa train/val/test ở cả 5 fold và cả 3 cặp split.
43% hàm test có bản gần-giống (SequenceMatcher > 0.75) trong train — hệ quả của corpus dựng từ
cặp vulnerable/fixed.

Hai bộ dự phòng: `sven_python_twin` (154/153/152/151/150, gần-giống 5%, gán fold **tất định**) và
`sven_python_random` (146/153/157/153/151, gần-giống 2%, `StratifiedGroupKFold(shuffle=True)`,
**chưa có run nào**).

**Source (Phase 1)**

| file | dòng | số CWE | % dòng thuộc 4 lớp {022,078,079,089} |
| --- | --- | --- | --- |
| `train_ccpp_js.jsonl` — **đang dùng** | 1284 | 4 | 100% |
| `train_js_filtered.jsonl` | 1138 | 4 | 100% |
| `train_ccpp_filtered.jsonl` | 146 | 4 | 100% |
| `ccpp_primevul_paired_4cwe.jsonl` | 178 | 4 | 100% |
| `ccpp_primevul_paired_common.jsonl` | 2975 | 73 | 6% |
| `ccpp_primevul_paired_full.jsonl` | 9408 | 121 | 2% |
| `ccpp_common_parent.jsonl` | 2975 | 9 pillar | `--cwe_vocab precomputed` |
| `train_ccpp_js_editsize.jsonl` | 1284 | 4 nhóm kích thước sửa | — |

`train_ccpp_js` = `train_ccpp_filtered` + `train_js_filtered` (146 + 1138). Head phụ 4 lớp tường
minh (`--cwe_vocab fixed4`) chỉ dùng được với source có đúng 4 CWE đó.

Phần JS của source trùng **1138/1138** hàm với target `js_twin`; `ccpp_primevul_paired_common`
trùng **0/1138**. Không ảnh hưởng khi target là Python.

**Không sửa hay xử lý lại `ccpp_primevul_paired_*`** — chúng có quy tắc lọc CWE riêng của người dùng.

## 3. Backbone

Đọc trực tiếp từ checkpoint bằng `src/inspect_backbone.py`.

| nhãn | model | lớp nạp về | hidden/lớp | vocab | tham số ngoài embedding | pooling |
| --- | --- | --- | --- | --- | --- | --- |
| `codebert` | `microsoft/codebert-base` | RobertaModel | 768/12 | 50,265 | 86,042,112 | `cls` |
| `unixcoder` | `microsoft/unixcoder-base` | RobertaModel | 768/12 | 51,416 | 86,442,240 | `cls` |
| `t5` | `Salesforce/codet5-base` | T5EncoderModel | 768/12 | 32,100 | 84,954,240 | `mean` |
| `t5p` | `Salesforce/codet5p-220m-bimodal` | T5Stack (`.encoder`) | 768/12 | 32,103 | **84,954,240** | `mean` |
| `t5pe` | `Salesforce/codet5p-110m-embedding` | T5Stack (`.encoder`) | 768/12 | 32,103 | **84,954,240** | `mean` |
| *(cũ)* | `Salesforce/codet5p-220m` | T5EncoderModel | 768/12 | 32,100 | **84,954,240** | `mean` |

**Ba checkpoint họ CodeT5+ có đúng cùng 84,954,240 tham số ngoài embedding và cùng shape output
`(batch, seq, 768)`** — kiểm bằng forward thật, không suy từ config. Khác nhau duy nhất ở giai đoạn
pretrain. `codet5p-*-bimodal` và `codet5p-*-embedding` cần `trust_remote_code=True` và phải lấy
`.encoder`; `build_backbone` có một nhánh khoá theo tên model cho đúng hai họ tên đó.

So encoder của `110m-embedding` với `codet5p-220m`: 98/99 tensor trùng tên và shape, **0/98 tensor
giống hệt**, lệch Frobenius toàn bộ **1.392**.

## 4. Kết quả đợt 22/08

Chung: source `train_ccpp_js.jsonl` · target Python bộ gốc · **seed 42, một seed** · 5 fold ·
`--cwe_vocab fixed4` · batch 16 · max_length 512 · lr 2e-5 · patience 5 · min_epochs 3 ·
truncation `head_middle_tail`. Chạy trên RTX 5060 Ti và RTX 5070 Ti.

Mọi Δ **ghép cặp theo fold, đủ 5 fold, không loại trừ gì**. Cột "fold sập" đếm số fold có
Macro-F1 < 0.55 nhưng **không** bỏ chúng khỏi trung bình.

### λ=0.2 · Phase 2 = RecAdam

| backbone | nhánh | F1 | Δ baseline | +/n | head phụ | +/n | fold sập |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **CodeBERT cls** | baseline | 0.7668 | — | | — | | |
| | `none` | 0.7956 | +0.0288 | 5/5 | — |  |  |
| | `cwe` | 0.8070 | +0.0402 | 5/5 | +0.0114 | 3/5 |  |
| | `latent_bottleneck` | 0.8295 | +0.0627 | 5/5 | +0.0339 | 5/5 |  |
| | `latent_proto` | 0.7650 | -0.0018 | 2/5 | -0.0306 | 1/5 |  |
| **UniXcoder cls** | baseline | 0.8511 | — | | — | | |
| | `none` | 0.8802 | +0.0291 | 3/5 | — |  |  |
| | `cwe` | 0.8828 | +0.0318 | 4/5 | +0.0027 | 2/5 |  |
| | `latent_bottleneck` | 0.8763 | +0.0252 | 4/5 | -0.0039 | 2/5 |  |
| | `latent_proto` | 0.8788 | +0.0278 | 4/5 | -0.0013 | 2/5 |  |
| **CodeT5-base mean** | baseline | 0.7995 | — | | — | | |
| | `none` | 0.8566 | +0.0570 | 5/5 | — |  |  |
| | `cwe` | 0.7758 | -0.0237 | 0/5 | -0.0807 | 0/5 |  |
| | `latent_bottleneck` | 0.7853 | -0.0142 | 1/5 | -0.0712 | 0/5 |  |
| | `latent_proto` | 0.5650 | -0.2345 | 1/5 | -0.2915 | 0/5 | 3 |
| **CodeT5+ 220m mean** | baseline | 0.8547 | — | | — | | |
| | `none` | 0.8504 | -0.0043 | 3/5 | — |  |  |
| | `cwe` | 0.8296 | -0.0252 | 1/5 | -0.0209 | 1/5 |  |
| | `latent_bottleneck` | 0.8163 | -0.0384 | 0/5 | -0.0341 | 0/5 |  |
| | `latent_proto` | 0.8392 | -0.0156 | 2/5 | -0.0113 | 1/5 |  |
| **t5pe 110m-emb mean** | baseline | 0.8005 | — | | — | | |
| | `none` | 0.8354 | +0.0349 | 5/5 | — |  |  |
| | `cwe` | 0.7997 | -0.0008 | 2/5 | -0.0357 | 0/5 |  |
| | `latent_bottleneck` | 0.7860 | -0.0145 | 1/5 | -0.0494 | 0/5 |  |
| | `latent_proto` | 0.8234 | +0.0229 | 3/5 | -0.0120 | 1/5 |  |

### λ=0.2 · Phase 2 = AdamW (chỉ hai backbone đã chạy)

| backbone | nhánh | F1 | Δ baseline | +/n | head phụ | +/n | fold sập |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **CodeBERT cls** | baseline | 0.7668 | — | | — | | |
| | `none` | 0.8127 | +0.0458 | 5/5 | — |  |  |
| | `cwe` | 0.8085 | +0.0417 | 4/5 | -0.0042 | 2/5 |  |
| | `latent_bottleneck` | 0.8272 | +0.0604 | 5/5 | +0.0145 | 3/5 |  |
| | `latent_proto` | 0.7700 | +0.0032 | 3/5 | -0.0427 | 1/5 |  |
| **CodeT5-base mean** | baseline | 0.7995 | — | | — | | |
| | `none` | 0.8535 | +0.0540 | 5/5 | — |  |  |
| | `cwe` | 0.7939 | -0.0057 | 1/5 | -0.0597 | 0/5 |  |
| | `latent_bottleneck` | 0.8033 | +0.0038 | 3/5 | -0.0502 | 0/5 |  |
| | `latent_proto` | 0.8353 | +0.0358 | 5/5 | -0.0182 | 1/5 |  |

### λ=0.05 · Phase 2 = RecAdam  (`none` dùng chung với λ=0.2 vì λ không vào loss khi aux_mode=none)

| backbone | nhánh | F1 | Δ baseline | +/n | head phụ | +/n | fold sập |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **CodeBERT cls** | baseline | 0.7668 | — | | — | | |
| | `cwe` | 0.8284 | +0.0616 | 5/5 | +0.0328 | 5/5 |  |
| | `latent_bottleneck` | 0.7874 | +0.0206 | 4/5 | -0.0082 | 2/5 |  |
| | `latent_proto` | 0.7807 | +0.0139 | 4/5 | -0.0149 | 1/5 |  |
| **UniXcoder cls** | baseline | 0.8511 | — | | — | | |
| | `cwe` | 0.8722 | +0.0211 | 4/5 | -0.0080 | 2/5 |  |
| | `latent_bottleneck` | 0.8709 | +0.0198 | 4/5 | -0.0093 | 1/5 |  |
| | `latent_proto` | 0.8681 | +0.0170 | 3/5 | -0.0121 | 1/5 |  |
| **CodeT5-base mean** | baseline | 0.7995 | — | | — | | |
| | `cwe` | 0.8377 | +0.0382 | 5/5 | -0.0189 | 0/5 |  |
| | `latent_bottleneck` | 0.8511 | +0.0516 | 5/5 | -0.0055 | 1/5 |  |
| | `latent_proto` | 0.8456 | +0.0461 | 4/5 | -0.0110 | 2/5 |  |
| **CodeT5+ 220m mean** | baseline | 0.8547 | — | | — | | |
| | `cwe` | 0.8484 | -0.0064 | 2/5 | -0.0021 | 2/5 |  |
| | `latent_bottleneck` | 0.8319 | -0.0228 | 1/5 | -0.0185 | 2/5 |  |
| | `latent_proto` | 0.8296 | -0.0251 | 1/5 | -0.0208 | 1/5 |  |
| **t5pe 110m-emb mean** | baseline | 0.8005 | — | | — | | |
| | `cwe` | 0.8394 | +0.0389 | 3/5 | +0.0040 | 3/5 |  |
| | `latent_bottleneck` | 0.8445 | +0.0440 | 4/5 | +0.0091 | 4/5 |  |
| | `latent_proto` | 0.8288 | +0.0283 | 4/5 | -0.0066 | 1/5 |  |

### λ=0.2 + SAM ρ=0.05 · RecAdam

| backbone | nhánh | F1 | Δ baseline | +/n | head phụ | +/n | fold sập |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **UniXcoder cls** | baseline | 0.8511 | — | | — | | |
| | `none` | 0.8945 | +0.0435 | 5/5 | — |  |  |
| | `cwe` | 0.8853 | +0.0342 | 4/5 | -0.0093 | 1/5 |  |
| **CodeT5+ 220m mean** | baseline | 0.8547 | — | | — | | |
| | `none` | 0.8590 | +0.0042 | 2/5 | — |  |  |
| | `cwe` | 0.8177 | -0.0370 | 0/5 | -0.0412 | 0/5 |  |
| **t5pe 110m-emb mean** | baseline | 0.7972 | — | | — | | |
| | `none` | 0.8403 | +0.0431 | 5/5 | — |  |  |
| | `cwe` | 0.8143 | +0.0171 | 4/5 | -0.0260 | 1/5 |  |

### RecAdam so với AdamW — ghép cặp trong cùng fold

`nora_*` dùng lại **đúng** checkpoint Phase 1 và **đúng** baseline của `fam1_*` (đã đối chiếu:
baseline giống hệt từng fold), nên hiệu dưới đây đổi **đúng một biến** là `--phase2_optimizer`.
Cột cuối là giá trị sau khi bỏ fold có |hiệu| lệch xa trung bình nhất.

| backbone | nhánh | AdamW − RecAdam | +/n | sd | bỏ 1 fold |
| --- | --- | --- | --- | --- | --- |
| CodeBERT | `none` | +0.0171 | 5/5 | 0.0243 | +0.0064 |
| CodeBERT | `cwe` | +0.0015 | 2/5 | 0.0104 | +0.0054 |
| CodeBERT | `latent_bottleneck` | -0.0023 | 3/5 | 0.0338 | +0.0103 |
| CodeBERT | `latent_proto` | +0.0050 | 2/5 | 0.0541 | -0.0122 |
| CodeT5-base | `none` | -0.0030 | 2/5 | 0.0177 | +0.0032 |
| CodeT5-base | `cwe` | +0.0180 | 3/5 | 0.0388 | +0.0011 |
| CodeT5-base | `latent_bottleneck` | +0.0179 | 5/5 | 0.0146 | +0.0126 |
| CodeT5-base | `latent_proto` | +0.2702 | 5/5 | 0.1409 | +0.3295 |
| **CodeBERT** | **gộp 20 ô** | **+0.0053** | **12/20** | 0.0325 | |
| **CodeT5-base** | **gộp 20 ô** | **+0.0758** | **15/20** | 0.1340 | |

### Chất lượng Phase 1 (val Macro-F1 trên tập val của source, seed 42, λ=0.2)

Đọc từ metadata checkpoint, lưu ở `records/phase1_checkpoints_2026-08-22.tsv` (58 dòng, giữ lại
trước khi xóa 25 GB checkpoint).

| nhánh | CodeBERT | UniXcoder | CodeT5-base | CodeT5+ 220m | t5pe 110m-emb |
| --- | --- | --- | --- | --- | --- |
| `none` | 0.6575 (ep 15) | 0.6912 (ep 11) | 0.6828 (ep 14) | 0.6771 (ep 10) | 0.7119 (ep 15) |
| `cwe` | 0.6586 (ep 12) | 0.6668 (ep 13) | 0.6449 (ep 14) | 0.7078 (ep 13) | 0.6589 (ep 14) |
| `latent_bottleneck` | 0.6476 (ep 14) | 0.7075 (ep 14) | 0.6356 (ep 14) | 0.7078 (ep 14) | 0.6586 (ep 14) |
| `latent_proto` | 0.5346 (ep 3) | 0.6778 (ep 13) | 0.5357 (ep 7) | 0.6709 (ep 14) | 0.6558 (ep 15) |

## 5. Độ nhọn cực tiểu Phase 1

Nhánh `cwe`, seed 42. `‖ε‖ = ρ` tuyệt đối, chuẩn L2 toàn cục, hướng đối kháng `ρ·g/‖g‖` — chính
là bước leo của SAM. Chỉ nạp checkpoint và chạy forward/backward, không huấn luyện gì.

| backbone | loss nền | Δ ρ=0.01 | Δ ρ=0.05 | Δ ρ=0.1 | Δ ρ=0.2 | Δ(ρ=0.2)/loss nền |
| --- | --- | --- | --- | --- | --- | --- |
| **t5pe 110m-emb** | 0.7238 | 0.0240 | 0.1180 | 0.2466 | 0.6212 | **86%** |
| CodeT5+ 220m | 0.6438 | 0.0282 | 0.1439 | 0.2945 | 0.6522 | 101% |
| CodeT5-base | 0.6463 | 0.0142 | 0.1010 | 0.2800 | 0.7526 | 116% |
| CodeBERT | 0.6830 | 0.0418 | 0.2892 | 0.6728 | 0.9675 | 142% |
| UniXcoder | 0.7425 | 0.0871 | 0.6168 | 1.5757 | 4.1487 | 559% |

Nhiễu loạn theo hướng **ngẫu nhiên** ở mọi backbone: 1e-6 đến 4.5e-5, tức gần như không làm gì.

Bốn dòng đầu đo trên máy vast 22/08; dòng `t5pe` đo trên RTX A4000 local ngày 24/08 từ checkpoint
đã tải về. Phép đo này tất định trên một checkpoint cố định (không huấn luyện, không early
stopping) nên chênh lệch phần cứng ở mức 1e-6, khác hẳn trường hợp huấn luyện.

Chưa đo cho `codet5p-220m-bimodal`.

## 6. Dịch chuyển trọng số sau Phase 1

`‖θ_phase1 − θ_pretrained‖ / ‖θ_pretrained‖`, trung bình có trọng số theo tham số, seed 42,
source `train_ccpp_js`.

| backbone | `none` | `cwe` |
| --- | --- | --- |
| CodeBERT | 0.011876 | 0.010556 |
| CodeT5+ 220m | 0.022105 | 0.020844 |

Chưa đo cho UniXcoder, CodeT5-base, t5pe. **Checkpoint đã xóa**, nên muốn có phải chạy lại Phase 1.

## 7. Kết quả các đợt trước (tóm tắt)

Bộ fold **gốc**, target Python, `head phụ` = Δ(nhánh) − Δ(`none`):

| backbone | pooling | λ | seed | Δ`none` | head phụ | ghi chú |
| --- | --- | --- | --- | --- | --- | --- |
| CodeBERT | cls | 0.2 | 36 | +0.0019 | +0.0270 (`cwe`) | |
| CodeBERT | cls | 0.05 | 42 | +0.0230 | +0.0169 (`cwe`) | |
| CodeT5-base | mean | 0.05 | 42 | +0.0079 | +0.0105 | |
| CodeT5-base | mean | 0.05 | 7 | +0.0344 | −0.0156 | |
| CodeT5-base | mean | 0.05 | 12 | +0.0285 | −0.0013 | |
| CodeT5+ 220m | cls | 0.2 | 42 | +0.0186 | −0.0027 | |
| CodeT5+ 220m | mean | 0.2 | 42 | +0.0194 | −0.0373 | |
| CodeT5+ · 73 lớp CWE | mean | 0.2 | 42 | +0.0027 | −0.0099 | source `primevul_common` |
| CodeT5+ · 9 pillar | mean | 0.2 | 42 | +0.0027 | −0.0212 | source `ccpp_common_parent` |

CodeT5-base λ=0.05 gộp 3 seed: **−0.0021**, sd giữa seed **0.0131**, dương 1/3 seed.

Bộ fold **twin** (thí nghiệm phụ): CodeBERT gộp 5 seed (n=25) — `cwe` +0.0362 (sd giữa seed
0.0117), `latent_bottleneck` +0.0258 (sd 0.0043), `none` +0.0119 (sd 0.0110). CodeT5+ gộp 3 seed —
`cwe` +0.0062 (sd giữa seed **0.0206**), `latent_bottleneck` −0.0086, `none` −0.0032.

**Target JavaScript** — hai run duy nhất. Ở cả hai, **mọi nhánh có ROC-AUC < 0.66**; run
`headroom_js_t5p` có baseline AUC 0.498–0.631, `none` 0.513–0.539, `cwe` 0.570–0.650. Không run
nào trên target JS đạt mức phân biệt được vul/non-vul.

**Seed 7, máy B, fold 1–3** (job bị `Terminated` giữa fold 3; fold 4–5 chưa chạy):

| | CodeT5+ 220m | UniXcoder |
| --- | --- | --- |
| Δ`none` seed 42 / seed 7 | −0.0114 / −0.0135 | +0.0375 / +0.0229 |
| head phụ `cwe` seed 42 / seed 7 | −0.0174 / −0.0384 | +0.0066 / +0.0089 |

## 8. Thời gian chạy

Một job = một nhánh (hoặc baseline) trên một fold, gồm train Phase 2 + inference. Trung vị theo
backbone trên RTX 5060 Ti / 5070 Ti: CodeT5-base 317 s · CodeBERT 201 s · t5pe 174 s ·
CodeT5+ 220m 157 s · UniXcoder 129 s. Con số lập kế hoạch an toàn: **~3 phút/job**.
Chi tiết ở `records/run_timing_2026-08-22.md`.

## 9. Khuyết tật đã tìm thấy trong các run cũ

| ở đâu | gì | hệ quả |
| --- | --- | --- |
| `same_emb` (SAM trên t5pe) | `sam-gate.sh` tìm Phase 1 và baseline ở `model/fam1_emb/`, nhưng run thật tên `emb1_emb`; cả hai lệnh `cp` trượt im lặng | Nhánh tự huấn luyện **Phase 1 khác** (val 0.6647/0.7197 vs 0.6589/0.7119) **và baseline khác** (0.7972 vs 0.8005). Phép so "chỉ đổi SAM" thực ra đổi **ba** biến — không đọc được |
| `sam_t5p`, `samu_unixcoder` | — | Tái dùng **đúng** (val và best_epoch trùng khít với `fam1_*`) |
| `fam2_codebert/latent_bottleneck` | Phase 1 dừng ở epoch 2 với val **0.5109** (ngang ngẫu nhiên) | Kết quả λ=0.05 của nhánh đó thừa hưởng một lần rút hỏng. `check_phase1.py` chỉ bắt `best_epoch ≤ 1` nên lọt |
| `results/*/fold*.json` trường `hyperparameters.data_path` | Ở bản ghi Phase 2 nó là giá trị mặc định, không phải source thật đã dùng | Source thật chỉ đọc được từ metadata checkpoint Phase 1 |
| Bảng gõ tay giữa các thư mục run | 4/8 ô của bảng RecAdam trong tài liệu cũ không dựng lại được từ file thô | Đã thay bằng `report_paired.py` đọc từ `records/results_all.jsonl` |

## 10. Công cụ

| file | dùng để |
| --- | --- |
| `run/matrix.sh` | driver chính — vòng ngoài là fold, Phase 1 ở kho dùng chung, in bảng sau mỗi fold |
| `run/gated.sh` | driver cũ có cổng chặn sau fold 3 |
| `run/queue_runner.sh` | hàng đợi tuần tự có mốc chốt; chốt chỉ chặn job mới, không giết job đang chạy |
| `src/build_records.py` | gộp mọi `fold*.json` thành `records/results_all.jsonl` |
| `src/report_paired.py` | so hai nhánh **ghép cặp theo fold**; in cả bản đủ fold lẫn bản bỏ fold sập |
| `src/report_fold.py` | bảng đọc-ngay trong lúc chạy, cờ `*` fold sập và `~` AUC < 0.65 |
| `src/audit_coverage.py` | kiểm kê run nào đủ/thiếu nhánh, thiếu fold, có fold sập |
| `src/check_phase1.py` | phát hiện checkpoint nguồn hỏng (`best_epoch ≤ 1`) |
| `src/measure_sharpness.py` | độ nhọn cực tiểu, cờ `--rho_mode {absolute,relative}` |
| `src/measure_drift.py` | dịch chuyển trọng số so với checkpoint pretrained gốc |
| `src/inspect_backbone.py` | thông số backbone, tách tham số embedding và ngoài embedding |
| `src/sam.py` | SAM, port từ mã JAX của Google; bật bằng `--sam_rho > 0` |
| `src/RecAdam.py` | RecAdam |
| `scripts/shutdown_vast.sh` | tải results + log về, đối chiếu từng file, chỉ hủy khi khớp |
| `src/archive/`, `run/archive/` | 14 công cụ báo cáo và 43 driver một-lần của các thí nghiệm đã đóng |

Tham khảo: `docs/SAM_REFERENCE.md` (mã gốc SAM nguyên văn + bốn chi tiết dễ port sai),
`RESEARCH_2026-08-20_0959.md` (kho trích dẫn đã xác minh).

## 11. Kho dữ liệu

| file | nội dung |
| --- | --- |
| `records/results_all.jsonl` | **1025 dòng** — mọi `(run, nhánh, seed, fold)` từng chạy, đủ metric, per-CWE, hyperparameter phân biệt, bộ fold suy ra được. Thay cho 1025 file JSON rời |
| `records/phase1_checkpoints_2026-08-22.tsv` | 57 checkpoint Phase 1: backbone, nhánh, λ, source, best_epoch, val Macro-F1 |
| `records/run_timing_2026-08-22.md` | thời gian mỗi job theo backbone, và các sự cố ghi trong log |

Checkpoint Phase 1 (25 GB) và toàn bộ log đã xóa ngày 24/08. Bản thô của các file kết quả vẫn nằm
trong lịch sử git ở commit `644dc46`.

## 12. Việc còn dở

**Đang chạy** — ma trận reset trên `ntat2` (RTX 4080S): 3 backbone họ CodeT5 (`t5`, `t5p` bimodal,
`t5pe`) × 4 nhánh × 2 optimizer × 5 fold, seed 42, λ=0.2. 12 Phase 1 + 135 job.

**Chưa chạy**

- `codebert` và `unixcoder` trong ma trận reset — chưa có máy.
- **λ=0.05 trong ma trận reset.** Số cũ cho thấy λ là biến hạng nhất: CodeBERT `cwe` head phụ
  +0.0328 (5/5) ở λ=0.05 so với +0.0114 (3/5) ở λ=0.2; t5pe `latent_bottleneck` +0.0091 (4/5) ở
  λ=0.05 so với −0.0494 (0/5) ở λ=0.2.
- **Đa seed.** Toàn bộ mục 4 là seed 42. Seed 7 mới có fold 1–3 trên hai backbone.
- **SAM** — mới chạy nhánh `none` và `cwe`, chưa chạy hai nhánh latent; và run trên t5pe hỏng
  (mục 9). Chưa quét ρ, mới thử ρ=0.05.
- **Độ nhọn của `codet5p-220m-bimodal`**; **dịch chuyển trọng số** của UniXcoder, CodeT5-base, t5pe.
- **Bộ fold `sven_python_random`** — chưa có run nào.
- **Pooling cho họ T5**: dùng `mean` theo quy ước học thuật. Đo trên bộ gốc với `codet5p-220m`:
  `cls` cho head phụ −0.0027, `mean` cho −0.0373.
- `check_phase1.py` chưa bắt được Phase 1 dừng sớm với val ngang ngẫu nhiên (mục 9).
