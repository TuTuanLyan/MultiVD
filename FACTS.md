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

## 4b. Ma trận reset — `codebert` và `unixcoder`, λ=0.2, đủ 5 fold

Máy `ntat` (RTX 4070S Ti), seed 42, bộ fold `goc`, 25/08. Mỗi backbone có baseline và nhánh `none`
của chính nó trên cùng máy, nên mọi Δ đều ghép cặp trong cùng phần cứng.

Giá trị **riêng của head phụ** = Δ(nhánh) − Δ(`none`), ghép cặp theo fold, **cùng optimizer**:

| | `cwe` | `latent_bottleneck` | `latent_proto` |
| --- | --- | --- | --- |
| **codebert** RecAdam | **+0.0542 (5/5)** | **+0.0514 (5/5)** | +0.0129 (3/5) |
| **codebert** AdamW | **+0.0325 (5/5)** | **+0.0322 (5/5)** | +0.0145 (3/5) |
| **unixcoder** RecAdam | +0.0038 (3/5) | +0.0039 (3/5) | −0.0145 (1/5) |
| **unixcoder** AdamW | −0.0015 (2/5) | +0.0079 (3/5) | −0.0197 (1/5) |

Và **pretrain một mình** (`none` vs baseline):

| | RecAdam | AdamW | baseline |
| --- | --- | --- | --- |
| codebert | −0.0014 (2/5) | +0.0112 (4/5) | 0.7811 |
| unixcoder | **+0.0294 (4/5)** | **+0.0333 (4/5)** | 0.8442 |

**Phân ly kép.** Hai backbone cùng họ RoBERTa, cùng máy, cùng fold, cùng seed, mà vai trò của hai
thành phần đảo ngược nhau:

- **codebert**: pretrain một mình ≈ 0 hoặc âm; **toàn bộ lợi ích đến từ head phụ**, và hai head
  **dùng nhãn CWE** dương ở **cả 5/5 fold** trên cả hai optimizer (bốn ô, p=0.0625 — tức mức sàn
  của Wilcoxon ở n=5, nghĩa là "cùng dấu ở cả 5 fold").
- **unixcoder**: **pretrain một mình lấy trọn** +0.03; head phụ thêm ~0 và ba ô đổi dấu khi bỏ một
  fold; `latent_proto` âm 4/4 ô.

Đối chiếu họ CodeT5 (mục 4): ở đó hai head dùng nhãn CWE âm **12/12 ô** và `latent_proto` là nhánh
**duy nhất** dương. Nên qua 5 backbone hiện có **ba chế độ khác nhau**, không phải một:

| chế độ | backbone | thứ mang lại lợi ích |
| --- | --- | --- |
| head phụ **có nhãn** | codebert | `cwe`, `latent_bottleneck`, 5/5 fold |
| **pretrain** thuần | unixcoder | `none`; head phụ không thêm gì |
| head phụ **không nhãn** | t5, t5p, t5pe | `latent_proto`; hai head có nhãn đều hại |

Câu "transfer chỉ chạy được với họ RoBERTa" trong tài liệu cũ vì thế **gộp hai cơ chế khác nhau vào
một tên**. Đây là seed 42, một bộ fold — chưa loại trừ được may rủi giữa seed.

## 4c. Trục λ trên họ RoBERTa — ngược chiều họ CodeT5

`ntat`, seed 42, đủ 5 fold ở **cả hai** giá trị λ. Giá trị riêng của head phụ (vs `none`, cùng optimizer):

| | λ=0.2 | λ=0.05 | |
| --- | --- | --- | --- |
| codebert `cwe` RecAdam | **+0.0542 (5/5)** | +0.0411 (4/5) | λ=0.2 hơn |
| codebert `cwe` AdamW | **+0.0325 (5/5)** | +0.0283 (5/5) | λ=0.2 hơn |
| codebert `latent_bottleneck` RecAdam | **+0.0514 (5/5)** | +0.0226 (4/5) | λ=0.2 hơn |
| codebert `latent_bottleneck` AdamW | **+0.0322 (5/5)** | +0.0150 (4/5) | λ=0.2 hơn |
| codebert `latent_proto` RecAdam | +0.0129 (3/5) | +0.0174 (4/5) | λ=0.05 hơn |
| codebert `latent_proto` AdamW | +0.0145 (3/5) | +0.0076 (2/5) | λ=0.2 hơn |
| unixcoder — cả 6 ô | ≈ 0 | ≈ 0 hoặc kém hơn | không đọc được |

**Bốn ô head có nhãn của codebert: λ=0.2 hơn λ=0.05 ở cả 4.** Họ CodeT5 thì ngược — giảm λ từ 0.2
xuống 0.05 cải thiện `cwe` và `latent_bottleneck` ở 6/6 ô. Nên trên trục λ, hai họ backbone đi
**ngược chiều nhau** cho cùng một loại head phụ:

| | dạng đường cong λ của head **có nhãn** |
| --- | --- |
| họ CodeT5 (3 điểm: 0.05, 0.2, ≈101) | **đơn điệu giảm** — tốt nhất là nhỏ nhất từng thử |
| codebert (2 điểm: 0.05, 0.2) | **tăng** trong khoảng này — cực đại nằm ở λ ≥ 0.2, chưa quét tới |

**Điều này bác một con số vẫn được trích trong tài liệu cũ**: "CodeBERT `cwe` head phụ +0.0328 (5/5)
ở λ=0.05 so với +0.0114 (3/5) ở λ=0.2". Trong ma trận reset, cùng backbone và cùng bộ fold, kết quả
đảo chiều. Số cũ đến từ run `fam2_*`, mà mục 9 đã ghi là có một nhánh Phase 1 hỏng.

### 4c-2. Trục λ ba điểm trên codebert — head CÓ NHÃN cũng có cực đại nội tại

Đủ 5 fold ở cả ba giá trị λ, cùng máy, `none` cùng optimizer làm đối chứng:

| | λ=0.05 | λ=0.2 | λ=0.5 |
| --- | --- | --- | --- |
| `cwe` RecAdam | +0.0411 (4/5) | **+0.0542 (5/5)** | +0.0395 (4/5) |
| `cwe` AdamW | +0.0283 (5/5) | **+0.0325 (5/5)** | +0.0136 (3/5) |
| `latent_bottleneck` RecAdam | +0.0226 (4/5) | +0.0514 (5/5) | **+0.0593 (5/5), p=0.0625** |
| `latent_bottleneck` AdamW | +0.0150 (4/5) | +0.0322 (5/5) | **+0.0333 (5/5), p=0.0625** |
| `latent_proto` RecAdam | +0.0174 (4/5) | +0.0129 (3/5) | *(Phase 1 bị từ chối)* |
| `latent_proto` AdamW | +0.0076 (2/5) | +0.0145 (3/5) | *(Phase 1 bị từ chối)* |

**`cwe` có cực đại nội tại tại λ=0.2 ở cả hai optimizer.**

`latent_bottleneck` được quét thêm **λ=1.0**, đủ 5 fold, và đỉnh đã được **kẹp**:

| `latent_bottleneck` | λ=0.05 | λ=0.2 | λ=0.5 | λ=1.0 |
| --- | --- | --- | --- | --- |
| RecAdam | +0.0226 (4/5) | +0.0514 (5/5) | **+0.0593 (5/5)** | +0.0472 (5/5) |
| AdamW | +0.0150 (4/5) | +0.0322 (5/5) | +0.0333 (5/5) | **+0.0456 (5/5)** |

RecAdam có cực đại nội tại tại **λ≈0.5**; AdamW vẫn đang lên ở λ=1.0 nên đỉnh của nó ở trên 1.0.

**Đây là kết quả bền nhất dự án có:** `codebert/latent_bottleneck` dương ở **5/5 fold tại mọi λ từ
0.2 đến 1.0, trên cả hai optimizer** — tám ô liên tiếp cùng dấu, mỗi ô p=0.0625 (mức sàn của Wilcoxon
ở n=5). So với `none` cùng optimizer, cùng máy, cùng bộ fold.

**Điều này rút lại một cách đọc đã dùng trong các mục trước.** Tài liệu này từng mô tả tín hiệu **có
nhãn** là "đơn điệu giảm theo λ" và tín hiệu **không nhãn** là "có cực đại nội tại", coi đó là khác
biệt **về chất**. Số liệu codebert bác điều đó: head có nhãn **cũng** có đỉnh, chỉ nằm ở λ cao hơn.

Cách đọc thay thế, đơn giản hơn và khớp cả hai họ: **λ tối ưu phụ thuộc backbone**, và cực đại của
họ CodeT5 nằm **dưới 0.05** — dưới giá trị nhỏ nhất từng thử — nên đường cong của họ đó *trông* đơn
điệu giảm chỉ vì cửa sổ quét nằm trọn bên phải đỉnh. Cùng một hình dạng, dịch chỗ.

Dự đoán kiểm được từ cách đọc này: **quét λ = 0.01 và 0.02 trên họ CodeT5 phải thấy đường cong quay
đầu.** Rẻ: 3 lần rút Phase 1 cho mỗi điểm, không cần chạm vào baseline hay `none`.

**λ=0.5 val nguồn (đo 25/08, `ntat`):** hai head dùng nhãn chịu được — `cwe` val nguồn 0.6393, `latent_bottleneck`
0.6363. Head **không nhãn** thì sập: `latent_proto` dừng ở **epoch 2, val 0.5234**, bị cổng từ chối.
Nhất quán với cực đại nội tại quanh λ≈0.2 của `latent_proto` đã thấy trên họ CodeT5 — đi lên quá thì
hỏng hẳn chứ không kém dần. Δ transfer của λ=0.5 xem kết quả Phase 2.

*Bẫy khi đọc bảng val nguồn:* val nguồn **giảm đều theo λ** ở cả ba nhánh (`cwe` 0.6519 → 0.6396 →
0.6393; `latent_bottleneck` 0.6889 → 0.6577 → 0.6363), nhưng mục 8 đã đo rằng val nguồn **không** dự
báo Δ transfer. Bảng đó **không** nói λ=0.05 tốt hơn — chính λ=0.2 mới cho Δ transfer cao nhất ở cả
4 ô head có nhãn.

Việc chưa làm: quét λ **lớn hơn 0.2** cho codebert. Cực đại của nó nằm ngoài khoảng đã đo, và đó là
backbone duy nhất trong 5 cái mà head phụ mang lại toàn bộ lợi ích transfer (mục 4b).

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

Chưa đo cho `codet5p-220m-bimodal` **trong bảng trên**; đo lại đủ 5 backbone ở mục 5b.

### 5b. Đo lại trên ma trận reset — đủ 5 backbone, kèm `‖w‖`

Nhánh `cwe`, seed 42, checkpoint của chính ma trận reset. Bảng trên thiếu `‖w‖`, mà **ρ tuyệt đối
trên hai mô hình có `‖w‖` khác nhau ba lần thì không phải cùng một phép đo** — đó là confound phải
xử lý trước khi so.

| backbone | `‖w‖` | loss nền | Δ ρ=0.01 | Δ ρ=0.05 | Δ ρ=0.1 | Δ ρ=0.2 |
| --- | --- | --- | --- | --- | --- | --- |
| CodeBERT | 957.09 | 0.8707 | 0.1393 | **0.8095** | 1.6489 | 2.5714 |
| UniXcoder | 640.54 | 0.6990 | 0.0775 | **0.5687** | 1.4846 | 4.0373 |
| CodeT5-base | 1874.83 | 0.6359 | 0.0073 | **0.0525** | 0.1370 | 0.3545 |
| codet5p-220m-bimodal | 1828.13 | 0.6442 | 0.0157 | **0.0806** | 0.1414 | 0.2404 |
| codet5p-110m-embedding | 1828.13 | 0.6400 | 0.0151 | **0.0782** | 0.1387 | 0.2401 |

Ở cùng ρ tuyệt đối 0.05, họ RoBERTa nhọn hơn họ CodeT5 **10–15 lần**. Nhưng `‖w‖` của họ CodeT5 lớn
gần gấp đôi đến gấp ba, nên ρ=0.05 với chúng là nhiễu loạn **tương đối** nhỏ hơn 2–3 lần. Ghép ở
**cùng ρ/‖w‖** thì khoảng cách co lại nhưng không mất:

| ghép ở cùng ρ/‖w‖ | | |
| --- | --- | --- |
| CodeBERT ρ=0.05 (5.2e-5) | vs CodeT5-base ρ=0.1 (5.3e-5) | 0.8095 vs 0.1370 → **5.9×** |
| UniXcoder ρ=0.05 (7.8e-5) | vs CodeT5-base ρ≈0.146 (nội suy) | 0.5687 vs ≈0.237 → **2.4×** |

**Kết luận đi ngược giả thuyết SAM ban đầu.** `measure_sharpness.py` in ra ngay trong output của nó:
*"SAM chỉ đáng chạy nếu backbone đang hỏng (CodeT5+) nhọn hơn rõ rệt"*. Họ CodeT5 — họ mà transfer
hỏng — lại **phẳng hơn**, không phải nhọn hơn, kể cả sau khi chuẩn hoá theo `‖w‖`. Làm phẳng thêm
không phải là thứ họ đang thiếu.

Đây là bản mạnh hơn của mục 4: không chỉ "thứ tự độ nhọn không khớp thứ tự nào", mà **khớp ngược** —
backbone nhọn nhất (CodeBERT) chính là backbone head phụ hoạt động tốt nhất (mục 4b). Tương quan
trên 5 điểm, không phải nhân quả.

Ghi chú cần kiểm: `‖w‖` của `codet5p-220m-bimodal` và `codet5p-110m-embedding` trùng nhau tới 6 chữ
số (1828.13) dù val và Δloss khác nhau. Dịch chuyển Phase 1 gần như trực giao với `w` nên `‖w‖` đổi
rất ít, nhưng trùng đến mức này thì đáng xác minh chứ chưa nên coi là đã hiểu.

## 5c. SAM ở PHASE 1 — ρ phải chia thang theo backbone

Watts et al. (ICML 2026, arXiv:2605.02105) đặt SAM ở giai đoạn pretrain. `src/train.py` nay cho phép
(`PHASE1_EXTRA="--sam_rho ..."`). Kết quả rút Phase 1, `ntat`, seed 42:

**ρ=0.05 (giá trị bài báo) làm hỏng họ RoBERTa.** Không phải suy đoán — đo được ở mục 5b: Δloss đối
kháng của CodeBERT tại ρ=0.05 là **0.8095** trong khi chính loss của nó là **0.8707**. Bước leo gần
như nhân đôi loss, tức nhảy ra khỏi lòng chảo chứ không thăm dò lân cận. Quan sát:

| | ρ=0.05 |
| --- | --- |
| `codebert/none` | train loss kẹt 0.703 (= ln 2), val 0.3333 (đoán một lớp), dừng sớm epoch 7 |
| `codebert/cwe` | val 0.4586 ở epoch 4, so với 0.6396 bản không SAM |

**ρ=0.01 đưa nhiễu loạn về ngang họ CodeT5 ở ρ=0.05** (codebert Δloss 0.1393 ≈ 16% loss, unixcoder
0.0775 ≈ 11%, họ CodeT5 ở ρ=0.05 là 8–12%), và 7/8 nhánh chạy được:

| nhánh | thường | + SAM ρ=0.01 | chênh |
| --- | --- | --- | --- |
| `codebert/none` | ep15 0.6283 | ep7 **0.3333** | **−0.2949 · BỊ TỪ CHỐI** |
| `codebert/cwe` | ep15 0.6396 | ep13 0.6209 | −0.0187 |
| `codebert/latent_bottleneck` | ep14 0.6577 | ep14 0.6423 | −0.0154 |
| `codebert/latent_proto` | ep15 0.6758 | ep14 0.6123 | −0.0635 |
| `unixcoder/none` | ep12 0.7023 | ep15 0.6944 | −0.0078 |
| `unixcoder/cwe` | ep13 0.6953 | ep12 0.6792 | −0.0161 |
| `unixcoder/latent_bottleneck` | ep13 0.6883 | ep13 **0.6912** | **+0.0029** |

Ba điều rút ra:

1. **ρ tuyệt đối không chuyển được giữa các backbone.** Cùng ρ=0.05 là nhiễu loạn nhẹ với họ CodeT5
   và là phá huỷ với họ RoBERTa. Bất kỳ so sánh SAM nào dùng chung một ρ cho nhiều backbone đều
   đang so hai chế độ khác nhau. Đây là lý do tag `_sam1r01` tách khỏi `_sam1`.
2. **Chỉ đúng một tổ hợp chết: `codebert × none`.** Đó là backbone nhọn nhất trong 5 (mục 5b) gặp
   mục tiêu Phase 1 yếu nhất trong bốn nhánh của chính nó (0.6283, thấp nhất). Mô tả, chưa phải cơ
   chế. Giả thuyết "thiếu head phụ nên SAM nhấn chìm tín hiệu nhị phân" **đã bị bác** bởi
   `unixcoder/none`, cũng không có head phụ mà vẫn về 0.6944.
3. **Giá phải trả ở val nguồn là nhỏ** (−0.008 đến −0.064, một ô dương). Và mục 8 đã đo rằng val
   nguồn **không** dự báo Δ transfer, nên con số này chưa nói SAM-Phase-1 tốt hay xấu — Phase 2 mới
   trả lời.

### 5c-1. SAM ở Phase 2 — đủ 5 fold, họ CodeT5, khối đầy đủ đầu tiên

`ntat2`, ρ=0.05 (đúng thang cho họ này, mục 5b), 3 backbone × 4 nhánh × 2 optimizer × 5 fold.
Δ = nhánh có SAM ở Phase 2 − nhánh tương ứng không SAM, **cùng checkpoint Phase 1**, ghép cặp theo fold.

| nhóm | kết quả |
| --- | --- |
| head **có nhãn** (`cwe`, `latent_bottleneck`), 12 ô | 9/12 dương, trung bình **+0.0009** |
| head **không nhãn** (`latent_proto`), 6 ô | **1/6 dương**, trung bình −0.0414 *(−0.0088 nếu bỏ ô sập)* |
| **`none`**, 6 ô | 3/6, trung bình **+0.0020** |

**SAM ở Phase 2 không làm gì.** +0.0009 và +0.0020 là số không. Thứ duy nhất nhất quán là nó **hại**
`latent_proto` (1/6 ô dương).

**Cú sập không phải nhiễu — nó tái lập.** `t5/latent_proto/RecAdam`:

| fold | không SAM | + SAM | |
| --- | --- | --- | --- |
| 1 | 0.8090 | 0.8476 | +0.0386 |
| 2 | 0.8417 | **0.4545** | −0.3872 |
| 3 | 0.8420 | **0.5060** | −0.3360 |
| 4 | 0.8880 | **0.5355** | −0.3525 |
| 5 | 0.5577 | 0.5709 | +0.0132 |

**3/5 fold sập xuống mức gần ngẫu nhiên.** Đây là hỏng lưỡng cực (chạy được hoặc sập), không phải
nhiễu quanh một trung bình — nên sd 0.2116 của ô đó khiến Δ trung bình vô nghĩa. `docs/SAM_REFERENCE.md`
đã cảnh báo SAM và RecAdam chồng lấn vì cả hai sửa bước cập nhật; đây là bằng chứng đo được.

*Cảnh báo về n:* ở n=4 nhóm có-nhãn là 11/12 dương với trung bình +0.0098; ở n=5 còn 9/12 và +0.0009.
Đây là lần thứ ba trong ngày 25/08 một hình dạng co lại khi thêm một fold.

### 5c-2. SAM ở Phase 1 có giúp transfer không — đủ 5 fold, họ RoBERTa

Δ = nhánh có SAM ở Phase 1 (ρ=0.01) − nhánh tương ứng không SAM, **cùng máy, cùng λ, cùng Phase 2**,
ghép cặp theo fold. `ntat`, seed 42.

| | `cwe` | `latent_bottleneck` | `latent_proto` | `none` |
| --- | --- | --- | --- | --- |
| **codebert** RecAdam | −0.0320 (0/5) | −0.0355 (1/5) | **+0.0135 (4/5)** | *(loại)* |
| **codebert** AdamW | −0.0277 (2/5) | −0.0395 (0/5) | −0.0063 (2/5) | *(loại)* |
| **unixcoder** RecAdam | −0.0145 (3/5) | −0.0052 (2/5) | **+0.0158 (4/5)** | −0.0027 (2/5) |
| **unixcoder** AdamW | −0.0063 (2/5) | −0.0263 (0/5) | **+0.0152 (4/5)** | +0.0027 (3/5) |

| nhóm | kết quả |
| --- | --- |
| head **có nhãn** (`cwe`, `latent_bottleneck`), 8 ô | **âm 8/8**, trung bình **−0.0234** |
| head **không nhãn** (`latent_proto`), 4 ô | dương 3/4, trung bình **+0.0096** |
| **`none`** (không có head phụ), 2 ô | **−0.0027 và +0.0027 — bằng không** |

**Kết luận cho họ RoBERTa: SAM ở Phase 1 không dùng được.** Nó hại đúng hai nhánh đang tốt trên
codebert (mục 4b: `cwe` và `latent_bottleneck` dương 5/5 fold) và chỉ giúp nhẹ nhánh vốn yếu. Biên
độ nhỏ so với sd giữa fold (0.014–0.041), và chỉ hai ô chạm p=0.0625 — **cả hai đều âm**.

**Đối chiếu họ CodeT5 — `none` KHÔNG bằng không.** Cùng phép so, trên `t5pe`, ρ=0.05, máy local, đủ 5 fold:

| | Δ | fold dương | p |
| --- | --- | --- | --- |
| `t5pe/none` RecAdam | **+0.0293** | 4/5 | 0.1250 |
| `t5pe/none` AdamW | +0.0056 | 3/5 | 0.4375 |

Từng fold của cột RecAdam: +0.0330, +0.0264, +0.0340, +0.0531, **−0.0002**. Bốn fold dương chắc
chắn và một fold **hoà tuyệt đối** — không phải một fold đi ngược. Nhưng p=0.1250 chứ không chạm mức
sàn 0.0625, chỉ một backbone, và AdamW không có gì. **Đáng theo tiếp, chưa đủ để kết luận.**

Cảnh báo về n, ghi lại vì nó lặp: ô này là +0.0311 (3/3) ở n=3 và +0.0366 (4/4) ở n=4 trước khi về
+0.0293 (4/5) ở n=5. Ngày 25/08 có **năm** hình dạng co lại khi thêm fold.

**Điều đáng chú ý hơn con số: `none` của họ RoBERTa bằng đúng không.** SAM đổi hẳn checkpoint Phase 1 của
`unixcoder/none` (val nguồn 0.7023 → 0.6944) mà Δ transfer **không nhúc nhích** qua 5 fold. Nên tác
dụng của SAM ở Phase 1 — dù dương hay âm — chỉ xuất hiện **khi có head phụ**, tức nó tác động qua
việc thay đổi cách tín hiệu phụ định hình biểu diễn, không phải qua độ phẳng tự thân.

Đây là mục thứ ba cùng nói một điều: **độ phẳng không phải là đại lượng điều khiển transfer** —
mục 4 (thứ tự độ nhọn không khớp thứ tự nào), mục 5b (họ hỏng lại phẳng hơn), và giờ là 5c-2 (cố ý
làm phẳng cũng không đổi gì khi không có head phụ).

### 5c-3. Nhánh `none` là nhánh dễ vỡ nhất dưới SAM ở Phase 1 — đếm được, chưa giải thích được

Đếm mọi lần rút Phase 1 có SAM đã chạy trong ngày 25/08, trên cả 5 backbone:

| nhánh | lần rút hỏng / tổng |
| --- | --- |
| có head phụ (`cwe`, `latent_bottleneck`, `latent_proto`) | **0 / 15** |
| `none` | **2 / 5** — `codebert` (ρ=0.01) và `t5` (ρ=0.05) |

Cả hai lần hỏng đều là cùng một dạng: val Macro-F1 kẹt ở mức đoán-một-lớp, dừng sớm, bị
`phase1_usable` từ chối.

**Giới hạn phải nói kèm.** Hai họ chạy ở **ρ khác nhau** (0.01 cho RoBERTa vì 0.05 phá huỷ chúng,
0.05 cho CodeT5), nên 20 lần rút này không nằm trên một trục so sánh duy nhất. Và giả thuyết dạng
mạnh — *"không có head phụ thì SAM luôn nhấn chìm tín hiệu nhị phân"* — **đã bị bác**: `unixcoder/none`,
`t5p/none`, `t5pe/none` đều sống. Thứ số liệu đỡ được chỉ là dạng yếu: mục tiêu Phase 1 **không có
head phụ** là mục tiêu **dễ vỡ nhất**, không phải luôn vỡ.

Cách kiểm rẻ cho lần sau: chạy riêng nhánh `none` của một backbone ở vài mức ρ (0.002, 0.005, 0.01,
0.02) và xem ngưỡng vỡ nằm ở đâu so với nhánh `cwe` của chính nó. Không cần Phase 2, chỉ cần val
nguồn — mỗi điểm ~5 phút.

**Hàng bị nhiễm, phải loại khỏi mọi tổng hợp SAM-Phase-1:** `n1_codebert/transfer_none_sam1r01`
và `..._sam1r01_adamw` (6 hàng, fold 1–3). Phase 1 của chúng là checkpoint val 0.3333 đã nêu ở
trên; Δ của chúng là **−0.4278 / −0.4436**, và đó là số đo của "fine-tune từ một checkpoint ngang
ngẫu nhiên", **không phải** số đo của SAM ở Phase 1. Giữ lại trong `results_all.jsonl` vì chúng là
phép đo thật của một hiện tượng khác, nhưng gộp chúng vào là sai.

**Hệ quả cho khối 07 (SAM ở Phase 2).** Khối đó chạy ρ=0.05 trên họ CodeT5, tức vùng nhẹ, nên nó
hợp lệ. Nhưng chưa từng chạy SAM Phase 2 trên họ RoBERTa, và nếu chạy thì **không được dùng ρ=0.05**.

## 6. Dịch chuyển trọng số sau Phase 1

`‖θ_phase1 − θ_pretrained‖ / ‖θ_pretrained‖`, trung bình có trọng số theo tham số, seed 42,
source `train_ccpp_js`.

| backbone | `none` | `cwe` | đợt |
| --- | --- | --- | --- |
| CodeBERT | 0.011876 | 0.010556 | 22/08 |
| CodeT5+ 220m | 0.022105 | 0.020844 | 22/08 |
| **CodeBERT** | **0.011894** | **0.012040** | ma trận reset 25/08 |
| **UniXcoder** | **0.012234** | **0.012563** | ma trận reset 25/08 |

Ba điều đọc được từ hai dòng mới:

1. CodeBERT lặp lại gần khít giữa hai đợt ở nhánh `none` (0.011876 → 0.011894), nên phép đo này
   **tái lập được** dù Phase 1 thì không (val nguồn sd 0.0521 giữa các lần rút — mục 9).
2. Head phụ gần như **không đổi độ dịch**: chênh `cwe` − `none` là +0.000146 trên CodeBERT và
   +0.000329 trên UniXcoder, tức khoảng 1–3%. Head phụ **không** hoạt động bằng cách kéo trọng số
   đi xa hơn.
3. Và độ dịch **không phân biệt được** hai backbone (0.0119 vs 0.0122) trong khi kết quả của chúng
   phân ly hoàn toàn (mục 4b). Nên đại lượng này, giống độ nhọn, **không dự báo transfer**.

Tensor dịch nhiều nhất trên CodeBERT đều là `attention.self.value.bias` của các lớp trên
(0.0335 ở lớp 11, 0.0298 ở lớp 5) — cao gấp ~3 lần mức tổng thể.

Chưa đo cho CodeT5-base và t5pe của ma trận reset.

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
| `run/matrix.sh` cổng chất lượng Phase 1 | `phase1_usable` **chỉ chạy trên đường tái dùng**, không chạy sau khi huấn luyện mới — nó chặn file bị cắt ngang, không chặn file hoàn chỉnh mà ngang ngẫu nhiên | `codebert__none_sam1r01` kết ở val **0.3333** (đoán một lớp), được công bố, **6 job Phase 2** chạy trên nó cho ra **−0.43**. Đã vá 25/08: cổng chạy ở cả hai đường, không đạt thì đổi thành `.rejected` và tính vào `FAILED` |
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

**Trạng thái cuối ngày 25/08.** Cả hai máy vast đã **HỦY** sau khi kéo về đầy đủ (kết quả +
checkpoint Phase 1), đối chiếu khớp. Ngày 26/08 thuê máy mới.

| máy | GPU | backbone | run | đã chạy xong |
| --- | --- | --- | --- | --- |
| `ntat` (đã hủy) | RTX 4070S Ti | `codebert`, `unixcoder` | `n1` | λ=0.2, λ=0.05, SAM-Phase-1 ρ=0.01 — **đủ 5 fold**. λ=0.5 và λ=1.0 chỉ `codebert` |
| `ntat2` (đã hủy) | RTX 4080S | `t5`, `t5p`, `t5pe` | `m1` | λ=0.2, λ=0.05, λ học được, SAM-Phase-2 ρ=0.05 — **đủ 5 fold**. SAM-Phase-1: 11/12 checkpoint + Phase 2 fold 1 |
| local | RTX A4000 | `t5pe` | `l1` | λ=0.2 và SAM-Phase-1 ρ=0.05 fold 1–3; nhánh `none` fold 1–5 |

Máy local là **server dùng chung**. Quy tắc riêng: đúng **một** process, `nice -n 15`,
`OMP_NUM_THREADS=2`, `num_workers 0`, và **không watchdog** — hàng đợi local luôn hữu hạn để nó tự
dừng thay vì giữ GPU vô thời hạn. Chủ máy luôn được nhường. Giờ nghỉ 23:00 **chỉ áp dụng cho vast**.

Hai máy vast nằm trên **cùng một host vật lý** nhưng **khác GPU**, nên không được so chéo. Mỗi
backbone giữ trọn baseline và `none` của chính nó trên MỘT máy.

### Việc cho ngày 26/08, theo thứ tự tôi đề xuất

1. **Quét λ = 0.01 và 0.02 trên họ CodeT5.** Mục 4c-2 đưa ra một cách đọc thống nhất — λ tối ưu chỉ
   *dịch chỗ* theo backbone chứ không đổi *hình dạng* — và dự đoán kiểm được là đường cong họ CodeT5
   phải **quay đầu** dưới 0.05. Rẻ nhất trong mọi việc còn lại: 3 lần rút Phase 1 mỗi điểm, dùng lại
   baseline và `none` của khối λ=0.2. Đây là phép thử có thể bác một kết luận trung tâm.
2. **λ ≥ 1.0 cho `codebert/latent_bottleneck`.** Ô mạnh nhất dự án (+0.0593, 5/5 fold, p=0.0625 ở
   λ=0.5) **vẫn đang lên**. Chạy λ=1.0 dở dang tối 25/08, cần chạy lại đủ 5 fold.
3. **Hoàn tất Phase 2 của khối SAM-Phase-1 họ CodeT5.** 11 checkpoint đã có sẵn ở
   `model_run_ntat2/m1/phase1/*_sam1/` — đẩy lên máy mới là chạy thẳng Phase 2, tiết kiệm ~4 giờ GPU.
   Mới có fold 1.
4. **Ngưỡng vỡ của nhánh `none` dưới SAM** (mục 5c-3): chạy `none` ở ρ ∈ {0.002, 0.005, 0.01, 0.02}
   trên một backbone, chỉ cần val nguồn, ~5 phút mỗi điểm.
5. **Đa seed.** Toàn bộ mục 4 là seed 42. Đây là cổng **cuối**, chỉ mở sau khi phương pháp đã vững
   trên nhiều backbone — không dùng để lấp GPU trống.

### Vẫn chưa chạy

- **SAM ở Phase 2 cho họ RoBERTa** — và **không được dùng ρ=0.05**: mục 5b đo được ρ đó phá huỷ họ
  này ở Phase 1. Nếu chạy thì dùng ρ=0.01.
- **Dịch chuyển trọng số** của CodeT5-base và t5pe (codebert/unixcoder đã có — mục 6).
- **Bộ fold `sven_python_random`** — chưa có run nào.
- **Pooling cho họ T5**: dùng `mean` theo quy ước học thuật. Đo trên bộ gốc với `codet5p-220m`:
  `cls` cho head phụ −0.0027, `mean` cho −0.0373.
- `check_phase1.py` chưa bắt được Phase 1 dừng sớm với val ngang ngẫu nhiên (mục 9). Cổng trong
  `run/matrix.sh` **đã** bắt (vá 25/08), nhưng công cụ độc lập thì chưa.

### Công cụ vận hành

| lệnh | làm gì |
| --- | --- |
| `bash scripts/status.sh` | trạng thái cả ba máy trong một lần gọi |
| `bash scripts/pull_results.sh [--with-phase1]` | kéo kết quả về; thêm cờ trước khi trả máy |
| `bash scripts/shutdown_vast.sh` | dừng hàng đợi → kéo về → đối chiếu → hủy. `ACTION=none\|stop\|destroy` |
| `bash scripts/restart_queue.sh` | (chạy TRÊN máy vast) khởi động lại hàng đợi với `STEPS` mới |
| `python src/build_records.py` | gộp mọi `fold*.json` vào `records/results_all.jsonl` (gộp, không ghi đè) |
| `python src/report_paired.py --run X` / `--vs A B` | ghép cặp theo fold, in n, sd, bỏ-1-fold, Wilcoxon |

## §13 — Ra soat du lieu nguon (26/08), truoc khi thue GPU

Do bang tokenizer that: microsoft/codebert-base (ho RoBERTa, dai dien ca unixcoder),
Salesforce/codet5-base, Salesforce/codet5p-110m-embedding. "Cap sap" = hai ve cua
mot cap tro thanh chuoi token GIONG HET nhau sau khi cat o 512, tuc input y het
nhung nhan nguoc nhau.

### 13.1 Ty le cap sap tren du lieu DANG dung

| file | ham > 512 token | cap sap |
|---|---|---|
| ccpp_primevul_paired_full.jsonl | 71.4% | 35.5% (1668/4694) |
| ccpp_primevul_paired_common.jsonl | 70.2% | 35.3% (521/1477) |
| ccpp_primevul_paired_4cwe.jsonl | 65.2% | 33.0% (29/88) |
| **train_ccpp_filtered.jsonl** (Phase 1 doc qua train_ccpp_js.jsonl) | 63.0% | **32.9% (24/73)** |
| train_js_filtered.jsonl | 40.1% | 13.9% (77/555) |
| ccpp_primevul_fit512.jsonl | 0% | 0% |

RevisitVD (arXiv:2507.16887) bao 27% tren PrimeVul; do tren repo nay ra 35.5%.
`ccpp_primevul_fit512.jsonl` (20/08) da khong con loi nay nhung CHUA BAO GIO duoc
noi vao pipeline — run/matrix.sh:66 van tro data/train_ccpp_js.jsonl.

### 13.2 Tran cung cua nguon C/C++ 4 CWE

Dem tren toan bo PrimeVul v0.1 tho (ca file khong ghep cap), so ham co nhan lo hong:

| | CWE-22 | CWE-78 | CWE-79 | CWE-89 | tong |
|---|---|---|---|---|---|
| train | 36 | 24 | 22 | 5 | 87 |
| valid | 4 | 2 | 6 | 0 | 12 |
| test | 6 | 1 | 2 | 1 | 10 |
| **tong** | 46 | 27 | 30 | **6** | **109** |

Sau ghep cap + bo cap sap: 59 cap / 118 hang, CWE-89 con 3 cap. Siet them
"khong ham nao bi cat": 26 cap. Day la tinh chat cua du lieu, khong phai loi
pipeline — CWE-22/78/79/89 la lo hong web, gan nhu khong ton tai trong C/C++.

### 13.3 Hai loi dung du lieu khac

- `train_js_filtered.jsonl` ghep cap theo THU TU DONG; 37/569 cap ke nhau khong
  phai cap that (khac CWE, tuong dong trung vi 0.41 vs 0.945 cua cap that).
- `train_ccpp_js_parent.jsonl` chi con 2 lop (132/1152) — nhan phu gan nhu vo dung.
- PrimeVul v0.1 tho: 38 cap co hai dong ke nhau doi nhan nhung KHAC commit_id.
  Phai ghep bang commit_id, khong duoc tin adjacency.

### 13.4 Nguon dung lai (src/build_sources.py, --drop collapse)

| file | hang | cap giu | cap bo vi sap | CWE | neu --drop truncated |
|---|---|---|---|---|---|
| src_ccpp_4cwe.jsonl | 118 | 59 | 30 (33.7%) | 4 | 58 hang (49%) |
| src_ccpp_common.jsonl | 2360 | 1180 | 722 (38.0%) | 74 | 986 hang (42%) |
| src_ccpp_full.jsonl | 6042 | 3021 | 1664 (35.5%) | 102 | 2424 hang (40%) |
| src_js_4cwe.jsonl | 956 | 478 | 77 (13.9%) | 4 | 648 hang (68%) |

Moi hang co `pair_id` (khoa theo commit_id phia ccpp), `n_tok_max`, `bi_cat`.
`train_transfer.py:40` da co pair_id dau GROUP_FIELDS nen tu dong dung
GroupShuffleSplit — khong can sua code.

`src_ccpp_common_parent.jsonl`: 8 pillar, dung duoc 100%. CWE-664 32.4%,
CWE-707 22.8%, CWE-682 15.8%, CWE-703 12.6%, CWE-284 8.9%, CWE-691 5.6%,
CWE-693 1.8%, CWE-697 0.2% (4 hang — nen gop hoac bo).

### 13.5 Con mo

- Nguon JS "common" chua dung duoc: train_js_filtered.jsonl chi chua dung 4 CWE
  nen js_common trung y het js_4cwe. Can CleanVul day du, KHONG co tren may.
  Hai ban dan xuat tren may mau thuan nhau: Archive/cwe_js_cleanfull.jsonl (1138)
  vs MAML/data/full_js_cleanvul.jsonl (1836, 96 CWE), giao nhau chi 562, khong
  cai nao chua cai nao. huggingface.co/api/datasets/yikun-li/CleanVul tra ve 200.
- Nhanh ccpp/4cwe chi 59 cap: chua quyet dinh bo, doi bo CWE, hay giu lam arm
  chung minh gioi han.

## §14 — Ngay 27/08, seed 42, Phase 1 (KHONG SAM). Head phu cuu backbone yeu tren nguon kho

Cau hinh: seed 42, lambda=0.05, 15 epoch, lr 2e-5, batch 16, max_len 512,
`--sam_rho 0`. Moi backbone chay tron tren MOT may (codebert=ntat,
t5p=ntat2, unixcoder=local) de baseline va moi nhanh cua no cung phan cung.
So duoi day la val macro-F1 cua HEAD NHI PHAN o Phase 1 — KHONG phai chi so dich.

### 14.1 Nguon `4cwe` (930 hang: 812 js + 118 ccpp) — nguon DE

| nhanh | codebert | t5p | unixcoder |
|---|---|---|---|
| none | 0.6874 | 0.7072 | 0.6768 |
| cwe | 0.6667 | 0.6874 | 0.7082 |
| latent_bottleneck | 0.6532 | 0.6976 | 0.6661 |
| latent_proto | 0.6870 | 0.6971 | 0.6976 |

`none` bang hoac hon cac nhanh phu o codebert va t5p. KHONG duoc doc thanh
"head phu vo dung" — day la val cua head nhi phan, con viec cua head phu la
nan bieu dien cho Phase 2.

### 14.2 Nguon `full` (7598 hang: 6042 ccpp + 1556 js) — nguon KHO

| nhanh | codebert | best_ep | t5p | unixcoder | best_ep (unix) |
|---|---|---|---|---|---|
| none | **0.3357** (TU CHOI) | 3 | 0.5612 | 0.5601 | 7 |
| latent_bottleneck | **0.5636** | 11 | 0.5557 | 0.5678 | 13 |
| latent_proto | 0.4557 (TU CHOI) | 4 | (dang chay) | (dang chay) | |

`latent_bottleneck` - `none`:
  codebert   +0.2279   (best_epoch 3 -> 11)
  t5p        -0.0055
  unixcoder  +0.0077

Sang 27/08 luc dau chi co 1 backbone duoc cuu / 1 khong doi, chua loai duoc
ngau nhien. Voi unixcoder thi thanh **1 duoc cuu / 2 khong doi**, va ca hai cai
"khong doi" deu nam trong +/-0.008 — nho hon nhieu mot bac.

Doc: tren nguon ma muc tieu nhi phan thuan SAP ve muc doan bua, loss phu CO NHAN
on dinh duoc toi uu hoa cho backbone yeu (huan luyen lau gap ~3 lan truoc khi
hong), con backbone manh thi khong can. Thu tu: co nhan (0.5636) > khong nhan
(0.4557) > khong co gi (0.3357).

Day la LAP LAI cua double dissociation o §4b (loi ich cua codebert den tu head
phu, cua unixcoder den tu pretrain) nhung o Phase 1 thay vi Phase 2 va tren mot
nguon khac — corroboration doc lap, khong phai cung mot phep do tinh hai lan.

GIOI HAN: n=1, mot seed. Chenh 0.2279 qua lon de la nhieu; chenh -0.0055 thi
hoan toan co the dao dau. Con thieu o `cwe` va toan bo unixcoder tren `full`.

### 14.3 Do kho cua nguon phu thuoc backbone

`4cwe` -> `full`, nhanh `none`:
  unixcoder 0.6768 -> 0.5601  (-0.117)
  t5p       0.7072 -> 0.5612  (-0.146)
  codebert  0.6874 -> 0.3357  (-0.352, sap han)

Backbone nao chiu thiet nhieu nhat tu nguon kho thi cung la backbone duoc head
phu cuu nhieu nhat — hai quan sat nay nhat quan ve co che, khong roi rac.

Khop voi PrimeVul ICSE 2025 (CodeBERT chi 20.86 F1 tren PrimeVul) va voi §13.1
(35.5% cap PrimeVul sap nhan duoi 512 token — sau khi loc bo, phan con lai van
rat kho).

## §15 — Ket qua Phase 2 day du, seed 42, 27-28/08

295 ket qua fold: codebert 85 (17 nhanh x 5 fold), t5p 105 (21x5), unixcoder 105
(21x5). codebert thieu 4 nhanh vi `none_full` va `latent_proto_full` bi cong
chat luong tu choi (xem §14.2). Moi backbone chay TRON tren mot may:
codebert=ntat, t5p=ntat2, unixcoder=local + ntat2 (fold 4,5). Ca hai may vast da
huy sau khi doi chieu tung file ke ca kich thuoc byte.

Cau hinh: seed 42 va CHI 42, lambda=0.05, KHONG SAM o Phase 1, Phase 2 30 epoch
lr 2e-5, dich `sven_python_folds_norm` (60/20/20). Chi so: test_macro_f1_at_0.5.
Delta ghep cap theo (backbone, nguon, optimizer, fold) so voi nhanh `none`.

### 15.1 Bang xep hang moi to hop nhanh x nguon x optimizer

| nhanh | nguon | opt | n | D tb | dau | sign p |
|---|---|---|---|---|---|---|
| **latent_bottleneck** | **4cwe** | **adamw** | **15** | **+0.0227** | **13/15** | **0.0074** |
| cwe | 4cwe | recadam | 15 | +0.0117 | 10/15 | 0.30 |
| latent_proto | 4cwe | adamw | 15 | +0.0104 | 10/15 | 0.30 |
| latent_bottleneck | full | adamw | 10 | +0.0099 | 8/10 | 0.11 |
| latent_bottleneck | 4cwe | recadam | 15 | +0.0098 | 9/15 | 0.61 |
| cwe | 4cwe | adamw | 15 | +0.0063 | 7/15 | 1.00 |
| latent_bottleneck | com | adamw | 15 | +0.0026 | 10/15 | 0.30 |
| latent_bottleneck | com | recadam | 15 | -0.0014 | 6/15 | 1.00 |
| latent_bottleneck | full | recadam | 10 | -0.0040 | 4/10 | 1.00 |

CHI MOT o duoi 0.05. Moi o con lai p >= 0.11.

### 15.2 O thang, tung backbone

| backbone | D tb | fold duong |
|---|---|---|
| codebert | +0.0266 | 4/5 |
| t5p | +0.0217 | 5/5 |
| unixcoder | +0.0198 | 4/5 |

15 o: 13 duong, 1 am (codebert fold 5, -0.0077), 2 hoa tuyet doi (t5p fold 5,
unixcoder fold 2 — giong het den 4 chu so).

### 15.3 Ba dieu bang nay noi ra

1. **Nguon quan trong hon nhanh.** `4cwe` la nguon DUY NHAT ma head phu co tac
   dung nhat quan. `com` phang li (-0.0014..+0.0026 o ca 4 o). `full` yeu.
   `4cwe` la nguon NHO NHAT (930 hang) nhung la nguon duy nhat co nhan 4 lop can
   bang — nen cai giup khong phai luong du lieu ma la CHAT LUONG tin hieu phu.
2. **Nhanh va optimizer khong tach roi.** `latent_bottleneck` thang khi ghep
   AdamW (13/15) nhung roi xuong 9/15 voi RecAdam. Nguoc lai `cwe` thi RecAdam
   (10/15) kha hon AdamW (7/15). Phai phat bieu thanh MOT CAP.
3. **Co mot o AM that.** `latent_bottleneck` + RecAdam + `full` tren unixcoder:
   0/5 fold duong, D -0.0318. Hai that, khong phai nhieu.

### 15.4 Gioi han

- 15 fold KHONG doc lap hoan toan: cung 5 fold dich dung lai cho 3 backbone, nen
  p=0.0074 la lac quan. Phat bieu chac hon: tung backbone cho 5/5, 4/5, 4/5 va
  CA BA cung huong.
- Mot seed duy nhat (42).
- Tap dich la ban ro ri `sven_python_folds_norm` (~40% hang test co twin trong
  train) — chon co chu y de so lien mach voi ket qua cu; so tuyet doi 0.75-0.89
  bi thoi phong.
- D +0.0227 nho hon san nhieu giua may (0.028) — chi co nghia TRONG cung may, ma
  thiet ke nay dung nhu vay (moi D ghep cap trong mot fold tren mot may).

## §16 — Dia day lam hong checkpoint, va cong chat luong dan nham nhan (30/08)

Khoi lambda=0.02 + ASAM rho=0.1 + RecAdam tren ntat. Ba nhanh Phase 1 cua t5p bi
`run/matrix.sh` doi ten thanh `best.pt.rejected` kem thong bao
"KHONG DAT (best_epoch<=1 hoac val<0.55)". **Khong nhanh nao that su kem.**

Loi that su trong log la khi ghi file:

```
RuntimeError: [enforce fail at inline_container.cc:672] . unexpected pos 388918848 vs 388918736
```

Kich thuoc file noi ro hon moi thong bao:

| Checkpoint | Byte | |
|---|---|---|
| `t5p__cwe_4cwe_l02` (lanh) | 438 505 969 | moc |
| `t5p__latent_proto_4cwe_l02.rejected` | 389 021 824 | cut ~49 MB |
| `t5p__latent_bottleneck_com_l02.rejected` | 33 270 | cut gan het |
| `t5p__latent_proto_com_l02.rejected` | 33 679 | cut gan het |

`torch.save` bi cat giua chung vi **/ day 100%** (20 GB, `model/` chiem 6.7 GB
trong do 4.8 GB la 10 checkpoint unixcoder da xong tu lau). Lan doc lai nem
RuntimeError, `train_transfer.py` tra ve that bai, va cong chat luong — chi phan
biet "co file hop le / khong" — ket luan la chat luong kem.

**Hai hau qua, ca hai deu im lang:**

1. `.rejected` la vinh vien theo thiet ke (de khong huan luyen lai vong lap), nen
   15 o Phase 2 se khong bao gio sinh ra.
2. Mot dot don dia truoc do da xoa `model/s42/phase1/t5p__none_*` (bo lambda=0.05)
   trong khi `t5p__none_*_l02/seed_42` la **symlink tro vao do** (dung lai hop le
   theo muc 5 CLAUDE.md: nhanh `none` khong phu thuoc lambda). Symlink treo ->
   `FileExistsError: [Errno 17] File exists` khi tao thu muc -> them 15 o nua mat.

Tong 30/50 o cua t5p bien mat vi mot su kien ha tang, khong o nao bao loi ro rang.

**Da sua**: xoa 10 checkpoint unixcoder tren may sau khi doi chieu 10/10 file khop
tung byte voi local (3.7 GB -> 8.4 GB trong); day lai 3 file `none` that tu local
thay cho symlink treo; xoa 3 nhan `.rejected` de huan luyen lai.

**Bai hoc cho cong xac minh**: mot cong chi kiem "load duoc khong" se quy MOI that
bai ve nguyen nhan no biet ten. Phan biet **file hong** (RuntimeError luc doc,
kich thuoc lech moc) voi **chat luong kem** (doc duoc, val thap) truoc khi dan
nhan vinh vien. Va don dia phai tu choi xoa bat ky thu muc nao dang la **dich cua
mot symlink**.

## §17 — Khoi λ=0.02 · ASAM ρ=0.1 (Phase 2) · RecAdam · seed 42, day du 150 o (31/08)

3 backbone × 3 nguon × 4 nhanh × 5 fold, tap dich `sven_python_folds_norm`,
SAM tat o Phase 1 (`--sam_rho 0`), ASAM chi o Phase 2 (ρ=0.1, η=0.01).

### Chuyen giao hai pha van an, va an chac

`none` (chuyen giao tran, khong head phu) so voi baseline, ghep cap theo
(backbone, nguon, fold), da loai 2 nhanh co Phase 1 sap:

| | |
|---|---|
| Δ trung binh | **+0.0280** |
| Fold cung dau | **35/40** |
| p (kiem dau, hai phia) | **< 0.0001** |
| Bien do | −0.0526 … +0.0850 |

### Head phu KHONG them gi o λ=0.02

Δ so voi `none` trong cung khoi/backbone/nguon/fold:

| nhanh | n | Δ vs none | cung dau | p |
|---|---|---|---|---|
| cwe | 15 | −0.0098 | 5/15 | 0.30 |
| latent_bottleneck | 40 | −0.0076 | 15/40 | 0.15 |
| latent_proto | 40 | −0.0105 | 15/40 | 0.15 |

Ca ba deu duoi san nhieu 0.010 ve do lon, nen doc la **rong**, khong phai "co hai".

### Ha λ tu 0.05 xuong 0.02 khong giup

Ghep cap tung o voi khoi D (λ=0.05, cung ASAM ρ=0.1, cung RecAdam) — doi dung mot bien:

| nhanh | n | Δ (F−D) | cung dau | p |
|---|---|---|---|---|
| none | 38 | +0.0028 | 22/38 | 0.42 |
| cwe | 15 | **−0.0122** | **3/15** | **0.035** |
| latent_bottleneck | 43 | −0.0099 | 19/43 | 0.54 |
| latent_proto | 37 | −0.0055 | 16/37 | 0.51 |
| **tat ca** | **133** | **−0.0053** | **60/133** | **0.30** |

Tong the rong. Rieng `cwe` te di that (3/15 fold, p=0.035) — hop ly, vi `cwe` la
nhanh phu thuoc nhieu nhat vao viec loss phu co trong so dang ke.

### Quet moi khoi: chi MOT o song sot

Δ so voi `none`, ghep cap trong cung khoi:

| khoi | cwe | latent_bottleneck | latent_proto |
|---|---|---|---|
| A λ=.05 khong SAM RecAdam | +0.0088 (12/18) | +0.0036 (23/43) | +0.0020 (25/43) |
| **B λ=.05 khong SAM AdamW** | +0.0046 (12/18) | **+0.0111 (33/43) p=0.0006** | +0.0051 (22/43) |
| C λ=.05 khong SAM RecAdam (doi chung) | +0.0052 (10/14) | +0.0058 (19/37) | +0.0040 (24/37) |
| D λ=.05 ASAM ρ=.1 RecAdam | +0.0033 (10/15) | −0.0032 (19/38) | −0.0025 (17/37) |
| F λ=.02 ASAM ρ=.1 RecAdam | −0.0098 (5/15) | −0.0076 (15/40) | −0.0105 (15/40) |

**`latent_bottleneck` + AdamW + λ=0.05 la o duy nhat co p < 0.05 va do lon vuot
san nhieu.** Doi optimizer sang RecAdam (khoi A, cung moi thu khac) ha no ve
+0.0036 va 23/43 — tuc bang khong. Them ASAM (D) hoac ha λ (F) cung xoa no.

Do lon +0.0111 chi nhinh hon san nhieu 0.010 mot chut; cai manh la **huong**:
33/43 fold cung dau. Phai neu ca hai so, khong duoc chi neu p.

### Bang chung truc tiep vi sao phai co cong chat luong Phase 1

`codebert` / nguon `full` / nhanh `none`, Phase 1 ket o val macro-F1 **0.3403**
(best_epoch 3, muc doan bua tren bai nhi phan). Phase 2 chay tu do:

**Δ = −0.4395, 0/5 fold, bien do −0.4904 … −0.3874.**

Trung khop voi truong hop cu `codebert__none_sam1r01` (val 0.3333 -> −0.43 ghi o
muc truoc). Neu khong nhin val Phase 1, con so nay doc y het "nhanh none khong hop
voi nguon full". Moi o trong so ket qua deu phai deo kem val Phase 1.

## §18 — Khoi NIGHT48: λ=0.05 · ρ ∈ {0, 0.1} · latent_bottleneck · RecAdam · 3 nguon × 3 seed × 5 fold (05/09)

**Cau hinh**: `codet5p-220m-bimodal` (mean pooling) · `latent_bottleneck` (H→8→C) ·
RecAdam · λ=0.05 · Pha 1 khong SAM, 15 epoch · Pha 2 hai nhanh ρ=0 (doi chung) va
ASAM ρ=0.1 (η=0.01) · nguon `4cwe`/`com`/`full` · seed 42/7/1234 · fold 1–5 ·
dich `sven_python_folds_norm` · `PHASE1_MIN_VAL=0`.

**90/90 o Pha 2 + 15/15 baseline, khong o nao thieu, khong o nao chong lan.**
Hai nhanh ρ dung chung dung mot checkpoint Pha 1 nen hieu giua chung doi mot bien.

### Pha 1 — khong nguon nao sap, ke ca `full`

| nguon | seed 42 | seed 7 | seed 1234 | do tan |
|---|---|---|---|---|
| `4cwe` | 0.6976 (ep 6) | 0.6684 (ep 8) | 0.7223 (ep 11) | 0.054 |
| `com` | 0.5897 (ep 11) | 0.5907 (ep 8) | 0.5684 (ep 14) | 0.022 |
| `full` | 0.5648 (ep 7) | 0.5834 (ep 14) | 0.5930 (ep 15) | 0.028 |

`full` giu 0.56–0.59 o **ba seed doc lap tren ba may khac nhau**. Truoc day chinh nguon
nay lam Pha 1 cua t5p sap ve ~0.34 voi `none` va `latent_proto` (§15, muc 7 CLAUDE.md).
Ba lan lap doc lap la bang chung chac hon han hai lan roi rac da ghi truoc do.

### Baseline tung seed (trung binh 5 fold)

seed 42 **0.7985** (0.7236–0.8618) · seed 7 **0.8073** (0.7630–0.8421) ·
seed 1234 **0.8140** (0.7821–0.8482). Do tan giua seed 0.0155, vuot san nhieu 0.010 —
bat buoc ghep cap theo seed.

### A) Chuyen giao vs baseline, n=15 moi o

| nguon | ρ | Δ macro-F1 | Δ ROC-AUC |
|---|---|---|---|
| `4cwe` | 0 | +0.0133 (11/15, p=0.119) | +0.0030 (8/15) |
| `4cwe` | 0.1 | +0.0120 (10/15, p=0.302) | +0.0100 (10/15) |
| `com` | 0 | +0.0092 (10/15, p=0.302) | +0.0056 (9/15) |
| `com` | 0.1 | **+0.0191 (12/15, p=0.0352)** | +0.0045 (10/15) |
| `full` | 0 | +0.0021 (8/15) | **−0.0094 (4/15)** |
| `full` | 0.1 | +0.0036 (7/15) | −0.0032 (8/15) |

**O duy nhat vua qua p<0.05 vua vuot san nhieu: `com` × ρ=0.1, +0.0191.**
`full` chet han tren ca hai chi so — nguon nhieu CWE nhat lai chuyen giao kem nhat.

### B) Hieu RIENG cua ASAM (ρ=0.1 vs ρ=0, chung checkpoint Pha 1)

| nguon | Δ macro-F1 | trai | Δ ROC-AUC |
|---|---|---|---|
| `4cwe` | −0.0012 (7/15) | −0.0335…+0.0555 | +0.0071 (11/15) |
| `com` | +0.0099 (12/15, p=0.0352) | −0.0459…+0.0691 | −0.0011 (10/15) |
| `full` | +0.0016 (10/15) | −0.0492…+0.0421 | +0.0062 (10/15) |
| **gop** | **+0.0034 (29/45, p=0.073)** | | +0.0040 (31/45, p=0.016) |

`com` qua p<0.05 nhung do lon **0.0099, tuc vua duoi san nhieu 0.010**. Va no khong lap
lai giua cac seed: +0.0056 (seed 42), +0.0106 (seed 7), **−0.0059** (seed 1234).

### Hieu ung nam o MOT FOLD, khong phai o may

Gop het seed, tach theo fold (moi o 9 diem = 3 nguon × 3 seed):

| fold | ΔASAM macro-F1 |
|---|---|
| 1 | −0.0040 (5/9) |
| 2 | −0.0035 (5/9) |
| **3** | **+0.0233 (8/9, p=0.0391)** |
| 4 | +0.0051 (6/9) |
| 5 | −0.0038 (5/9) |

Fold 3 duong o **ca ba seed** (+0.0281 / +0.0240 / +0.0177), tren **ba may khac nhau**
(vast cu, vast moi, 158). Khong phai hien vat phan cung.

Cung **khong** phai "ASAM cuu fold kho": tuong quan giua baseline cua fold va ΔASAM chi
r = −0.202 (n=15), va baseline cua fold 3 rat khac nhau giua ba seed (0.7236 / 0.8352 /
0.7950). Day la dac tinh cua **mot lat cat cu the** cua tap dich, khong phai cua phuong
phap.

### Ket luan

ASAM ρ=0.1 **khong mua duoc gi vuot san nhieu tren macro-F1** — cung ket luan da co o
ρ=0.2 va ρ=0.5 (§ dot xac nhan n=15 truoc, `results_confirm47_com/`). Ba ban kinh, ba
lan, cung mot cau tra loi. Giu ASAM TAT.

Con so gop +0.0034 la mot hieu ung cua rieng fold 3 bi pha loang; bao cao bang trung
binh 5 fold ma khong tach fold se giau mat dieu do.

### Ha tang

Khoi chay tren 4 may: seed 42 (vast A4000, torch 2.9.1+cu130), seed 7 fold 1–2
(161, 2.9.1+cu128), seed 7 fold 3–5 (vast A4000, 2.9.1+cu128, **dung lai dung 3
checkpoint Pha 1 cua seed 7 tu 161**), seed 1234 (158, 2.9.1+cu128). transformers
4.57.1 tren moi may. Chia theo **fold tron ven** nen moi Δ ghep cap nam gon trong mot
may. Du lieu goc: `results_night48/`, `results/n48_t5p/`, `results_night48b/`,
`results_night48_158/`. Bao cao: `python3 tools/n48_report.py`.
