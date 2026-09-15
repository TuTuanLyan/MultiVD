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


## 20. Đối chứng FINETUNE HAI LẦN THUẦN (08/09/2026) — kết quả đứng, không chạy lại

Người dùng yêu cầu 08/09: Pha 1 **không head** (`aux_mode=none`), Pha 2 **AdamW**, **SAM=0
ở cả hai pha** — tức chỉ backbone + head phân loại, fine-tune hai lần. Seed 42, đủ 5 fold.
**Đã có sẵn trong `results/s42_*`, không cần chạy lại**; giữ làm kết quả đứng kể cả khi
so với ô chạy trên máy khác.

### 20.1 So với baseline (chỉ fine-tune Python), ghép cặp cùng cây/fold

| backbone | nguồn | n | ΔF1@0.5 | ΔF1@val | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|---|---|
| t5p | 4cwe | 5 | +0.0106 (3/5) | +0.0051 (2/5) | +0.0017 (3/5) | −0.0012 (3/5) |
| t5p | com | 5 | **+0.0294 (5/5)** | +0.0276 (4/5) | +0.0104 (4/5) | **+0.0140 (5/5)** |
| t5p | full | 5 | +0.0106 (3/5) | +0.0118 (3/5) | **−0.0086 (0/5)** | −0.0136 (1/5) |
| codebert | 4cwe | 5 | +0.0352 (3/5) | +0.0232 (3/5) | +0.0102 (3/5) | −0.0019 (3/5) |
| codebert | com | 5 | **+0.0549 (5/5)** | **+0.0455 (5/5)** | +0.0284 (4/5) | +0.0266 (4/5) |
| codebert | full | — | ô thiếu: checkpoint Pha 1 là file `.rejected` **0 byte** (sự cố đầy đĩa tháng 8) — đang chạy lại ở `run/ft2.sh` |

**Fine-tune hai lần thuần đã lấy được phần lớn lợi ích**: +0.011 đến +0.055 macro-F1 so với
baseline. Bất kỳ thành phần nào của phương pháp cũng phải **vượt mốc này**, không phải vượt
baseline.

**Và một chỗ chỉ AUC nhìn thấy:** `t5p × full` cho ΔF1 +0.0106 nhưng **ΔAUC −0.0086, 0/5
fold**. Đọc một chỉ số thì tưởng nguồn `full` vô hại; đọc cả hai thì thấy nó làm hỏng thứ
hạng ở mọi fold. Đây là ví dụ sống cho `CLAUDE.md` mục 2b.

### 20.2 Head mua thêm được gì TRÊN mốc đó — `latent_bottleneck` − `none`, cùng optimizer

| backbone | opt | nguồn | ΔF1@0.5 | ΔROC-AUC |
|---|---|---|---|---|
| t5p | adamw | **4cwe** | **+0.0217 (5/5)** | +0.0039 (2/5) |
| t5p | adamw | com | −0.0016 (3/5) | −0.0090 (1/5) |
| t5p | adamw | full | +0.0095 (4/5) | +0.0132 (4/5) |
| codebert | adamw | **4cwe** | **+0.0266 (4/5)** | +0.0189 (3/5) |
| codebert | adamw | com | +0.0000 (3/5) | −0.0019 (3/5) |
| unixcoder | adamw | **4cwe** | **+0.0198 (4/5)** | +0.0093 (4/5) |
| unixcoder | adamw | com | +0.0092 (4/5) | **+0.0085 (5/5)** |
| unixcoder | adamw | full | +0.0103 (4/5) | **+0.0104 (5/5)** |

**Head + AdamW + nguồn `4cwe` là thứ duy nhất dương ở CẢ BA backbone trên F1** (5/5, 4/5,
4/5) — đó là đóng góp thật của phương pháp so với fine-tune hai lần thuần, và nó chỉ xảy ra
ở **nguồn đã lọc**. Trên `com` thì phẳng ở mọi backbone.

**Nhưng trên AUC thì head yếu hẳn** (+0.004 … +0.019, hiếm khi 5/5). Ghép với phát hiện ASAM
cùng ngày (AUC +0.0037, 119/190, p=0.0006 nhưng F1 phẳng), hai thành phần **bù nhau**: head
nâng F1, ASAM nâng AUC. Nếu ASAM xác nhận được ở n=5 thì đó là một câu chuyện phương pháp
mạch lạc chứ không phải hai mẩu rời.

Với **RecAdam** thì head yếu và thất thường, kể cả `unixcoder × full` −0.0318 (0/5) — thêm
một lý do nữa để không neo.


---

## §21 — TRỤC ρ CỦA ASAM Ở PHASE 2: đường cong có ĐỈNH NỘI TẠI (08–09/09/2026)

**Bậc 2 (xác nhận), n=5 fold × 3 nguồn = 15 ô mỗi mức ρ.** t5p ·
`latent_bottleneck` · λ=0.05 · RecAdam · seed 42. Đối chứng là **chính nhánh ρ=0
trong cùng cây, cùng máy**, ghép cặp theo (cây, nguồn, seed, fold) — CLAUDE.md §4.
Fold 1,2,4 chạy trên `ntat`; fold 3,5 trên `ntat2`; báo cáo bằng
`tools/report2.py` nên không có cặp nào bắc cầu qua hai máy.

Chỉ số CHÍNH khai báo trước khi chạy: **ROC-AUC**.

| ρ | n | ΔF1@0.5 | ΔF1@val | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|---|
| 0.1 | 15 | −0.0146 (6/15) | −0.0140 (7/15) | −0.0080 (5/15) | −0.0013 (6/15) |
| 0.2 | 15 | −0.0023 (9/15) | +0.0004 (9/15) | −0.0024 (7/15) | +0.0020 (7/15) |
| 0.5 | 15 | +0.0020 (9/15) | +0.0007 (10/15) | +0.0033 (10/15) | +0.0085 (10/15) |
| 1.0 | 15 | −0.0011 (8/15) | +0.0006 (10/15) | +0.0065 (11/15) | **+0.0159 (12/15, p=0.035)** |
| **2.0** | 15 | **+0.0124 (10/15)** | +0.0136 (10/15) | +0.0088 (11/15) | **+0.0155 (13/15, p=0.007)** |
| 4.0 | 15 | **−0.1562 (3/15, p=0.035)** | **−0.1579 (2/15, p=0.007)** | **−0.1719 (3/15, p=0.035)** | −0.1690 (4/15) |
| 8.0 | 15 | **−0.2747 (0/15, p<0.001)** | **−0.2787 (0/15)** | **−0.3197 (0/15, p<0.001)** | **−0.3147 (0/15)** |

> **09/09 — bảng này giờ ĐỦ n=15 ở CẢ BẢY MỨC.** Bậc 2 hoàn chỉnh. Đối chứng `r0`
> ở fold 3/5 đã được `asam5.sh` chạy bù nên không còn ô nào bị bỏ vì thiếu cặp.

### Đọc gì được từ bảng này

1. **Có cực đại nội tại quanh ρ = 1.0–2.0.** Đây là dạng bằng chứng khác hẳn một
   ô lẻ: sáu mức, tăng đơn điệu tới 2.0 rồi **đổ**. Một biến nhiễu không tạo ra
   hình dạng đó.
2. **ρ = 0.1 — mức dự án dùng suốt từ 31/08 — nằm ở ĐÁY.** Nó là mức duy nhất âm
   trên cả bốn chỉ số, cách đỉnh 20 lần. Nó được chốt từ một quét chỉ nhìn
   macro-F1@0.5 (§17: +0.0020, 70/130, p=0.43 → kết luận "ASAM null"). Đây là
   lần thứ hai cùng một lỗi đọc gây hậu quả — xem CLAUDE.md §2b.
3. **Ở ρ = 2.0 thì F1 cũng dương** (+0.0124, 10/15), lần đầu trong cả trục. Ở
   ρ ≤ 1.0 lợi ích chỉ nằm ở **xếp hạng** (AUC) chứ không qua được ngưỡng 0.5;
   ρ đủ lớn thì nó chuyển thành lợi ích cả ở quyết định.
4. **Vượt sàn nhiễu.** +0.0155 PR-AUC và +0.0124 F1 đều trên sàn 0.010 (cùng loại
   GPU). Δ ở ρ ≤ 0.5 thì **không** — đừng đọc chúng như hiệu ứng.

### Cảnh báo phải nêu kèm mọi lần trích số này

- **15 ô KHÔNG độc lập**: 5 fold × 3 nguồn, dùng lại cùng bộ fold đích và cùng
  seed. p là **lạc quan**. Phần chắc là *hình dạng đường cong* và *số fold cùng dấu*.
- **ρ = 4.0 mới n=8**, và nó lệch nặng: `full` chỉ có 2 ô nhưng sập −0.30…−0.37,
  kéo trung bình xuống. Tách nguồn: 4cwe −0.0094 (n=3), com −0.0636 (n=3),
  full −0.3658 (n=2). Nhánh giảm **là thật ở cả ba nguồn** nhưng biên độ thì chưa
  chốt được. ρ=8.0 đang chạy để xác nhận.
- Chỉ đo trên **t5p**. §7 đã ghi ρ phụ thuộc backbone, nên KHÔNG suy ra ρ=2.0 là
  tối ưu cho codebert/unixcoder.
- Trục cũ trong ghi chú `run/asam4.sh` (đọc từ `results/sw_t5p`, n=3, một nguồn)
  cho ΔAUC **tăng đơn điệu tới +0.0572 ở ρ=2.0**. Trục mới n=15 xác nhận hướng
  nhưng biên độ nhỏ hơn ~6 lần. Lại là một ví dụ n=3 phóng đại (CLAUDE.md §1).

### Vì sao 4 ô ρ=4.0 bị bỏ, và cách sửa

`asam4.sh` chạy đủ 5 fold trên `ntat`, nhưng đối chứng ρ=0 trên `ntat` chỉ có
fold 1,2,4 (fold 3,5 nằm ở `ntat2`). Ghép cặp trong-máy nên 4 ô (4cwe/com ×
fold 3,5) không có cặp → `report2.py` báo "BỎ QUA 4 ô". Sửa: thêm
`r0|recadam|--sam_rho 0` vào CONFIGS của `asam5.sh`; `matrix.sh:371` bỏ qua ô đã
có nên nó chỉ chạy bù đúng 6 ô còn thiếu, và mở khoá ghép cặp cho cả ρ=4.0 lẫn
ρ=8.0.

### Lỗi công cụ đã trả giá ở khối này

`run/asam5.sh` bản đầu có **dấu nháy lệch** trong một dòng `echo`:

```bash
echo "  truc rho: 8.0 — neu van chua tut thi "cang lon cang tot" va phai noi ro dieu do
```

Dấu `"` thứ hai **đóng** chuỗi, phần sau thành lệnh, và dấu `"` ở **dòng kế tiếp**
mở chuỗi mới → cả vòng `for` bị nuốt. `bash -n` vẫn báo **OK** vì một dấu nháy
phía dưới cân lại. Chạy thật thì `recadam: command not found`,
`CONFIGS: unbound variable`, **0 ô sinh ra** — mà driver worklist vẫn ghi vào
`worklist.done`. Bắt được trước khi nó tới lượt.

**Quy tắc rút ra:** `bash -n` KHÔNG đủ để tin một runner. Phải chạy thử với lời
gọi huấn luyện thay bằng `echo` và **đếm số lần gọi**, đúng cả hai chiều: bản
hỏng phải cho 0, bản đúng phải cho đúng số ô kỳ vọng.

### §21.1 — Lợi ích của ρ=2.0 rơi vào ĐÂU: tách theo nhóm rò rỉ (09/09, không tốn GPU)

Bộ `sven_python_folds_norm` chia theo từng dòng nên ~15–18% hàng test có **bản đối
nghịch gần trùng nằm trong TRAIN**. `tools/leak_groups.py` đã gán nhãn nhóm cho
từng hàng; các ô ASAM có `test_probabilities` nên tách được **mà không chạy lại gì**.

ρ=2.0 − ρ=0, cùng 15 cặp:

| nhóm | % hàng | ΔF1@0.5 | ΔROC-AUC |
|---|---|---|---|
| `train` (có bản gần trùng trong TRAIN) | ~16% | **+0.0509 (11/15)** | +0.0148 (10/15) |
| `none` (không có bản gần trùng) | **~73%** | +0.0071 (11/15) | +0.0075 (11/15) |
| `test` (bản gần trùng nằm trong TEST) | ~5% | −0.0175 (2/9) | −0.0131 (2/9) |

Ghép lại theo trọng số số hàng cho ≈ +0.0129, khớp với Δ tổng +0.0124 — phân rã
nhất quán, không phải ba con số rời rạc.

**Phải nêu kèm mọi lần trích §21:** con số F1 tổng +0.0124 **chủ yếu đến từ nhóm
dễ học vẹt**. Trên 73% hàng sạch, lợi ích F1 chỉ còn **+0.0071 — dưới sàn nhiễu
0.010** — dù 11/15 fold cùng dấu ở *cả hai* chỉ số. Lợi ích **ROC-AUC thì trải đều
hơn** (+0.0148 nhóm `train` so với +0.0075 nhóm `none`), tức ASAM cải thiện **xếp
hạng** trên cả hàng sạch, chỉ là biên độ nhỏ.

Ở ρ=1.0 bức tranh cùng hướng nhưng yếu hơn: `train` +0.0264 F1, `none` −0.0013 F1
/ +0.0069 ROC, và `test` **−0.0170 ROC (1/9, p=0.039)** — tức ở nhóm khó nhất thì
ρ>0 làm **xấu đi**.

Nhóm `test` chỉ 4–12 hàng mỗi fold nên n=9 và phương sai lớn; đừng đọc nó thành
kết luận. Nhóm `none` ~112 hàng/fold thì đọc được.

**Câu hỏi tự nhiên tiếp theo** (CHƯA chạy, cần người dùng quyết vì `twin` là tập
phụ theo CLAUDE.md §6): chạy ρ ∈ {0, 2.0} trên `data/sven_python_twin` — tập gom
cụm gần trùng rồi mới chia, nên không có nhóm `train` để hưởng lợi. Nếu ρ=2.0 vẫn
dương ở đó thì kết luận vững hẳn; nếu về 0 thì §21 phải phát biểu lại.

---

### §21.2 — ρ=8.0 chốt nhánh giảm (09/09, n=9)

ρ=8.0 cho **0/9 fold dương trên cả bốn chỉ số, p=0.004**, ΔROC-AUC −0.3006. Tách
nguồn: 4cwe −0.3078 (0/5), com −0.2915 (0/4). Kể cả `4cwe` — nguồn gần như không
hề hấn ở ρ=4.0 (−0.0094) — cũng sập ở ρ=8.0.

Trục ρ đầy đủ trên ROC-AUC:

```
ρ      0.1     0.2     0.5     1.0     2.0     4.0      8.0
n       15      15      15      15      15       9        9
ROC  −0.008  −0.002  +0.003  +0.007  +0.009  −0.140   −0.301
```

**Đơn đỉnh, hai phía đều chốt.** Nhánh giảm không còn là "một ô ngoại lệ": hai mức
liên tiếp (4.0 và 8.0) đều âm, và ở 8.0 thì 0/9 fold trên cả bốn chỉ số.

Nhưng đỉnh thì **nhỏ** và đã bị §21.1 hạn định: trên 73% hàng không rò rỉ, lợi ích
ở ρ=2.0 chỉ +0.0075 ROC / +0.0071 F1, dưới sàn nhiễu 0.010. Phát biểu an toàn nhất:
**ρ ∈ [1, 2] là vùng an toàn và ρ ≥ 4 phá mô hình**; còn "ASAM có đóng góp dương"
thì chỉ đúng ở mức xếp hạng và biên độ nhỏ.

### §21.3 — Lưới nguồn trên codebert (n=5): hai backbone BẤT ĐỒNG về hình dạng

Cùng lưới `pur*`, cùng máy ntat2, đối chứng `pur100_n930` của chính backbone đó:

| nhánh | codebert ΔROC | t5p ΔROC | codebert ΔPR | t5p ΔPR |
|---|---|---|---|---|
| pur75 | −0.0015 (2/5) | +0.0118 (4/5) | **+0.0282** (4/5) | +0.0018 (3/5) |
| pur50 | −0.0086 (1/5) | −0.0301 (0/5) | +0.0105 (2/5) | −0.0396 (0/5) |
| pur25 | −0.0211 (0/5) | −0.0007 (1/5) | +0.0128 (2/5) | −0.0002 (2/5) |
| pur12 | −0.0294 (0/5) | −0.0082 (2/5) | −0.0211 (2/5) | −0.0109 (1/5) |

Trên codebert, ROC-AUC giảm **đơn điệu** theo pha loãng (0/5 ở pur25 và pur12);
trên t5p thì gấp khúc. Hai backbone chỉ **thống nhất ở đầu pha loãng nặng**
(pur12 âm ở cả hai trên ROC và F1).

Thêm một cảnh báo bốn-chỉ-số: trên codebert **PR-AUC đi NGƯỢC ROC-AUC** ở
pur75/50/25 (dương trong khi ROC âm). Dữ liệu đích cân bằng 50/50 nên hai chỉ số
này thường đồng thuận — chỗ chúng tách nhau là chỗ phải cẩn thận, chưa đọc được
ở n=5.

Sự bất đồng này **ủng hộ** lập luận ở RESEARCH §B.1: lưới `pur*` không đo một
biến nào cả (pha loãng kéo tỉ lệ js sập theo), nên hình dạng của nó không có lý do
gì phải giống nhau giữa hai backbone. Phép đo sạch là lưới `lm*` — đang chạy trên
codebert (`pool_cb_lm`).

---

## §22 — Đọc lại phát biểu CHÍNH của dự án (head vs `none`) bằng CẢ BỐN chỉ số (09/09/2026)

CLAUDE.md §7 chốt: "**AdamW**: head ăn về điểm — Δ vs `none` +0.0111, 33/43 fold,
p=0.0006. Ô duy nhất trong toàn lưới vừa qua vừa p<0.05 vừa vượt sàn nhiễu."

Con số đó tính trên **macro-F1 và chỉ macro-F1** — đúng cách đã gây lỗi ASAM suốt
ba tuần (§2b). Đọc lại từ 213 cặp ghép được trên đĩa, ghép cặp theo
(cây, nhánh-nguồn, optimizer, seed, fold). **Không cần chạy lại gì.**

| optimizer | n | ΔF1@0.5 | ΔF1@val | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|---|
| **AdamW** | 43 | +0.0103 32/43 **p=0.002** | +0.0122 34/43 **p=0.0003** | +0.0053 27/43 p=0.126 | +0.0033 27/43 p=0.126 |
| RecAdam | 170 | +0.0113 83/170 p=0.82 | +0.0109 80/170 p=0.49 | +0.0110 89/170 p=0.59 | +0.0075 81/170 p=0.59 |
| gộp | 213 | +0.0111 115/213 p=0.27 | +0.0112 114/213 p=0.34 | +0.0098 116/213 p=0.22 | +0.0067 108/213 p=0.89 |

### Ba điều phải sửa vào cách phát biểu

1. **Head cải thiện QUYẾT ĐỊNH, không cải thiện XẾP HẠNG** (dưới AdamW). ΔF1 là
   +0.010…+0.012 với p≤0.002; ΔAUC chỉ bằng một nửa và **không có ý nghĩa**
   (27/43, p=0.126). Đây là **ngược hẳn** với ASAM (§21: AUC có, F1 không). Hai
   kỹ thuật **bổ trợ nhau chứ không trùng** — nói được điều đó là nhờ đọc bốn chỉ số.
2. **RecAdam có TRUNG BÌNH gần y hệt AdamW (+0.0113 vs +0.0103) nhưng đếm dấu là
   tung đồng xu** (83/170, p=0.82). Đây là minh hoạ sạch nhất cho lý do `report2.py`
   luôn in `+/n` cạnh trung bình: hai con số trung bình bằng nhau, hai kết luận trái ngược.
3. **Gộp hai optimizer thì hiệu ứng biến mất** ở cả bốn chỉ số. Vì thế §7 bắt buộc
   nêu cả hai optimizer là đúng, và mọi bảng gộp đều sai.

### Tách theo backbone (AdamW) — hiệu ứng KHÔNG đồng nhất

| backbone | n | ΔF1@0.5 | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|
| unixcoder | 15 | +0.0131 12/15 p=0.04 | **+0.0094 14/15 p=0.001** | **+0.0136 13/15 p=0.01** |
| t5p | 15 | +0.0099 12/15 p=0.04 | +0.0027 7/15 p=1.00 | −0.0006 9/15 p=0.61 |
| codebert | 10 | +0.0133 7/10 p=0.34 | +0.0085 6/10 p=0.75 | −0.0024 4/10 p=0.75 |
| t5pe | 3 | −0.0116 1/3 | −0.0133 0/3 | −0.0103 1/3 |

**Chỉ `unixcoder` được lợi trên cả bốn chỉ số.** Trên `t5p` và `codebert`, lợi ích
nằm hoàn toàn ở F1.

### Tách theo nguồn (AdamW) — và một phát hiện CỦNG CỐ lập luận chống sập

| nguồn | n | ΔF1@0.5 | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|
| 4cwe | 15 | **+0.0227 13/15 p=0.01** | +0.0107 9/15 p=0.61 | +0.0067 10/15 p=0.30 |
| **full** | 10 | +0.0099 8/10 p=0.11 | **+0.0118 9/10 p=0.02** | **+0.0127 9/10 p=0.02** |
| com | 15 | +0.0026 10/15 p=0.30 | −0.0008 9/15 p=0.61 | −0.0037 7/15 p=1.00 |

Hai nguồn cho hiệu ứng ở **hai chỉ số khác nhau**:
- Trên `4cwe` head ăn ở **F1** (+0.0227, 13/15) nhưng không ở AUC.
- Trên `full` — nguồn khó nhất, chỗ Phase 1 hay sập — head ăn ở **AUC**
  (+0.0118/+0.0127, 9/10, p=0.02) chứ không ở F1.

Điều này **củng cố** lập luận của §7 rằng giá trị chắc nhất của head là **chống sập
Phase 1**, bằng đúng chỉ số §7 chưa bao giờ xem: ở nguồn khó, head giữ được **thứ
hạng điểm** — dấu hiệu biểu diễn không sụp — chứ không phải đẩy điểm qua ngưỡng.

### Giới hạn

213 cặp gộp qua nhiều khối và nhiều λ; khoá ghép cặp có `(cây, nhánh-nguồn,
optimizer, seed, fold)` nên λ khác nhau vẫn nằm chung một dòng. Các fold **không
độc lập** (dùng lại cùng bộ fold đích), nên p lạc quan; phần chắc là **số fold
cùng dấu**. Nhóm `t5pe` chỉ n=3, đừng đọc.

---

## §23 — TRANSFER CỨU HAI LỚP CWE HIẾM: hiệu ứng dồn đúng chỗ đích yếu (09/09/2026)

**Đọc lại 507 cặp đã có trên đĩa. Không chạy lại ô nào.** Công cụ: `tools/percwe.py`.

### Vì sao nhìn theo CWE

Tập đích Python **lệch nặng**, và nguồn Phase 1 giàu đúng chỗ đích nghèo:

| CWE | đích: hàng test (5 fold) | đích: tỉ lệ | nguồn `4cwe` | baseline macro-F1 |
|---|---|---|---|---|
| CWE-089 | 408 | 54% | 46 | **0.9217** |
| CWE-078 | 204 | 27% | 100 | 0.7782 |
| **CWE-079** | 82 | **11%** | **692** | 0.5780 |
| **CWE-022** | 66 | **9%** | 92 | **0.3032** |

Huấn luyện chỉ trên đích cho một mô hình **rất tốt ở lớp phổ biến và hỏng ở lớp
hiếm**: CWE-022 đạt macro-F1 **0.3032** — thấp hơn cả đoán bừa.

### Δ (latent_bottleneck − baseline) theo từng CWE, 507 cặp

| CWE | ΔF1 | fold cùng dấu | p | ΔROC-AUC | fold cùng dấu |
|---|---|---|---|---|---|
| **CWE-022** | **+0.1611** | 396/507 | <0.001 | **+0.1688** | 58/78 |
| **CWE-079** | **+0.1653** | 430/507 | <0.001 | **+0.1758** | 67/78 |
| CWE-078 | −0.0217 | 179/507 | <0.001 | −0.0459 | 12/78 |
| CWE-089 | −0.0236 | 217/507 | 0.001 | −0.0003 | 43/78 |

**Lợi ích dồn trọn vào hai lớp hiếm**, biên độ +0.16…+0.18 — gấp **16 lần** sàn
nhiễu 0.010. Hai lớp phổ biến phẳng hoặc hơi âm.

### Không phụ thuộc backbone — cả bốn đều dương ở cả hai lớp hiếm

| backbone | Δ CWE-022 | Δ CWE-079 |
|---|---|---|
| codebert | +0.2408 (72/75) | **+0.2609 (75/75)** |
| t5p | +0.1564 (260/348) | +0.1482 (297/348) |
| unixcoder | +0.1010 (56/75) | +0.1461 (51/75) |
| t5pe | +0.1787 (8/9) | +0.1894 (7/9) |

### Không phụ thuộc nguồn — cả ba đều dương, p<0.001

| nguồn | Δ CWE-022 | Δ CWE-079 |
|---|---|---|
| 4cwe | +0.1425 (150/198) | +0.1981 (181/198) |
| com | +0.2080 (114/131) | +0.1523 (103/131) |
| full | +0.1889 (94/109) | +0.1644 (97/109) |

Kể cả `full` — 7 598 dòng, 123 CWE, chủ yếu ccpp — cũng cho hiệu ứng. Vậy **không
phải** do đã chọn sẵn một nguồn khớp 4 CWE của đích.

### Đây là TRANSFER, không phải cái head

Nhánh `none` (Phase 1 **không** head, 225 cặp) cho **+0.1492 CWE-022 (190/225)** và
**+0.1799 CWE-079 (188/225)**. Hiệu ứng đến từ bản thân việc học Phase 1 trên nguồn,
head không cần thiết cho nó.

### Sống qua phép tách nhóm rò rỉ

| CWE | nhóm SẠCH (~73% hàng) ΔF1 |
|---|---|
| CWE-022 | +0.1196 (28/44, p=0.096) |
| **CWE-079** | **+0.1213 (43/58, p<0.001)** |
| CWE-078 | −0.0302 (21/78) |
| CWE-089 | +0.0022 (31/78) |

Và hai lớp hiếm **không đủ 8 hàng rò rỉ mỗi fold để lập nhóm** — nên lợi ích của
chúng *không thể* là hiện vật rò rỉ. Đây là điểm khác căn bản so với §21.1 (ASAM),
nơi lợi ích phần lớn nằm ở nhóm rò rỉ.

### Vì sao con số TỔNG chỉ +0.02

Vì trung bình bị CWE-089 (54% hàng, baseline đã 0.92, không còn chỗ tăng) và
CWE-078 (27%) chi phối. Hai lớp có hiệu ứng lớn chỉ chiếm **20% hàng**. Báo cáo
**chỉ bằng macro-F1 tổng là giấu mất phát hiện chính** — cùng họ với lỗi ở §2b.

### Cơ chế: baseline ĐOÁN NGƯỢC ở CWE-022, transfer kéo nó về mức ngẫu nhiên

| CWE | | recall | precision | F1 dương | accuracy |
|---|---|---|---|---|---|
| **CWE-022** | baseline | 0.384 | 0.328 | 0.333 | **0.338** |
| | transfer | 0.477 | 0.474 | 0.447 | **0.501** |
| | Δ | +0.092 | **+0.146** | +0.114 | +0.163 |
| **CWE-079** | baseline | 0.621 | 0.569 | 0.570 | 0.561 |
| | transfer | 0.716 | 0.724 | 0.708 | **0.721** |
| | Δ | +0.096 | **+0.155** | +0.138 | +0.160 |
| CWE-078 | Δ | −0.007 | −0.032 | −0.018 | −0.020 |
| CWE-089 | Δ | −0.019 | −0.009 | −0.016 | −0.020 |

Tập con CWE-022 **cân bằng đúng 33/33** qua 5 fold, nên accuracy **0.338** của
baseline là **thấp hơn đoán bừa**. Mô hình chỉ huấn luyện trên đích không "bỏ qua"
lớp này — nó **đoán ngược**: nó đã học một quy tắc phản tác dụng với path traversal.

**Phải phát biểu đúng mức cho từng lớp:**

- **CWE-022**: transfer kéo từ **đoán ngược (0.338) về mức ngẫu nhiên (0.501)**.
  Đây là *gỡ một lỗi hệ thống*, KHÔNG phải "dò được path traversal". Nói quá lên
  là chỗ reviewer sẽ bắt ngay.
- **CWE-079**: 0.561 → **0.721**. Đây mới là biến một mô hình gần-ngẫu-nhiên thành
  một bộ dò dùng được.
- Đổi lại, transfer **mất ~0.02** ở hai lớp đích vốn đã làm tốt (078: 0.772→0.752;
  089: 0.933→0.913).

**Precision tăng nhiều hơn recall** ở cả hai lớp hiếm (+0.146 / +0.155 so với
+0.092 / +0.096), nên đây không phải hiệu ứng "đoán dương nhiều hơn" — khả năng
phân biệt thật sự tăng, cả hai vế cùng lên.

### Giới hạn

- CWE-022 trên hàng sạch p=0.096 (28/44) — dương nhưng chưa dưới 0.05. CWE-079 thì chắc.
- 507 cặp gộp qua nhiều khối/λ/optimizer; các fold **không độc lập** nên p lạc quan.
  Phần chắc là **số fold cùng dấu** và việc nó lặp trên 4 backbone × 3 nguồn.
- CWE-022 chỉ 66 hàng test qua cả 5 fold (13 hàng/fold) — phương sai lớn.
- Chưa tách được ảnh hưởng của độ tinh khiết nguồn (§B.2e) trong lát cắt này:
  `com` cho CWE-022 cao nhất (+0.2080) dù kém tinh khiết hơn `4cwe`.

### §23.1 — `cwe_class` của `com`/`full` là PILLAR, không phải từng CWE (đo 09/09)

Không phải lỗi dữ liệu — là thiết kế. 123 CWE của `full` gộp thành **10 lớp theo
pillar** của CWE-1000:

| class | pillar | CWE tiêu biểu |
|---|---|---|
| 0 | CWE-284 Improper Access Control | 269–295, 732, 798, 862, 863 |
| **2** | CWE-664 Improper Control of a Resource | **22**, 59, 119–125, 401, 415, 416, 787 |
| 3 | CWE-682 Incorrect Calculation | 190, 191, 193, 369 |
| 4 | CWE-691 Control Flow Management | 362, 617, 670, 834, 835 |
| 5 | CWE-693 Protection Mechanism Failure | 311, 326, 327, 338, 352 |
| 7 | CWE-703 Improper Check of Exceptional Conditions | 252, 476, 754, 755 |
| **8** | CWE-707 Improper Neutralization | 20, 74, 77, **78**, **79**, 88, **89**, 116 |

Chỉ `4cwe` mới dùng vocab `fixed4` với đúng 4 lớp = 4 CWE.

**Hệ quả cho §23:** CWE-022 (pillar 2) và CWE-079 (pillar 8) nằm ở **hai pillar
khác nhau** mà cùng được lợi lớn; CWE-089 **cùng pillar 8 với 079** và nguồn có 692
dòng pillar-8, nhưng **không** được lợi. Vậy pillar cũng không dự báo được lợi ích —
chỉ **độ yếu của baseline** dự báo được (Pearson −0.886).

**Hệ quả vận hành:** mọi phép so cắt nguồn theo nhóm CWE phải chạy nhánh **`none`**,
vì head phụ sẽ thấy số pillar khác nhau ở hai nhánh và phép so đổi hai biến.

---

## §24 — ASAM ρ=2.0 MẠNH HƠN khi BỎ neo RecAdam (09/09/2026, đang chạy)

> **CHỐT 04:30 09/09 — n=15 đã đủ trên ntat, hai máy đồng thuận.**
>
> | máy | n | F1@0.5 | F1@val | ROC-AUC | PR-AUC |
> |---|---|---|---|---|---|
> | **ntat (đủ 5 fold × 3 nguồn)** | **15** | +0.0161 (10/15, p=0.30) | +0.0176 (11/15, p=0.12) | **+0.0200 (13/15, p=0.007)** | **+0.0318 (15/15, p<0.001)** |
> | ntat2 (fold 1–3, độc lập) | 9 | +0.0164 (4/9) | +0.0152 (4/9) | +0.0135 (7/9) | +0.0216 (7/9) |
>
> Trên ntat, **cả ba nguồn** dương trên **cả bốn** chỉ số, và PR-AUC đạt **5/5 fold ở từng nguồn**
> (4cwe +0.0328, com +0.0290, full +0.0335). Máy thứ hai độc lập cùng dấu trên cả bốn.
>
> **Phát biểu đúng — hiệu ứng ở XẾP HẠNG, không ở quyết định tại ngưỡng 0.5.** F1@0.5 chỉ 10/15
> (p=0.30) và F1@val 11/15 (p=0.12); ntat2 còn cho F1 chỉ 4/9. Đúng y hệt kết luận §2b về ASAM.
> Viết là *"ASAM ρ=2.0 trên nền AdamW cải thiện ROC-AUC và PR-AUC"*, **không** phải *"cải thiện F1"*.
>
> So với nền RecAdam (§21, cùng ρ=2.0): ROC +0.0088 / PR +0.0155. **Bỏ neo thì hiệu ứng mạnh hơn
> ~2,3× trên ROC và ~2,1× trên PR** — neo RecAdam kìm ASAM chứ không phối hợp với nó.
>
> _(Ghi lúc 02:56: bất đồng hai máy — ntat2 khi đó null ở n=6. Đã giải khi ntat2 lên n=9.)_

Trục ρ ở §21 đo **trên nền RecAdam**. Nhưng RecAdam null ở mọi γ (§20), nên cấu
hình chốt sẽ dùng AdamW. Khối `asam_aw` đo cùng ρ=2.0 nhưng **Phase 2 dùng AdamW,
không neo**; đối chứng `aw_r0` = AdamW ρ=0, cùng máy cùng phiên.

**Bậc 1 (n=3 fold × 3 nguồn = 9 cặp), máy `ntat`:**

| nền Phase 2 | n | ΔF1@0.5 | ΔF1@val | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|---|
| **AdamW** | 9 | +0.0246 (7/9) | +0.0272 (7/9) | **+0.0260 (8/9, p=0.039)** | **+0.0379 (9/9, p=0.004)** |
| RecAdam (§21) | 15 | +0.0124 (10/15) | +0.0136 (10/15) | +0.0088 (11/15) | +0.0155 (13/15) |

Tách nguồn dưới AdamW — **cả ba đều dương trên cả bốn chỉ số**:
4cwe +0.0394 ROC (3/3) · com +0.0129 (2/3) · full +0.0258 (3/3).

**Đọc được gì:** ASAM ρ=2.0 cho hiệu ứng **gấp ~3 lần** khi bỏ neo (ROC +0.0260 so
với +0.0088; PR +0.0379 so với +0.0155). Neo RecAdam **kìm** ASAM chứ không phối
hợp với nó. Nếu đứng vững, cấu hình chốt là **AdamW + ASAM ρ=2.0**, và §21 phải
được đọc như một *cận dưới* bị neo làm hụt.

### CHƯA CHỐT — máy thứ hai chưa xác nhận

`ntat2` chạy đúng khối này song song và ở n=6 cho **null**: ΔROC +0.0004 (4/6),
ΔF1 −0.0081 (2/6). Hai máy chưa hợp nhau. Đây đúng kiểu bất đồng đã thấy ở lưới
`lm` (§B.2e) nơi máy thứ hai lật dấu ở mức pha loãng nhẹ.

`ntat` đang chạy nốt fold 4–5 để lên n=15 trên máy có hiệu ứng; `ntat2` vẫn đang
chạy fold 1–3. Chưa được trích §24 khi chưa có cả hai.

### Lỗi vận hành ghi kèm

Khi `asam_aw` xong, driver ntat chạy tiếp `pool_lm.sh` và chết ngay:
`bash: run/pool1.sh: No such file or directory` — **`pool1.sh` chưa từng được đẩy
lên ntat**, dù `pool_lm.sh`/`pool_pur.sh` (chỉ là wrapper `exec bash run/pool1.sh`)
thì có. Cùng đợt, `e60_cb.sh` lại được ghi `done` với **0 ô** vì máy này cũng thiếu
checkpoint Pha 1 codebert. Cả hai đều bị driver **cũ** xử lý (phóng lúc 20:58,
trước bản vá đếm hiện vật), nên không có cảnh báo nào. Máy trống ~3 phút.

**Quy tắc rút ra:** đẩy một wrapper mà không đẩy thứ nó `exec` là một lỗi im lặng
nữa. Trước khi xếp một script vào worklist của máy xa, phải kiểm **mọi file nó gọi
đến** đã có trên máy đó chưa.

---

## §25 — TRỘN ĐỀU baseline ⊕ chuyển giao: vượt CẢ HAI đầu mút (09/09/2026, 0 GPU)

**Đây là câu trả lời cho yêu cầu 09/09 của người dùng**: một phương thức *không phụ thuộc
backbone*, *không quét siêu tham số*, chứng minh chuyển giao có lợi cho đích cùng miền nhưng
yếu, và nâng riêng hai CWE hiếm 022/079.

### Phép đo

`tools/ensemble2.py`. Với mỗi ô đã có, trộn xác suất từng mẫu:
`p(α) = (1−α)·p_baseline + α·p_chuyển_giao`, rồi chấm lại. **Không tốn GPU** — mọi ô đã lưu
`test_probabilities`, `test_labels`, `test_cwe_classes`.

Đơn vị độc lập là **KHỐI** `(cây kết quả, run, seed, fold)`, không phải ô: 536 nhánh chia nhau
83 baseline, nên một baseline yếu kéo cả chùm nhánh của nó cùng dấu. Trung bình các nhánh
**trong** một khối trước, rồi mới đếm dấu trên **83 khối**. Đếm trên 536 ô là thổi phồng.

Lọc: `latent_bottleneck` (cấu hình đã chốt §7), Pha 1 val ≥ 0.40, bỏ các nhánh đã biết là hỏng
(`lm12 lm25 pur12 pur25 pur50 r4p0 r8p0 e60`) — giữ chúng lại thì α=1.0 xấu đi vì lý do không
liên quan đến chuyển giao.

### Kết quả — 83 khối, hai backbone

| | F1@0.5 | ROC-AUC | PR-AUC |
|---|---|---|---|
| **α=0.5 vs baseline** | **+0.0311 (72/83, p<1e-4)** | **+0.0126 (65/83, p<1e-4)** | **+0.0136 (67/83, p<1e-4)** |
| α=1.0 (chuyển giao thuần) vs baseline | +0.0163 (54/83, p=0.008) | **−0.0039** (53/83) | **−0.0098** (47/83, p=0.27) |
| **α=0.5 vs α=1.0** (ghép cặp) | +0.0148 (50/83, **p=0.078 — KHÔNG có ý nghĩa**) | **+0.0165 (59/83, p=0.0002)** | **+0.0235 (62/83, p<1e-4)** |

Bản trộn **vượt cả hai đầu mút** trên hai chỉ số xếp hạng. Trên F1@0.5 nó vượt baseline nhưng
**không** vượt được chuyển giao thuần một cách có ý nghĩa — phải nêu, không được gộp.

### Vì sao điều đó là một LẬP LUẬN, không chỉ một con số

Phản biện lớn nhất của nhánh này là *"finetune hai lần thì tất nhiên hơn một lần"*. Nếu Pha 1
chỉ là "huấn luyện thêm", hai mô hình **thừa** nhau và điểm bản trộn phải nằm **giữa** hai đầu
mút. Nó nằm **trên cả hai** ⇒ hai mô hình sai ở **chỗ khác nhau** ⇒ Pha 1 đưa vào thông tin mà
mô hình chỉ-đích không có. Đó là bằng chứng trực tiếp chống lại phản biện đó.

### Không phụ thuộc backbone

| backbone | khối | ΔF1@0.5 | ΔROC | ΔPR | ΔROC vs α=1 |
|---|---|---|---|---|---|
| codet5p | 68 | +0.0319 (58/68) | +0.0126 (53/68) | +0.0143 (56/68) | +0.0172 (46/68, p=0.005) |
| codebert | 15 | +0.0275 (14/15) | +0.0129 (12/15) | +0.0106 (11/15) | +0.0132 (13/15, p=0.007) |

Hai backbone cho **cùng biên độ**, sai khác dưới sàn nhiễu 0.010.

### Cơ chế: bản trộn giữ phần được, bỏ phần mất (ΔROC-AUC theo CWE)

| CWE | tỉ lệ test | α=1.0 (chuyển giao thuần) | **α=0.5 (trộn đều)** |
|---|---|---|---|
| **022** | 9% | +0.2275 (75/83) | **+0.0967 (74/82, p<1e-4)** |
| **079** | 11% | +0.2379 (81/83) | **+0.1513 (81/83, p<1e-4)** |
| 078 | 27% | **−0.0288 (17/83, p<1e-4 — hại có ý nghĩa)** | −0.0014 (40/81, **p=1.00 — đúng null**) |
| 089 | 54% | −0.0213 (31/83) | **+0.0036 (56/83, p=0.0019 — lợi**) |

Chuyển giao thuần **hại có ý nghĩa** trên CWE-078 và hại trên 089 — hai lớp chiếm 81% hàng test,
nên chúng nuốt hết phần được ở hai lớp hiếm và làm ΔROC tổng thành âm. Trộn đều **xoá sạch phần
hại** (078 về đúng null, 089 thành lợi) mà vẫn giữ 43% phần được ở CWE-022 và 64% ở CWE-079.
Đó là lý do ΔROC tổng lật từ −0.0039 sang +0.0126.

### Hai phép kiểm ổn định (thêm 09/09 04:1x)

**(a) Không cây kết quả nào chi phối.** 87 khối trải trên **~20 cặp `(cây, run)` khác nhau**; cây
đóng góp nhiều nhất chỉ có **5 khối** (5 fold × 1 seed). Không có chuyện một khối chạy lớn kéo cả
kết luận. Danh sách đầu bảng: `results/pool1_t5p` 5, `results_asam1_ntat/.` 5,
`results_asam1_ntat/asam1_t5p` 5, `results_asam1_ntat/e60_t5p` 5, `results_asam1_ntat2/pool1_t5p`
5, `results_asam1_ntat2/poolcb_codebert` 5, `results_asamcb_158/.` 5, `results_e60_ntat/.` 5,
`results_pool1_ntat2/.` 5, `results_asamaw_ntat/.` 4… (`tools/ensemble2.py --by-tree`).

**(b) Ổn định khi thêm dữ liệu.** Đo lại sau khi ntat sinh thêm ô của `asam_aw` fold 4–5, số khối
83 → 87: `trộn − chuyển giao thuần` trên ROC đi từ +0.0165 (59/83, p=0.0002) sang **+0.0156
(61/87, p=0.0002)**; PR giữ nguyên +0.0235 → +0.0224 (63/87). Biên độ nhích nhẹ, đếm dấu và p
không đổi hạng.

### Ba giới hạn phải nêu

1. **α=0.5 là lựa chọn KHÔNG THAM SỐ khai báo trước** (hai mô hình một phiếu ngang nhau), không
   phải giá trị dò trên test. Đường α đầy đủ chỉ để chẩn đoán hình dạng. Không ô cũ nào lưu xác
   suất trên **val** nên không chọn α trên val được ngoài tuyến — đã vá `src/train_transfer.py`
   và `src/train_baseline.py` ghi thêm `val_probabilities`/`val_labels` từ 09/09, mọi ô mới
   chọn được α trên val.
2. **F1@ngưỡng-val không tính được ngoài tuyến** vì ngưỡng của bản trộn phải hiệu chỉnh trên val.
   Bảng trên chỉ có ba chỉ số. Bịa ngưỡng từ test là rò rỉ — không làm.
3. **Chưa loại được phản biện "trộn hai mô hình nào cũng lợi"** (trung bình hoá phương sai).
   Không kiểm ngoài tuyến được: 21 cặp baseline đa-seed duy nhất trong kho là của khối 47, chạy
   trước khi lưu xác suất từng mẫu. `run/ensctl.sh` đã xếp vào worklist ntat: baseline seed 7 và
   1234, 5 fold, ghi vào đúng cây `asamaw_t5p` đã có baseline seed 42 → trộn baseline⊕baseline
   ghép cặp cùng máy cùng fold. **§25 chưa được trích khi chưa có đối chứng đó.**

### Giá phải trả

Suy luận **hai lần** (hai mô hình). `train_transfer.py:1086 sweep_source_interpolation` đã có sẵn
phép nội suy **trọng số** θ_Pha1 ⊕ θ_Pha2 (WiSE-FT, chọn α trên val) — nhưng đó là **cặp khác**.
Cặp ở đây là baseline ⊕ chuyển giao, hai mô hình *cùng tác vụ đích*, và nội suy trọng số của cặp
đó chưa đo. Nếu nó chạy được thì chi phí suy luận về lại một mô hình.

---

## §25.1 — Đối chứng ÂM cho §25: trộn với nguồn PHA LOÃNG thì KHÔNG lợi (09/09/2026, 0 GPU)

Phản biện "trộn hai mô hình nào cũng lợi, đó là trung bình hoá phương sai" **loại được một
phần ngay ngoài tuyến**, không cần đợi `run/ensctl.sh`.

Các cây pool chứa **cả hai loại nhánh trong CÙNG một khối**, so với **CÙNG một baseline**:
nguồn nguyên chất (`pur100_n930`, `lm100_n930`) và nguồn pha loãng (`pur12/25`, `lm12/25` —
§B.2e đo được là âm, 0/10 fold). Cả hai đều là "một mô hình thứ hai" đã finetune trên cùng dữ
liệu đích, nên giả thuyết trung bình-hoá dự đoán **cùng một mức lợi**.

`tools/ens_dilute.py`, 13 khối, ghép cặp trong khối:

| | F1@0.5 | ROC-AUC | PR-AUC |
|---|---|---|---|
| **nguyên chất** · trộn α=0.5 | +0.0240 (12/13) | **+0.0109** (9/13) | **+0.0154** (10/13) |
| **nguyên chất** · thuần α=1.0 | +0.0315 (12/13) | +0.0054 (7/13) | +0.0003 (6/13) |
| **pha loãng** · trộn α=0.5 | +0.0062 (6/13) | **−0.0036** (4/13) | **−0.0004** (4/13) |
| **pha loãng** · thuần α=1.0 | −0.0060 (3/13) | −0.0162 (2/13, p=0.023) | −0.0140 (2/13, p=0.023) |
| **hiệu ghép cặp (nguyên − loãng), trộn** | **+0.0177 (12/13, p=0.0034)** | **+0.0145 (12/13, p=0.0034)** | **+0.0157 (11/13, p=0.023)** |

**Trộn không lợi bừa.** Trộn với mô hình chuyển giao từ nguồn pha loãng cho **−0.0036 ROC**,
tức không lợi gì so với chính baseline. Chỉ khi Pha 1 thấy nguồn nguyên chất thì bản trộn mới
dương. Hiệu ghép cặp trong khối là **+0.0145 ROC, 12/13 khối, p=0.0034**.

**Đọc thêm một tầng.** Ở *cả hai* nhóm, trộn tốt hơn chuyển giao thuần — đó chỉ là nội suy về
phía baseline, không có gì lạ. Điều phân biệt hai giả thuyết là: trộn tốt hơn **baseline** thì
**chỉ xảy ra ở nhóm nguyên chất**. Chú ý riêng cột PR-AUC nhóm nguyên chất: chuyển giao thuần
+0.0003 (đúng bằng không) nhưng bản trộn +0.0154 — mô hình chuyển giao *một mình không hơn gì*
vẫn **đóng góp** được khi trộn. Đó là định nghĩa của bổ trợ, không phải của trung bình hoá.

**Vẫn còn thiếu** đối chứng baseline⊕baseline khác seed (`run/ensctl.sh`, đang xếp trên ntat):
nguồn pha loãng dù sao cũng là mô hình *kém hơn*, nên chưa loại triệt để. Hai đối chứng bổ nhau.

---

## §25.2 — Driver worklist bỏ sót mục và để máy vast nằm không (09/09/2026 02:40)

`scripts/vast_worklist.sh` trên ntat2 in `WORKLIST xong | da chay 8 muc` trong khi danh sách chỉ
có **6** mục, và `run/pool_cb_lm.sh` **chưa bao giờ chạy** mà cũng không nằm trong `worklist.done`.
Máy nằm không cho tới khi phóng lại tay (~2 phút).

**Nguyên nhân**: `log/worklist.txt` bị **ghi đè** trong lúc driver đang giữ nó trên fd 3. Offset
byte của driver rơi vào giữa nội dung mới → đọc ra dòng rác/lặp, đếm 8, rồi gặp EOF và **thoát**.
Đây là họ hàng của bẫy stdin đã sửa hôm qua: cả hai đều làm driver **thoát sớm** chứ không báo lỗi.

**Sửa**: bọc vòng đọc bằng một **vòng ngoài**. Hết một lượt thì **mở lại** worklist và đối chiếu
`todo − done`; còn việc thì chạy lượt nữa. Hai cửa chặn quay vòng vô hạn: hết `MAXPASS_WL` (20),
hoặc một lượt **không chạy được mục nào** (mọi thứ còn lại đều thiếu file) — lúc đó in rõ mục nào
còn và dừng.

**Kiểm cả hai chiều** (bản cũ *phải* hỏng, bản mới *phải* chạy đúng):

| tình huống | bản CŨ | bản MỚI |
|---|---|---|
| danh sách không đổi | done=2, còn 0, 1 lượt | done=2, còn 0, 1 lượt — **y hệt** |
| một mục thiếu file ở lượt 1, file xuất hiện sau | **done=2, CÒN 1, chỉ 2 ô** | **done=3, còn 0, 3 ô, 2 lượt** |
| một mục thiếu file **vĩnh viễn** | — | dừng ở lượt 2, in `con: run/khong_ton_tai.sh`, **không** quay vòng |

Bài học chung: **ghi đè một file mà tiến trình khác đang đọc thì phải coi như tiến trình đó sẽ
đọc ra rác** — nối thêm (`>>`) thì an toàn, ghi đè thì không. Và mọi driver giữ máy tính tiền
phải kết thúc bằng một phép **đối chiếu trạng thái**, không bằng "đã tới EOF".

---

## §25.3 — Hai CWE hiếm được gì ở ĐIỂM VẬN HÀNH, không chỉ ở AUC (09/09/2026, 0 GPU)

§23/§25 phát biểu bằng ROC-AUC. Câu người đọc bài thật sự muốn là *"bắt thêm được bao nhiêu lỗ
hổng"* — tức **recall tại một ngưỡng cụ thể**. `tools/percwe_op.py`, 86 khối, α=0.5, cùng bộ lọc
như §25.

**Ngưỡng 0.5** (không hiệu chỉnh gì, không thể rò rỉ):

| CWE | hàng dương/fold | recall baseline | recall trộn | **Δ recall** | Δ precision |
|---|---|---|---|---|---|
| **022** | 6.7 | 0.359 | 0.432 | **+0.073 (54/63, p<1e-3)** | **+0.107 (68/77)** |
| **079** | 8.2 | 0.519 | 0.665 | **+0.146 (58/73, p<1e-3)** | **+0.170 (83/85)** |
| 078 | 20.8 | 0.779 | 0.789 | +0.010 (42/73, p=0.24) | −0.001 (36/86) |
| 089 | 40.6 | 0.931 | 0.934 | +0.003 (36/70, p=0.91) | +0.016 (52/84) |

**Ngưỡng hiệu chỉnh trên VAL của chính baseline**, dùng chung cho cả ba mô hình — tức *"giữ
nguyên điểm vận hành đang triển khai, chỉ đổi mô hình"*:

| CWE | Δ recall | Δ precision |
|---|---|---|
| **022** | **+0.079 (53/72, p<1e-3)** | +0.079 (49/81) |
| **079** | **+0.072 (49/64, p<1e-3)** | +0.083 (61/72) |
| 078 | +0.011 (38/70, p=0.55) | −0.015 (31/85) |
| 089 | +0.001 (42/69, p=0.09) | −0.002 (32/81) |

**Ba điều đọc được:**

1. **Recall VÀ precision cùng tăng** trên cả hai CWE hiếm, ở cả hai ngưỡng. Nếu bản trộn chỉ
   "đoán dương nhiều hơn" thì recall tăng còn precision phải giảm. Nó không giảm.
2. **Hai CWE thường không bị đụng tới** — Δ recall +0.010/+0.003, đếm dấu là tung đồng xu. Đúng
   với §25: bản trộn không đánh đổi lớp thường lấy lớp hiếm, nó chỉ thêm vào lớp hiếm.
3. Ngưỡng val cho biên độ **nhỏ hơn** ngưỡng 0.5 ở CWE-079 (+0.072 so với +0.146). Phải nêu con
   số nhỏ hơn khi phát biểu, vì ngưỡng val mới là điểm vận hành thật.

**Cảnh báo phải in kèm mọi lần trích:** CWE-079 chỉ có **~8 hàng dương mỗi fold**, CWE-022 **~6.7**.
`+0.146 recall` nghĩa là **bắt thêm khoảng 1,2 lỗ hổng mỗi fold** — nhỏ về tuyệt đối. Thứ làm nó
đáng tin **không phải biên độ** mà là **58/73 khối cùng dấu** (và 83/85 ở precision). Trích biên độ
mà không trích số khối cùng dấu là đọc sai theo đúng kiểu CLAUDE.md mục 2b đã cấm.

---

## §25.4 — §25 SỐNG SÓT phép kiểm rò rỉ, và lợi ích của phép trộn nằm ĐÚNG ở hàng sạch (09/09/2026)

Phép kiểm này đã từng **đánh sập** một kết luận của chính dự án (§21.1: lợi của ASAM ρ=2.0 hoá ra
dồn vào nhóm `train`, còn nhóm `none` chiếm 73% dữ liệu thì dưới sàn nhiễu). Bắt buộc chạy cho §25.

`tools/ens_leak.py`, 86 khối, α=0.5, Δ so với baseline **trên cùng nhóm hàng**:

| nhóm (hàng/fold) | mô hình | F1@0.5 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| **`none`** (~111, 73%) | **trộn** | **+0.0264 (74/86, p<1e-4)** | **+0.0080 (58/86, p=0.0016)** | **+0.0070 (62/86, p=0.0001)** |
| | chuyển giao thuần | +0.0089 (59/86) | **−0.0102 (46/86, p=0.59)** | −0.0159 (42/86) |
| **`test`** (~9) | **trộn** | **+0.0742 (35/52, p<1e-4)** | **+0.0432 (37/52, p=0.0002)** | **+0.0333 (37/52, p=0.0002)** |
| | chuyển giao thuần | +0.0429 (26/52, p=0.29) | +0.0242 (31/52, p=0.21) | +0.0208 (31/52, p=0.21) |
| `train` (~23) | trộn | +0.0569 (58/86) | +0.0289 (57/86) | +0.0287 (52/86) |
| | chuyển giao thuần | **+0.0692 (67/86)** | **+0.0436 (53/86)** | +0.0339 (53/86) |
| `val` (~11) | trộn | +0.0292 (39/66) | +0.0164 (27/66) | +0.0141 (29/66) |
| | chuyển giao thuần | +0.0016 (33/66) | −0.0056 (26/66) | −0.0113 (24/66) |

**Đọc theo hiệu `trộn − chuyển giao thuần` trên ROC-AUC — đây mới là chỗ đáng nhìn:**

| nhóm | `none` | `test` | `val` | **`train`** |
|---|---|---|---|---|
| trộn − chuyển giao | **+0.0182** | **+0.0190** | **+0.0220** | **−0.0147** |

Phép trộn **hơn** chuyển giao thuần ở ba nhóm phải khái quát hoá, và **kém hơn** ở đúng nhóm
`train` — nhóm có bản đối nghịch nằm trong TRAIN, tức nhóm học vẹt được. Nói cách khác: **cái mà
chuyển giao thuần được nhiều nhất lại là nhóm dễ học vẹt nhất**, còn cái mà phép trộn thêm vào
nằm đúng ở hàng sạch. Đây là ngược hẳn với §21.1.

**Nhóm `test` là ô mạnh nhất cho bài**: đó là bài toán của bộ `twin` thu nhỏ — hàng chưa từng
thấy cặp nào tương tự. Trộn cho **+0.0432 ROC (37/52, p=0.0002)**, trên sàn nhiễu 4 lần; chuyển
giao thuần +0.0242 nhưng **không có ý nghĩa** (31/52, p=0.21).

**Phải nêu — biên độ ở nhóm `none` dưới sàn nhiễu.** ROC +0.0080 và PR +0.0070 nằm **dưới** sàn
0.010 (cùng loại GPU). Đếm dấu có ý nghĩa (58/86, 62/86) nhưng biên độ thì không vượt sàn. Chỉ
F1@0.5 (+0.0264) là trên sàn rõ ràng. Phát biểu đúng: *trên hàng sạch, phép trộn dương ổn định về
dấu ở cả ba chỉ số và vượt sàn ở F1; chuyển giao thuần thì âm ở cả hai chỉ số AUC.*

**Lỗi đã sửa trong chính công cụ trước khi trích**: nhãn `val` trong `data/leak_groups.json`
**không** có nghĩa "hàng này thuộc val" mà là "bản đối nghịch của hàng test này nằm trong VAL".
Danh sách có đúng một mục cho mỗi hàng TEST. Bản đầu tôi lọc bỏ `val` → độ dài lệch → **100/100
khối bị bỏ im lặng**, công cụ in bảng rỗng mà không báo lỗi. Đã thêm nhóm `val` thành một nhóm
riêng và đếm số ô bị bỏ ra đầu bảng.

---

## §25.5 — Nội suy TRỌNG SỐ baseline ⊕ chuyển giao: cơ chế đã kiểm, khối đã xếp (09/09/2026)

§25 cho một phương thức nhưng nó tốn **hai lần suy luận**. Nếu trộn được trong không gian
**trọng số** thì chi phí về lại một mô hình — đó là khác biệt giữa một mẹo ensemble và một
phương thức triển khai được. Câu này **chưa ai trong dự án trả lời**.

**Cơ chế đã kiểm trước khi tiêu GPU** (nguyên tắc "đo cơ chế trước"):

1. **Đọc `src/model.py`, 0 GPU**: `BaselineModel` và `TransferModel` chia đúng `backbone.*` +
   `vul_head.{weight,bias}` — cùng tên, cùng shape. `latent_proj.*` và `cwe_head.*` chỉ có ở
   bản chuyển giao và **không nằm trên đường tính `vul_logits`**.
2. **Chạy thật `tools/wblend.py`**: **201 tensor nội suy được, đúng 4 bị bỏ** — và bốn cái đó
   đúng là `latent_proj.{weight,bias}`, `cwe_head.{weight,bias}`. Vậy phép nội suy xác định
   được cho **mọi** tensor có tác dụng lên dự đoán.

**Kiểm công cụ hai chiều** (bắt buộc, theo bài học `bash -n` không đủ):

| kiểm | kỳ vọng | kết quả |
|---|---|---|
| cùng MỘT checkpoint hai đầu | mọi α cho số **y hệt** | α=0/0.5/1 đều F1 0.4818, ROC 0.6431 ✓ |
| HAI checkpoint khác nhau | các α phải **khác nhau** | ROC 0.9765 / 0.9608 / 0.6431 ✓ |

**Kết quả phụ đáng giá từ kiểm 2**: trộn 50/50 hai bản fine-tune **khác nhau** (khác fold, khác
bộ dữ liệu) vẫn ra **mô hình chạy được** — ROC 0.9608, nằm giữa hai đầu mút, không sập về ngẫu
nhiên. Nếu hai bản nằm ở hai lòng chảo khác nhau thì α=0.5 phải sập. Vậy hai bản fine-tune từ
cùng pretrained init của kiến trúc này **nằm trong vùng nối tuyến tính** — điều kiện mà WiSE-FT
cần. Nội suy trọng số là khả thi, không phải bắn mò.

> Con số trong hai phép kiểm trên **KHÔNG có nghĩa khoa học**: hai checkpoint khác fold, khác bộ
> dữ liệu, và chỉ chấm trên 32 hàng (`--max_eval 32`, cờ chỉ dùng để kiểm đường ống). Chúng chỉ
> chứng minh công cụ đúng.

**Đã xếp trên ntat**, sau `ensctl`: `run/wblend.sh|4cwe com full|1 2 3` — bậc 1 (3 fold, seed 42),
α chọn trên **VAL**, báo trên TEST, và in kèm phép trộn **xác suất** α=0.5 trên **cùng cặp** để so
sánh trực tiếp. Khối này cũng sinh ra ô có `val_probabilities` nên gỡ luôn **giới hạn số 2 của
§25** (α chỉ chọn được trên test).

**Thay đổi kèm theo**: `run/matrix.sh` thêm cờ `KEEP_CKPT=1` để giữ checkpoint Pha 2 (mặc định
vẫn **XOÁ** — `model/` đã 41GB và đĩa từng đầy 98% làm cụt `torch.save`, §16). `run/wblend.sh`
dọn checkpoint **ngay sau mỗi fold** và từ chối chạy nếu đĩa dưới 12GB. Cổng đã kiểm hai chiều.

---

## §25.6 — Phép trộn có chỉ CỨU bản chuyển giao yếu không? Chia theo tiêu chí ĐỘC LẬP (09/09/2026)

Nếu lợi của phép trộn chỉ là kéo một mô hình kém về phía baseline thì nó là **chính quy hoá**,
không phải **bổ trợ**. Cách kiểm tầm thường — chia khối theo Δ của chính nhánh chuyển giao — **bị
lệch**: chọn nhóm theo x rồi đo y trên **cùng** dữ liệu test là hồi quy về trung bình, nhóm x<0
sẽ tự động cho y>0. Phải chia theo đại lượng **độc lập với tập test đích**.

### Chia theo `phase1_val_macro_f1` (đo trên tập val của chính Pha 1), trung vị 0.5897

| nhóm | | ROC-AUC | PR-AUC |
|---|---|---|---|
| **Pha 1 val TRÊN trung vị** (87 khối) | trộn vs baseline | +0.0128 (69/87, p<1e-4) | +0.0130 (72/87) |
| | chuyển giao thuần vs baseline | −0.0027 (57/87) | −0.0094 (52/87) |
| | **trộn − thuần** | **+0.0154 (58/87, p=0.0025)** | **+0.0224 (62/87, p=0.0001)** |
| **Pha 1 val DƯỚI trung vị** (57 khối) | trộn vs baseline | +0.0180 (44/57, p<1e-4) | +0.0201 (50/57) |
| | chuyển giao thuần vs baseline | +0.0129 (37/57, p=0.033) | +0.0102 (41/57) |
| | **trộn − thuần** | **+0.0051 (30/57, p=0.79 — KHÔNG có ý nghĩa)** | +0.0099 (35/57, p=0.11) |

**Ngược hẳn với giả thuyết "chỉ cứu bản yếu".** Phần phép trộn **thêm** vào có ý nghĩa ở nhóm Pha 1
**mạnh** và **không** có ý nghĩa ở nhóm yếu. Nếu là chính quy hoá thuần thì phải ngược lại.

(Chú ý phụ, khớp §B.2c: ở nhóm Pha 1 **yếu**, chuyển giao thuần lại **dương** (+0.0129) còn ở nhóm
Pha 1 **mạnh** thì **âm** (−0.0027). `phase1_val` không dự báo được chất lượng chuyển giao —
Spearman −0.191 — nên phép chia này thật sự trực giao với kết quả test.)

### Chia theo NGUỒN (cũng độc lập với kết quả test)

| nguồn | khối | chuyển giao thuần (ROC) | trộn (ROC) | trộn − thuần (ROC) |
|---|---|---|---|---|
| `4cwe` | 62 | **−0.0209** (40/62) | **+0.0078** (48/62) | +0.0287 (35/62, p=0.37) |
| `com` | 61 | +0.0102 (46/61) | +0.0198 (53/61) | +0.0096 (37/61, p=0.12) |
| `full` | 33 | +0.0256 (26/33) | +0.0272 (29/33) | +0.0016 (15/33, p=0.73 — **null**) |

Trên trục **nguồn** thì mẫu hình lại đúng là "thêm được nhiều nhất ở chỗ chuyển giao thuần tệ
nhất" và **biến mất** ở `full` — nơi chuyển giao thuần vốn đã tốt. **Phải nêu cả hai trục, chúng
nói hai điều khác nhau.**

### Phát biểu đúng sau khi ghép hai trục

**Phép trộn cho một SÀN.** Nó dương so với baseline ở **cả ba nguồn** (+0.0078, +0.0198, +0.0272)
và ở **cả hai** nhóm Pha 1 — tức chưa bao giờ tệ hơn baseline. Nó **sửa** trường hợp chuyển giao
thuần gây hại (`4cwe`: −0.0209 → +0.0078) và **trung tính** khi chuyển giao đã tốt (`full`).

Đó mới là câu đáng viết vào bài: *không cần biết trước nguồn có chuyển giao tốt hay không — bản
trộn đều không bao giờ kém hơn mô hình chỉ-đích, và lấy lại phần lớn thiệt hại khi nguồn kém.*
Nó **không** phải là "trộn luôn tốt hơn": ở `full` nó không thêm được gì.

---

## §26 — Lưới `lm` trên codebert: pha loãng nặng lặp lại qua BACKBONE (09/09/2026, bậc 1 → **bậc 2**)

Chi tiết và cách đọc ở `RESEARCH_2026-09-08_transfer.md` §B.2f. Tóm tắt: khối `pool_cb_lm` xong
trên 161 lúc 04:08 UTC, đủ 15 ô + 3 baseline. Đối chứng `lm100_n930` cùng máy cùng phiên.

| pool | F1@0.5 | F1@val | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| `lm75` | −0.0300 (1/3) | −0.0180 (1/3) | +0.0019 (2/3) | +0.0185 (3/3) |
| `lm50` | −0.0553 (0/3) | −0.0465 (0/3) | −0.0287 (1/3) | −0.0089 (2/3) |
| `lm25` | −0.0441 (0/3) | −0.0350 (0/3) | −0.0187 (0/3) | +0.0064 (2/3) |
| `lm12` | −0.0548 (0/3) | −0.0553 (0/3) | −0.0287 (0/3) | +0.0177 (2/3) |

Pha loãng nặng lặp lại hướng của §B.2e trên backbone thứ hai. Ở n=3 thì p=0.250 là **sàn**, và
**PR-AUC ngược dấu** ở `lm75`/`lm25`/`lm12` — đúng tình huống mục 2b.

### BẬC 2 — n=5 trên ntat2 (09/09 05:52). Bất đồng PR-AUC ĐÃ HẾT

`pool_cb_lm` fold 4–5 chạy trên **ntat2** (máy này đã có lm grid fold 1–3 và baseline fold 1–5,
nên Δ ghép cặp trọn vẹn trong cùng máy). Đối chứng `lm100_n930`:

| pool | F1@0.5 | F1@val | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| `lm75` | +0.0071 (3/5) | −0.0033 (2/5) | −0.0005 (2/5) | −0.0035 (2/5) |
| `lm50` | −0.0092 (2/5) | −0.0154 (2/5) | −0.0110 (2/5) | −0.0152 (3/5) |
| `lm25` | −0.0293 (1/5) | −0.0293 (2/5) | −0.0252 (1/5) | −0.0026 (1/5) |
| **`lm12`** | **−0.0500 (0/5)** | **−0.0476 (0/5)** | **−0.0560 (0/5)** | **−0.0611 (0/5)** |

**`lm12` âm ở 0/5 fold trên CẢ BỐN chỉ số**, mỗi cái p=0.0625 — tức **kết quả tốt nhất có thể đạt
ở n=5**. Bất đồng PR-AUC ở bậc 1 (khi đó `lm12` cho PR **+0.0177**) **biến mất hoàn toàn** khi lên
n=5: PR thành −0.0611, cũng 0/5. Đây là ví dụ sạch cho quy tắc *"n=3 chỉ đủ để DỪNG, không đủ để
KẾT LUẬN"* — nếu chốt ở bậc 1 thì đã ghi vào bài một bất đồng không có thật.

Và liều–đáp ứng **đơn điệu** theo mức pha loãng trên cả bốn chỉ số: `lm75` ≈ 0 → `lm50` âm nhẹ →
`lm25` âm rõ → `lm12` âm mạnh nhất. Trên t5p (§B.2e) `lm50`/`lm75` từng **lệch dấu giữa hai máy**;
ở codebert thì thang này sạch.

**Phát biểu được phép dùng**: pha loãng nguồn dưới ~25% hàng CWE đích phá khái quát hoá — **lặp
lại ở bậc 2 trên backbone thứ hai, trên cả bốn chỉ số**, với `lm12` đạt sàn thống kê của n=5.

---

## §25.7 — `ensctl` hỏng cả ba fold vì MỘT tham số, và cổng đếm hiện vật đã bắt được (09/09 04:21)

`run/ensctl.sh` chạy trên ntat2, **hỏng cả ba fold trong 65 giây**, ghi `done` với **0 ô**, và máy
nằm không ~3 phút. Lỗi thật, nguyên văn:

```
train_baseline.py: error: argument --pooling: invalid choice: 'enc' (choose from cls, mean)
```

**Hai lỗi trong một dòng**, cả hai đều do tôi đặt `BB="t5p=Salesforce/codet5p-220m:enc"`:

1. **`enc` không phải lựa chọn hợp lệ của `train_baseline.py`.** `train_transfer.py` nhận `enc`
   nhưng `train_baseline.py` chỉ nhận `(cls, mean)`. Tôi đã kiểm **mọi file** script gọi tới đều
   tồn tại — nhưng **không kiểm tham số có hợp lệ với đúng trình huấn luyện đó không**.
2. **Sai cả backbone.** Baseline seed 42 trong cây `asamaw_t5p` dùng
   `Salesforce/codet5p-220m-**bimodal**` với `pooling=mean` (đọc thẳng từ `hyperparameters` của ô
   đã có). Nếu script cứ thế chạy được thì Δ **không ghép cặp được** — hỏng theo kiểu im lặng,
   tệ hơn nhiều so với hỏng ồn ào.

**Cổng đếm hiện vật (thêm 08/09) đã làm đúng việc**: nó thấy `196 → 196` ô, in
`!! MUC NAY KHONG SINH RA O NAO ... CAN NGUOI XEM` và ghi vào `log/worklist.noop`. Không có cổng
đó thì mục này bị đánh dấu xong vĩnh viễn mà chưa bao giờ chạy.

**Đã sửa**: `BB` mặc định của `ensctl.sh` đổi thành `t5p=Salesforce/codet5p-220m-bimodal:mean`,
khớp đúng ô đã có; kiểm hai chiều bằng chính `train_baseline.py --help` (`mean` chấp nhận, `enc`
từ chối). Xoá dòng `done` giả trên cả hai máy rồi phóng lại.

**Quy tắc bổ sung**: xếp một script vào máy xa thì ngoài "mọi file nó gọi tới có tồn tại không"
phải kiểm thêm **"mọi tham số nó truyền có hợp lệ với đúng trình nhận không"** — hai trình huấn
luyện trong cùng dự án này **không** nhận cùng tập lựa chọn. Và với khối đối chứng, đọc
`hyperparameters` của ô sẽ-được-ghép-cặp để lấy cấu hình, đừng đặt tay.

### Phụ: `vast_endpoint` trả rỗng một lần, và vì sao KHÔNG được kết luận máy chết

Cùng lúc đó `vast_endpoint ntat2` trả chuỗi rỗng. Theo quy tắc đã ghi trong `scripts/endpoints.sh`,
tôi **không** kết luận máy chết mà hỏi thẳng API: `state=running/running`, và cổng SSH thật
(`ports['22/tcp']` = 52121) **không hề đổi**. Nối trực tiếp thì vào được ngay. Cache
`/tmp/vast_endpoints.json` kiểm lại vẫn hợp lệ (JSON, 4 mục) và lần gọi sau trả đúng — đây là
trục trặc thoáng qua. Nếu lúc đó tin vào chuỗi rỗng thì đã huỷ nhầm một máy đang chạy.

---

## §25.8 — ĐỐI CHỨNG ÂM: trộn hai baseline khác seed. §25 phải PHÁT BIỂU LẠI (09/09, n=32)

Đây là phép kiểm tôi đã ghi là *"chưa có thì chưa trích §25"*. `run/ensctl.sh` sinh baseline seed
7 và 1234 vào **đúng cây `asamaw_t5p`** đã có baseline seed 42 — cùng model, cùng pooling, cùng lr,
cùng epochs (đã đối chiếu `hyperparameters`) — nên trộn baseline⊕baseline ghép cặp được cùng máy
cùng fold. **n=48 cặp trên hai máy — khối đã chạy xong.** Số ổn định suốt từ n=8 lên n=48.

### Tổng thể: đối chứng KHÔNG null — §25 mất phần lớn biên độ tổng

| trộn α=0.5 | F1@0.5 | ROC-AUC | PR-AUC |
|---|---|---|---|
| **baseline ⊕ baseline′** (đối chứng, **n=48 — đã chốt**) | +0.0103 (30/46, p=0.054) | **+0.0079 (37/48, p=0.0002)** | **+0.0111 (41/48, p<1e-4)** |
| baseline ⊕ chuyển giao (§25, 87 khối) | +0.0317 (76/87) | +0.0129 (69/87) | +0.0136 (71/87) |
| ~~phần dôi ra của chuyển giao~~ | ~~+0.0214~~ | ~~≈ +0.0050~~ | ~~≈ +0.0025~~ |

> **CẢNH BÁO — dòng “phần dôi ra” ở trên là HIỆU CỦA HAI TRUNG BÌNH và KHÔNG được dùng.** Nó lấy
> +0.0129 (trên 87 khối, nhiều cây) trừ +0.0084 (trên 32 cặp, một cây) — hai tập khác nhau. Đó
> đúng là điều CLAUDE.md mục 2 cấm. Con số ghép cặp đúng cách nằm ở §25.9 và nó **lớn gấp ba**.

Trộn **hai mô hình ngang tài chỉ khác seed** cũng cho +0.0079 ROC / +0.0111 PR. Điều này đứng
vững và có ý nghĩa: **một phần mức tăng của phép trộn là trung bình hoá phương sai**, nên
*"trộn hơn baseline"* một mình **không** đủ làm bằng chứng cho chuyển giao. Nhưng **bao nhiêu**
phần thì phải đo bằng phép ghép cặp trực tiếp — xem §25.9.

### Theo CWE thì đối chứng PHẲNG hoặc ÂM — đây mới là chỗ phân biệt

ΔROC-AUC theo CWE, α=0.5:

| CWE | **đối chứng** baseline⊕baseline′ (n=32) | §25 baseline⊕chuyển giao (86 khối) | tỉ lệ |
|---|---|---|---|
| **022** | **−0.0257 (12/48, p=0.047)** — *âm CÓ Ý NGHĨA* | **+0.0967 (74/82, p<1e-4)** | **ngược dấu** |
| 078 | +0.0045 (27/48, p=0.47) — tung đồng xu | −0.0014 (40/81, p=1.00) | — |
| **079** | **+0.0297 (30/48, p=0.054)** — ranh giới | **+0.1513 (81/83, p<1e-4)** | **5,1×** |
| 089 | +0.0042 (26/48, p=0.46) — tung đồng xu | +0.0036 (56/83, p=0.0019) | — |

Đối chứng: hai lớp thường đúng **26–27/48, tức tung đồng xu**; CWE-022 **âm có ý nghĩa**
(−0.0257, 12/48, p=0.047) và CWE-079 chỉ ở **ranh giới** (30/48, p=0.054). Trộn với mô hình chuyển giao thì CWE-022 **dương
+0.097 với 74/82 fold** và CWE-079 **+0.151 với 81/83 fold**. Ngược dấu ở một lớp, gấp 5,3 lần ở
lớp kia, và mẫu hình cùng dấu gần tuyệt đối — trung bình hoá phương sai **không** tạo ra được.

### Phát biểu lại §25 — hẹp hơn, đứng vững hơn

> ~~Trộn đều baseline ⊕ chuyển giao vượt cả hai đầu mút~~ **← không dùng nữa làm luận điểm tổng.**
>
> **Giá trị của Pha 1 không nằm ở điểm tổng — điểm tổng là cách đo sai.** Phần mà chuyển giao đóng
> góp thêm so với chỉ trộn hai bản chạy lại (≈+0.0045 ROC) nằm **dưới sàn nhiễu**. Nhưng tính riêng
> từng CWE thì đối chứng **phẳng hoặc âm** ở cả bốn lớp, còn chuyển giao dồn **+0.097 / +0.151**
> vào đúng hai lớp hiếm mà mô hình chỉ-đích yếu nhất, với 74/82 và 81/83 fold cùng dấu.
> **Đó** là bằng chứng chuyển giao mang vào thông tin mới, và nó nằm ở **phân bố**, không ở trung bình.

Ba mục vẫn đứng vì **không** phải phát biểu về biên độ tổng: §25.1 (nguồn pha loãng cho −0.0036
còn nguyên chất +0.0109 — trung bình hoá phương sai không phân biệt được nguồn), §25.3 (recall
**và** precision cùng tăng ở hai CWE hiếm), §25.4 (hiệu trộn−chuyển giao dồn vào hàng **sạch**,
nhóm `train` thì âm).

**Bài học chung**: đối chứng này đáng giá đúng bằng cả khối §25 — nếu bỏ qua nó thì đã viết vào bài
một phát biểu mà 2/3 biên độ đến từ việc chạy lại cùng một mô hình. Mọi phát biểu dạng *"gộp hai
thứ thì tốt hơn"* phải có đối chứng *"gộp hai bản của cùng một thứ"*.


---

## §25.9 — GHÉP CẶP TRỰC TIẾP hai phép trộn: con số đúng lớn gấp ba (09/09/2026)

§25.8 so hai đại lượng đo trên **hai tập khác nhau** rồi trừ nhau. Đó là *hiệu của hai trung bình*
— chính điều CLAUDE.md mục 2 cấm, và tôi đã mắc. Bản ghép cặp đúng:

Trong **cùng một `(cây, fold)`**, với **cùng một baseline seed 42**, lấy
`A = trộn(base42, chuyển_giao)` và `B = trộn(base42, base_khác_seed)`, rồi đo **A − B**. Mọi thứ
triệt tiêu trừ đúng một câu hỏi: *mô hình thứ hai là bản chuyển giao hay chỉ là một bản chạy lại?*

`tools/ens_headtohead.py` · **48 cặp ghép trực tiếp** (cập nhật khi `ensctl` chạy thêm fold; ở
n=36 các số là +0.0215 / +0.0148 / +0.0135 — ổn định):

| A − B | F1@0.5 | ROC-AUC | PR-AUC |
|---|---|---|---|
| | **+0.0200 (37/48, p=0.0002)** | **+0.0121 (42/48, p<1e-4)** | **+0.0117 (43/48, p<1e-4)** |

**+0.0121 ROC vẫn vượt sàn nhiễu 0.010** với 42/48 fold cùng dấu, và **F1@0.5 giờ cũng có ý nghĩa**
(p=0.0002) — tức không chỉ ở xếp hạng. So với con số sai ở §25.8 (“≈+0.0045, dưới sàn nhiễu”) thì
lớn **gấp 2,7 lần** và **đổi hẳn kết luận**. (Biên độ ROC đi từ +0.0148 ở n=36 xuống +0.0121 ở
n=48 — vẫn trên sàn, nhưng phải nêu là nó co lại khi thêm dữ liệu.)

Theo CWE (A − B, ΔROC):

| CWE | A − B | |
|---|---|---|
| **022** | **+0.1076 (40/48, p<1e-4)** | trộn với chuyển giao hơn hẳn |
| **079** | **+0.1323 (42/48, p<1e-4)** | trộn với chuyển giao hơn hẳn |
| 078 | +0.0055 (27/48, p=0.47) | null |
| 089 | +0.0021 (26/48, p=0.46) | null |

Toàn bộ khoảng cách giữa hai phép trộn nằm ở **đúng hai lớp hiếm**, và **đúng bằng không** ở hai
lớp thường. Đây là phiên bản mạnh nhất của luận điểm: **mô hình chuyển giao mang vào thứ mà một
bản chạy lại của baseline không có, và thứ đó chỉ nằm ở hai lớp mà baseline yếu nhất.**

**Giới hạn**: chỉ những khối có **đủ cả ba thành phần** mới vào được phép so (48 cặp) (baseline seed 42 + baseline khác
seed + nhánh chuyển giao cùng cây cùng fold), toàn bộ từ cây `asamaw_t5p` trên hai máy. Hẹp hơn
nhiều so với 87 khối của §25 — nhưng **hẹp mà ghép cặp đúng** thì dùng được, còn **rộng mà lấy
hiệu hai trung bình** thì không. `ensctl` chạy xong trên ntat sẽ nâng số khối lên.

**Bài học**: cùng một dữ liệu, đọc bằng hiệu-hai-trung-bình cho “dưới sàn nhiễu, kết luận sập”,
đọc bằng ghép-cặp cho “+0.0148, 33/36, p<1e-4”. Khoảng cách giữa hai cách đọc lớn hơn khoảng cách
giữa có hiệu ứng và không.

---

## §25.10 — Cache endpoint hỏng vì HAI tiến trình ghi chung một file tạm (09/09 05:35)

`fleet_status.sh` báo **"ntat2: không giải được địa chỉ | trạng thái instance = không-đọc-được"**
trong khi máy **đang chạy bình thường** (`job=1`, đang huấn luyện `lm75` fold 5). Đây là lần thứ
hai trong ngày dấu hiệu này xuất hiện, và lần này truy được nguyên nhân.

**Nguyên nhân**: `/tmp/vast_endpoints.json` hỏng — `Extra data: line 884 column 1`. Hàm
`_vast_refresh` ghi ra `"$VAST_CACHE.tmp"` — **một đường dẫn cố định**. Watchdog chạy **mỗi 10
phút** và phiên tương tác **cũng** gọi hàm này; hai tiến trình ghi cùng một file tạm nên nội dung
đan xen, tạo ra JSON có hai tài liệu nối nhau. Phép kiểm JSON *trước khi thay cache* không cứu
được vì bản thân file tạm đã bị tiến trình kia ghi thêm **sau** khi kiểm.

**Vì sao nguy hiểm**: đọc thành "máy đã mất" dẫn thẳng tới quyết định **huỷ nhầm một máy đang làm
việc**. Quy tắc *"chưa giải được địa chỉ ⇒ TUYỆT ĐỐI KHÔNG huỷ"* đã cứu cả hai lần — tôi hỏi thẳng
API và thấy `state=running/running`, cùng IP cùng cổng.

**Đã sửa hai chỗ**:
1. File tạm **riêng cho từng tiến trình** (`$VAST_CACHE.tmp.$$`) — hết đan xen.
2. **Kiểm cache lúc ĐỌC**, không chỉ lúc ghi: cache hỏng từ trước thì xoá và làm mới, thay vì im
   lặng trả về rỗng (đọc y hệt "máy đã mất").

**Kiểm ba chiều**: (a) cache hỏng sẵn → tự làm mới, trả đúng địa chỉ; (b) cache lành → trả địa
chỉ, không báo gì; (c) cache hợp lệ nhưng rỗng → trả **rỗng**, không bịa ra địa chỉ.

**Bài học chung**: mọi mẫu `cmd > file.tmp && mv file.tmp file` đều **không an toàn khi có hơn một
tiến trình**, dù `mv` là nguyên tử. Cái nguyên tử là `mv`, không phải việc ghi. Đường dẫn tạm phải
mang `$$` hoặc dùng `mktemp`. Và mọi cache phải kiểm lúc **đọc** — chỉ kiểm lúc ghi thì một lần
hỏng là hỏng vĩnh viễn.

---

## §27 — Nội suy TRỌNG SỐ: một mô hình, ngang trộn xác suất (09/09, **n=15, bậc 3**)

> **TỰ SỬA (giữ lại để nhớ).** Bản đầu viết ở **n=1**: *"nội suy trọng số nói KHÔNG — đường α lõm,
> val chọn α=1.0, hai mô hình có hàng rào"*, kèm cả cơ chế. Ở n=15 thì **α chọn trên val nội tại ở
> 13/15 ô** (0.10–0.90, trung bình 0.627). Tôi đã vơ đúng ô ngoại lệ làm kết luận.

`run/wblend.sh`, t5p, **3 nguồn × 5 fold = 15 ô**. Δ so với **baseline** cùng ô:

| | F1@0.5 | ROC-AUC | PR-AUC | chi phí suy luận |
|---|---|---|---|---|
| chuyển giao thuần (α=1) | +0.0091 (10/15) | +0.0112 (11/15, p=0.12) | +0.0064 (8/15) | **1 mô hình** |
| **nội suy TRỌNG SỐ, α chọn trên VAL** | +0.0150 (11/15, p=0.057) | **+0.0174 (12/15, p=0.013)** | **+0.0205 (14/15, p=0.001)** | **1 mô hình** |
| nội suy trọng số, α=0.5 **cố định** | **−0.0149 (4/15)** | +0.0094 (8/15) | +0.0154 (9/15) | 1 mô hình |
| **trộn XÁC SUẤT, α=0.5 khai báo trước** | **+0.0197 (15/15, p=0.0001)** | +0.0170 (12/15, p=0.035) | +0.0158 (13/15, p=0.007) | 2 mô hình |

Ghép cặp trực tiếp trong cùng ô:

| | F1@0.5 | ROC-AUC | PR-AUC |
|---|---|---|---|
| trọng số@val − chuyển giao thuần | +0.0059 (8/15, ns) | **+0.0062 (11/15, p=0.023)** | **+0.0141 (13/15, p=0.0002)** |
| trọng số@val − xác suất@0.5 | −0.0047 (3/15, **p=0.092**) | +0.0005 (9/15, ns) | +0.0048 (8/15, ns) |

**Ba kết luận ở bậc 3:**

1. **Nội suy trọng số với α chọn trên val HƠN chuyển giao thuần** — ROC +0.0062 (11/15, p=0.023),
   PR +0.0141 (13/15, p=0.0002). Có ý nghĩa, và **không tốn thêm mô hình nào**.
2. **Không phân biệt được với trộn xác suất trên cả ba chỉ số** (F1 −0.0047 p=0.092, ROC +0.0005,
   PR +0.0048). Ở n=9 F1 còn thua đều (1/9); lên n=15 thì **thế thua đó biến mất**. Nghĩa là
   **thu chi phí suy luận từ hai mô hình xuống một mà không mất gì đo được**.
3. **α=0.5 CỐ ĐỊNH trong không gian trọng số thì KHÔNG dùng được** (F1 −0.0149, 4/15) — khác hẳn
   không gian xác suất, nơi α=0.5 cố định lại là lựa chọn tốt nhất (F1 **15/15**). Trọng số
   **bắt buộc** hiệu chỉnh α trên val; xác suất không cần tham số nào.

**Đánh đổi để nêu trong bài**: hai mô hình + không tham số (trộn xác suất), hay một mô hình + một
lần hiệu chỉnh α trên val (nội suy trọng số). Hai cái ngang nhau về điểm.

**Bản lặp backbone** (`wblend_cb` trên codebert) đang chạy trên ntat2 — mọi phát biểu khác của dự
án đều phải lặp trên backbone thứ hai trước khi viết, và §25.11 vừa cho thấy **biên độ tổng có thể
không lặp**.

## §27.1 — Pha 1 `full` trên ntat2 hỏng vì THIẾU FILE DỮ LIỆU, không phải vì huấn luyện (09/09 07:33)

`run/p1fill_cb.sh` tạo được Pha 1 codebert cho `4cwe` (498 700 261 byte) và `com`
(498 700 453 byte) nhưng **hỏng ở `full`**. Nguyên nhân, nguyên văn:

```
FileNotFoundError: [Errno 2] No such file or directory: 'data/phase1_full.jsonl'
```

**Không phải Pha 1 sập** — `full` là nguồn khó nhất (123 CWE, lệch mạnh về ccpp) và là chỗ Pha 1
của backbone yếu hay sập thật (CLAUDE.md mục 6), nên rất dễ đọc nhầm thành "codebert không học nổi
`full`" rồi ghi vào bài một kết luận sai. Log dừng ở **0.00s**, trước cả khi tải mô hình — đó là
dấu hiệu phân biệt: sập vì huấn luyện thì phải có ít nhất một epoch.

Đã đẩy `data/phase1_full.jsonl` (18 569 689 byte, **7 598 dòng** — khớp mục 6) lên ntat2 và xếp bù
`p1fill_cb.sh|full|-` rồi `wblend_cb.sh|full|1 2 3` vào **cuối** hàng đợi, không cắt ngang khối
đang chạy. `p1fill_cb.sh` tự bỏ qua nguồn đã có nên chỉ chạy đúng `full`.

**Quy tắc bổ sung**: khi xếp một khối lên máy xa, kiểm cả **file DỮ LIỆU** nó đọc, không chỉ file
mã nó gọi. Danh sách kiểm trước đó của tôi có `run/*.sh`, `src/*.py`, `tools/*.py` và một file
fold của tập đích — nhưng **không** có ba file nguồn Pha 1.

---

## §28.1 — §28 qua phép kiểm RÒ RỈ: phần CWE hiếm gần như không mất gì (09/09)

Bộ đích chia theo từng dòng nên ~16% hàng test có bản gần trùng trong TRAIN. Phép kiểm này đã
**đánh sập §21.1**, nên bắt buộc chạy cho con số đầu bài.

**ntat, n=15** — cấu hình chốt vs baseline, tách theo nhóm rò rỉ:

| nhóm | hàng/fold | ΔF1@0.5 | ΔROC-AUC |
|---|---|---|---|
| tất cả | 152 | +0.0535 (12/15, p=0.035) | +0.0337 (14/15, p=0.001) |
| `train` (rò rỉ vào TRAIN) | 23 | **+0.1527 (13/15)** | **+0.1308 (13/15)** |
| `val` | 10 | +0.0281 (6/12) | +0.0243 (4/12) |
| `test` | 9 | +0.1242 (5/9) | +0.0887 (7/9) |
| **`none` — hàng SẠCH, 73% dữ liệu** | **111** | **+0.0369 (14/15, p=0.001)** | +0.0202 (10/15, **p=0.30**) |

**Theo CWE, CHỈ trên hàng sạch (`none`):**

| CWE | ntat n=15 | ntat2 n=9 |
|---|---|---|
| **022** | **+0.2567 (13/15, p=0.002)** | +0.2354 (6/9) |
| **079** | **+0.3158 (14/15, p<0.001)** | **+0.2673 (9/9, p=0.004)** |
| 078 | −0.0154 (5/15, ns) | +0.0135 (6/9, ns) |
| 089 | +0.0064 (8/15, ns) | +0.0088 (4/9, ns) |

### Đọc cho đúng — hai vế, cả hai đều phải nêu

**Vế tốt**: phần CWE hiếm **gần như không mất gì** khi bỏ hết hàng có bản gần trùng.
CWE-022 đi từ +0.2238 (tất cả) xuống +0.2567 (sạch) — *tăng*; CWE-079 từ +0.3638 xuống +0.3158,
tức **giữ 87%**, và vẫn **14/15 fold** trên ntat, **9/9** trên ntat2. Đây **không** phải hiệu ứng
do rò rỉ.

**Vế phải thừa nhận**: *biên độ tổng* thì có phần dựa vào rò rỉ. Nhóm `train` (23 hàng/fold) cho
ΔROC **+0.1308**, cao hơn hẳn nhóm sạch (+0.0202), và trên hàng sạch **đếm dấu ROC tụt xuống
10/15 (p=0.30)** dù F1 vẫn 14/15 (p=0.001). Nghĩa là con số tổng +0.0337 **được rò rỉ giúp một
phần**; con số per-CWE thì không.

> Lại đúng mẫu hình của cả đêm: **phát biểu ở mức TỔNG yếu đi khi kiểm; phát biểu ở mức PHÂN BỐ
> thì đứng vững.** Ba lần trước là đối chứng seed, bản lặp backbone, và hiệu-hai-trung-bình — lần
> này là rò rỉ. Bốn phép kiểm độc lập, cùng một kết luận về *cách đo nào dùng được*.

**Phải viết vào bài**: nêu con số **trên hàng sạch** (CWE-022 +0.2567, CWE-079 +0.3158) làm số
chính, và nêu rõ số tổng có phần dựa vào rò rỉ của bộ `norm`. Bộ `twin` (chia theo cụm gần trùng)
là câu trả lời trực diện nếu reviewer hỏi — chưa chạy, người dùng đã nêu là để sau.

---

## §27.2 — Bản lặp §27 trên CODEBERT (n=6) và một lỗi THIẾT KẾ của `run/wblend.sh` (09/09)

### Kết quả, n=6 (3 fold × 2 nguồn — thiếu `full`, lý do ở dưới)

Δ so với baseline cùng ô:

| | F1@0.5 | ROC-AUC | PR-AUC |
|---|---|---|---|
| chuyển giao thuần | **+0.0365 (6/6, p=0.031)** | +0.0039 (3/6) | +0.0106 (4/6) |
| nội suy trọng số, α trên val | +0.0154 (4/6) | **+0.0150 (6/6, p=0.031)** | +0.0218 (5/6) |
| **trộn xác suất, α=0.5** | **+0.0427 (6/6, p=0.031)** | **+0.0147 (6/6, p=0.031)** | **+0.0263 (6/6, p=0.031)** |

Ghép cặp trực tiếp: trọng số@val − chuyển giao thuần = F1 **−0.0211 (0/6)**, ROC +0.0111 (5/6);
trọng số@val − xác suất@0.5 = F1 **−0.0273 (1/6)**, ROC +0.0003, PR −0.0046.

**Không lặp lại hoàn toàn.** Trên t5p (n=15) nội suy trọng số **không phân biệt được** với trộn
xác suất trên cả ba chỉ số. Trên codebert nó **thua đều ở F1** — 0/6 so với chuyển giao thuần và
1/6 so với trộn xác suất, hướng nhất quán dù n nhỏ. Còn **trộn xác suất thì 6/6 trên cả ba chỉ số**.

**Phát biểu phải thu hẹp**: *trộn xác suất α=0.5* là lựa chọn an toàn trên **cả hai** backbone;
*nội suy trọng số* mới chỉ chứng minh được trên t5p, và trên codebert nó mất F1. Muốn nêu "một mô
hình là đủ" thì phải chạy thêm.

### Lỗi thiết kế: khối KHÔNG chạy lại được từng phần

Nguồn `full` **đã sinh đủ 3 ô Pha 2**, nhưng bước nội suy bị bỏ:

```
!! thieu checkpoint baseline model/wbcb_codebert/baseline/seed_42/fold3/best.pt — bo fold 3
```

`run/wblend.sh` dọn checkpoint **ngay sau mỗi fold** (đúng, vì mỗi cái ~450MB). Nhưng khi chạy lại
cho một nguồn khác, `matrix.sh` thấy **file kết quả JSON đã có** nên **không huấn luyện lại**
baseline — mà checkpoint thì đã bị xoá. Kết quả: có ô Pha 2 nhưng không có cặp để nội suy.

**Sửa cho lần sau**: hoặc (a) giữ checkpoint baseline đến hết khối rồi mới dọn, hoặc (b) khi chạy
bù một nguồn, xoá luôn **file kết quả JSON** của baseline để `matrix.sh` huấn luyện lại. Đây là
biến thể của bẫy "mã thoát 0 không có nghĩa là việc đã thành": khối in `xong ... (+3 o)` và cổng
đếm hiện vật **không bắt được**, vì nó đếm `fold*.json` — mà 3 ô Pha 2 đúng là đã sinh ra.
Cổng đếm hiện vật nên đếm **thứ mà khối sinh ra để dùng** (ở đây là file nội suy), không phải thứ
dễ đếm nhất.

---

## §29 — Huấn luyện lại Pha 1 một cách THỪA vì tra theo TÊN thay vì theo NỘI DUNG (09/09)

Người dùng nhắc: *"Phase 1 dùng lại được thì nên dùng nhé."* Đúng — và tôi đã **huấn luyện lại
thừa ~27 phút GPU**: codebert/`4cwe` (11 phút, 161) và codebert/`com` (16 phút, vast, phải giết
giữa chừng).

**Nguyên nhân**: tôi tìm checkpoint bằng đường dẫn `*/codebert__latent_bottleneck_<src>_l0p05/`
và kết luận "không có". Nhưng khối **s42** (27–31/08) đặt tên **không có hậu tố λ** khi λ=0.05, và
chỉ thêm `_l02` khi λ=0.02:

| kho | tên | λ thật (đọc từ `training_args`) |
|---|---|---|
| `s42/phase1` | `codebert__latent_bottleneck_com` | **0.05** |
| `s42/phase1` | `codebert__latent_bottleneck_com_l02` | 0.02 |
| `n48/phase1` | `codebert__latent_bottleneck_4cwe_l0p05` | 0.05 |

**Hai quy ước tên cho cùng một λ**, đặt ra ở hai đợt khác nhau. Tra theo tên thì trượt.

Đối chiếu `training_args` của bản s42 và bản tôi vừa train: **trùng khít mọi trường** —
`microsoft/codebert-base` · `cls` · `latent_bottleneck` · λ 0.05 · `fixed4` · 4 lớp · 15 epoch ·
lr 2e-5 · `sam_rho 0` · seed 42 · `num_latent 8`. Khác duy nhất là `best_val_macro_f1`, dao động
giữa các lần chạy — và §B.2c đã cho thấy val Pha 1 **không** dự báo transfer, nên nó không phải
lý do để chọn bản nào.

> **Đính chính**: bản ghi đầu của mục này nêu "0.6532 vs 0.6976". Con số 0.6976 **không truy được
> nguồn**: bản tôi tự huấn luyện lại đã bị ghi đè bằng bản s42 và không còn dòng log nào giữ val
> của nó. Trùng hợp là **0.697621 đúng bằng val Pha 1 của `t5p`/`4cwe`** — nhiều khả năng tôi đọc
> nhầm dòng của backbone kia. Giá trị s42 (0.6532) thì kiểm lại được bất cứ lúc nào từ chính
> checkpoint. Đã kiểm cả sáu checkpoint đang dùng: mỗi cái mang đúng `model_name` và `pooling` của
> backbone nó, không có chuyện lẫn đường dẫn.

| | 4cwe | com | full |
|---|---|---|---|
| codebert (`cls`) | 0.6532 | 0.5598 | 0.5636 |
| t5p (`mean`) | 0.6976 | 0.5897 | 0.5648 |

Đã dùng bản **s42 cho cả ba nguồn** (cùng xuất xứ, tránh confound phiên bản thư viện) và đẩy lên
vast; log xác nhận `phase1 ... | da co, dung lai`.

**Quy tắc bổ sung**: tìm checkpoint Pha 1 dùng lại được thì **đọc `training_args` của mọi
checkpoint có cùng backbone + aux_mode**, đừng lọc theo tên thư mục. Một lệnh
`torch.load(...)['training_args']['lambda_cwe']` rẻ hơn 27 phút GPU. Chỉ **λ** mới buộc huấn
luyện lại (CLAUDE.md mục 5); optimizer, SAM/ASAM, cách chia fold thì dùng lại được hết.

---

## §30 — Head phụ trên `com`/`full` học **10 pillar**, KHÔNG phải 94/123 CWE (09/09)

Đi kiểm một chuyện nhỏ — vì sao Pha 1 `com` và `full` **cùng đúng 498 700 197 byte** trong khi
một cái có 94 CWE còn cái kia 123 — thì ra `cwe_head` của **cả hai** đều là `(10, 8)`:

| nguồn | `cwe_vocab` | số CWE-ID | **số lớp head thật** | dòng bị `-100` |
|---|---|---|---|---|
| `4cwe` | `fixed4` | 4 | **4** | 0 |
| `com` | `precomputed` | 94 | **10** | 0 |
| `full` | `precomputed` | 123 | **10** | **764 (10.1%)** |

`precomputed` (`train_transfer.py:562`) lấy `cwe_class` **nguyên văn** từ file dữ liệu. Và
`cwe_class` trong `data/phase1_{common,full}.jsonl` là **pillar của CWE-1000 Research Concept**,
không phải CWE cụ thể. Mười pillar, `sorted()` nên chỉ số là tất định:

| lớp | pillar | ví dụ CWE trong đó | dòng (`com`) | dòng (`full`) |
|---:|---|---|---:|---:|
| 0 | CWE-284 Improper Access Control | 284, 269, 862, 287 | 266 | 266 |
| 1 | CWE-435 Improper Interaction Between Entities | 115 | **2** | **2** |
| 2 | CWE-664 Improper Control of a Resource | 119, 125, 787, 200, 401, 22 | 1 026 | 3 616 |
| 3 | CWE-682 Incorrect Calculation | 190, 369 | 372 | 380 |
| 4 | CWE-691 Insufficient Control Flow Management | 835, 617, 362 | 136 | 258 |
| 5 | CWE-693 Protection Mechanism Failure | 352, 347, 345 | 216 | 216 |
| 6 | CWE-697 Incorrect Comparison | 697 | **4** | **4** |
| 7 | CWE-703 Improper Check of Exceptional Conditions | 703, 755, 754, 476 | 310 | 680 |
| 8 | CWE-707 Improper Neutralization | 79, 20, 78, 89 | 1 410 | 1 410 |
| 9 | CWE-710 Improper Adherence to Coding Standards | 1125 | **2** | **2** |

**Ba hệ quả phải nêu khi viết bài:**

1. **Mô tả phương pháp không được viết "head phụ 94 lớp".** Nó là head **10 pillar** — nói *kiểu
   sai lầm*, không nói *lỗ hổng nào*. Đúng như `src/build_parent_labels.py` đặt ra chủ ý.
2. **Ba lớp gần như rỗng** (1, 6, 9 với 2–4 dòng trên 3 744/7 598). Thực tế head chỉ học được
   **7 lớp**, và hai lớp 2 và 8 chiếm 65% (`com`) đến 66% (`full`). Cái head "10 lớp" này gần
   với nhị phân *neutralization vs resource-control* hơn là một bộ phân loại mười lớp.
3. **`full` ném đi 10.1% dòng khỏi loss phụ** (`cwe_class = -100`: CWE-264, 189, 399, 310 — các
   CWE loại "category"/đã bỏ, không có pillar). Nhánh `none` không bị vì nó không dùng loss phụ.
   Đây là một khác biệt thật giữa `full` và `com` ngoài chuyện kích thước, và nó nằm đúng chỗ
   Pha 1 của backbone yếu hay sập.

**Không có confound.** `data/phase1_{4cwe,common,full}.jsonl` đều còn nguyên mtime **27/08 09:28**
— tức mọi Pha 1 của mọi khối từ trước tới nay đều dùng đúng nhãn này. Không hề có chuyện file bị
ghi đè giữa dự án làm hai checkpoint khác nhãn bị đem so với nhau.

**Cách phát hiện**: hai file lẽ ra phải khác kích thước (94 vs 123 lớp ⇒ lệch ~1 KB) mà **giống
nhau từng byte**. Kích thước bằng nhau ở nơi lẽ ra phải khác là một tín hiệu, y như hai số khác
`n` bị đem trừ nhau ở §25.9. md5 xác nhận chúng vẫn là hai checkpoint **khác nhau** — cùng cỡ,
khác nội dung.

### §30.1 — Chưa từng đo head phụ có HỌC được gì không (09/09)

Đi tiếp từ §30: cả phương pháp dựa trên giả thiết head phụ dạy encoder một tín hiệu hữu ích.
Nhưng **không chỗ nào ghi lại độ chính xác của head phụ** — `src/train.py` không log, checkpoint
không lưu. Khoá duy nhất về chất lượng trong checkpoint là `best_val_macro_f1`, và đó là F1 của
**đầu ra lỗ hổng nhị phân**, không phải của head CWE.

Sàn mà head phải vượt (bộ đoán luôn lớp đa số, không học gì):

| nguồn | K | n có nhãn | độ chính xác sàn | macro-F1 sàn |
|---|---:|---:|---:|---:|
| `4cwe` | 4 | 930 | **0.744** | 0.2133 |
| `com` | 10 | 3 744 | 0.377 | 0.0547 |
| `full` | 10 | 6 834 | 0.529 | 0.0692 |

Sàn của `4cwe` **0.744** là con số đáng chú ý: CWE-79 chiếm 692/930. Một head "đoán CWE-79" đã
đúng 74%, nên λ·loss_phụ có thể gần như bằng hằng số ngay từ đầu và không ép encoder học gì.

**Đây là câu hỏi mở, không phải kết luận.** Đo được bằng một lượt forward trên CPU với checkpoint
đã có — không tốn GPU, không cắt ngang khối nào. Chưa chạy vì cả hai máy đang bận và CPU của 161
còn phải nuôi dataloader.

**Kèm theo: `cwe_mapping` trong checkpoint `com`/`full` là RÁC.** Nó ghi
`{CWE-022:0, CWE-078:1, CWE-079:2, CWE-089:3}` trong khi `num_cwes=10` và lớp 2 thật ra là pillar
**CWE-664**, không phải CWE-079. `train_transfer.py:462` ghi một hằng số `CWE_MAPPING` cứng bất kể
`cwe_vocab` là gì. **Không tai nạn nào đã xảy ra**: grep toàn bộ `src/ tools/ run/ scripts/` cho
thấy trường này **chỉ được ghi, chưa bao giờ được đọc**. Bảng per-CWE trong báo cáo lấy nhãn từ
`test_cwe_classes` của tập **đích** Python (đúng là bộ 4 CWE), không đi qua trường này. Nhưng nó là
bẫy cho bất kỳ phân tích nào sau này tin vào siêu dữ liệu của checkpoint.

---

## §31 — Quét mồ côi bắt LÁ mà không bắt VỎ: một `matrix.sh` sống sót rồi đẻ lại python (09/09)

Khởi động lại driver 161 để đổi `FOLD_LIST` sang `1 2 3`. Script khởi động lại có đủ hai bước
đúng sách: giết theo **PID** (không `pkill -f`), rồi **quét mồ côi**. Vẫn sót.

Sau 12 phút, `log/chot_t5p.log` đứng ở một dòng, `job=1`, mà **không có `train_transfer` nào của
tôi** — trong khi GPU 13 320 MiB / 100%. Tra ra:

| pid | ppid | là gì | tuổi |
|---|---|---|---|
| 2302684 | 1 | `bash run/chot2bb.sh` (driver MỚI) | đang kẹt ở `wait_vram` (`wchan=do_wait`) |
| **2302594** | **1** | **`bash run/matrix.sh` — MỒ CÔI sót lại** | 608 s |
| 2302799 | 2302594 | `train_baseline.py --run_name chot_t5p --fold 2` | 600 s, **12 628 MiB** |

**Lỗi ở đâu**: `kill_tree` duyệt con trước rồi mới giết cha — đúng ý định, nhưng trong lúc nó
đang giết ở dưới sâu thì cha **vẫn sống** và kịp sinh một `matrix.sh` mới mà lần liệt kê đầu
không hề thấy. Bước quét mồ côi lẽ ra vớt được, nhưng mẫu của nó là

```
$2==1 && /train_(transfer|baseline)\.py/
```

tức **chỉ bắt cái LÁ**. Kẻ sống sót là `bash run/matrix.sh` — một **vỏ trung gian**, và chính nó
đẻ ra một python mới *sau* khi quét đã chạy xong. Quét lá không bao giờ dọn được thứ đẻ ra lá.

**Sửa**: mẫu quét phải phủ **cả chuỗi** — `train_*.py` **và** `run/matrix.sh` **và** `run/opt1.sh`
— và phải **lặp cho đến khi không còn gì**, không phải quét đúng một lượt.

**Cái đã cứu**: `wait_vram` của `chot2bb.sh`. Driver mới thấy chỉ còn 3 056 MiB trống (cần
13 000) nên **ngồi chờ 12 phút** thay vì nhảy vào. Nếu không có nó, đây là đúng cấu hình đã hỏng
26 ô hôm nay: **hai chuỗi một GPU**, vì `run/matrix.sh` KHÔNG giữ lock. Nghịch lý đáng nhớ:
cổng chờ VRAM làm máy nằm không 12 phút, và đó là điều tốt nhất nó có thể làm.

**Giá phải trả**: ~10 phút GPU của ô `baseline fold 2` bị bỏ dở khi tôi giết mồ côi. Tôi chọn
giết thay vì để nó chạy nốt, vì thời điểm nó nhả bộ nhớ giữa hai ô chính là lúc driver kia thoát
`wait_vram` — cửa sổ đâm nhau. Máy local được phép nằm không; hai chuỗi một GPU thì không.

**Dấu hiệu nhận ra sớm**: `job=1` (lock có người giữ) **nhưng** không có tiến trình huấn luyện
nào của mình, **và** GPU vẫn 100%. Ba dữ kiện đó cùng lúc = có chuỗi thứ hai ngoài tầm kiểm soát.

### §31.1 — Công cụ giết không tự từ chối khi mẫu khớp quá nhiều: suýt xoá cả phiên làm việc (09/09)

Viết `scripts/kill_my_chain.sh` để sửa §31. Trong lúc **thử chính nó**, một lệnh gọi hỏng làm
`PAT` thành **rỗng**, và bản đầu của script không chặn gì cả — nó khớp **mọi tiến trình tôi sở
hữu** và in ra danh sách sẽ giết:

```
[DRY] se giet 31344  (systemd --user)        [DRY] se giet 2728629 (claude)
[DRY] se giet 2726411 (code-server)          [DRY] se giet 1272876 (tmux new -s ml4vd)
[DRY] se giet 3459844 (ssh -L 8080 ...)      [DRY] se giet 2302684 (bash run/chot2bb.sh)  <-- driver dang chay
[DRY] se giet 2323984 (train_transfer.py fold 2)                                          <-- o dang huan luyen
```

**Chỉ `DRY=1` cứu.** Nếu là lần chạy thật thì mất phiên VS Code, tmux, daemon Claude, kết nối ssh,
**và** ô `fold 2` đang huấn luyện của chính khối ưu tiên.

Điều đáng nói không phải lệnh gọi hỏng — lệnh gọi lúc nào chẳng hỏng được. Điều đáng nói là
**công cụ chấp nhận nó**. Một công cụ giết mà không tự từ chối khi mẫu khớp quá nhiều thì
không được phép tồn tại. Đã dựng **ba chặn**, thử cả hai chiều:

| chặn | từ chối cái gì | đã thử |
|---|---|---|
| 1. mẫu phải có thật, ≥6 ký tự, không phải mẫu bắt-tất-cả | `""`, `sh`, `.*`, `*` | 4/4 từ chối |
| 2. khớp > `MAXROOTS` (mặc định 4) ⇒ gần như chắc chắn sai mẫu | `/bin/bash` khớp 12 | từ chối, in ra danh sách |
| 3. danh sách cấm tuyệt đối (systemd, code-server, tmux, claude, ssh, dbus, pipewire, sshd) | `tmux new -s ml4vd`, `server-main.js` | BẢO VỆ, bỏ qua cả cây |

Chiều ngược lại cũng phải đúng: `'run/chot2bb.sh'` khớp **đúng 1** gốc và chỉ ra đủ 4 tầng
(`chot2bb.sh → opt1.sh → matrix.sh → train_transfer.py`). Driver sống nguyên sau mọi phép thử.

**Quy tắc rút ra**: mọi công cụ có sức phá phải có **ngưỡng bán kính nổ**. Không phải "mẫu này
đúng không" mà "mẫu này khớp bao nhiêu, và con số đó có hợp lý với thứ tôi định làm không". Một
mẫu driver hợp lệ khớp 1–2 tiến trình; khớp 44 nghĩa là mẫu sai, không phải là hôm nay có nhiều
việc.

---

## §32 — Ma trận λ × ρ: codebert và t5p có vẻ muốn hai chỗ NGƯỢC nhau (09/09)

`tools/hp_matrix.py` gộp **3 804 ô ghép cặp** từ 46 cây kết quả (mọi khối từ 24/08 đến nay),
Δ luôn ghép cặp theo `(cây, backbone, seed, fold)` với `baseline` cùng chỗ. Trang tương tác:
xem ARTIFACTS.md.

### ρ trên t5p có cực đại nội tại, và sập rất dứt khoát

λ=0.05, nhánh nút thắt, ΔROC-AUC:

| ρ | 0 | 0.05 | 0.1 | 0.2 | 0.5 | **1.0** | **2.0** | 4.0 | 8.0 |
|---|---|---|---|---|---|---|---|---|---|
| ΔROC | −0.0101 | +0.0312 | +0.0109 | +0.0187 | +0.0229 | **+0.0359** | **+0.0352** | −0.1450 | **−0.2982** |
| fold + | 619/982 | 6/6 | 74/100 | 43/51 | 55/59 | 20/21 | 52/57 | 9/25 | **0/15** |

Đỉnh ở ρ=1–2, rồi **sập** ở ρ=4 và ρ=8 (0/15 fold dương ở ρ=8). Đây không phải "càng lớn càng
tốt", cũng không phải "ASAM null" — nó là một cực đại nội tại, đo được trên hơn 1 300 ô.

### codebert thì ngược ở cả hai trục

| trục | mức | ΔF1@0.5 | ΔROC-AUC |
|---|---|---|---|
| **λ** (ρ=0) | **0.05** ← đang chạy | +0.0263 220/295 | **+0.0068 177/295** — dưới sàn nhiễu |
| | 0.2 | +0.0572 19/20 | +0.0302 18/20 |
| | 0.5 | +0.0512 10/10 | **+0.0350 8/10** |
| | 1.0 | +0.0513 10/10 | +0.0290 8/10 |
| **ρ** (λ=0.05) | 0 | +0.0263 220/295 | +0.0068 177/295 |
| | **0.1** | +0.0419 37/40 | **+0.0244 40/40** |
| | **2.0** ← đang chạy | −0.0070 8/12 | **−0.0439 6/12** |

Trên t5p, λ0.05 (−0.0101) hơn λ0.2 (−0.0555); trên codebert thì λ0.05 là mức **kém nhất** trong
bốn mức và là mức duy nhất không vượt sàn nhiễu 0.010. Tương tự với ρ: t5p thích ρ=1–2 còn
codebert có 40/40 fold dương ở ρ=0.1 và âm ở ρ=2.0.

### Vì sao đây là GIẢ THUYẾT chứ không phải phát hiện

Các mức λ đến từ **các khối khác nhau** (λ0.2/0.5/1.0 là khối `n1`, tháng 8) với n rất lệch
(295 / 20 / 10). Mỗi Δ **có** ghép cặp sạch trong cây của nó, nhưng **so giữa các mức λ thì
không ghép cặp** — đó là lấy hiệu của hai Δ tính riêng, đúng cái §25.9 đã trả giá một lần.
Muốn chắc phải chạy một khối λ trên codebert **cùng máy cùng fold**.

Chưa xếp khối đó: λ nằm ở Pha 1 nên đổi λ là phải huấn luyện lại Pha 1 (CLAUDE.md mục 5), và
người dùng chưa duyệt.

### §32.1 — "Chưa thử" bị tôi đọc thành "tốt nhất" (09/09)

Người dùng yêu cầu *"chạy 2 backbone với cái tốt nhất của nó"*. Tôi đặt **cả hai** ở ρ=2.0, lý do
tự nhủ là *"codebert chưa hề chạy ở ρ=2.0"* — tức đã lặng lẽ đổi câu hỏi từ **cái tốt nhất của nó**
sang **cái chưa thử**. Số cần để chọn đúng thì **đã nằm sẵn trong dữ liệu cũ** từ trước khi khối
bắt đầu: codebert ρ=0.1 cho ΔROC +0.0244 (**40/40 fold**), ρ=2.0 cho −0.0439 (6/12).

Người dùng bắt được sau khi khối đã chạy **8 ô** ở nhánh sai. Giá: ~40 phút GPU, và 8 ô đó vẫn
giữ vì chúng trả lời một câu có thật (ρ=2.0 không hợp codebert) — nhưng phải **loại khỏi phép
đếm** của khối, nếu không watchdog tưởng xong sớm 8 ô.

**"Tốt nhất" là phát biểu về số ĐÃ ĐO, không phải về khoảng trống trong lưới.** Chưa đo thì là
*chưa biết*. Muốn thăm dò ô chưa đo thì đó là thí nghiệm riêng và phải hỏi (CLAUDE.md mục 9),
không được gói vào khối "chạy cấu hình tốt nhất".

**Mặt được của cái lỗi đó**: 8 ô `r2p0` và các ô `r0p1` của codebert nằm trong **cùng một cây,
cùng fold, cùng nguồn, cùng baseline, cùng máy, cùng phiên**. Nên khi `r0p1` đủ 4 fold × 2 nguồn,
sẽ có **8 cặp ghép được thật** cho câu hỏi "ρ nào hợp codebert" — mạnh hơn hẳn §32, nơi buộc phải
lấy hiệu của hai Δ tính ở hai khối khác nhau (đúng chỗ yếu §25.9 đã cảnh báo). Đọc bằng
`tools/rho_paired.py results/chot_codebert r2p0 r0p1`. Ở n=8 sàn phép thử dấu là p=0.0078.

---

## §33 — Khối `chot`, nửa codebert (GD1, n=5): optimizer null, nhưng theo CWE thì LẶP LẠI (09/09)

Cấu hình: `latent_bottleneck` λ=0.05, seed 42, đích `sven_python_folds_norm`, nguồn `4cwe`+`com`,
5 fold, tất cả trên **một máy một phiên** (vast 5060 Ti). Δ ghép cặp với `baseline` cùng fold.
**A** = RecAdam + ASAM **ρ=0.1** (mức tốt nhất của chính codebert, §32). **B** = AdamW, không SAM.

| nhánh | ΔF1@0.5 | ΔF1@val | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|
| **B** `plain` | **+0.0508 10/10** p=0.002 | **+0.0599 10/10** p=0.002 | +0.0239 9/10 p=0.021 | +0.0058 6/10 |
| **A** ρ=0.1 | +0.0410 9/10 p=0.021 | +0.0463 9/10 p=0.021 | +0.0173 8/10 p=0.109 | +0.0013 6/10 |
| A cũ ρ=2.0 (n=8) | −0.0153 5/8 | +0.0026 5/8 | −0.0449 4/8 | −0.0511 3/8 |

### A − B ghép cặp trong CÙNG ô — phần optimizer đóng góp riêng

| nguồn | ΔF1@0.5 | ΔF1@val | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|
| 4cwe | −0.0146 1/5 | **−0.0199 0/5** p=0.062 | −0.0069 1/5 | −0.0024 2/5 |
| com | −0.0049 2/5 | −0.0073 3/5 | −0.0062 3/5 | −0.0065 3/5 |
| GỘP | −0.0098 3/10 | −0.0136 3/10 | −0.0066 4/10 | −0.0044 5/10 |

**Trên codebert, bật RecAdam+ASAM không thêm gì so với AdamW trần.** Âm ở cả bốn chỉ số nhưng
biên độ −0.004 đến −0.020, phần lớn ở hoặc dưới sàn nhiễu 0.010; chỉ F1@val trên `4cwe` chạm sàn
phép thử (0/5). Gọi đúng là **null đến hơi âm**, không phải "có hại". **Ngược với t5p** (§28), nơi
ASAM ρ=2.0 làm ROC hơn 2,5×.

### Theo CWE thì mẫu hình LẶP LẠI qua backbone — chỗ đáng viết

A(ρ=0.1) − baseline, ΔROC-AUC theo CWE, so với §28 trên t5p:

| | CWE-022 | CWE-078 | CWE-079 | CWE-089 |
|---|---|---|---|---|
| **codebert** (n=10, đây) | **+0.3965 10/10** p=0.002 | +0.0104 5/10 | **+0.3474 10/10** p=0.002 | −0.0036 3/10 |
| **t5p** (n=15, §28) | **+0.2238 13/15** | — | **+0.3638 15/15** | — |

**Cùng hai CWE thắng, cùng hai CWE null, trên hai backbone khác nhau với hai cấu hình optimizer
khác nhau.** Đây là lần thứ **sáu** trong dự án này mẫu hình ở mức **phân phối** giữ vững trong
khi phát biểu ở mức **tổng hợp** yếu đi.

### Chưa chắc

- Mới là **GD1** (`4cwe`+`com`); nguồn `full` đang chạy.
- Cột GỘP n=10 dùng **chung 5 baseline** nên p=0.002 ở đó **lạc quan** — mỗi nguồn thực chất
  n=5, sàn p=0.0625.
- **PR-AUC null ở mọi nhánh** (6/10, 6/10, 3/8). Ba chỉ số kia đồng thuận, PR thì không.
- Nửa t5p của khối chưa xong.

---

## §34 — KHỐI `chot` TRỌN VẸN: 2 backbone × 2 điều kiện × 3 nguồn × 5 fold = 70 ô (09/09)

Người dùng đặt: *"chạy 2 backbone với cái tốt nhất có đầy đủ optimizer hiện tại của phương pháp"*
và *"cái này sẽ tổng hợp riêng kĩ nhé"*. Đây là bản tổng hợp đó.

**Cấu hình.** `latent_bottleneck` (nút thắt 8 chiều), λ=0.05 ở Pha 1, seed 42, đích
`sven_python_folds_norm` (152 hàng test/fold). Pha 1 **dùng lại cả sáu checkpoint**, không huấn
luyện lại cái nào.

| | Pha 2 | t5p | codebert |
|---|---|---|---|
| **A** | RecAdam + ASAM, ρ **tốt nhất của chính backbone đó** | ρ=2.0 | ρ=0.1 |
| **B** | AdamW, tắt cả SAM lẫn RecAdam | ρ=0 | ρ=0 |
| đối chứng | `baseline` — không Pha 1, dùng chung cho A và B trong cùng fold | | |

**Máy.** codebert trọn trên vast 5060 Ti. t5p chia theo **fold trọn vẹn**: fold 1,2,3,5 trên 161
(A4000), fold 4 trên vast. Δ luôn ghép cặp trong cùng cây/fold nên không bao giờ lấy hiệu giữa
hai máy; nhưng độ tản **giữa các fold** của t5p có thêm phần phần cứng.

### 1. Cả hai điều kiện đều hơn baseline, ở cả hai backbone

| | nguồn | ΔF1@0.5 | ΔROC-AUC |
|---|---|---|---|
| **codebert A** (ρ0.1) | 4cwe / com / full | +0.0389 4/5 · +0.0432 5/5 · +0.0617 5/5 | +0.0187 · +0.0159 · +0.0338 |
| | **GỘP** | **+0.0479 14/15** | **+0.0228 12/15** |
| **codebert B** | **GỘP** | **+0.0540 15/15** | **+0.0264 13/15** |
| **t5p A** (ρ2.0) | 4cwe / com / **full** | +0.0541 4/5 · +0.0580 5/5 · **−0.0165 4/5** | +0.0445 · +0.0254 · **−0.0364** |
| | **GỘP** | +0.0319 13/15 | +0.0111 13/15 |
| **t5p B** | **GỘP** | +0.0337 12/15 | +0.0198 12/15 |

### 2. Optimizer đóng góp ĐÚNG BẰNG KHÔNG — ở CẢ HAI backbone

A − B ghép cặp trong cùng ô (tách riêng phần optimizer):

| backbone | nguồn | ΔF1@0.5 | ΔROC-AUC |
|---|---|---|---|
| codebert | 4cwe / com / full | −0.0146 1/5 · −0.0049 2/5 · +0.0012 3/5 | −0.0069 · −0.0062 · +0.0024 |
| | **GỘP** | **−0.0061 6/15** | **−0.0036 8/15** |
| t5p | **4cwe** | **+0.0305 5/5** p=0.0625 | +0.0298 4/5 |
| | com | +0.0093 2/5 | −0.0029 2/5 |
| | **full** | **−0.0450 3/5** | **−0.0530 4/5** |
| | **GỘP** | **−0.0018 10/15** | **−0.0087 10/15** |

**Đây là kết quả quan trọng nhất của khối, và nó sửa lại §28.** Trên codebert, optimizer null
(mọi số dưới sàn nhiễu 0.010, đếm dấu quanh 50%). Trên t5p, **gộp cả ba nguồn cũng null** — lợi
ích ASAM chỉ có ở `4cwe` (+0.0305, **5/5**) và bị `full` triệt tiêu (−0.0450). §28 đo ASAM chỉ
trên một nhánh nguồn nên thấy hiệu ứng lớn; trải đủ ba nguồn thì nó **phụ thuộc nguồn**, không
phải một hiệu ứng chung.

Trị tuyệt đối cho thấy rõ mức thiệt: `t5p × full × A` ROC **0.8532** trong khi baseline 0.8896 và
B 0.9061 — ASAM ρ=2.0 **làm hỏng** t5p ở nguồn `full`.

### 3. Theo CWE: mẫu hình LẶP LẠI qua cả hai backbone VÀ cả hai nhánh

ΔROC-AUC so với baseline:

| backbone | nhánh | CWE-022 (8 hàng) | CWE-078 (42) | CWE-079 (19) | CWE-089 (83) |
|---|---|---|---|---|---|
| codebert | A | **+0.4151 15/15** | +0.0147 9/15 | **+0.3745 15/15** | −0.0036 5/15 |
| codebert | B | **+0.4038 15/15** | +0.0059 7/15 | **+0.3818 15/15** | −0.0047 6/15 |
| t5p | A | **+0.2224 15/15** | −0.0184 7/15 | **+0.2456 15/15** | −0.0168 11/15 |
| t5p | B | **+0.2349 14/15** | −0.0371 3/15 | **+0.2495 15/15** | +0.0070 12/15 |

**Bốn dòng, hai backbone, hai cấu hình optimizer — CWE-022 và CWE-079 dương ở 14–15/15 fold mọi
lần; CWE-078 và CWE-089 null mọi lần.** Vì A và B trùng nhau trong sai số, lợi ích này đến từ
**Pha 1 + head nút thắt**, không từ optimizer.

### 4. Chỗ phải nói kèm, nếu không sẽ đọc sai

- **Hai CWE thắng là hai nhóm NHỎ NHẤT**: CWE-022 chỉ **8 hàng test**, CWE-079 **19** — cộng lại
  18% dữ liệu. Hai nhóm null là hai nhóm lớn nhất (42 và 83). Đó chính là lý do ROC tổng thể chỉ
  ~+0.02 trong khi per-CWE tới +0.4. ROC trên 8 hàng rất nhiễu **trong một fold**; điều đỡ cho
  phát biểu là **đếm dấu 15/15 qua fold**, không phải biên độ.
- Cột GỘP n=15 dùng **chung 5 baseline** cho 3 nguồn ⇒ không phải 15 quan sát độc lập. Mỗi nguồn
  thực chất n=5, **sàn p=0.0625**.
- **PR-AUC yếu nhất trong bốn chỉ số** ở codebert (11/15 và 10/15).
- t5p fold 4 chạy trên GPU khác ba fold kia.
- 8 ô codebert ở ρ=2.0 giữ lại làm bằng chứng (ROC **−0.0439**, 6/12 so với baseline) — không
  thuộc 35 ô của khối.

### §34.1 — ROC-AUC **hoà thật** trên tập đích, và dấu phẩy động đang âm thầm phá vỡ thế hoà

Khi dựng trang cho khối, trang và `chot_report.py` lệch nhau đúng một fold (7/15 vs 8/15 ở
`codebert A−B` ROC). Truy ra không phải lỗi làm tròn mà là một hiện tượng thật:

| | codebert `4cwe` fold 5 | codebert `full` fold 3 |
|---|---|---|
| ROC-AUC nhánh A | 0.8781249999999999 | 0.896701388888889 |
| ROC-AUC nhánh B | 0.878125 | 0.8967013888888888 |
| **hiệu** | **−1.11e-16** | **+1.11e-16** |
| xác suất hai nhánh có giống nhau không | **không** (lệch tối đa 0.958) | **không** (0.958) |
| F1@0.5 | 0.7889 vs 0.8078 | 0.8485 vs 0.8618 |

Hai mô hình **khác hẳn nhau** nhưng ROC-AUC bằng nhau đến epsilon. Lý do: **ROC-AUC là thống kê
thứ hạng.** Hai vector xác suất rất khác nhau mà sinh **cùng một thứ tự** dương/âm thì cho đúng
cùng một AUC. Trên 152 hàng test (76/76) thì AUC chỉ nhận các giá trị rời rạc `k/5776`, nên hoà
là chuyện thường — đo được **2/15 ô**.

**Cái nguy hiểm**: hiệu không phải 0 chẵn mà là ±1.1e-16 do thứ tự cộng dồn dấu phẩy động, và
**dấu của nó là ngẫu nhiên**. `stat()` cũ lọc `v != 0` nên cái +1.1e-16 lọt qua và được đếm thành
một fold "thắng" — `full` thành 4/5 thay vì 3/5. Một phép thử dấu đang lấy dữ liệu từ nhiễu làm
tròn của phép cộng.

**Sửa**: coi mọi `|Δ| < 1e-12` là **HOÀ** — bỏ khỏi phép thử dấu (cách xử lý hoà chuẩn) và **in ra
số ô hoà** (`~k`) thay vì giấu. Tìm thấy hoà ở cả F1 nữa (F1 cũng là hàm bậc thang trên 152 hàng).

**Quy tắc chung**: mọi phép so hai mô hình bằng thống kê **thứ hạng hoặc bậc thang** trên tập nhỏ
đều phải có ngưỡng hoà tường minh. `x != 0` không phải phép kiểm hoà — nó là phép kiểm "có khác
nhau ở bit cuối cùng không".

---

## §30.2 — ĐÃ ĐO: head phụ gần như không học được gì, ở **mọi nguồn và cả hai backbone** (10/09)

Trả lời câu hỏi mở §30.1 bằng `tools/aux_head_probe.py` — chỉ đọc, một lượt forward, không
huấn luyện. Tập val được chia lại **bằng chính `split_source_records` của `train_transfer.py`
với đúng seed của checkpoint**, nên nó trùng tập đã dùng để chọn checkpoint.

| checkpoint | độ chính xác head | sàn lớp đa số | macro-F1 head | sàn | **số lớp head thực dùng** |
|---|---|---|---|---|---|
| codebert/`4cwe` | 0.6667 | **0.6667** | 0.2000 | **0.2000** | **1** / 4 |
| t5p/`4cwe` | 0.6667 | **0.6667** | 0.2000 | **0.2000** | **1** / 4 |
| codebert/`com` | 0.4891 | 0.3804 | 0.1864 | 0.0787 | 3 / 10 |
| t5p/`com` | 0.4891 | 0.3804 | 0.1629 | 0.0787 | 2 / 10 |
| codebert/`full` | 0.5740 | 0.4793 | 0.1900 | 0.0810 | 3 / 10 |
| t5p/`full` | 0.5740 | 0.4793 | 0.1644 | 0.0810 | 2 / 10 |

**Trên `4cwe`, head SẬP HOÀN TOÀN ở cả hai backbone** — bằng sàn đến từng chữ số, và dự đoán
**một lớp duy nhất** cho mọi hàng val. Đúng như §30.1 đã cảnh báo: CWE-79 chiếm 692/930 nên
"đoán CWE-79" đã đúng 74%, và λ·loss_phụ gần như là hằng số ngay từ đầu.

Trên `com`/`full` head **có vượt sàn** (macro-F1 gấp ~2,2×) nhưng chỉ đúng ở **hai lớp lớn nhất**
— lớp 2 (pillar CWE-664, kiểm soát tài nguyên) và lớp 8 (pillar CWE-707, neutralization). Mọi lớp
còn lại **0% đúng**, kể cả lớp 7 (CWE-703) có tới **84 hàng** trong `full`:

```
full   lop  n    codebert dung   t5p dung
        0   30        0             0
        2  324      306           303
        3   40        0             0
        4   24        0             0
        5   24        3             0
        7   84        0             0     <- 84 hang, khong dung mot hang nao
        8  148       79            85
        9    2        0             0
```

Head phụ thực chất là **một bộ phân biệt hai pillar lớn nhất**, không phải bộ phân loại CWE.

**Hệ quả cho cách viết bài.** Không được nói "head phụ dạy encoder cấu trúc CWE" — nó không học
được cấu trúc đó. Lợi ích transfer (§34) phải đến từ chỗ khác: huấn luyện nhị phân lỗ hổng ở Pha 1
(tức domain-adaptive pretraining), và/hoặc chính cái phân biệt hai-pillar thô mà head có học,
và/hoặc tác dụng chính quy hoá của một số hạng loss thêm vào.

**Một quan sát, chưa phải kết luận**: hai CWE ăn đậm ở Pha 2 là CWE-022 và CWE-079, ánh xạ về
đúng hai pillar mà head học được (664 và 707). Nhưng CWE-078 và CWE-089 cũng thuộc pillar 707 mà
lại null — nên mối liên hệ này **chưa giải thích được** và không được viết như một cơ chế.

**Đã kiểm dấu hiệu "giống nhau ở chỗ lẽ ra phải khác"** (bài học §30): codebert và t5p cho độ
chính xác **trùng khít** trên `com` (0.4891) và `full` (0.5740). Không phải lỗi — số đúng từng lớp
khác nhau, chỉ trùng **tổng** (180=180, 388=388); macro-F1 cũng khác nhau.

---

## §35 — THÊM HAI SEED: hiệu ứng tổng thể của t5p ĐỔI DẤU, per-CWE giữ nguyên (10/09)

Người dùng yêu cầu nâng khối `chot` lên n=15 (seed 42 + 7 + 1234). Đọc khi còn 11/140 ô chưa xong
(seed 1234 dở dang) — **đủ để thấy một điều phải ghi ngay**.

### t5p nhánh A (ASAM ρ=2.0): seed 42 dương, hai seed mới ÂM

| seed | ΔF1@0.5 | ΔROC-AUC | fold + |
|---|---|---|---|
| **42** | **+0.0319** 13/15 | **+0.0111** 13/15 | dương |
| **7** | **−0.0357** 9/15 | **−0.0607** 8/15 | **âm** |
| **1234** | **−0.0253** 9/14 | **−0.0342** 10/14 | **âm** |
| GỘP 44 | −0.0093 31/44 | **−0.0278** 31/44 | |

**Đây là lần thứ tám một mẫu hình co lại khi thêm dữ liệu — và lần đầu nó ĐỔI DẤU.** §28 và §34
đều đo trên seed 42; con số ROC +0.0337 của §28 không sống sót khi thêm hai seed.

codebert nhánh A: +0.0228 (s42) → +0.0014 (s7) → +0.0241 (s1234), gộp **+0.0155, 28/42** — vẫn
dương nhưng yếu hơn hẳn con số một-seed.

### Hai thứ GIỮ VỮNG

**1. F1 vẫn chắc qua cả ba seed**: codebert A **+0.0440, 39/42** (p<0.0001); codebert B **+0.0445,
40/41**; t5p B +0.0231, 33/43. Chỉ ROC/PR của t5p A là đổi dấu.

**2. Per-CWE lặp lại y nguyên** — bốn dòng độc lập (2 backbone × 2 nhánh), ~42 ô mỗi dòng:

| | CWE-022 (8 hàng) | CWE-078 (42) | CWE-079 (19) | CWE-089 (83) |
|---|---|---|---|---|
| codebert A | **+0.3605 42/42** | +0.0259 29/42 | **+0.3452 42/42** | −0.0098 15/42 |
| codebert B | **+0.3462 41/41** | +0.0198 22/41 | **+0.3457 41/41** | −0.0168 10/41 |
| t5p A | **+0.2211 42/44** | −0.0416 19/44 | **+0.2267 42/44** | −0.0595 20/44 |
| t5p B | **+0.2584 39/43** | −0.0431 6/43 | **+0.2326 43/43** | −0.0077 16/43 |

**42/42, 41/41, 43/43 — không một fold nào đi ngược.** Ở n=15 với ba seed độc lập, đây là phát
biểu mạnh nhất dự án này có.

### Optimizer: null ở codebert, ÂM ở t5p

A − B ghép cặp, gộp ba seed: codebert **−0.0005 F1** (18/41), t5p **−0.0329 F1** (22/43) và
**−0.0347 ROC**. Trên t5p, bật RecAdam+ASAM ρ=2.0 giờ **hại** chứ không phải null.

### Điều này nói gì về cách viết bài

Lần thứ **bảy** dự án này thấy: **phát biểu ở mức TỔNG HỢP yếu đi hoặc đổi dấu khi thêm dữ liệu;
phát biểu ở mức PHÂN PHỐI (per-CWE) thì giữ.** Kết quả đầu bài phải là bảng per-CWE, không phải Δ
tổng thể. Và **ASAM phải rút khỏi cấu hình chốt** — nó không sống sót đa seed trên t5p.

### §35.1 — Giả thuyết "thắng vì nguồn nhiều dữ liệu cùng CWE" BỊ BÁC (10/09)

Kiểm giả thuyết đơn giản nhất cho §35: CWE-022 và CWE-079 thắng vì nguồn Pha 1 có nhiều mẫu
thuộc đúng hai CWE đó. **Sai.**

| CWE | trong nguồn `4cwe` | hàng test (fold 1) | kết quả |
|---|---|---|---|
| CWE-079 | **692 (74,4%)** | 14 | **+0.35 thắng** |
| CWE-078 | 100 (10,8%) | 39 | null |
| **CWE-022** | **92 (9,9%)** | 19 | **+0.36 thắng** |
| CWE-089 | 46 (4,9%) | 80 | null |

CWE-022 chiếm **ít hơn** CWE-078 trong nguồn (9,9% vs 10,8%) mà vẫn thắng, còn CWE-078 thì null.
Lượng dữ liệu nguồn không giải thích được.

**Một cách giải thích cạnh tranh, phải loại trừ trước khi viết cơ chế**: hai CWE thắng là hai
CWE **ít hàng test nhất** (14 và 19 trên 152); hai CWE null là hai nhiều nhất (39 và 80). Đây có
thể là hiện tượng về **kích thước nhóm** chứ không phải về nội dung — ROC trên 14–19 hàng biến
động mạnh hơn nhiều, nên chênh lệch dễ lớn hơn về biên độ. Cái chống lại cách giải thích này là
**đếm dấu 42/42 qua fold**, nhưng cần kiểm thẳng.

**Đính chính**: các mục trước ghi "CWE-022 chỉ 8 hàng test" — con số đó lấy từ `test_cwe_classes`
gộp, không phải một fold thật. Số đúng ở fold 1: **CWE-022 19, CWE-078 39, CWE-079 14, CWE-089 80**.

### §35.2 — Khối n=15 TRỌN VẸN: 210 ô, ba seed (10/09 17:10)

Đủ 210 ô (3 seed × 2 backbone × 3 nguồn × 5 fold), không ô nào thiếu, không `.rejected` mới.
Máy: seed 42 trên vast 5060 Ti + 161; seed 7 và 1234 trên 161 (codebert) và 158 (t5p).

**So với baseline, gộp 45 ô mỗi dòng:**

| | ΔF1@0.5 | ΔROC-AUC |
|---|---|---|
| codebert A (ρ0.1) | **+0.0450 42/45** | +0.0138 29/45 |
| codebert B (AdamW) | **+0.0441 44/45** | +0.0105 26/45 |
| t5p A (ρ2.0) | **−0.0088 32/45** | **−0.0274 31/45** |
| t5p B (AdamW) | **+0.0214 33/45** | +0.0047 27/45 |

**Bốn chỉ số nói khác nhau.** F1 chắc (p<0.00001 ở codebert); ROC/PR yếu, và t5p A **âm**.
Phương pháp cải thiện **quyết định ở ngưỡng** rõ hơn cải thiện **thứ hạng điểm**.

**ΔROC theo seed — không ổn định:**

| | s42 | s7 | s1234 |
|---|---|---|---|
| codebert A | +0.0228 | +0.0014 | +0.0172 |
| codebert B | +0.0264 | **−0.0049** | +0.0100 |
| t5p A | **+0.0111** | **−0.0607** | **−0.0326** |
| t5p B | +0.0198 | −0.0073 | +0.0015 |

**A − B ghép cặp (phần optimizer):** codebert +0.0010 F1 (21/45) — null; t5p **−0.0302 F1**
(24/45) và **−0.0321 ROC** (29/45) — **có hại**.

**Per-CWE — thứ duy nhất giữ vững:** codebert A: CWE-022 **+0.3677 45/45**, CWE-079 **+0.3500
45/45**; codebert B: +0.3569 45/45 và +0.3474 45/45; t5p A: +0.2204 43/45 và +0.2209 42/45;
t5p B: +0.2528 41/45 và +0.2207 44/45. **CWE-078 và CWE-089 null ở mọi dòng.**

**Số hàng test đúng** (trung bình qua fold, kèm min–max): CWE-022 **13 (8–19)**, CWE-078 41
(38–44), CWE-079 **16 (13–20)**, CWE-089 82 (79–87). Tổng 152.

**Kết luận cho cấu hình chốt: RÚT ASAM.** Cấu hình nên là `latent_bottleneck` + λ=0.05 +
**AdamW trần**. Trang kết quả đã cập nhật (ARTIFACTS.md).

---

## §36 — ĐO TRỰC TIẾP: **đặc trưng CHUYỂN GIAO, hàm quyết định KHÔNG** (10/09, 0 GPU)

Suốt dự án, câu "chỉ có đặc trưng là chuyển được, hàm quyết định thì không" được **suy ra** từ
kết quả cuối (§12.2 của RESEARCH_2026-09-06), chưa lần nào **đo thẳng**. Nay đo được, và nó là
phát biểu cơ chế mạnh nhất dự án có.

### Phép đo — `tools/feature_probe.py`, chạy CPU, không tốn GPU

Trích đặc trưng pooled (CLS cho codebert, mean cho t5p, **trước dropout**) của **760 dòng Python**
từ hai mô hình **đóng băng hoàn toàn**:

* **(a)** backbone pretrained nguyên bản (chưa hề thấy dữ liệu lỗ hổng nào)
* **(b)** checkpoint Pha 1 (`latent_bottleneck`, `4cwe`, λ=0.05, seed 42)

Mỗi fold: chuẩn hoá theo train, fit logistic regression trên 456 dòng train, chọn C trên val theo
ROC-AUC, chấm test. Δ ghép cặp theo fold. Thêm một cột thứ ba: **zero-shot của chính `vul_head`
Pha 1** trên đặc trưng (b) — không fit gì.

### Kết quả

| | ΔF1@0.5 | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|
| **codebert** (Pha1 LP − pretrained LP) | **+0.0849 5/5** | **+0.1125 5/5** | **+0.1010 5/5** |
| t5p | +0.0198 4/5 | +0.0202 3/5 | +0.0097 3/5 |

**codebert: cả ba chỉ số dương ở CẢ NĂM fold, biên độ gấp 4–11 lần sàn nhiễu 0.010.** Đây là
bằng chứng trực tiếp rằng Pha 1 làm không gian đặc trưng của Python **tách được tuyến tính hơn
hẳn** — dù Pha 1 chưa từng thấy một dòng Python nào.

**Còn hàm quyết định thì không chuyển:** `vul_head` của chính Pha 1, chấm thẳng trên Python,

| | F1@0.5 (dải qua 5 fold) | ROC-AUC |
|---|---|---|
| codebert | **0.537** (0.488–0.591) | 0.645 |
| t5p | **0.503** (0.474–0.542) | 0.545 |

t5p ROC 0.545 ≈ ngẫu nhiên, khớp với INT1 (§12.1 RESEARCH: α=0 cho 0.5205).

### Hai điều mới, không suy ra được từ số cũ

1. **Hiệu ứng đặc trưng KHÔNG đều giữa hai backbone.** codebert +0.1125 ROC (5/5); t5p +0.0202
   (3/5) — dưới sàn nhiễu. Nghĩa là "Pha 1 cải thiện đặc trưng" là phát biểu về **codebert**,
   chưa lặp được trên t5p. Và nó đúng chiều với biên độ end-to-end (codebert +0.045 F1 vs t5p
   +0.021, §35.2).
2. **Theo CWE, hai backbone học hai thứ khác nhau.** ΔROC-AUC của LP:

   | | 022 | 078 | 079 | 089 |
   |---|---|---|---|---|
   | codebert | +0.169 **5/5** | +0.199 **5/5** | +0.159 **5/5** | +0.046 4/5 |
   | t5p | −0.023 2/5 | +0.027 2/5 | **+0.173 5/5** | −0.017 2/5 |

   Trên **t5p, thứ DUY NHẤT chuyển giao trong không gian đặc trưng là CWE-079** — đúng lớp chiếm
   **74%** nguồn `4cwe` (692/930 dòng CWE-79). Trên codebert thì cả bốn lớp đều lên.

### Nghịch lý đáng ghi: 078 lên mạnh nhất ở đặc trưng nhưng NULL ở kết quả cuối

codebert được **+0.199 ROC 5/5** cho CWE-078 trong không gian đặc trưng, nhưng end-to-end
(§34/§35) CWE-078 **null ở mọi phép đo**. Nghĩa là fine-tune Pha 2 **không dùng** phần đặc trưng
đã tốt lên đó — 456 dòng train đủ để mô hình tự tìm lời giải riêng cho lớp lớn (078 có 40 hàng
test, 089 có 82) và chỉ giữ lợi thế nguồn ở hai lớp hiếm. Đây là giả thuyết đọc được, chưa phải
kết luận.

### Cảnh báo khi trích số

Bộ `norm` chia theo dòng nên ~40% hàng test có bản gần trùng trong train; **cả hai vế của Δ đều
chịu chung**, nên Δ ghép cặp vẫn hợp lệ, nhưng **con số tuyệt đối là lạc quan**. LP tuyệt đối
thấp hơn fine-tune đầy đủ nhiều (codebert LP ROC 0.73–0.82 so với FT ~0.88), đúng như kỳ vọng.

Số đầy đủ: `results/probe/codebert_4cwe.json`, `results/probe/t5p_4cwe.json`.

---

## §37 — KHỐI `bridge3`: replay nguồn và cầu CWE ở Pha 2 (10/09, **kiểm chứng n=3, seed 42**)

Trả lời câu hỏi mở của §11.4 RESEARCH_2026-09-06 (Đ5, chưa ai chạy): **đưa dữ liệu nguồn vào
chính Pha 2** thì tri thức nguồn có giúp đích không? Mã: `src/replay.py` + cờ `--replay_*` /
`--phase2_lambda_cwe`. 4 nhánh × 3 fold × 2 backbone + baseline = 30 ô, hai máy local, 0 vast.

| tag | cơ chế |
|---|---|
| `plain` | **đối chứng** — fine-tune hai lần thuần |
| `cwe05` | head phụ 4 lớp học tiếp trên nhãn CWE **của Python** (λ=0.05) |
| `rp50` | mỗi bước thêm một batch **nguồn 4cwe**, μ giảm tuyến tính 0.5→0 sau 6 epoch, cân tầng (nhãn, CWE) |
| `rpc` | cả hai + head phụ học cả trên nguồn |

### Δ so với `plain` (ghép cặp cùng fold, n=3 — **chỉ để sàng lọc**)

| | | F1@0.5 | F1@val | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **codebert** | `cwe05` | +0.0196 2/3 | +0.0040 2/3 | −0.0108 1/3 | −0.0012 1/3 |
| | `rp50` | −0.0019 1/3 | −0.0159 1/3 | +0.0029 1/3 | +0.0070 1/3 |
| | `rpc` | +0.0023 2/3 | −0.0131 1/3 | −0.0117 **0/3** | −0.0196 **0/3** |
| **t5p** | `cwe05` | +0.0305 1/3 | +0.0394 2/3 | −0.0104 **0/3** | −0.0372 **0/3** |
| | **`rp50`** | **+0.0217 2/3** | **+0.0323 3/3** | **+0.0122 2/3** | **+0.0099 2/3** |
| | `rpc` | +0.0002 1/3 | −0.0068 1/3 | −0.0168 **0/3** | −0.0327 **0/3** |

### Ba điều đọc được

1. **`cwe05` là hiệu ứng NGƯỠNG, không phải hiệu ứng xếp hạng.** Trên **cả hai** backbone nó nâng
   F1@0.5 (+0.0196 / +0.0305) mà **hạ** ROC và PR (0–1/3 fold). Đúng mẫu hình đã cảnh báo ở
   RESEARCH §2.3. Cho head phụ học nhãn CWE của đích **không** làm mô hình xếp hạng tốt hơn.
2. **`rpc` (cầu đầy đủ) HẠI trên AUC ở cả hai backbone** — ROC 0/3 và PR 0/3 ở cả codebert lẫn
   t5p. Ghép hai cơ chế lại thì phần `cwe05` kéo xuống nhiều hơn phần replay kéo lên.
3. **`rp50` (replay THUẦN) là nhánh duy nhất dương cả bốn chỉ số — nhưng chỉ trên t5p.** Trên
   codebert nó null (1/3 ba lần). **Không lặp qua backbone ⇒ giả thuyết, không phải phát hiện**
   (CLAUDE.md mục 2b). Đáng lên n=5 để xem có sống không.

### Per-CWE: replay ăn đúng CWE-022, và chỉ trên codebert

So với `plain`, codebert: `rp50` cho CWE-022 **F1 +0.125 3/3, ROC +0.113 3/3**; `rpc` cho
**+0.118 3/3 / +0.157 3/3**. t5p không lặp (022 chỉ 1/3). CWE-022 chỉ **14 hàng test/fold** nên
biên độ không đáng tin; thứ đáng chú ý là **3/3 cùng dấu ở cả hai nhánh có replay**.

Đọc bằng `python3 tools/bridge_report.py results/bridge3_codebert` (và `_t5p`) — in cả bốn chỉ số
và per-CWE, ghép cặp theo `(cây, seed, fold)`, ngưỡng hoà 1e-12.

---

## §38 — `feat3` + `lpft3` trên CODEBERT: neo đặc trưng KHÔNG ăn, nhưng **probe head Pha 1 rồi mới fine-tune** thì có (11/09, **kiểm chứng n=3, seed 42**)

Sáu nhánh Pha 2, tất cả trên `adamw --sam_rho 0`, đối chứng `plain` **cùng máy cùng fold cùng
ngày** (dùng lại của khối `bridge3`), cây `results/bridge3_codebert`. 161, A4000.

| nhánh | cờ | ΔF1@0.5 | ΔF1@val | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|---|
| **`lp3`** | `--lp_epochs 3` | **+0.0283 3/3** | +0.0223 2/3 | **+0.0204 3/3** | **+0.0307 3/3** |
| `fd1` | `--feat_distill_beta 1` | +0.0110 3/3 | +0.0053 1/3 ~1 | +0.0066 2/3 | +0.0245 2/3 |
| `rhlp3` | `--lp_epochs 3 --phase2_reinit_head` | −0.0127 1/3 | −0.0193 0/3 | +0.0058 2/3 | +0.0070 2/3 |
| `rh` | `--phase2_reinit_head` | −0.0113 0/3 ~1 | −0.0108 1/3 | +0.0000 1/3 | +0.0047 1/3 |
| `fd10` | `--feat_distill_beta 10` | −0.0112 1/3 | −0.0157 1/3 | −0.0136 1/3 | +0.0121 2/3 |
| `fd10rh` | cả hai | −0.0115 1/3 | −0.0123 1/3 | −0.0135 1/3 | +0.0119 2/3 |

### Ba điều đọc được (bậc 1 — chỉ SÀNG LỌC)

1. **`lp3` là nhánh mạnh nhất từng thấy ở khối này**: dương **cả bốn** chỉ số, ba trong bốn ở
   **3/3 fold**, biên độ ROC +0.0204 gấp đôi sàn nhiễu. Cơ chế đúng như Kumar et al. ICLR 2022
   (LP-FT) mô tả: fit head trước với backbone đóng băng thì gradient đầu tiên của toàn mô hình
   không còn bị một head hỗn loạn kéo đi, nên đặc trưng tốt không bị méo.

2. **Nhưng biến thể THẮNG lại KHÔNG phải LP-FT sách giáo khoa.** `lp3` giữ **head của Pha 1** rồi
   tinh chỉnh nó; `rhlp3` khởi tạo lại head rồi probe — đúng công thức gốc của Kumar et al. — và
   nó **null/âm** (F1 −0.0127, F1@val 0/3). Nghĩa là head Pha 1, dù §36 đo được chỉ ~0.537 F1
   zero-shot trên Python, **vẫn là điểm xuất phát tốt hơn ngẫu nhiên** cho bước probe. Đây là
   điểm khác biệt phải nêu rõ nếu viết: bài gốc giả định head **ngẫu nhiên**.

3. **Neo không gian đặc trưng KHÔNG ăn, và càng neo chặt càng tệ.** β=1 còn dương yếu, β=10 âm
   trên ba chỉ số. Khởi tạo lại head một mình cũng âm. Xem §38.1 cho lý do cơ chế.

### §38.1 — Vì sao neo đặc trưng hại trên t5p mà không hại trên codebert

t5p (n=2 fold, đang chạy): `fd1` **−0.0516 ROC 0/2**, `fd10` **−0.0848 0/2**, `fd10rh`
**−0.1189 0/2**. Ngược hẳn codebert.

Điều này **khớp đúng** §36: đặc trưng Pha 1 của codebert chuyển giao mạnh (**+0.1125 ROC, 5/5**),
còn của t5p thì gần như không (**+0.0202, 3/5**, hai fold âm). Neo mô hình vào một không gian đặc
trưng **không tốt hơn** điểm xuất phát thì chỉ còn là ràng buộc thuần tuý — nó cấm mô hình đi tìm
đặc trưng tốt hơn mà không đổi lại được gì.

Đây **không phải giải thích nghĩ ra sau**: §36 được đo **trước** khi khối `feat3` chạy, và dấu
của nó dự báo đúng dấu của kết quả trên cả hai backbone. Tài liệu có ghi nhận cơ chế này nhưng
chưa ai biến thành quy tắc — BSS (NeurIPS 2019) thấy L2-SP hại ở đích nhỏ **trừ** trên Stanford
Dogs (gần nguồn nhất), và quy cho *"the transferability of pre-trained knowledge"*; Plested et al.
(ICONIP 2021) dùng đúng margin probe-đóng-băng để chọn có neo hay không, nhưng chỉ n=4 dataset,
một backbone, không kiểm định. Chi tiết ở `RESEARCH_2026-09-10_dactrung.md` §5.4.

**CHƯA ĐƯỢC KẾT LUẬN**: `lp3` mới có trên **một** backbone. Luật leo bậc đêm nay là dương cả bốn
chỉ số **và lặp trên cả hai backbone** ở n=3 mới được lên n=5. `lpft3` trên t5p đang xếp hàng sau
`feat3`. Nếu t5p lặp lại thì mới chạy fold 4–5.

## §38.2 — MẪU HÌNH: **mọi nhánh thắng ở backbone này đều thua ở backbone kia**, và §36 dự báo đúng chiều (11/09, n=3)

Đủ 9 nhánh × 2 backbone × 3 fold, đối chứng `plain` cùng máy cùng fold. Đánh dấu ✓ khi **dương
cả bốn** chỉ số:

| nhánh | can thiệp thuộc loại | codebert | t5p |
|---|---|---|---|
| `lp3` | **giữ/khai thác** đặc trưng sẵn có (fit head trên backbone đóng băng) | **✓** +0.0283 / +0.0204 ROC | ✗ (F1 −0.0058) |
| `fd1` | **giữ** (neo đặc trưng nhẹ) | **✓** +0.0110 / +0.0066 ROC | ✗ (ROC −0.0351 0/3) |
| `rp50` | **thêm** thông tin mới (replay dữ liệu nguồn) | ✗ (F1 −0.0019) | **✓** +0.0217 / +0.0122 ROC |
| `rhlp3` | **thay** (khởi tạo lại head rồi probe) | ✗ (F1 −0.0127, F1@val 0/3) | **✓** +0.0021 / **+0.0147 ROC 3/3** |
| `rh` | thay (chỉ khởi tạo lại head) | ✗ | ✗ (2/4 dương) |
| `cwe05` | thêm (head CWE học nhãn đích) | ✗ (AUC âm) | ✗ (AUC âm) |
| `rpc` | thêm (cả hai) | ✗ (0/3 hai AUC) | ✗ (0/3 hai AUC) |
| `fd10` | giữ **chặt** (β=10) | ✗ | ✗✗ (−0.0659 ROC) |
| `fd10rh` | giữ chặt + thay | ✗ | ✗✗ (−0.0920 ROC) |

**KHÔNG nhánh nào ✓ ở cả hai backbone ⇒ theo luật leo bậc đêm nay, KHÔNG nhánh nào được lên n=5.**

### Nhưng cái ✗/✓ đó không ngẫu nhiên — nó tách đúng theo LOẠI can thiệp

- **codebert** — đặc trưng Pha 1 **chuyển giao mạnh** (§36: +0.1125 ROC, **5/5 fold**). Hai nhánh
  thắng đều thuộc loại **giữ/khai thác cái đã có** (`lp3`, `fd1`). Ba nhánh loại **thay/thêm** đều
  thua.
- **t5p** — đặc trưng Pha 1 **gần như không chuyển giao** (§36: +0.0202, 3/5, hai fold âm). Đảo
  ngược hoàn toàn: hai nhánh thắng đều thuộc loại **thêm thông tin mới hoặc thay** (`rp50`,
  `rhlp3`); hai nhánh **giữ** thua nặng nhất trong cả bảng.
- **Neo quá chặt thì hại ở CẢ HAI** (`fd10`, `fd10rh`) — không phụ thuộc đặc trưng có tốt hay không.

**Giả thuyết đọc được:** phép đo probe của §36 nói cho ta biết nên **giữ** hay nên **thay**. Đặc
trưng nguồn tốt ⇒ giữ nó, đừng để head làm méo. Đặc trưng nguồn không tốt ⇒ giữ nó là tự trói,
phải bơm thêm thông tin hoặc bỏ phần hỏng đi.

**Vì sao đáng chú ý:** §36 đo **trước** khi bốn khối này chạy, và dấu của nó dự báo đúng chiều ở
**4/4 nhánh có kết luận rõ trên cả hai backbone**. Tài liệu có ghi nhận cơ chế (BSS NeurIPS 2019
thấy neo hại ở đích nhỏ **trừ** khi nguồn gần đích; Plested et al. ICONIP 2021 dùng đúng margin
probe-đóng-băng để quyết định có neo hay không) nhưng **chưa ai biến thành quy tắc** — bản ICONIP
chỉ có n=4 dataset, một backbone, không kiểm định.

### CẢNH BÁO — đây là GIẢ THUYẾT, không phải phát hiện

- **n=3, seed 42, một nguồn (`4cwe`).** Dự án này đã **bốn lần** thấy n=3 đổi dấu ở n=5.
- Bảng ✓/✗ được đọc **sau khi** nhìn số. Phép kiểm sạch phải là: **khai báo trước** rằng probe dự
  báo loại can thiệp nào thắng, rồi chạy trên backbone/nguồn thứ ba.
- `rhlp3` trên t5p có ROC 3/3 và PR 3/3 nhưng F1@0.5 chỉ +0.0021 — biên độ dưới sàn nhiễu.

### Ba đề xuất (CHƯA CHẠY, chờ người dùng duyệt)

1. **Phép kiểm khai báo trước**: chạy probe §36 trên `com`/`full` (0 GPU, ~45 phút CPU mỗi ô),
   **ghi dự đoán ra file trước**, rồi mới chạy `lp3` vs `rp50` trên nguồn đó. Đây là cách duy nhất
   biến §38.2 từ giả thuyết thành phát hiện.
2. `lp3` lên n=5 **chỉ trên codebert**, nêu rõ là phát biểu **theo backbone**, không phải phát biểu
   chung. Cần thêm `plain` + `baseline` fold 4–5 làm đối chứng ghép cặp.
3. Probe theo **từng tầng** — LDIFS (TMLR 2024) đo được neo chỉ-tầng-cuối kém hơn neo nhiều tầng,
   mà `fd*` hiện chỉ neo tầng cuối. 0 GPU.

## §36.1 — CONTROL TASK: probe KHÔNG tự học được tác vụ, nên §36 đứng vững (11/09, 0 GPU)

Phản biện chuẩn với mọi phép probe (Hewitt & Liang, EMNLP 2019): *"probe của anh đủ mạnh để tự
học tác vụ từ bất kỳ đặc trưng nào, nên con số đó không nói gì về đặc trưng."* Cách chặn: chạy
**đúng probe đó** trên **nhãn xáo trộn cố định theo hàng** (cùng phân phối), rồi báo **độ chọn
lọc** = thật − ngẫu nhiên.

| | ROC thật | ROC nhãn ngẫu nhiên | **độ chọn lọc** |
|---|---|---|---|
| codebert pretrained | 0.6575 | **0.5107** | 0.1468 |
| **codebert Pha 1** | 0.7700 | **0.5047** | **0.2652** |
| t5p pretrained | 0.6222 | **0.5240** | 0.0981 |
| **t5p Pha 1** | 0.6423 | **0.5120** | 0.1303 |

**Ba điều đọc được:**

1. **Sàn ngẫu nhiên nằm ở 0.505–0.524 trên cả bốn ô** — probe tuyến tính trên 456 dòng train
   **không** thuộc lòng được nhãn ngẫu nhiên. Vậy điểm thật là do cấu trúc trong đặc trưng, không
   phải do sức chứa của probe. Phản biện được trả lời bằng số.
2. **Δ độ chọn lọc ≈ Δ ROC thô**: codebert +0.1184 (so với +0.1125 thô), t5p +0.0322 (so với
   +0.0202). Sàn ngẫu nhiên gần như không nhúc nhích giữa hai mô hình, nên **con số đầu bài của
   §36 không đổi** sau khi trừ sàn.
3. Khoảng cách codebert/t5p **rộng ra** khi tính bằng độ chọn lọc (0.2652 so với 0.1303, gấp hơn
   hai lần), củng cố §38.2.

Số đầy đủ: `results/probe/{codebert,t5p}_4cwe_ctl.json`. Đặc trưng đã đệm ở
`results/probe/cache/*.npz` nên các phép probe sau **không phải trích lại** (45 phút CPU/backbone).

> **Bẫy đã mắc lại đêm nay:** tôi báo "probe đang chạy" bốn lần trong khi nó **đã xong từ 02:25**,
> vì `pgrep -f "tools/feature_probe.py"` khớp **chính dòng lệnh của tôi**. Đây đúng là mục memory
> `pkill-kills-own-ssh-session` đã ghi. Cách kiểm đúng: lọc theo `comm` (`ps -o comm=` bằng
> `python`), hoặc đếm hiện vật (file kết quả), không đếm tiến trình theo dòng lệnh.

## §38.3 — PHÉP KIỂM KHAI BÁO TRƯỚC: probe tách theo **BACKBONE**, không tách theo **NGUỒN** — và điều đó làm §38.2 YẾU ĐI (11/09, 0 GPU)

Dự đoán viết **trước khi đo** ở `records/prediction_2026-09-11_probe_chon_can_thiep.md`
(22:40 UTC, ngưỡng 0.06 khai báo trước, không sửa sau).

### Đủ 6 ô: 2 backbone × 3 nguồn, mỗi ô 5 fold

| backbone | nguồn | ΔROC probe | fold+ | ΔF1 | độ chọn lọc pretrained → Pha 1 | nhóm |
|---|---|---|---|---|---|---|
| codebert | 4cwe | **+0.1125** | 5/5 | +0.0849 | 0.1468 → **0.2652** | GIỮ |
| codebert | com | **+0.0849** | 5/5 | +0.0506 | 0.1468 → **0.2679** | GIỮ |
| codebert | full | **+0.0849** | 5/5 | +0.0486 | 0.1468 → **0.2374** | GIỮ |
| t5p | 4cwe | +0.0202 | 3/5 | +0.0198 | 0.0981 → 0.1303 | THÊM/THAY |
| t5p | com | +0.0434 | 4/5 | +0.0519 | 0.0981 → 0.1372 | THÊM/THAY |
| t5p | full | +0.0122 | 3/5 | **−0.0150** | 0.0981 → 0.1225 | THÊM/THAY |

### Kết quả của phép kiểm — ghi trung thực cả phần SAI

1. **Ngưỡng 0.06 tách SẠCH, nhưng tách theo BACKBONE.** Cả ba nguồn codebert ≥ 0.0849 (5/5 fold);
   cả ba nguồn t5p ≤ 0.0434. Khoảng trống giữa hai nhóm là **0.0415** — rộng gấp 4 lần sàn nhiễu.
2. **Dự đoán (2) của tôi SAI một nửa.** Tôi đoán `com`/`full` sẽ **thấp hơn** `4cwe` vì val Pha 1
   của chúng thấp hơn nhiều (codebert 0.56 so với 0.65). Đúng trên codebert (0.1125 → 0.0849), nhưng
   **sai trên t5p**: `com` cho **+0.0434**, cao gấp đôi `4cwe`. Val Pha 1 **không** dự báo Δ probe.
3. **Và đây mới là điều quan trọng: §38.2 YẾU ĐI, không mạnh lên.** Nếu Δ probe chỉ thay đổi theo
   backbone mà gần như không theo nguồn, thì phát biểu *"probe chọn can thiệp"* rút gọn thành
   *"backbone chọn can thiệp"* — và ta chỉ có **HAI** backbone. Hai điểm dữ liệu không dựng được
   quy tắc. Phép kiểm mà §38.2 đề xuất (chạy `lp3` vs `rp50` trên nguồn có Δ thấp) **không thực
   hiện được như thiết kế**, vì không nguồn nào của codebert rơi xuống dưới ngưỡng.
4. **Ô đáng chú ý nhất: `t5p`/`full`** — ΔF1 **âm** (−0.0150) và độ chọn lọc theo F1 **tụt**
   (0.0736 → 0.0572). Đây là ô duy nhất trong sáu ô mà Pha 1 làm đặc trưng **kém tách hơn** backbone
   gốc. Khớp với việc `full` là nguồn gây negative transfer đã biết.
5. Per-CWE lặp lại ở **cả sáu ô**: CWE-079 lên mạnh nhất (từ +0.16 đến +0.25 ROC), **CWE-089 âm ở
   5/6 ô** dù chiếm quá nửa hàng test. Đây là mẫu hình bền nhất của toàn bộ phép đo.

### Muốn kiểm §38.2 cho đúng thì phải làm gì (CHƯA CHẠY, cần duyệt)

Trục "nguồn" không tách được nhóm, nên phải tìm trục khác **trong cùng một backbone**:

- **Backbone thứ ba** (`unixcoder` đã có trong dự án) — cho thêm một điểm dữ liệu ở trục duy nhất
  thật sự biến thiên. Cần Pha 1 mới (~40 phút GPU/nguồn).
- **Làm hỏng đặc trưng có kiểm soát**: lấy checkpoint codebert tốt rồi nội suy về pretrained ở vài
  mức α, đo Δ probe (0 GPU, có cờ `--source_interpolation` sẵn), chọn α làm Δ probe tụt xuống dưới
  0.06, rồi chạy `lp3` vs `rp50` ở đó. Đây là cách **duy nhất** biến Δ probe thành biến độc lập
  điều khiển được thay vì thuộc tính cố hữu của backbone.

Số đầy đủ: `results/probe/*_ctl.json`. Đặc trưng đã đệm ở `results/probe/cache/`.

## §39 — CẤU HÌNH CHỐT, đọc theo TỪNG NGUỒN: `4cwe` là nguồn DUY NHẤT dương cả bốn chỉ số trên CẢ HAI backbone (11/09, n=15)

§35.2 kết luận "rút ASAM" từ con số **gộp ba nguồn**. Đọc lại theo từng nguồn thì kết luận đó
**vẫn đúng nhưng vì lý do khác**, và một chi tiết quan trọng đã bị con số gộp làm mờ.

### B = `latent_bottleneck` λ0.05 + AdamW trần, so với `baseline` cùng ô (n=15 mỗi dòng)

| backbone | nguồn | ΔF1@0.5 | ΔF1@val | ΔROC-AUC | ΔPR-AUC | đủ bốn dương? |
|---|---|---|---|---|---|---|
| codebert | **4cwe** | **+0.0503 15/15** | **+0.0563 15/15** | +0.0148 10/15 | +0.0035 7/15 | **✓** |
| codebert | com | +0.0326 14/15 | +0.0360 12/15 | +0.0033 7/15 | **−0.0073** 7/15 | ✗ |
| codebert | full | +0.0493 15/15 | +0.0515 14/15 | +0.0133 9/15 | +0.0088 9/15 | ✓ |
| t5p | **4cwe** | **+0.0261 13/15** | +0.0241 12/15 | +0.0095 9/15 | +0.0084 9/15 | **✓** |
| t5p | com | +0.0298 13/15 | +0.0249 12/15 | +0.0124 11/15 | +0.0105 10/15 | ✓ |
| t5p | full | +0.0083 7/15 ~1 | +0.0117 8/15 | **−0.0079** 7/15 | **−0.0175** 7/15 | ✗ |

**`4cwe` là nguồn DUY NHẤT dương cả bốn chỉ số trên CẢ HAI backbone.** `com` hỏng ở codebert
(PR âm), `full` hỏng ở t5p (ROC và PR đều âm).

### Chi tiết bị con số gộp làm mờ: ASAM trên t5p là hiệu ứng THEO NGUỒN

A − B (tách riêng phần optimizer đóng góp, ghép cặp trong cùng ô):

| backbone | nguồn | ΔF1@0.5 | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|
| t5p | **4cwe** | +0.0087 10/15 ~1 | **+0.0166 11/15 (p=0.057)** | **+0.0253 13/15 (p=0.0074)** |
| t5p | com | **−0.0584** 4/15 ~1 | **−0.0672** 6/15 | −0.0582 8/15 |
| t5p | full | −0.0410 10/15 | **−0.0456 12/15 (p=0.035)** | −0.0408 10/15 |
| t5p | **gộp** | −0.0302 | −0.0321 | −0.0245 |
| codebert | gộp | +0.0010 21/45 ~2 | +0.0033 28/45 (p=0.066) | +0.0029 29/45 (p=0.073) |

Trên **`4cwe` thì ASAM ρ=2.0 GIÚP t5p** (+0.0253 PR, 13/15, p=0.0074). Toàn bộ con số âm của
dòng gộp đến từ `com` và `full`, nơi nó làm **sập** t5p: ROC tuyệt đối 0.8442 và 0.8454 so với
0.9114 và 0.8911 của AdamW trần.

**Nên phát biểu lại:** ASAM không phải "vô dụng"; nó là một **núm vặn mong manh phụ thuộc nguồn**.
Bỏ nó đi mất ~0 trên codebert, mất một ít AUC trên `t5p`/`4cwe`, nhưng loại bỏ được nguy cơ sập
0.07 ROC khi đổi nguồn. **Với một bài báo, đổi lấy sự ổn định là đúng.**

### Trị tuyệt đối — cấu hình chốt so với baseline

| backbone | nhánh | F1@0.5 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| codebert | baseline (không Pha 1) | 0.7674 | 0.8752 | 0.8756 |
| codebert | **4cwe + AdamW trần** | **0.8176** | 0.8901 | 0.8791 |
| t5p | baseline | 0.8040 | 0.8990 | 0.8972 |
| t5p | **4cwe + AdamW trần** | **0.8301** | 0.9085 | 0.9056 |

### CẢNH BÁO khi viết: đầu bài là F1, KHÔNG phải AUC

ΔROC-AUC của cấu hình chốt chỉ **+0.0148** (codebert) và **+0.0095** (t5p) — con số t5p **nằm
dưới sàn nhiễu 0.010**. Phát biểu trung thực là về **quyết định ở ngưỡng 0.5**, không phải về
**thứ hạng**. Thứ mạnh và lặp lại là **per-CWE**: CWE-022 +0.3677 (45/45) và CWE-079 +0.3500
(45/45) trên codebert; +0.2204 (43/45) và +0.2209 (42/45) trên t5p. CWE-078 và CWE-089 null ở
**mọi** phép đo, và chúng chiếm **121 trên 152** hàng test.

---

## §40 — ĐƯỜNG CONG THEO CỠ TẬP TRAIN ĐÍCH: lợi ích transfer TĂNG MẠNH khi dữ liệu đích co lại (11/09, **kiểm chứng n=3, seed 42**)

Phép kiểm khai báo trước ở `records/prediction_2026-09-11_duong_cong_co_dich.md` (03:05 UTC,
viết **trước khi chạy một ô nào**). Cắt ngẫu nhiên tập train Python xuống N = 228 / 152 / 76,
**giữ nguyên val và test**; chạy **cả** nhánh chuyển giao **lẫn** baseline ở cùng N với cùng seed
nên **cùng một tập con**. N = 456 lấy từ khối `bridge3`. 36 ô mới + 6 ô cũ = 42 ô ghép cặp.

### Δ (chuyển giao − baseline), ghép cặp TRONG cùng ô

| backbone | N | ΔF1@0.5 | ΔF1@val | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|---|
| **codebert** | 456 | +0.0309 3/3 | +0.0344 3/3 | +0.0150 2/3 | +0.0029 2/3 |
| | 228 | +0.0415 3/3 | +0.0460 3/3 | +0.0453 3/3 | +0.0334 2/3 |
| | 152 | +0.0882 3/3 | +0.0875 3/3 | +0.0816 3/3 | +0.0584 2/3 |
| | **76** | **+0.1608 3/3** | **+0.1606 3/3** | **+0.1761 3/3** | **+0.1622 3/3** |
| **t5p** | 456 | +0.0397 3/3 | +0.0241 3/3 | +0.0183 2/3 | +0.0090 2/3 |
| | 228 | +0.0523 3/3 | +0.0365 3/3 | +0.0347 3/3 | **−0.0146** 1/3 |
| | 152 | **+0.0102** 2/3 | +0.0194 2/3 | +0.0307 2/3 | +0.0381 2/3 |
| | **76** | **+0.0810 3/3** | **+0.0931 3/3** | **+0.1184 3/3** | **+0.0888 3/3** |

### Kết quả của phép kiểm — ghi cả phần SAI

**Dự đoán 1 (Δ tăng đơn điệu khi N giảm, trên CẢ HAI backbone): ĐÚNG MỘT NỬA.**
- codebert: **đơn điệu chặt trên cả bốn chỉ số**, không một chỗ lùi. ROC đi từ +0.0150 lên
  +0.1761 — **gấp 11,7 lần**, và cả bốn chỉ số đều 3/3 fold ở N=76.
- t5p: **KHÔNG đơn điệu** — F1 tụt ở N=152 (+0.0102), PR âm ở N=228 (−0.0146).
- Nhưng dạng **hai đầu mút** thì đúng ở cả hai: Δ tại N=76 lớn hơn hẳn Δ tại N=456 trên cả bốn
  chỉ số, và đều **3/3 fold**.

**Dự đoán 2 (CWE-078 phải đạt ΔROC ≥ +0.10 tại N=152): BỊ BÁC.**
codebert +0.025 (2/3), t5p **−0.065 (0/3)**. Ngưỡng +0.10 khai báo trước, **không sửa**.

**Dự đoán 3 (CWE-022 và 079 giữ lợi ích ở mọi N): ĐÚNG.** Dương ở cả 8 ô (2 backbone × 4 N).

### ΔROC-AUC theo CWE — cột trong ngoặc là SỐ HÀNG TRAIN của lớp đó tại N

| bb | N | 022 | 078 | 079 | 089 |
|---|---|---|---|---|---|
| codebert | 456 | +0.234 3/3 [40] | −0.002 2/3 [124] | +0.098 3/3 [47] | +0.019 2/3 [245] |
| | 228 | +0.364 3/3 [19] | +0.044 3/3 [60] | +0.154 2/3 [23] | +0.017 3/3 [126] |
| | 152 | +0.265 3/3 [9] | +0.025 2/3 [37] | +0.195 3/3 [15] | **+0.095 3/3** [91] |
| | 76 | +0.160 3/3 [7] | +0.088 2/3 [18] | +0.185 3/3 [8] | **+0.174 3/3** [43] |
| t5p | 456 | +0.141 2/3 [40] | −0.043 0/3 [124] | +0.249 3/3 [47] | +0.006 3/3 [245] |
| | 228 | +0.129 2/3 [19] | +0.001 2/3 [60] | +0.380 3/3 [23] | +0.003 1/3 [126] |
| | 152 | +0.045 2/3 [9] | −0.065 0/3 [37] | +0.227 3/3 [15] | +0.018 1/3 [91] |
| | 76 | +0.158 3/3 [7] | **+0.176 3/3** [18] | +0.270 3/3 [8] | +0.063 3/3 [43] |

**Ngưỡng đáp ứng KHÁC NHAU theo từng CWE, không phải một con số hàng train chung:**

| CWE | đáp ứng từ khoảng | ghi chú |
|---|---|---|
| 022, 079 | **≥ 40 hàng** (tức ngay ở dữ liệu đầy đủ) | dương ở cả 8 ô |
| 089 | ~**43–91 hàng** | codebert rõ (+0.095 rồi +0.174, 3/3); t5p yếu hơn |
| **078** | **~18 hàng** — muộn nhất | t5p nhảy từ −0.065 lên **+0.176 (3/3)** giữa N=152 và N=76 |

CWE-078 là lớp **khó giúp nhất ở mọi mức**, dù nó **không** phải lớp nhiều hàng train nhất — và
đúng nó là lớp mà §36 đo được đặc trưng Pha 1 cải thiện **mạnh nhất** (+0.199 ROC, 5/5, codebert).
Nghịch lý §36 giờ có **bốn** điểm dữ liệu thay vì một, và nó **không** giải thích được bằng số
hàng train.

### CẢNH BÁO — ngưỡng loại ô tôi đặt trước đã QUÁ LỎNG

Trị tuyệt đối ở N=76: codebert baseline 0.5599 F1 / 0.6040 ROC (yếu nhưng trên ngẫu nhiên);
**t5p baseline 0.4907 F1 / 0.4815 ROC — ĐÚNG mức ngẫu nhiên**. Ngưỡng loại khai báo trước là
`baseline F1@0.5 < 0.40`, mà macro-F1 của một bộ phân loại ngẫu nhiên trên tập cân bằng là
**~0.49 chứ không phải 0.33**, nên cổng không bắt được. **Không sửa ngưỡng sau khi thấy số**;
thay vào đó ghi rõ: ô `t5p`/`N=76` phải đọc là *"chuyển giao còn chạy được, baseline thì không"*,
không phải một phép so có thang. Lần sau cổng phải đặt trên **ROC-AUC < 0.55**, không phải F1.

### Ý nghĩa

Con số đầu bài của dự án (+0.05 F1, ROC quanh sàn nhiễu ở 456 hàng) là **một điểm trên một đường
cong**, và nó nằm ở chỗ **thoải nhất** của đường cong đó. Ở 76 hàng train — vẫn là chế độ thực tế
cho một CWE mới hoặc một ngôn ngữ mới — lợi ích là **+0.16 F1 và +0.18 ROC trên codebert, 3/3
fold**. Đây là cách biến một phát biểu yếu thành một chế độ được đặc tả, **mà không cần phát minh
thêm phương pháp nào**.

Đang leo lên **n=5** (fold 4, 5) cho cả bốn mức N và cả hai backbone.
Đọc bằng `python3 tools/tsize_report.py`.

## §40.1 — Cổng môi trường lại bắt được một lần gọi thiếu `PYTHON`, và lần này KHÔNG mất gì (11/09)

Khi leo đường cong §40 lên n=5, tôi gọi thẳng `run/opt1.sh` cho mức N=456 fold 4–5 mà **quên
export `PYTHON`**. Khác `tsize.sh` / `feat3.sh` / `chot2bb.sh`, `opt1.sh` **không tự dò env**, nên
`matrix.sh` rơi về `python` của conda base không có torch.

**Cổng đã làm đúng việc:**

```
!! DUNG: PYTHON='python' khong import duoc torch/numpy/sklearn.
########## OPT1 seed 42 fold 4 xong | 0/1 o + 0/1 baseline ##########
```

Thoát 2, **không xoá một file nào**, in đường dẫn env đúng cho từng máy. So với 08/09 khi đúng
lớp lỗi này **xoá mất hai checkpoint Pha 1 tốt** (CLAUDE.md mục 3) thì ba lớp chặn dựng sau đó đã
trả đủ giá trị. Cổng **đếm hiện vật** (`0/1 o`) là thứ làm lỗi lộ ra ngay ở dòng log, không phải
sau nhiều giờ.

**Thiệt hại:** mức N=456 fold 4–5 ra 0 ô trên **cả hai** máy; ba mức kia đủ 12/12 vì `tsize.sh`
có tự dò. Đã xếp chạy bù.

**Sửa gốc:** `run/opt1.sh` nay **tự dò env** giống mọi runner khác. Kiểm hai chiều bằng `env -i`:
với môi trường rỗng nó vẫn tìm ra `/home/ntat/miniconda3/envs/vdenv/bin/python`.

> **Quy tắc rút ra:** một runner **không được** để người gọi tự nhớ `PYTHON`. Mọi script chạy
> được trực tiếp phải tự dò, vì cách gọi sẽ thay đổi theo thời gian còn trí nhớ thì không.

---

## §41 — ĐẢO CHIỀU (Python → JavaScript): hiệu ứng **MẠNH HƠN** chiều thuận, và nó khớp §40 (11/09, **n=1 fold — chỉ sàng lọc**)

Người dùng yêu cầu 11/09. Mọi số của dự án tới giờ đều đo **một chiều** (C/C++ + JS → Python).
Nếu hiệu ứng là tính chất của **phương pháp** thì đảo chiều vẫn phải thấy; nếu là tính chất của
riêng **cặp** (nguồn này, đích này) thì đảo chiều sẽ tắt. Chưa phép đo nào phân biệt được hai
khả năng đó.

**Thiết kế**: nguồn = 760 dòng Python (SVEN gộp, Pha 1 tự chia train/val); đích = 812 dòng JS của
`phase1_4cwe`, chia 486/162/164 phân tầng theo (nhãn, CWE), **chia theo dòng** đúng quy ước bộ
`norm` để biến duy nhất đổi là **chiều**. Rò rỉ cặp 227/406 (56%), cùng dạng `norm` (~40%).
`baseline` huấn luyện trên **chính tập train JS đó**, cùng fold cùng máy cùng phiên. 1 fold, seed 42.

### Trị tuyệt đối — và đây mới là con số quan trọng nhất

| backbone | nhánh | F1@0.5 | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| codebert | **baseline (chỉ JS)** | 0.5285 | **0.4927** | 0.5380 |
| codebert | plain (có Pha 1) | 0.5942 | 0.6536 | 0.6585 |
| codebert | r0p1 (RecAdam+ASAM) | 0.5352 | 0.6365 | 0.6559 |
| t5p | **baseline (chỉ JS)** | 0.4553 | **0.4686** | 0.5083 |
| t5p | plain | 0.6152 | 0.6425 | 0.6632 |
| t5p | r2p0 | 0.5729 | **0.6823** | 0.7085 |

**Baseline trên JS nằm Ở HOẶC DƯỚI mức ngẫu nhiên** (ROC 0.4927 và 0.4686). Huấn luyện từ đầu
trên 486 dòng JS **gần như không học được gì**. So sánh: baseline trên 456 dòng Python đạt
0.7674 F1 / 0.8752 ROC. **JS là đích khó hơn hẳn Python.**

### Δ so với `baseline` cùng ô

| backbone | nhánh | ΔF1@0.5 | ΔF1@val | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|---|
| codebert | plain | +0.0657 | +0.0812 | **+0.1609** | +0.1205 |
| codebert | r0p1 | +0.0067 | +0.0495 | +0.1438 | +0.1179 |
| t5p | plain | +0.1598 | +0.1657 | +0.1739 | +0.1550 |
| t5p | r2p0 | +0.1176 | +0.1611 | **+0.2137** | +0.2003 |

**Hiệu ứng lớn hơn chiều thuận một bậc**: +0.16 đến +0.21 ROC, so với +0.015 (codebert) và
+0.018 (t5p) ở chiều thuận với dữ liệu đích đầy đủ.

### Điều này KHÔNG phải bất đối xứng — nó là §40 nhìn từ góc khác

Đừng đọc thành *"chiều ngược tốt hơn"*. §40 đo được: lợi ích transfer **tăng khi đích một mình
không học nổi**. Ở đây baseline JS **đúng mức ngẫu nhiên**, tức đích đang ở chế độ cực đoan nhất
của đường cong đó — nên lợi ích lớn là **điều §40 dự báo**, không phải điều mới.

Nói cách khác: hai phép đo độc lập (cắt dữ liệu đích ở §40; đổi hẳn đích sang JS ở đây) cùng chỉ
về **một** biến giải thích — **đích một mình học được bao nhiêu**. Không phải cặp ngôn ngữ, không
phải chiều.

### ASAM: LẶP LẠI LẦN THỨ BA đúng mẫu hình cũ

A − B (RecAdam+ASAM trừ AdamW trần), cùng ô:

| backbone | ΔF1@0.5 | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|
| codebert | **−0.0590** | −0.0171 | −0.0026 |
| t5p | −0.0422 | **+0.0399** | **+0.0453** |

Khớp §39: ASAM **null/âm trên codebert**, còn trên t5p thì **nâng AUC mà hạ F1**. Đây là lần lặp
thứ ba của mẫu hình đó trên một cặp nguồn–đích **hoàn toàn khác**, nên nó là tính chất của
**backbone**, không phải của dữ liệu.

### BỐN cảnh báo — con số này CHƯA kết luận được

1. **n = 1 fold.** Bậc 1 tối thiểu. Dự án đã bốn lần thấy n=3 đổi dấu; n=1 còn yếu hơn nữa.
2. **Baseline dưới ngẫu nhiên** ⇒ Δ phải đọc là *"chuyển giao chạy được, baseline thì không"*,
   không phải một phép so có thang. Đúng bài học §40.1.
3. **Đích JS lệch cực mạnh**: CWE-079 chiếm **134/164** hàng test (82%); CWE-022 và CWE-089 chỉ
   **8 hàng** mỗi lớp. Per-CWE bên JS **không đọc được**, chỉ số tổng mới có nghĩa.
4. **Trị tuyệt đối thấp ở mọi nhánh** (ROC 0.64–0.68). Không nhánh nào thật sự "giải" được JS.

Muốn dùng được thì phải lên **n=3 fold** (chia lại JS thành 3 fold) trước khi viết bất cứ câu nào.

## §40.2 — Đường cong ở **n=5**: ROC-AUC ĐƠN ĐIỆU CHẶT trên **cả hai** backbone (11/09, 40 ô ghép cặp)

Đủ fold 4–5 cho cả bốn mức. **Kết luận đổi so với n=3, và đổi theo hướng SẠCH HƠN.**

| backbone | N | ΔF1@0.5 | ΔF1@val | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|---|
| codebert | 456 | +0.0381 5/5 | +0.0362 5/5 | +0.0114 3/5 | **−0.0087 2/5** |
| | 228 | +0.0261 4/5 | +0.0318 5/5 | +0.0362 5/5 | +0.0269 3/5 |
| | 152 | +0.0845 5/5 | +0.0688 5/5 | +0.0610 5/5 | +0.0287 3/5 |
| | **76** | **+0.1468 5/5** | **+0.1416 5/5** | **+0.1940 5/5** | **+0.1801 5/5** |
| t5p | 456 | +0.0264 4/5 | +0.0130 4/5 | +0.0173 4/5 | +0.0198 4/5 |
| | 228 | +0.0518 5/5 | +0.0378 5/5 | +0.0395 4/5 | +0.0219 3/5 |
| | 152 | +0.0489 4/5 | +0.0626 4/5 | +0.0897 4/5 | +0.0834 4/5 |
| | **76** | **+0.0696 4/5** | **+0.0747 4/5** | **+0.1003 5/5** | **+0.0682 5/5** |

### Chỉ số đơn điệu là ROC-AUC, không phải F1

```
codebert ROC:  +0.0114 → +0.0362 → +0.0610 → +0.1940      ĐƠN ĐIỆU CHẶT
t5p      ROC:  +0.0173 → +0.0395 → +0.0897 → +0.1003      ĐƠN ĐIỆU CHẶT
```

Ở N=76, **cả hai backbone đều 5/5 fold** trên ROC và PR — tức `p = 0.0625`, **sàn của phép kiểm
dấu ở n=5**. Không thể chặt hơn với n này.

F1@0.5 **gần** đơn điệu nhưng có chỗ lùi nhỏ (codebert tụt ở N=228, t5p ở N=152). Ở n=3 thì
ngược lại — codebert đơn điệu trên cả bốn còn t5p không. **Thêm hai fold đã đổi chỉ số nào là
chỉ số sạch.** Phát biểu đúng là về **ROC-AUC**.

### Một số bị xấu đi khi lên n=5, phải ghi

codebert ở **N=456** (dữ liệu đầy đủ): ΔPR-AUC **−0.0087, chỉ 2/5 fold**. Ở n=3 nó là +0.0029.
Nghĩa là **ở dữ liệu đích đầy đủ, lợi ích của chuyển giao chỉ có trên F1 và F1@val; trên PR-AUC
nó thực ra hơi ÂM**. Đây là phiên bản chặt hơn của cảnh báo ở §39 ("đầu bài là F1, không phải
AUC") và phải nêu đúng như vậy.

### Per-CWE ở n=5 — CWE-078 CUỐI CÙNG cũng đáp ứng, trên CẢ HAI backbone

ΔROC-AUC, cột trong ngoặc là số hàng train của lớp đó tại N:

| bb | N=456 | N=228 | N=152 | **N=76** |
|---|---|---|---|---|
| codebert · 078 | −0.016 2/5 [124] | +0.043 4/5 [60] | +0.022 4/5 [37] | **+0.198 4/5 [18]** |
| t5p · 078 | −0.035 0/5 [124] | −0.002 3/5 [60] | −0.002 2/5 [37] | **+0.146 4/5 [18]** |
| codebert · 089 | +0.013 3/5 [245] | +0.006 3/5 [126] | +0.059 4/5 [91] | **+0.163 5/5 [43]** |
| codebert · 079 | +0.182 5/5 | +0.225 4/5 | +0.263 5/5 | **+0.301 5/5** |

**CWE-078 chuyển từ âm sang +0.198 / +0.146 (4/5 fold) trên CẢ HAI backbone** khi số hàng train
của nó rơi xuống ~18. Ở n=3 điều này chỉ thấy rõ trên t5p. Vậy **mọi lớp đều đáp ứng**, chỉ là
**ngưỡng khác nhau theo lớp** — 022/079 từ ~40 hàng, 089 từ ~43, 078 phải xuống ~18.

078 vẫn là lớp **khó giúp nhất**, và nó vẫn đúng là lớp mà §36 đo được đặc trưng Pha 1 cải thiện
**mạnh nhất**. Nghịch lý đó **chưa giải thích được**, và giờ nó đứng trên 40 ô thay vì 3.

### Phát biểu dùng được cho bài

> Lợi ích của tiền huấn luyện xuyên ngôn ngữ **tăng đơn điệu theo ROC-AUC khi dữ liệu đích co
> lại**, trên cả hai họ backbone: từ +0.011 / +0.017 ở 456 hàng train lên **+0.194 / +0.100 ở 76
> hàng**, với **5/5 fold** ở đầu mút nhỏ. Ở dữ liệu đầy đủ thì lợi ích chỉ nằm ở ngưỡng quyết
> định (F1), không ở thứ hạng.

Đang kiểm **độ bền theo seed** ở hai đầu mút (seed 7, N ∈ {456, 76}, 5 fold, hai backbone).

---

## §42 — Máy thuê mới: một lỗi che một lỗi, và bản ghim thư viện là thứ bị bỏ quên (11/09)

Dựng vast `50570168` (nhãn `ntat`) cho khối `auxb` mất **~25 phút** vì **ba** lỗi xếp chồng,
mỗi lỗi chỉ lộ ra sau khi sửa xong lỗi trước. Ghi lại vì cả ba đều sẽ lặp ở máy thuê tiếp theo.

**Lỗi 1 — địa chỉ SSH trong API là PROXY.** `ssh_host`/`ssh_port` từ `vastai show instance --raw`
trả về proxy và cho `kex_exchange_identification: Connection closed`. `vastai ssh-url <id>` trả
địa chỉ **trực tiếp** và vào được ngay. Mất ~10 phút.

**Lỗi 2 — HF Hub tải đứng.** Blob trọng số nằm ở `.incomplete` = **0 byte sau 3 phút**; cache chỉ
lớn thêm 1,2 KB trong 25 s. Thêm một chỗ đi lạc: image đặt `HF_HOME=/workspace/.hf_home`, không
phải `~/.cache/huggingface`. Cách chữa đã dùng: đẩy 1,8 GB cache từ local rồi chạy với
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

**Lỗi 3 — và đây mới là cái đắt: `pip install transformers` trần kéo về 5.17.0.**
`requirements-pin.txt` đã ghi đúng cái bẫy này từ **29/08** (`transformers==4.57.1`), nhưng lúc
cài gói thiếu trên image tôi gõ `pip install transformers` chứ không `pip install -r`. Ở
transformers 5.x các lớp tokenizer **chậm** thành khúc cụt, nên `AutoTokenizer.from_pretrained`
chết bằng:

```
AttributeError: RobertaTokenizer has no attribute build_inputs_with_special_tokens
```

**Điều đáng sợ không phải là nó chết.** Nó chết nên ta biết. Nếu nó *chạy được*, thì các
checkpoint Pha 1 mới của khối `auxb` sẽ được huấn luyện bằng một thư viện **khác major version**
so với mọi checkpoint trước — và Δ ghép cặp giữa chúng với các khối cũ mất nghĩa mà không có dấu
hiệu nào. Đúng lý do `requirements-pin.txt` tồn tại.

**Vì sao lỗi 3 nấp được sau lỗi 2**: khi HF Hub còn treo, job chết ở bước *tải*, chưa bao giờ
chạm tới bước *dựng tokenizer*. Chữa xong treo thì lỗi phiên bản mới lộ. Một lỗi che một lỗi.

**Cách chặn, đã kiểm CẢ HAI CHIỀU trước khi phóng lại** (`CLAUDE.md` mục 8):

| phép thử | kết quả phải có | đo được |
|---|---|---|
| dựng tokenizer + backbone **có** cache | chạy | `RobertaTokenizerFast`, 124.6M / 223.1M tham số |
| dựng tokenizer **không** cache (`HF_HOME` trỏ chỗ rỗng) | **phải hỏng** | `OSError` — đúng như mong đợi |

Chiều thứ hai là chiều quan trọng: nếu không cache mà vẫn dựng được thì cổng offline **vô nghĩa**,
job vẫn đang lén ra mạng.

**Việc phải làm ở máy thuê tiếp theo**: cài bằng `pip install -r requirements-pin.txt`, rồi chạy
một **phép thử khói đi đúng đường mã thật** (tokenizer + backbone + một bước huấn luyện) **trước**
khi phóng khối. Job đầu tiên của khối là phép thử khói tồi: nó mất vài phút mới tới chỗ chết, và
vết lỗi bị chôn giữa log.

**Bẫy `pgrep -f` — lần thứ TƯ.** Dọn runner cũ bằng
`ssh host 'for t in "bash run/matrix.sh" ...; do pgrep -f "$t" ...'` thì **chính dòng lệnh ssh
chứa chuỗi đó**, nên `pgrep -f` khớp shell đang chạy lệnh của mình và tự giết phiên: lệnh trả về
**không một dòng nào**. Cách chữa dùng được: đưa script qua **stdin** (`ssh host 'bash -s' <<'EOF'`)
— nội dung khi đó không nằm trên `argv` của shell từ xa nên không thể tự khớp. Lá chắn phụ:
bỏ qua mọi PID nằm trong cây tổ tiên của `$$`.

---

## §43 — Head phụ CÂN BẰNG LỚP: cứu được trên t5p, **hỏng thêm** trên codebert (11/09, val 96 hàng, seed 42)

§30.2 đo được head phụ chỉ đoán **một lớp**. Nguyên nhân giả định: cross-entropy tràn trên nguồn
`4cwe` lệch 74% về CWE-79 (692/930 dòng), λ chỉ 0.05. Can thiệp: **trọng số nghịch tần suất,
chuẩn hoá về trung bình 1** (`--aux_class_balanced`) — tổng độ lớn loss **không đổi** nên λ vẫn so
sánh được với mọi khối cũ, chỉ **phân bổ lại** giữa các lớp. Pha 1 huấn luyện lại từ đầu cho cả
hai backbone trên vast `50570168`.

| checkpoint | độ chính xác | **macro-F1** | số lớp head dùng | val nhị phân |
|---|---|---|---|---|
| **codebert** cân bằng | 0.2500 | **0.1917** | **4/4** | 0.6870 (ep 8) |
| codebert không cân bằng | 0.6667 | 0.2000 | 1/4 | 0.6532 (ep 6) |
| **t5p** cân bằng | 0.6458 | **0.5996** | **4/4** | 0.6875 (ep 7) |
| t5p không cân bằng | 0.6667 | 0.2000 | 1/4 | 0.6976 (ep 6) |
| sàn đoán-lớp-đa-số | 0.6667 | 0.2000 | 1 | |
| sàn đoán ngẫu nhiên | 0.4870 | — | | |

**Đọc được ba điều, và điều thứ ba là điều quan trọng.**

**1. §30.2 giờ đã đo trên CẢ HAI backbone, không phải một.** Không cân bằng thì cả codebert lẫn
t5p đều cho **đúng** 0.6667 / 0.2000 và dùng **đúng một lớp** — trùng khít sàn đoán-lớp-đa-số tới
từng chữ số. Đây không còn là quan sát trên một backbone nữa.

**2. Trên t5p, can thiệp THÀNH CÔNG rõ ràng.** macro-F1 từ 0.2000 lên **0.5996** — gấp **ba lần**
sàn — trong khi độ chính xác chỉ tụt 0.0209 dưới sàn. Đó đúng là đánh đổi mà cân bằng lớp phải
tạo ra: bỏ một ít độ chính xác trên lớp đa số để lấy lại recall của lớp hiếm. Và val **nhị phân**
gần như không đổi (0.6976 → 0.6875), nên head học được không phải trả bằng nhiệm vụ chính.

**3. Trên codebert, can thiệp HỎNG, và hỏng theo một kiểu đáng ghi.** macro-F1 **0.1917 < 0.2000**
và độ chính xác **0.2500 < 0.4870**, tức **dưới cả sàn ngẫu nhiên**. Nhưng nó không phải là "không
học gì": tách theo lớp thì CWE-022 đúng **8/10 = 80%** (ngẫu nhiên là 25%), CWE-089 đúng 33%, còn
lớp đa số CWE-079 chỉ 21.88% và CWE-078 **0/16**. Head đổi lớp đa số lấy lớp hiếm — quá tay. Điều
trớ trêu: val **nhị phân** lại TĂNG (0.6532 → 0.6870), nên nhìn từ nhiệm vụ chính thì can thiệp
này có vẻ tốt lên.

### Mẫu hình §38.2 lặp lại LẦN THỨ NĂM

Danh sách các can thiệp thắng ở một backbone và thua ở backbone kia giờ là: `lp3`, `rh`, `fd1`,
`fd10`, ASAM theo nguồn (§39), và nay `--aux_class_balanced`. **Không có một can thiệp nào** trong
toàn dự án dương trên cả hai họ backbone ở cùng một cấu hình. Đây đã là phát biểu mạnh nhất mà
dữ liệu hiện có cho phép, và nó là một phát biểu **âm**.

### Cổng mảnh 1: KẾT QUẢ TÁCH ĐÔI

Cổng khai báo trước (`scripts/vast_auxb_run.sh`): *"head phải VƯỢT sàn đoán-lớp-đa-số và dùng >1
lớp"*. **t5p ĐẠT** (macro-F1 gấp 3 sàn, 4 lớp). **codebert TRƯỢT** (dưới sàn ở cả hai chỉ số).

Theo luật leo bậc (một nhánh chỉ lên bậc khi lặp trên **cả hai** backbone), cổng này **không mở
đường** cho mảnh 2 ở dạng "định tuyến qua `cwe_head`".

**Nhưng cổng đó đo sai thứ cho mảnh 2 như đã cài đặt.** Mảnh 2 không dùng `cwe_head`: nó đóng
băng `latent_proj` (768→8) rồi đặt một head nhị phân **MỚI** lên đầu ra 8 chiều, huấn luyện trên
nhãn lỗ hổng của **đích**. Câu hỏi quyết định vì thế là *"ảnh 8 chiều còn tách được lỗ hổng tuyến
tính không, và có hơn một phép chiếu 8 chiều NGẪU NHIÊN không"* — khác hẳn. Phép đo đó đã được
**khai báo trước** ở `records/prediction_2026-09-11_nut_that_8_chieu.md` và chạy bằng
`tools/latent_probe.py` (0 GPU). Ghi rõ ở đây rằng cổng cũ đã trượt trên codebert **trước** khi
đo cái mới, để không ai đọc thành dời cột gôn.

---

## §41.1 — Đảo chiều, đích JS **common** (1 384 dòng): Δ dương cả 8 ô, nhưng **mọi baseline đều ở hoặc dưới mức ngẫu nhiên** (11/09, **n=1 fold — chỉ sàng lọc**)

Người dùng 11/09: *"Thử với cái đảo source nhưng js có thêm full và common giúp tôi nhé"*, và
*"so baseline là baseline của train js tương ứng nhé"* — nên đối chứng là baseline huấn luyện trên
**đúng** tập JS đó, cùng máy cùng fold.

Δ ghép cặp (chuyển giao − baseline), fold 1, seed 42:

| đích | backbone | nhánh | ΔF1@0.5 | ΔF1@val | **ΔROC** | ΔPR |
|---|---|---|---|---|---|---|
| `js_4cwe` (812) | codebert | AdamW trần | +0.0657 | +0.0812 | **+0.1609** | +0.1205 |
| | codebert | ASAM+RecAdam ρ=0.1 | +0.0067 | +0.0495 | +0.1438 | +0.1179 |
| | t5p | AdamW trần | +0.1598 | +0.1657 | **+0.1739** | +0.1550 |
| | t5p | ASAM+RecAdam ρ=2.0 | +0.1176 | +0.1611 | **+0.2137** | +0.2003 |
| `js_com` (1 384) | codebert | AdamW trần | +0.0615 | +0.0648 | **+0.0925** | +0.0712 |
| | codebert | ASAM+RecAdam ρ=0.1 | +0.0789 | +0.1078 | +0.1036 | +0.0636 |
| | t5p | AdamW trần | +0.0881 | +0.1238 | **+0.1617** | +0.1324 |
| | t5p | ASAM+RecAdam ρ=2.0 | +0.1731 | +0.1567 | **+0.1828** | +0.1482 |

**8/8 ô dương trên cả bốn chỉ số.** Nhưng phải đọc kèm vế sau, nếu không là đọc sai:

### Mọi baseline JS đều ở hoặc DƯỚI mức ngẫu nhiên

| đích | backbone | baseline ROC | baseline F1@0.5 |
|---|---|---|---|
| `js_4cwe` | codebert | 0.4927 | 0.5285 |
| `js_4cwe` | t5p | 0.4686 | 0.4553 |
| `js_com` | codebert | 0.4866 | 0.4672 |
| `js_com` | t5p | **0.3946** | 0.3935 |

Ô cuối **dưới hẳn** mức ngẫu nhiên: mô hình phản tương quan với nhãn. Theo ngưỡng đã rút ra ở
§40 (*"lần sau đặt cổng trên ROC-AUC < 0.55"*), **cả tám ô đều nằm dưới cổng đó**. Nghĩa là Δ ở
đây đo **"Pha 1 cứu được một đích mà mô hình không tự học nổi"**, KHÔNG phải *"phương pháp thắng
một baseline đang chạy được"*. Hai phát biểu đó khác nhau, và chỉ phát biểu thứ nhất có bằng chứng.

### Điều đáng chú ý: ΔROC của codebert **CO LẠI** khi đích to hơn

codebert: `4cwe` (812 dòng) **+0.1609** → `com` (1 384 dòng) **+0.0925**. Cùng hướng với §40
(*lợi ích transfer tăng khi đích co lại*), và là lần đầu mẫu hình đó xuất hiện ở **chiều ngược**.
Trên t5p thì gần như không đổi (+0.1739 → +0.1617), nên **chưa** lặp trên cả hai backbone.

**Cảnh báo phải giữ**: `com` không chỉ *to hơn* `4cwe`, nó còn là một **phân phối khác** (94 CWE
gộp thay vì 4 CWE). Đây không phải phép đo cỡ tập sạch như khối `tsize`. Ghi ra như một quan sát
gợi ý, **không** phải bằng chứng.

### ASAM: lần thứ TƯ cùng một mẫu hình

ASAM+RecAdam ở ρ tốt nhất của từng backbone hơn AdamW trần về ROC ở **3/4** cặp
(4cwe t5p +0.0398, com codebert +0.0111, com t5p +0.0211) và thua ở 1/4 (4cwe codebert −0.0171).
Trùng hướng với §39 và với phép đo 190 ô ở CLAUDE.md mục 2b: **ASAM cải thiện thứ hạng điểm**.

**n=1 fold. Chỉ sàng lọc, không viết vào bài.** `js_full` (1 556 dòng) đang chạy trên cả hai máy.

---

## §44 — NÚT THẮT 8 CHIỀU **KHÔNG** là một biểu diễn: nó thua PCA ở cả bốn checkpoint, và chất lượng của nó KHÔNG liên quan tới độ chính xác của head phụ (11/09, n=5 fold, 0 GPU)

Khai báo trước ở `records/prediction_2026-09-11_nut_that_8_chieu.md`, viết **trước** khi chạy.

Đặc trưng **đóng băng**, 760 dòng Python, logistic regression, `C` chọn trên val, chấm test.
Bốn bộ đặc trưng từ **cùng một checkpoint**: `p768` = pooled 768 chiều; `lat8` = `latent_proj(pooled)`;
`rnd8` = chiếu Gauss 768→8 của **cùng** pooled (**đối chứng**); `pca8` = 8 thành phần chính đầu,
fit **chỉ trên train** của từng fold.

| checkpoint | macro-F1 head phụ (§43) | `p768` | **`lat8`** | `rnd8` | `pca8` |
|---|---|---|---|---|---|
| codebert **cân bằng** | 0.1917 | 0.7827 | **0.6586** | 0.6504 | 0.6845 |
| codebert không cân bằng | 0.2000 (1 lớp) | 0.7700 | **0.6702** | 0.6603 | 0.6778 |
| t5p **cân bằng** | **0.5996** | 0.6118 | **0.5210** | 0.5076 | 0.5540 |
| t5p không cân bằng | 0.2000 (1 lớp) | 0.6405 | **0.5057** | 0.5164 | 0.5280 |

(ROC-AUC, trung bình 5 fold)

Δ **ghép cặp theo fold**, ROC-AUC:

| checkpoint | `lat8 − rnd8` | `lat8 − pca8` | `lat8 − p768` |
|---|---|---|---|
| codebert cân bằng | +0.0083 **3/5** | **−0.0259 0/5** | −0.1241 0/5 |
| codebert không cân bằng | +0.0099 **2/5** | −0.0076 1/5 | −0.0997 0/5 |
| t5p cân bằng | +0.0134 **3/5** | **−0.0330 1/5** | −0.0908 1/5 |
| t5p không cân bằng | **−0.0107** 3/5 | −0.0223 1/5 | −0.1348 0/5 |

Ngưỡng khai báo trước: `lat8 − rnd8 ≥ +0.02` **và** ≥ 4/5 fold. **Không checkpoint nào đạt** —
cao nhất là +0.0134 ở 3/5. Ngưỡng và số fold **giữ nguyên**, không sửa sau khi thấy số.

### Bốn điều đọc được

**1. `lat8` thua `pca8` ở CẢ BỐN checkpoint** (−0.0259 / −0.0076 / −0.0330 / −0.0223). Phép giảm
chiều tầm thường nhất — PCA trên chính đặc trưng ấy — **luôn** tốt hơn nút thắt đã học. Không có
ngoại lệ nào để bấu víu.

**2. Làm cho head phụ HỌC ĐƯỢC không làm nút thắt hữu ích hơn.** Đây là phép đo trực tiếp nhất, và
nó âm: trên t5p, head đi từ macro-F1 0.2000 (đoán một lớp) lên **0.5996** (gấp ba sàn), mà `lat8`
chỉ nhích **0.5057 → 0.5210**, vẫn ở mức ngẫu nhiên. Trên codebert head **tệ đi** (0.2000 → 0.1917)
và `lat8` cũng nhích xuống (0.6702 → 0.6586). **Không có quan hệ nào** giữa hai đại lượng.

**3. Trên t5p, `lat8` ở mức NGẪU NHIÊN** (0.5210 và 0.5057). Ngưỡng "bác thẳng" khai báo trước là
0.55 — **cả hai checkpoint t5p nằm dưới**. Nhánh quyết định thứ hai đi qua nút thắt ở t5p sẽ là
nhiễu thuần, và một cổng học được sẽ (đúng đắn) dìm nó về 0.

**4. Cân bằng lớp cho head phụ lại TÁCH THEO BACKBONE ở đặc trưng 768 chiều nữa**: codebert
`p768` 0.7700 → 0.7827 (**+0.013, tốt lên**), t5p 0.6405 → 0.6118 (**−0.029, tệ đi**). Lần thứ sáu.

### Hệ quả: MẢNH 2 DỪNG. Không phóng `run/gate3.sh`.

Hệ quả đã ghi **trước khi đo**: *"mảnh 2 vẫn có thể làm điểm số đẹp lên, nhưng không được viết là
cơ chế mới — phải viết là 'một nhánh ít tham số làm chính quy hoá'."* Kèm luật leo bậc (phải lặp
trên **cả hai** backbone), và kèm điều 3 ở trên (t5p dưới ngưỡng bác thẳng), khối `gate3` **không
được phóng**. Mã và bộ kiểm giữ lại (`--phase2_gate`, 12/12 phép, mặc định TẮT nên đường cũ không
đổi một byte) để nếu sau này có checkpoint mà `lat8` thật sự vượt đối chứng thì chạy được ngay.

**Phát biểu âm mang đi được**: trong hai pha, head phụ có nút thắt tiềm ẩn **không** tạo ra một
biểu diễn dùng lại được — nó là một phép nén mà PCA làm tốt hơn, và độ chính xác của head **không
dự báo** chất lượng nén. Giá trị đo được của head vẫn đúng như §7 đã ghi từ 31/08: **chống sập
Pha 1**, không phải chất lượng biểu diễn. Đây là kết luận có đối chứng (Hewitt & Liang), không
phải suy đoán.

### Một lỗi thiết kế bắt được TRƯỚC khi đốt GPU

Ô thô 16 mẫu trên CPU cho thấy cổng vô hướng huấn luyện ở **đúng learning rate của backbone**
(2e-5) **đứng yên ở g = 0.5000** sau cả hai epoch. Tính ra: AdamW mỗi bước dịch ~lr, cả 30 epoch ×
114 bước = 3 420 bước chỉ cho logit dịch tối đa ~0.068, tức `g ∈ [0.483, 0.517]`. Phép đo *"g tăng
khi tập đích co lại"* khi đó **vô nghĩa ngay từ gốc**, và sẽ tốn cả khối GPU để ra một bảng toàn
0.500. Đã sửa bằng nhóm tham số riêng (`--phase2_gate_lr`, mặc định 1e-2) và một phép kiểm **tái
hiện đúng lỗi này** (lr 2e-5 dịch < 0.01; lr 1e-2 dịch > 10 lần).

---

## §41.2 — Đảo chiều, đích JS **full**: Pha 1 đặt một **SÀN** dưới đích, chứ không phải "giúp nhiều hơn khi đích yếu" (11/09, **n=1 fold — chỉ sàng lọc**)

Bảng đầy đủ ba đích JS, Δ ghép cặp (chuyển giao − baseline), fold 1, seed 42, `plain` = AdamW trần:

| đích JS | n | backbone | ΔF1@0.5 | ΔF1@val | **ΔROC** | ΔPR | baseline ROC |
|---|---|---|---|---|---|---|---|
| `4cwe` | 812 | codebert | +0.0657 | +0.0812 | +0.1609 | +0.1205 | 0.4927 |
| | | t5p | +0.1598 | +0.1657 | +0.1739 | +0.1550 | 0.4686 |
| `com` | 1 384 | codebert | +0.0615 | +0.0648 | +0.0925 | +0.0712 | 0.4866 |
| | | t5p | +0.0881 | +0.1238 | +0.1617 | +0.1324 | 0.3946 |
| `full` | 1 556 | codebert | +0.0803 | +0.0803 | +0.1154 | +0.0877 | 0.4747 |
| | | **t5p** | **−0.0009** | +0.0038 | **−0.0138** | +0.0181 | **0.5375** |

Nhánh ASAM+RecAdam ở `full`: t5p +0.0330 ROC; **codebert THIẾU — ô trống**, job OOM lúc 09:39 vì
user `cuongtm` nở VRAM giữa chừng (GPU còn 3 MiB trống). Ghi ra đây thay vì im lặng (CLAUDE.md mục 3).

### Ô duy nhất transfer THẤT BẠI cũng là ô duy nhất baseline HỌC ĐƯỢC

`full`×t5p là cặp duy nhất trong sáu cặp có baseline **trên** mức ngẫu nhiên (0.5375), và nó là
cặp duy nhất Δ ROC **âm**. Nhìn qua thì đây là bằng chứng đẹp cho §40. **Nhưng nó không phải.**

### Cái bẫy đã kiểm và đã gỡ: tương quan đó là ẢO

Spearman(baseline ROC, ΔROC) = **−0.771** trên sáu ô. Hấp dẫn — và **vô nghĩa**, vì
`Δ = T − B` nên Δ chứa `−B` theo định nghĩa: tương quan âm sinh ra **từ chính phép trừ**, không
cần hiệu ứng nào cả. Mô phỏng 20 000 lần với `T`, `B` **độc lập hoàn toàn** (cùng trung bình và
SD như đo được, n=6):

| | giá trị |
|---|---|
| `corr(B, T−B)` khi T ⟂ B | trung bình **−0.643**, khoảng 90% **[−0.949, −0.038]** |
| đo được | **−0.708** |

Đo được **nằm gọn trong** khoảng của giả thuyết độc lập. Tương quan này **không** là bằng chứng.

### Điều đo được THẬT, và nó tốt hơn

| | min | max | biên độ | SD |
|---|---|---|---|---|
| baseline | 0.3946 | 0.5375 | 0.1428 | 0.0466 |
| **chuyển giao** | 0.5237 | 0.6536 | 0.1299 | 0.0499 |

Spearman(baseline, **chuyển giao**) = **−0.086** — điểm tuyệt đối của nhánh chuyển giao **gần như
không liên quan** tới việc baseline làm tốt hay tệ. Phát biểu đúng là:

> **Pha 1 nguồn Python đặt một SÀN dưới đích JavaScript.** Nhánh chuyển giao rơi vào dải
> 0.524–0.654 bất kể đích là `4cwe`, `com` hay `full`, trong khi baseline dao động 0.395–0.538.

Đây là phát biểu về **phương sai**, không phải về **trung bình**, và nó không bị phép trừ làm hỏng.

**Điều này KHÔNG động tới §40.** §40 đổi **cỡ tập train** trong **cùng** bộ dữ liệu, ghép cặp
trong cùng fold, và **cả hai** nhánh nhận **cùng** tập con vì cùng seed. Đó là một phép **can
thiệp có kiểm soát**, không phải tương quan cắt ngang, nên nó không mắc bẫy trên.

**n=1 fold mỗi ô, 6 ô. Giả thuyết, không phải phát hiện.** Cần n=3 mới được viết.

---

## §45 — Head phụ giỏi gấp ba KHÔNG đổi được gì sau fine-tune, và probe đóng băng KHÔNG dự báo được dấu (11/09, n=3 fold, đối chứng CÙNG MÁY)

Khai báo trước ở `records/prediction_2026-09-11_probe_du_bao_duoc_khong.md`, viết lúc khối đối
chứng mới xong 3/6 ô và **chưa đọc ô nào**.

Khối `auxb` trên vast `50570168`: ba nhánh trong **cùng cây, cùng fold, cùng phiên, cùng GPU** —
`baseline` (không Pha 1), `unbal` (Pha 1 thường), `bal` (Pha 1 `--aux_class_balanced`). 18 ô.

### Δ so với `baseline`, ghép cặp theo fold

| backbone | nhánh | ΔF1@0.5 | ΔF1@val | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|---|
| codebert | `bal` | +0.0501 3/3 | +0.0183 2/3 | +0.0329 3/3 | +0.0507 3/3 |
| codebert | `unbal` | +0.0592 3/3 | +0.0366 3/3 | +0.0290 3/3 | +0.0119 2/3 |
| t5p | `bal` | −0.0073 1/3 | +0.0039 2/3 | +0.0096 2/3 | +0.0090 2/3 |
| t5p | `unbal` | −0.0045 2/3 | −0.0101 2/3 | +0.0017 2/3 | −0.0063 2/3 |

### Ghép cặp TRỰC TIẾP `bal` − `unbal` (cùng fold)

| backbone | ΔF1@0.5 | ΔF1@val | **ΔROC-AUC** | ΔPR-AUC |
|---|---|---|---|---|
| codebert | −0.0090 1/3 | −0.0183 0/3 | **+0.0040 2/3** | +0.0389 3/3 |
| t5p | −0.0028 2/3 | +0.0140 2/3 | **+0.0079 3/3** | +0.0154 3/3 |

### Hai điều đọc được

**1. Cân bằng lớp gần như KHÔNG đổi gì sau fine-tune, trên cả hai backbone.** +0.0040 và +0.0079
trên ROC-AUC đều **ở hoặc dưới sàn nhiễu cùng-GPU 0.010**, và dấu trên F1@0.5 thì **âm** ở cả hai.
Trong khi đó §43 đo được can thiệp này đưa chính head phụ của t5p từ macro-F1 **0.2000** (đoán một
lớp) lên **0.5996** (gấp ba sàn, dùng cả bốn lớp). **Head giỏi gấp ba, mô hình không đổi.**

**2. Probe trên đặc trưng ĐÓNG BĂNG không dự báo được DẤU sau fine-tune.** §44 đo `p768` và dự báo:

| backbone | probe dự báo (`bal − unbal`) | đo được sau fine-tune | dấu |
|---|---|---|---|
| codebert | +0.0127 | +0.0040 2/3 | khớp |
| t5p | **−0.0287** | **+0.0079 3/3** | **SAI** |

Và nó sai ở đúng chỗ nó **tự tin nhất**: −0.0287 là gần gấp ba sàn nhiễu, không phải một con số
biên. **Hệ quả đã khai báo trước, giữ nguyên**: không dùng probe để sàng lọc thay GPU; §36 và §44
vẫn đúng nguyên văn nhưng chỉ phát biểu về **đặc trưng đóng băng**, không suy sang mô hình đã
fine-tune. Từ nay mọi câu trích §36/§44 phải kèm phạm vi đó.

### Mảnh cuối đóng lại mạch head phụ

Ba phép đo độc lập, cùng một kết luận:

| phép đo | kết quả |
|---|---|
| §43 — làm head phụ học được | được trên t5p (0.2000 → 0.5996), hỏng trên codebert |
| §44 — nút thắt 8 chiều có phải biểu diễn | **không**: thua PCA ở cả bốn checkpoint, ngang chiếu ngẫu nhiên |
| §45 — head giỏi hơn có làm mô hình tốt hơn | **không**: +0.004 / +0.008 ROC, trong nhiễu |

**Chất lượng của head phụ không phải một đòn bẩy.** Giá trị đo được của nó vẫn đúng như §7 ghi từ
31/08 — **chống sập Pha 1** (ở `codebert × full`, hai lần độc lập tại hai λ, `none` sập về ~0.34
còn `latent_bottleneck` giữ 0.545–0.564) — và chỉ thế.

### Sự cố kèm theo: vast nằm không 7 phút vì một file chưa được đẩy

Wrapper nối `auxb-ctl → seed15` chết bằng `scripts/queue_seed15.sh: No such file or directory`.
Nguyên nhân: tôi tạo file đó **sau** lần `rsync scripts/`, và phép đối chiếu byte sau đó chỉ phủ
`src`, `run`, `tools` — **không phủ `scripts`**. Lỗi chỉ lộ ra 50 phút sau, lúc chuỗi chạy tới.
Mất 10:46:48 → 10:54:08 ≈ **7 phút** máy tính tiền. **Bài học: đối chiếu phải phủ MỌI thư mục mà
chuỗi sẽ chạm tới, không chỉ những thư mục mình nhớ ra.**

---

## §40.3 — Hai seed, hai backbone: ở dữ liệu ĐẦY ĐỦ hiệu ứng thứ hạng là **ĐÚNG MỘT ĐỒNG XU** (11/09, **n=10 ô ghép cặp mỗi đầu mút**; seed 1234 đang chạy)

Seed 42 và seed 7, 5 fold mỗi seed, hai đầu mút đã **đủ** trên cả hai backbone (các mức N giữa
còn đang chạy). Δ ghép cặp trong cùng ô:

| backbone | N | F1@0.5 | F1@val | **ROC-AUC** | PR-AUC |
|---|---|---|---|---|---|
| codebert | 456 | +0.0463 **10/10** | +0.0458 **10/10** | +0.0126 **5/10** | **−0.0084 3/10** |
| codebert | 76 | +0.1346 **10/10** | +0.1378 **10/10** | **+0.1859 10/10** | **+0.1821 10/10** |
| t5p | 456 | +0.0126 7/10 | +0.0079 6/10 | +0.0056 **5/10** | +0.0091 6/10 |
| t5p | 76 | +0.0755 9/10 | +0.0742 9/10 | **+0.1008 10/10** | **+0.0766 10/10** |

### Phát biểu, và nó chặt hơn §40 cũ

> **Ở dữ liệu đích đầy đủ, phương pháp đổi NGƯỠNG QUYẾT ĐỊNH chứ không đổi THỨ HẠNG.**
> ROC-AUC ở N=456 là **5/10 trên codebert và 5/10 trên t5p** — đúng một đồng xu, ở cả hai họ
> backbone, qua hai seed độc lập. PR-AUC của codebert còn **âm** (−0.0084, 3/10). Trong khi
> F1@0.5 là **10/10** trên codebert.
>
> **Ở N=76, nó đổi tất cả**: ROC-AUC và PR-AUC đều **10/10** trên **cả hai** backbone.

§40 cũ nói "lợi ích tăng đơn điệu khi đích co lại" — đúng, nhưng nói nhẹ. Với hai seed thì thấy
được điều mạnh hơn: ở đầu mút lớn hiệu ứng thứ hạng **không nhỏ dần, mà bằng không**, và đó là
kết luận rút từ **20 ô** (2 backbone × 2 seed × 5 fold), không phải từ một trung bình.

### Vì sao một seed không đủ để thấy điều này

Tách theo seed ở N=456, ROC-AUC:

| backbone | seed 7 | seed 42 |
|---|---|---|
| codebert | +0.0139 **2/5** | +0.0114 **3/5** |
| t5p | **−0.0060 1/5** | **+0.0173 4/5** |

Trên t5p hai seed **đổi dấu nhau**: seed 42 cho 4/5 dương (nhìn như một hiệu ứng thật), seed 7 cho
1/5 (nhìn như hiệu ứng ngược). Gộp lại mới ra 5/10. **Một seed ở n=5 không phân biệt được
"hiệu ứng nhỏ" với "không có hiệu ứng"** — đây là ca cụ thể nhất của bài học đó trong dự án.

Ở N=76 thì ngược lại, hai seed trùng khít: codebert +0.1777 (5/5) và +0.1940 (5/5); t5p +0.1013
(5/5) và +0.1003 (5/5). Sai khác giữa hai seed ở t5p là **0.0010**, tức mười lần nhỏ hơn sàn nhiễu.

**Cách viết vào bài**: đừng viết "phương pháp thắng ở mọi cỡ dữ liệu". Viết rằng lợi ích **về thứ
hạng** là một hiện tượng **dữ liệu-ít**, và ở dữ liệu đầy đủ cái còn lại chỉ là hiệu chỉnh ngưỡng.

---

## §40.4 — ĐƯỜNG CONG ĐẦY ĐỦ ở **n=10 ô mỗi mức** (2 seed × 5 fold), cả bốn mức N, cả hai backbone (11/09)

Seed 42 và seed 7 đã **xong đủ** cả bốn mức N trên cả hai backbone. Δ ghép cặp trong cùng ô
(cùng cây, cùng seed, cùng fold; hai nhánh nhận **cùng** tập con vì cùng seed).

**codebert**

| N | F1@0.5 | F1@val | **ROC-AUC** | PR-AUC |
|---|---|---|---|---|
| 456 | +0.0463 10/10 | +0.0458 10/10 | **+0.0126 5/10** | **−0.0084 3/10** |
| 228 | +0.0444 9/10 | +0.0544 10/10 | **+0.0379 10/10** | +0.0257 7/10 |
| 152 | +0.0775 10/10 | +0.0642 9/10 | **+0.0726 10/10** | +0.0576 8/10 |
| 76 | +0.1346 10/10 | +0.1378 10/10 | **+0.1859 10/10** | +0.1821 10/10 |

**t5p**

| N | F1@0.5 | F1@val | **ROC-AUC** | PR-AUC |
|---|---|---|---|---|
| 456 | +0.0151 8/11 | +0.0126 7/11 | **+0.0064 6/11** | +0.0092 7/11 |
| 228 | +0.0356 9/10 | +0.0303 8/10 | **+0.0347 8/10** | +0.0338 7/10 |
| 152 | +0.0488 9/10 | +0.0650 9/10 | **+0.0809 9/10** | +0.0901 9/10 |
| 76 | +0.0755 9/10 | +0.0742 9/10 | **+0.1008 10/10** | +0.0766 10/10 |

### Ba tính chất, và cả ba đều giữ trên CẢ HAI backbone

**1. ROC-AUC đơn điệu chặt qua cả bốn mức.**
codebert **+0.0126 → +0.0379 → +0.0726 → +0.1859**; t5p **+0.0064 → +0.0347 → +0.0809 → +0.1008**.
Không có một chỗ lùi nào, ở **n=10** chứ không phải n=5.

**2. Số fold cùng dấu đi từ ĐỒNG XU sang TUYỆT ĐỐI.** ROC-AUC: codebert 5/10 → 10/10 → 10/10 →
10/10; t5p 6/11 → 8/10 → 9/10 → 10/10. Ở N=456 đó **đúng nghĩa một đồng xu**, không phải "hiệu
ứng nhỏ".

**3. PR-AUC của codebert ÂM ở N=456** (−0.0084, 3/10) rồi dương dần: 7/10 → 8/10 → **10/10**.
Nên phát biểu "dương trên cả bốn chỉ số" **chỉ đúng từ N=152 trở xuống trên codebert, và từ
N=152 trên t5p** — tự nó cũng là hiện tượng dữ liệu-ít.

### Điều duy nhất KHÔNG đơn điệu, và phải nêu

F1@0.5 của codebert: +0.0463 → +0.0444 → +0.0775 → +0.1346. Có một chỗ **lùi nhẹ** ở N=228
(−0.0019, nhỏ hơn sàn nhiễu 0.010 năm lần). Đừng viết "mọi chỉ số đều đơn điệu"; chỉ **ROC-AUC**
là đơn điệu chặt trên cả hai backbone.

### Cách viết vào bài

> Lợi ích của chuyển giao đa ngôn ngữ **về mặt thứ hạng** là một hiện tượng **dữ liệu-ít**. Ở dữ
> liệu đích đầy đủ nó bằng không (ROC-AUC 5/10 và 6/11 ô cùng dấu, PR-AUC âm trên codebert), và
> cái còn lại chỉ là hiệu chỉnh **ngưỡng quyết định**. Khi tập đích co xuống 1/6, nó thành
> +0.19 / +0.10 ROC-AUC với **10/10** ô cùng dấu trên cả hai họ backbone.

Seed 1234 đang chạy để lên n=15. Bậc hiện tại: **n=10 ô mỗi mức, 2 seed × 5 fold**.

---

## §41.3 — Đảo chiều ở **n=3 fold**: dương trên CẢ BỐN chỉ số, **3/3 fold**, CẢ HAI backbone. Và ASAM cải thiện **thứ hạng** lần thứ năm (11/09)

Nguồn **Python** → đích **JavaScript `js_4cwe`** (812 dòng, chia 486/162/164). Fold 1 chạy trên
161/158 hôm trước, fold 2–3 chạy trên vast `50570168`. Seed 42. Pha 1 nguồn Python **dùng lại**,
không huấn luyện lại (`PHASE1_TAG` tách khỏi `ARM_TAG`).

### Δ so với baseline, ghép cặp theo fold

| backbone | nhánh | ΔF1@0.5 | ΔF1@val | **ΔROC-AUC** | ΔPR-AUC |
|---|---|---|---|---|---|
| codebert | `plain` (AdamW trần) | +0.0648 **3/3** | +0.0834 **3/3** | **+0.1127 3/3** | +0.1098 **3/3** |
| codebert | ASAM+RecAdam ρ=0.1 | +0.0813 **3/3** | +0.0933 **3/3** | **+0.1400 3/3** | +0.1436 **3/3** |
| t5p | `plain` | +0.1338 **3/3** | +0.1220 **3/3** | **+0.1756 3/3** | +0.1536 **3/3** |
| t5p | ASAM+RecAdam ρ=2.0 | +0.1297 **3/3** | +0.1366 **3/3** | **+0.2066 3/3** | +0.1868 **3/3** |

**12/12 ô dương, cả bốn chỉ số, cả hai backbone, cả hai optimizer.** Theo luật leo bậc thì nhánh
này **đủ điều kiện lên n=5** — nhưng `data/js_4cwe_folds` **chỉ có 3 fold**, nên muốn lên n=5
phải dựng thêm fold 4–5 trước. Ghi ra đây để lần sau khỏi tưởng là đã bỏ sót.

### Trị tuyệt đối — và vì sao vẫn phải đọc theo kiểu "SÀN"

| backbone | baseline | `plain` | ASAM |
|---|---|---|---|
| codebert | **0.5119** | 0.6247 | 0.6520 |
| t5p | **0.4223** | 0.5979 | 0.6288 |

Baseline JS vẫn ở (codebert) hoặc **dưới** (t5p) mức ngẫu nhiên, đúng như §41.1/§41.2 đã đo ở
n=1. Nên phát biểu vẫn là **"Pha 1 nguồn Python đặt một SÀN dưới đích JavaScript"**, không phải
"thắng một baseline đang chạy được". Số fold cùng dấu giờ là 3/3 thay vì 1/1, nhưng bản chất phép
đo không đổi.

### ASAM so TRỰC TIẾP với `plain` trong cùng fold — lần thứ NĂM cùng một mẫu hình

| backbone | ΔF1@0.5 | ΔF1@val | **ΔROC-AUC** | **ΔPR-AUC** |
|---|---|---|---|---|
| codebert | +0.0165 2/3 | +0.0099 2/3 | **+0.0273 2/3** | **+0.0339 2/3** |
| t5p | **−0.0041 1/3** | +0.0146 2/3 | **+0.0310 3/3** | **+0.0332 3/3** |

Đây là khối đầu tiên trong dự án mà ASAM **dương trên CẢ HAI chỉ số thứ hạng, trên CẢ HAI
backbone, trong cùng một khối**. Và F1@0.5 thì lệch: 2/3 trên codebert, **1/3 và âm** trên t5p.

Trùng khít điều CLAUDE.md mục 2b đã rút ra từ 190 ô: **ASAM cải thiện THỨ HẠNG ĐIỂM, không cải
thiện QUYẾT ĐỊNH Ở NGƯỠNG 0.5.** Danh sách lần lặp: khối C/D (2026-08), đo lại 190 ô (08/09),
§39 (theo nguồn), §41.1 (đảo chiều n=1, 3/4 cặp), và nay §41.3. **Năm lần, không lần nào ngược.**

Đây là phát biểu **âm về F1 và dương về AUC** — phải viết cả hai vế, vì chính việc chỉ đọc F1 đã
giữ kết luận "ASAM null" sai suốt ba tuần.

### Vì sao đảo chiều đáng nằm trong bài

Mọi số khác của dự án đo **một chiều** (C/C++ + JS → Python). Nếu hiệu ứng là tính chất của
**phương pháp** thì đảo chiều vẫn phải thấy; nếu là tính chất của riêng cặp (nguồn này, đích này)
thì đảo chiều sẽ tắt. Nó **không tắt**: 12/12 ô dương. Trước §41.3 chưa phép đo nào của dự án
phân biệt được hai khả năng đó.

---

## §40.5 — **BẬC 3 (n = 15 = 5 fold × 3 seed), XONG TRỌN VẸN**: lợi ích về THỨ HẠNG là hiện tượng dữ liệu-ít, và ở dữ liệu đầy đủ nó BẰNG KHÔNG (11/09, 120 ô ghép cặp)

Đây là con số **đưa vào bài**, không phải sàng lọc. Seed 42 / 7 / 1234, 5 fold mỗi seed, bốn mức
N, hai backbone. **24/24 ô đủ cả 5 fold ghép cặp; 0 job hỏng.** Chạy trên 161 (codebert),
158 (t5p) và vast `50570168` (t5p seed 7 fold 4–5).

### codebert

| N | F1@0.5 | F1@val | **ROC-AUC** | PR-AUC |
|---|---|---|---|---|
| 456 | +0.0442 **15/15** p=0.000 | +0.0466 **15/15** p=0.000 | +0.0127 **8/15** p=1.000 | **+0.0002 5/15** |
| 228 | +0.0375 13/15 p=0.007 | +0.0451 14/15 p=0.001 | **+0.0354 15/15** p=0.000 | +0.0257 11/15 |
| 152 | +0.0844 13/15 p=0.007 | +0.0783 12/15 p=0.035 | **+0.0922 13/15** p=0.007 | +0.0769 11/15 |
| 76 | +0.1344 **14/14** p=0.000 | +0.1388 **14/14** p=0.000 | **+0.1849 14/14** p=0.000 | +0.1786 **14/14** p=0.000 |

### t5p

| N | F1@0.5 | F1@val | **ROC-AUC** | PR-AUC |
|---|---|---|---|---|
| 456 | +0.0229 12/15 p=0.035 | +0.0185 11/15 | +0.0094 **8/15** p=1.000 | +0.0101 9/15 |
| 228 | +0.0523 14/15 p=0.001 | +0.0500 13/15 p=0.007 | **+0.0504 13/15** p=0.007 | +0.0537 11/15 |
| 152 | +0.0710 12/14 p=0.013 | +0.0792 12/14 p=0.013 | **+0.1069 12/14** p=0.013 | +0.1135 12/14 p=0.013 |
| 76 | +0.0884 13/14 p=0.002 | +0.0832 13/14 p=0.002 | **+0.1188 14/14** p=0.000 | +0.0955 **14/14** p=0.000 |

### Ba phát biểu, cả ba giữ trên CẢ HAI backbone ở n=15

**1. ROC-AUC đơn điệu chặt qua cả bốn mức, không một chỗ lùi.**
codebert **+0.0127 → +0.0354 → +0.0922 → +0.1849**; t5p **+0.0094 → +0.0504 → +0.1069 → +0.1188**.

**2. Ở dữ liệu đích ĐẦY ĐỦ, hiệu ứng thứ hạng BẰNG KHÔNG — không phải "nhỏ".**
ROC-AUC ở N=456 là **8/15 trên cả hai backbone**, p=1.000 cả hai. PR-AUC của codebert là
**+0.0002 với 5/15** — bằng không tới bốn chữ số. Trong khi F1@0.5 là **15/15, p=0.000**.
Tức ở dữ liệu đầy đủ phương pháp **chỉ dời NGƯỠNG QUYẾT ĐỊNH, không đổi THỨ HẠNG**.

**3. Ở N=76, nó đổi tất cả**: cả bốn chỉ số **14/14, p=0.000** trên **cả hai** backbone.

### Ba seed độc lập nói cùng một chuyện (ROC-AUC, tách theo seed)

| backbone | N | seed 7 | seed 42 | seed 1234 |
|---|---|---|---|---|
| codebert | 456 | +0.0139 **2/5** | +0.0114 **3/5** | +0.0129 **3/5** |
| codebert | 76 | +0.1777 5/5 | +0.1940 5/5 | +0.1824 4/4 |
| t5p | 456 | **−0.0060 1/5** | +0.0173 4/5 | +0.0170 3/5 |
| t5p | 76 | +0.1013 5/5 | +0.1003 5/5 | +0.1638 4/4 |

Ở N=456 ba seed cho 2/5, 3/5, 3/5 (codebert) và 1/5, 4/5, 3/5 (t5p) — **đồng xu**, và trên t5p
seed 7 còn **đổi dấu**. Ở N=76 thì cả ba seed 5/5 hoặc 4/4 trên cả hai backbone. **Một seed ở
n=5 không phân biệt được "hiệu ứng nhỏ" với "không có hiệu ứng"**; ba seed thì phân biệt được.

### Ba ô bị LOẠI, theo đúng ngưỡng khai báo trước

`baseline F1@0.5 < 0.40` (khai báo ở `records/prediction_2026-09-11_duong_cong_co_dich.md`):
`t5p N=152 seed 1234 fold 4`, `codebert N=76 seed 1234 fold 4`, `t5p N=76 seed 1234 fold 3`.
Cả ba đều là baseline **sập** ở N nhỏ — đúng trường hợp cổng này sinh ra để bắt. Vì thế vài ô
ghi n=14 thay vì 15. **Không sửa ngưỡng hồi tố.**

### Cách viết vào bài — và cái KHÔNG được viết

> Lợi ích của chuyển giao đa ngôn ngữ **về mặt thứ hạng** là một hiện tượng **dữ liệu-ít**. Ở dữ
> liệu đích đầy đủ nó **bằng không** (ROC-AUC 8/15 ô cùng dấu trên cả hai backbone, PR-AUC
> +0.0002); cái còn lại chỉ là hiệu chỉnh **ngưỡng quyết định** (F1@0.5 15/15). Khi tập đích co
> xuống 1/6, nó thành **+0.185 / +0.119 ROC-AUC** với **14/14** ô cùng dấu trên cả hai họ backbone.

**KHÔNG được viết** "phương pháp thắng ở mọi cỡ dữ liệu" — số nói ngược. Và **không được** chỉ
báo F1: chính việc chỉ đọc F1 sẽ khiến N=456 nhìn như một thắng lợi 15/15 p=0.000, trong khi
thứ hạng không đổi chút nào.

---

## §46 — "FINETUNE HAI LẦN THUẦN" ĐÃ CHIẾM GẦN HẾT HIỆU ỨNG: head phụ thêm **+0.0005** trên 132 ô ghép cặp (12/09, **0 GPU — đọc lại dữ liệu đã có**)

**Câu hỏi người dùng nêu 12/09:** nhánh `none` (Pha 1 **không head**, Pha 2 AdamW) chính là
đối chứng *"finetune hai lần thuần"* — Pha 1 fine-tune backbone trên ccpp+js nhị phân
vul/non-vul, Pha 2 fine-tune tiếp trên Python. `latent_bottleneck` khác nó **đúng một thứ**:
có thêm head phụ ở Pha 1. Vậy Δ(`latent_bottleneck` − `none`) là **giá trị riêng của HEAD**,
còn Δ(`none` − `baseline`) là giá trị của **bản thân việc fine-tune hai lần**.

Đối chứng này **đã có sẵn 230 ô** (3 backbone × 3 nguồn × {adamw, recadam, recadam+ASAM},
seed 42, 5 fold) — không cần chạy thêm gì. Đọc bằng `tools/head_vs_none.py`.

### Giá trị riêng của head phụ — 137 ô ghép cặp trong cùng (cây, backbone, tag Pha 1, optimizer, ρ, seed, fold)

| tập | n | F1@0.5 | F1@val | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| TẤT CẢ | 137 | +0.0163 72/137 | +0.0168 77/137 | +0.0138 74/137 | +0.0095 71/137 |
| (i) ô `none` **SẬP** (ROC < 0.55) | **5** | **+0.4326 5/5** | +0.4313 5/5 | **+0.3546 5/5** | +0.2993 5/5 |
| (ii) ô `none` **bình thường** | **132** | **+0.0005 67/132** | +0.0011 72/132 | **+0.0009 69/132** | −0.0015 66/132 |

> **Toàn bộ trung bình dương của head đến từ 5 ô.** Năm ô đó là `codebert × full × RecAdam`
> ở `s42_codebert`, nơi Pha 1 `none` sập hẳn (F1 0.312–0.345, **ROC 0.442–0.535 — dưới mức
> ngẫu nhiên**) còn head giữ được 0.67–0.87. Bỏ 5 ô đó ra thì head cho **+0.0005**, tức nhỏ
> hơn sàn nhiễu cùng-GPU **0.010 khoảng hai mươi lần**, và 67/132 fold là đúng một đồng xu.

Đây **đúng bẫy số 2 của mục 2b** ("một ô cực trị kéo được trung bình nhưng không kéo được
đếm dấu") — và lần này nó kéo trung bình lên gấp **33 lần** giá trị thật.

### Tách theo optimizer, chỉ tập bình thường (ii)

| optimizer | n | F1@0.5 | F1@val | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **AdamW** (không SAM) | 46 | **+0.0071 32/46 p=0.01** | +0.0093 34/46 p=0.00 | +0.0036 27/46 p=0.30 | +0.0012 27/46 p=0.30 |
| RecAdam (không SAM) | 46 | −0.0010 20/46 p=0.46 | +0.0006 23/46 | +0.0000 24/46 | −0.0013 19/46 |
| RecAdam + ASAM ρ=0.1 | 40 | −0.0053 15/40 p=0.15 | −0.0076 15/40 | −0.0012 18/40 | −0.0047 20/40 |

**Head chỉ còn ăn ở AdamW, và chỉ ăn ở NGƯỠNG chứ không ở THỨ HẠNG**: F1@0.5 +0.0071
(32/46, p=0.01) nhưng ROC-AUC +0.0036 (27/46, p=0.30) và PR-AUC +0.0012 (27/46, p=0.30).
Và +0.0071 vẫn **dưới sàn nhiễu 0.010**.

> **Sửa §7.** Mục 7 ghi *"AdamW: head ăn về điểm — Δ vs `none` +0.0111, 33/43 fold,
> p=0.0006"*. Con số đó (a) tính trên **macro-F1 và chỉ macro-F1**, (b) **không tách** các ô
> `none` sập. Đo lại trên cả bốn chỉ số và tách ô sập thì còn **+0.0071, 32/46** ở F1@0.5 và
> **null ở cả hai chỉ số thứ hạng**. Đây là lỗi **đối xứng** với lỗi "ASAM null" ở mục 2b:
> lần đó F1 nói không còn AUC nói có; lần này F1 nói có còn AUC nói không.

### Và bản thân "finetune hai lần" thì ăn bao nhiêu? — Δ vs `baseline`, chỉ trên ô có CẢ HAI nhánh

| optimizer | nhánh | n | F1@0.5 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| AdamW | **`none` = finetune 2 lần** | 46 | **+0.0253 34/46** | +0.0054 23/46 | −0.0019 27/46 |
| AdamW | `latent_bottleneck` | 46 | **+0.0324 37/46** | +0.0089 29/46 | −0.0008 24/46 |
| RecAdam | `none` | 46 | +0.0252 38/46 | +0.0040 23/46 | −0.0017 25/46 |
| RecAdam | `latent_bottleneck` | 46 | +0.0242 33/46 | +0.0041 27/46 | −0.0030 23/46 |
| RecAdam+ASAM | `none` | 40 | +0.0279 34/40 | +0.0064 27/40 | +0.0006 24/40 |
| RecAdam+ASAM | `latent_bottleneck` | 40 | +0.0226 31/40 | +0.0052 22/40 | −0.0041 23/40 |

**Ba điều đọc thẳng:**

1. **Finetune hai lần thuần đã lấy ~78% hiệu ứng** ở cấu hình tốt nhất (+0.0253 / +0.0324),
   và **lấy hơn 100%** ở hai cấu hình còn lại (head âm).
2. **Mọi nhánh đều null ở thứ hạng**: ROC-AUC +0.0040…+0.0089 (tất cả **dưới sàn nhiễu
   0.010**), fold cùng dấu 22–29/46 ≈ đồng xu; PR-AUC âm ở 5/6 dòng. Toàn bộ cái gọi là
   "lợi ích transfer" ở dữ liệu đích **đầy đủ** là một phép **dời ngưỡng**. Khớp hoàn toàn
   với §40.5 (ở N=456 hiệu ứng thứ hạng là 8/15 p=1.000 trên cả hai backbone).
3. **Cài đặt tốt nhất ở F1@0.5 là `latent_bottleneck` + AdamW** (+0.0324, 37/46) — nhưng
   khoảng cách của nó với `none` + AdamW là +0.0071, dưới sàn nhiễu. Nói "tốt hơn" được;
   nói "tốt hơn có ý nghĩa" thì **không**.

### Giá trị DUY NHẤT còn đứng của head phụ vẫn đúng như §7 ghi từ 31/08

**Chống sập Pha 1** — và nó lớn: +0.4326 F1@0.5, 5/5 fold, ở `codebert × full` nơi `none`
rơi xuống **dưới mức ngẫu nhiên**. Đây là phát biểu về **phương sai**, không phải về trung
bình, và nó cùng họ với §41.2 ("Pha 1 đặt một SÀN dưới đích"). Cộng với §44 (nút thắt 8
chiều không hơn chiếu ngẫu nhiên) và §45 (head phụ giỏi gấp ba không đổi được gì), mạch
"head phụ là một đòn bẩy độ chính xác" **đóng hẳn**.

### LỖ HỔNG: cấu hình CHỐT chưa từng có đối chứng `none` cùng máy cùng phiên

Mọi ô ghép cặp ở trên nằm trong các khối **cũ** (kho Pha 1 tag `4cwe`/`com`/`full`/`_l02`,
λ mặc định). Ở **cấu hình chốt** của mục 7 — λ=0.05, SAM tắt **cả hai** pha, tag `_l0p05`,
cây `results/chot*` và `results/n48_*` — **không có nhánh `none` nào**. Kiểm tra:

```
$ python3 tools/head_vs_none.py results --tags
1666 o doc duoc; 378 o cau hinh goc; 0 va cham khoa con lai (OK)

  tag Pha 1 co nhanh none              : ['-', '4cwe', '4cwe_l02', 'com', 'com_l02', 'full', 'full_l02', 'sam1']
  ... trong do tag l0p05 cua none          : KHONG CO
```

(`latent_bottleneck` ở λ=0.05 thì có, nhưng nằm dưới tên nhánh biến thể —
`transfer_latent_bottleneck_<nguon>_l0p05_plain_adamw` và `..._r2p0` của khối `chot` —
nên bộ lọc "cấu hình gốc" của công cụ không đếm chúng; điều đó không đổi kết luận vì
vế `none` **trống hoàn toàn** ở λ=0.05.)

`model/s42/phase1/<bb>__none_{4cwe,com,full}` **dùng lại được ở mọi λ** (mục 5: `none` trả
`aux_loss=None` nên λ không vào loss), nên **chỉ cần chạy lại Pha 2**, không cần Pha 1 —
trừ `codebert__none_full` vốn là file `.rejected` 0 byte và đã được dựng lại ở
`model/ft2/phase1`. Nhưng đối chứng phải **cùng cây, cùng máy, cùng phiên** (mục 4), mà
`ft2_codebert` (có `none`) và `chot161_codebert` (có head) là **hai cây khác nhau với hai
baseline khác nhau** — lấy hiệu hai Δ đó là đúng thao tác bị cấm ở
`never-difference-two-means-even-for-controls`.

**Khối cần chạy khi có GPU** (đã đặc tả, **chưa chạy** — 12/09 cả 161 lẫn 158 đều đang bị
`cuongtm`/`tranmanhcuong` chiếm, và vast không thuê được):

| | |
|---|---|
| nhánh | `baseline`, `none`, `latent_bottleneck` — **cùng một cây** |
| Pha 1 | dùng lại, **không huấn luyện lại**: `none` ← `model/s42/phase1` (+ `model/ft2` cho codebert×full); head ← `model/n48/phase1/*_l0p05` |
| Pha 2 | AdamW, `--sam_rho 0`, λ=0.05 |
| nguồn | `4cwe`, `com`, `full` (đều là ccpp+js) |
| đích | `data/sven_python_folds_norm`, python |
| quy mô | **bậc 1: 3 fold, seed 42** → 2 backbone × 3 nguồn × 3 fold × 2 nhánh + 6 baseline = **42 ô** |


---

## §47 — ADAPTER FUSION (arXiv:2005.00247): mạnh và ổn định trên **codebert**, nhưng **KHÔNG lặp** trên t5p ở n=5 (13/09, 26 ô, vast 5060 Ti)

Người dùng yêu cầu 13/09 thử hướng AdapterFusion cho chuyển giao `ccpp+js → python`.

| | |
|---|---|
| **Pha 1** | `latent_bottleneck` λ=0.05 **như cũ**, thêm adapter bottleneck `src` (dim 48, reduction 16) sau **mỗi** lớp transformer; fine-tune **cả** backbone + adapter + head |
| **Pha 2** | thêm adapter `tgt` + lớp fusion; adapter `src` **đóng băng ở cả hai biến thể** |
| `fusft` | fine-tune **cả** backbone pretrained |
| `fusfrz` | **không** đụng backbone — chỉ adapter đích + fusion + head (~22M/132M tham số) |
| optimizer | AdamW, SAM tắt hẳn cả hai pha. Nguồn `com`, đích `sven_python_folds_norm` |
| đối chứng | `latent_bottleneck` (phương pháp chốt mục 7) — **cùng cây, cùng máy, cùng phiên** |

### Pha 1 có adapter KHÔNG làm hỏng gì

| backbone | có adapter | ba mốc không-adapter |
|---|---|---|
| codebert | 0.5583 | 0.5598 / 0.5626 / 0.5730 |
| t5p | **0.5980** | 0.5684 / 0.5897 / 0.5907 |

t5p còn cao hơn cả ba mốc. Checkpoint lớn hơn đúng 3,6 MB = 894 528 tham số adapter × 4 byte.

### Bậc 1 (n=3): `fusft` qua cổng leo bậc, `fusfrz` trượt

| backbone | biến thể | F1@0.5 | F1@val | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| codebert | `fusft` | +0.0207 3/3 | +0.0153 2/3 | +0.0257 3/3 | +0.0222 2/3 |
| t5p | `fusft` | +0.0175 3/3 | +0.0059 2/3 | +0.0129 3/3 | +0.0160 3/3 |
| codebert | `fusfrz` | +0.0095 1/3 | +0.0270 3/3 | +0.0041 2/3 | +0.0039 2/3 |
| t5p | `fusfrz` | +0.0107 2/3 | **−0.0024** 1/3 | +0.0078 1/3 | +0.0069 1/3 |

`fusft` dương trên **cả bốn** chỉ số trên **cả hai** backbone ⇒ đủ điều kiện lên n=5.
`fusfrz` bị loại. Đáng ghi: fold 1 của `fusfrz` trên codebert cho **+0.0406** rồi tụt xuống
**dưới** đối chứng ở fold 3 — lại một lần nữa đúng `never-conclude-from-the-first-cell`.

### Bậc 2 (n=5) — **ĐÂY MỚI LÀ KẾT QUẢ**

| backbone | n | F1@0.5 | F1@val | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **codebert** | 5 | **+0.0259 5/5** p=0.062 | +0.0222 4/5 | **+0.0224 5/5** p=0.062 | +0.0192 3/5 |
| **t5p** | 5 | +0.0036 3/5 | +0.0007 3/5 | +0.0057 3/5 | +0.0140 4/5 |

**Hiệu ứng trên t5p co lại gần hết khi thêm fold** — LẦN THỨ NĂM trong dự án:

| | n=3 | → n=5 |
|---|---|---|
| codebert F1@0.5 | +0.0207 (3/3) | **+0.0259 (5/5)** |
| codebert ROC-AUC | +0.0257 (3/3) | **+0.0224 (5/5)** |
| t5p F1@0.5 | +0.0175 (3/3) | **+0.0036 (3/5)** |
| t5p ROC-AUC | +0.0129 (3/3) | **+0.0057 (3/5)** |

Trên t5p, fold 4 **và** fold 5 đều âm; +0.0036 và +0.0057 nằm **dưới sàn nhiễu cùng-GPU 0.010**;
3/5 fold là đúng một đồng xu.

### Kết luận

> **`fusft` là hiệu ứng của codebert, không phải hiệu ứng của phương pháp.**

Trên codebert nó rất chắc: 5/5 fold ở **cả** F1@0.5 **và** ROC-AUC (tức cả ngưỡng lẫn thứ hạng,
khác hẳn phần lớn can thiệp khác vốn chỉ được một trong hai), biên độ trên sàn nhiễu, và chạm
sàn Wilcoxon p=0.0625 — mức tốt nhất n=5 có thể đạt. Nhưng **không lặp trên t5p**, nên theo luật
mục 1 **KHÔNG được lên n=15**, và không viết được thành phát biểu chung.

Nó rơi đúng họ **§38.2**: *"mọi nhánh thắng ở backbone này đều thua ở kia"*. Tính tới nay
**chỉ §40 (đường cong cỡ tập đích)** là qua được cổng "lặp trên cả hai backbone".

### Ba ghi chú kỹ thuật

1. **t5p + `fusft` OOM ở 16 GB.** Đã gỡ bằng hai bước, **không** hạ batch (hạ batch làm phép so
   đổi hai biến): (a) `W_V` tuyến tính nên `Σ αₙ·W_V(zₙ) = W_V(Σ αₙ·zₙ)` — gộp trước rồi chiếu,
   bỏ một tensor `(B,T,N,H)` mỗi lớp; (b) `--grad_checkpointing`. Cả hai cho gradient y hệt.
2. **Checkpoint Pha 2 bị xoá** nên **trọng số attention của lớp fusion đã mất**. Đó lại chính là
   phần diễn giải được của bài báo — nó cho biết mô hình dùng adapter nguồn bao nhiêu. Lần sau
   phải ghi trung bình trọng số fusion theo lớp vào file kết quả.
3. **Baseline (không Pha 1) chưa chạy** — người dùng hoãn có chủ ý: baseline chắc chắn trên 0.74
   và fusion đã hơn `latent_bottleneck` rồi nên nó không đổi được kết luận.

---

## §48 — Một nửa lợi ích của AdapterFusion là **SỨC CHỨA**, và §47 không lặp trọn vẹn trên máy thứ hai (13/09, 15 ô + Pha 1, vast `ntat` 50882617)

**Khai báo trước**: `records/prediction_2026-09-13_fusion_transfer_hay_suc_chua.md`, viết
**trước** khi chạy ô nào, kèm ngưỡng và **cả vùng "không kết luận"**.

### Câu hỏi

§47 cho `fusft` hơn đối chứng **+0.0259 F1@0.5 (5/5)** trên codebert. Nhưng `fusft` **thêm
~22M tham số fusion + 1,8M tham số adapter**. Hai cách giải thích không phân biệt được bằng
số của §47: **(T)** lợi ích đến từ tri thức adapter nguồn học ở Pha 1; **(C)** lợi ích đến từ
việc có thêm tham số, nội dung adapter nguồn không quan trọng.

### Can thiệp

`fusftrnd`: giữ **nguyên** kiến trúc, số tham số, optimizer, Pha 1 checkpoint, fold, máy —
**chỉ thay** trọng số adapter **nguồn** bằng nhiễu Gauss **khớp std từng tensor**, rồi đóng
băng y hệt. Khớp thang độ là bắt buộc: dùng khởi tạo gốc (`up = 0`) thì adapter nguồn thành
ánh xạ đồng nhất, chỉ trả lời "bỏ hẳn adapter nguồn thì sao", chưa loại được (C).

Cả ba nhánh chạy **cùng một máy mới** (máy của §47 đã huỷ; mục 4 cấm ghép cặp qua hai máy).

### Kết quả — n=5, codebert, seed 42

| nhánh | F1@0.5 | F1@val | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| `fusft` (adapter **đã học**) | +0.0226 4/5 | +0.0214 4/5 | +0.0113 3/5 | +0.0056 4/5 |
| `fusftrnd` (adapter **ngẫu nhiên**) | **+0.0111 4/5** | +0.0032 2/5 | +0.0069 3/5 | +0.0117 2/5 |
| **`fusft` − `fusftrnd`** (trực tiếp) | +0.0115 **3/5** | +0.0182 3/5 | +0.0044 3/5 | **−0.0061** 3/5 |

### Phán quyết theo đúng luật đã chốt trước: **KHÔNG KẾT LUẬN**

`D_rnd = +0.0111` (dưới ngưỡng +0.0130) **nhưng 4/5 fold** (ngưỡng đòi ≤3/5) ⇒ rơi vào vùng
giữa đã khai báo trước. Vùng đó nói rõ: *"không được đọc theo hướng có lợi; phải lên n=15
hoặc bỏ"*.

### Hai điều ĐỌC ĐƯỢC, và cả hai làm YẾU §47

**1. Khoảng một nửa lợi ích là sức chứa.** Adapter **ngẫu nhiên** tái tạo `+0.0111` trong
tổng `+0.0226`, với **cùng 4/5 fold**. Phần còn lại — chính là phần "tri thức Pha 1" — chỉ
`+0.0115` với **3/5 fold** và **âm ở PR-AUC (−0.0061)**, tức **không tách được khỏi nhiễu** ở n=5.

**2. §47 không lặp trọn vẹn trên máy thứ hai.** Cùng `fusft`, cùng codebert, cùng loại GPU
(5060 Ti, sàn nhiễu 0.010):

| | máy §47 | máy này |
|---|---|---|
| F1@0.5 | +0.0259 **5/5** | +0.0226 4/5 |
| **ROC-AUC** | **+0.0224 5/5** | **+0.0113 3/5** |

F1@0.5 giữ được; **ROC-AUC thì không** — từ 5/5 xuống 3/5, biên độ còn một nửa và dưới sàn
nhiễu. Tức phát biểu mạnh nhất của §47 (*"ăn ở CẢ ngưỡng LẪN thứ hạng"*) **chỉ đúng ở một lần chạy**.

### Pha 1 `codebert × com` + adapter NẰM NGAY RANH GIỚI — phát hiện phụ nhưng quan trọng

Huấn luyện lại Pha 1 trên máy mới với **đúng seed 42, đúng mã, đúng phiên bản thư viện**
(torch 2.11.0+cu128, transformers 4.57.1, sklearn 1.7.2) thì nó **SẬP**: val kẹt **0.3333**
(đoán một lớp) 7 epoch rồi cạn patience. Đối chiếu log từng epoch: hai lần chạy bám sát nhau
tới epoch 5 (train loss lệch < 0.003) rồi **tách ở epoch 6** — lần cũ vớ được một nhịp val
loss giảm nên patience reset và thoát cao nguyên; lần mới không. Nới `PHASE1_PATIENCE=10`
(riêng Pha 1) thì nó thoát ở epoch 6→7 và đạt **0.5776**.

> **Phi tất định của GPU đủ để quyết định Pha 1 này học được hay không.** Mọi kết quả xây
> trên một checkpoint Pha 1 đơn lẻ ở cấu hình này đều thừa hưởng sự bấp bênh đó — kể cả §47.

### Kết luận thực dụng

Hướng adapter-fusion: **không đáng lên n=15**. Lý do cộng dồn — (a) không lặp trên t5p (§47),
(b) một nửa lợi ích là sức chứa, (c) phần "tri thức" không tách được khỏi nhiễu, (d) chỉ số
thứ hạng không lặp trên máy thứ hai, (e) Pha 1 nền tảng thì bấp bênh.

### §48.1 — Trọng số fusion: chỉ **HÌNH DẠNG theo lớp** khác, còn **ĐỘ LỚN thì ngược chiều trực giác** (n=5)

Đo trọng số attention của lớp fusion trên tập test thật, **cả 5 fold**, mỗi fold một cặp
checkpoint (`tools/fusion_weights.py`, `run/fusion_w5.sh`). Cột đo là trọng số fusion gán cho
adapter **nguồn**.

| | độ tản giữa các lớp | trọng số TB cho nguồn |
|---|---|---|
| src **đã học** | **0.1776** | 0.5566 |
| src **ngẫu nhiên** | 0.1075 | **0.7065** |
| hiệu (đã học − ngẫu nhiên) | **+0.0702, 5/5 fold**, p=0.062 | −0.1498, **1/5 fold** |

**Hai phát biểu, và chúng nói ngược nhau:**

1. **Hình dạng**: adapter đã học làm fusion trộn **phụ thuộc lớp mạnh hơn** — 5/5 fold, chạm
   sàn Wilcoxon p=0.0625. Nhưng tỉ lệ chỉ **1,65×** (từng fold: 1.0 / 1.8 / 1.9 / 1.7 / 2.0×).
2. **Độ lớn**: fusion gán trọng số trung bình **CAO HƠN** cho adapter **ngẫu nhiên** (0.7065 so
   với 0.5566), 4/5 fold theo chiều đó.

> Điểm (2) bác thẳng cách đọc trực giác *"trọng số fusion cao = mô hình đang dùng tri thức"*.
> Một đường nhiễu cùng thang độ lại được ưu ái hơn một đường đã học. Trọng số fusion **không**
> đo được "hữu ích"; chỉ **hình dạng theo lớp** mới phân biệt được hai loại adapter.

### §48.2 — Bản n=1 của mục này đã SAI, và sai theo cách đáng ghi

Bản đầu của §48.1 (viết khi mới có **fold 1**) phát biểu: *"độ tản của bản đã học gấp **4,9 lần**
bản ngẫu nhiên; khi adapter nguồn là nhiễu, fusion nằm phẳng ~0.54 ở cả 12 lớp, nó không tìm
thấy gì để chọn."* Đo đủ 5 fold thì tỉ lệ là **1,65×**, và bản ngẫu nhiên **không** phẳng.

Bằng chứng sắc nhất nằm ở chỗ khác: **cùng fold 1**, hai checkpoint huấn luyện **độc lập** cùng
seed cho ra

| fold 1, cùng cấu hình | lần đo 1 | lần đo 2 |
|---|---|---|
| độ tản, src đã học | 0.1850 | 0.1218 |
| độ tản, src ngẫu nhiên | **0.0379** | **0.1209** |
| tỉ lệ | **4,9×** | **1,0×** |

> **Biến thiên GIỮA HAI LẦN CHẠY cùng cấu hình lớn hơn hiệu ứng cần đo trên một fold.** Con số
> 0.0379 của lần đầu là một lần bốc bài may, không phải tính chất của adapter ngẫu nhiên.

Đây là lần thứ **ba trong cùng một ngày** fold 1 vẽ ra bức tranh sạch hơn thực tế ở khối này —
trước đó là `fusfrz` cho `+0.0406` rồi tụt xuống dưới đối chứng, và ấn tượng đầu về `fusftrnd`.
Cộng với §41/§40 thì luật *"n=3 mới là sàn để DỪNG, không phải để KẾT LUẬN"* nên đọc chặt hơn nữa:
**n=1 không đủ để mô tả cả một cơ chế**, kể cả khi cơ chế đó nghe rất hợp lý.


---

## §49 — `RecAdam + ASAM ρ=2.0` trông tốt nhất nếu nhìn MỘT ô, nhưng trung bình nó **thấp hơn baseline** (14/09, đọc lại dữ liệu cũ, 0 GPU)

**Vì sao có mục này.** Xếp các cấu hình theo số tuyệt đối thì `latent_bottleneck + RecAdam +
ASAM ρ=2.0` (`r2p0`) đứng đầu: F1@0.5 **0.8498**, cao hơn cả nhánh adapter-fusion. Tôi đã suýt
đề xuất đưa nó vào hàng kiểm chứng như một ứng viên thay cấu hình chốt. **Đó là sai.**

### Phép gộp phải khử trùng trước

Lần tính đầu cho "17 tổ hợp `r2p0` / 52 tổ hợp chốt". **Sai**: đơn vị khi đó là nhóm
*(cây, nhánh, seed)*, nên **cùng một điều kiện** `(backbone, nguồn, seed)` bị đếm nhiều lần —
`t5p × 4cwe × seed 42` xuất hiện 5 lần ở `r2p0` và 9 lần ở chốt. Khử trùng về đúng
`(backbone, nguồn, seed)`: **11** tổ hợp `r2p0`, **33** tổ hợp chốt, **11** có cả hai.

### So công bằng — chỉ trên 11 tổ hợp CÓ CẢ HAI

| backbone | nguồn | seed | `r2p0` | `chốt` |
|---|---|---|---|---|
| codebert | 4cwe | 42 | **−0.1025** | +0.0516 |
| codebert | com | 42 | +0.0718 | +0.0515 |
| t5p | 4cwe | 7 | +0.0174 | +0.0319 |
| t5p | 4cwe | 42 | +0.0457 | +0.0349 |
| t5p | 4cwe | 1234 | +0.0327 | +0.0978 |
| t5p | com | 7 | −0.0269 | +0.0329 |
| t5p | com | 42 | +0.0580 | +0.0330 |
| t5p | com | 1234 | **−0.1169** | +0.0076 |
| t5p | full | 7 | **−0.0976** | −0.0049 |
| t5p | full | 42 | −0.0165 | +0.0227 |
| t5p | full | 1234 | +0.0159 | +0.0012 |

| | Δ F1@0.5 trung bình | số tổ hợp **âm** |
|---|---|---|
| `r2p0` | **−0.0108** | **5/11** |
| `chốt` (AdamW, SAM tắt) | **+0.0328** | **1/11** |

`r2p0` **trung bình còn thấp hơn baseline**, và chỉ thắng chốt ở **4/11** tổ hợp.

> Cùng backbone t5p, cùng nguồn `com`, **chỉ đổi seed** 42 → 1234: `r2p0` nhảy từ
> **+0.0580** xuống **−0.1169**. Trên `codebert × 4cwe` nó là **−0.1025**.

### Vì sao cấu hình chốt là cấu hình chốt

Không phải vì đỉnh cao nhất, mà vì **ổn định qua mọi điều kiện**: âm ở **1/11** tổ hợp so với
**5/11**. Mục 7 ghi *"RecAdam ổn định nhất giữa backbone"* — nhưng ổn định giữa **backbone**
khác ổn định giữa **seed và nguồn**, và ở chiều thứ hai `r2p0` rất tệ.

### Bài học phương pháp — đây mới là phần đáng giữ

Bảng xếp theo số tuyệt đối đã đẩy `r2p0` lên đầu vì nó được tính trên **đúng ô may nhất của
nó** (`codebert × com × seed 42`, n=4, một khối, một máy). Đúng **thiên lệch chọn lọc** mà mục 3
cảnh báo, chỉ khác chiều: lần trước là cổng chất lượng loại ô xấu, lần này là bảng xếp hạng
chọn ô tốt.

**Hai phép chặn, dùng cho mọi bảng xếp hạng cấu hình về sau:**

1. **Khử trùng về điều kiện thật** `(backbone, nguồn, seed)` trước khi đếm — nếu không, một
   cấu hình chạy nhiều khối sẽ được đếm nhiều lần.
2. **Chỉ so trên phần giao**. Hai cấu hình phủ khác nhau thì trung bình của chúng không so
   được: cấu hình nào tình cờ không chạy ở điều kiện khó sẽ trông tốt hơn.
3. Báo **số điều kiện ÂM**, không chỉ trung bình. `r2p0` và chốt cách nhau 0.044 ở trung bình
   nhưng **5/11 so với 1/11** ở đếm dấu mới là thứ nói lên bản chất.

---

## §50 — Tách theo NHÓM RÒ RỈ: phương pháp chốt **KHÔNG** sống nhờ học vẹt, nhưng adapter-fusion **không** mang thêm tri thức (14/09, **0 GPU**, đọc lại dữ liệu đã có)

**Vì sao nhóm này là phép kiểm đúng.** Reviewer yêu cầu giữ split **ngẫu nhiên** vì nó cố ý
chứa những hàng test có **bản đối nghịch gần trùng** — cùng code, **NGƯỢC NHÃN** — nằm trong
train. Đoán đúng ở đó nghĩa là mô hình phân biệt được khác biệt nhỏ giữa hàm lỗi và hàm đã vá,
tức học **đặc trưng lỗ hổng** chứ không khớp **mẫu văn bản**. Điểm tổng trộn ba nhóm nên không
đọc được điều đó. `tools/leak_groups.py` đã gán nhãn nhóm từ 07/09; `tools/leak_groups_pair.py`
(mới) mở rộng để so **hai nhánh bất kỳ**, vì các cây fusion không có baseline.

Nhóm mỗi fold: `train` 16–27 hàng · `test` 4–12 · `none` 107–113 (trên 152).

### (1) Phương pháp CHỐT vs BASELINE — n=15, codebert

| nhóm | số hàng TB | Δ macro-F1@0.5 |
|---|---|---|
| `train` (bản đối nghịch trong TRAIN) | 23.4 | **+0.0591 11/15** p=0.118 |
| `test` (bản đối nghịch trong TEST) | 7.6 | +0.1347 10/15 |
| **`none`** (không có bản đối nghịch) | **111.2** | **+0.0459 14/15 p=0.001** |
| TẤT CẢ | 152 | +0.0493 **15/15** p=0.000 |

> **Đây là kết quả đáng giá nhất của mục này.** Lợi ích **KHÔNG** tập trung ở nhóm dễ học vẹt:
> trên **73% hàng sạch** (`none`) nó vẫn **+0.0459 với 14/15 fold, p=0.001** — trên sàn nhiễu
> 0.010 gấp bốn lần. Tức phương pháp chốt **khái quát hoá thật**, không sống nhờ rò rỉ.
> Đối chiếu §21.1: ASAM ρ=2.0 thì ngược lại — lợi ích của nó **tập trung ở nhóm `train`**
> (+0.0509) còn trên hàng sạch chỉ +0.0071, **dưới** sàn nhiễu.

### (2) `fusft` vs phương pháp CHỐT — n=10

| nhóm | Δ macro-F1@0.5 |
|---|---|
| `train` | +0.0332 7/10 |
| `none` | +0.0299 8/10 |
| TẤT CẢ | +0.0243 **9/10** p=0.021 |

Fusion cộng thêm khá đều trên cả hai nhóm.

### (3) `fusft` vs `fusftrnd` — tách TRI THỨC khỏi SỨC CHỨA, n=5

| nhóm | Δ macro-F1@0.5 |
|---|---|
| **`train`** | **−0.0595 1/5** |
| `test` | −0.0007 2/5 |
| `none` | +0.0273 3/5 |
| TẤT CẢ | +0.0115 3/5 |

> **Trên đúng nhóm đo "có học được đặc trưng lỗ hổng không", adapter nguồn ĐÃ HỌC lại THUA
> adapter NGẪU NHIÊN** (−0.0595, 1/5). Phần lợi ích của `fusft` so với đối chứng nhiễu nằm ở
> nhóm `none` (+0.0273), tức ở **khái quát hoá thường**, không ở **phân biệt lỗ hổng**.

Cộng với §48 (một nửa lợi ích tái tạo được bằng nhiễu) và §48.1 (trọng số fusion ưu ái adapter
ngẫu nhiên hơn): ba phép đo độc lập cùng nói một điều — **fusion không mang thêm tri thức
chuyển giao**.

**Giới hạn**: n=5 ở mục (3), nhóm `train` chỉ ~23 hàng/fold nên phương sai lớn. Đọc là **dấu
hiệu mạnh**, không phải kết luận đóng.

### Đọc gộp — trạng thái thật của đóng góp hiện tại

| | khái quát hoá (`none`, 73% hàng) | phân biệt lỗ hổng (`train`) |
|---|---|---|
| chốt vs baseline | **+0.0459 14/15 p=0.001** | +0.0591 11/15 |
| fusion vs chốt | +0.0299 8/10 | +0.0332 7/10 |
| fusion: tri thức vs sức chứa | +0.0273 3/5 | **−0.0595 1/5** |

---

## §51 — Nhãn Pha 1 NHIỄU, và độ nhiễu đó dự đoán đúng thứ tự chuyển giao (14/09/2026)

**Nguồn của mục này:** hai đồng nghiệp chạy **độc lập** baseline in-domain trên CleanVul js và
CleanVul cpp, **cả hai đều dưới 0.6**. Người dùng báo 14/09. Con số của chính dự án này khớp.

### Tác vụ NGUỒN chỉ học được tới ~0.57, trong khi tác vụ ĐÍCH học được tới 0.80

Phase-1 val macro-F1, **ba seed trên ba máy khác nhau** (§40.5):

| nguồn | seed 42 | seed 7 | seed 1234 | độ tản |
|---|---|---|---|---|
| `4cwe` (930 dòng, 4 CWE, 87% js) | 0.6976 | 0.6684 | 0.7223 | 0.054 |
| `com` (3 744) | 0.5897 | 0.5907 | 0.5684 | 0.022 |
| `full` (7 598) | 0.5648 | 0.5834 | 0.5930 | 0.028 |

Baseline trên **đích** (python, không Pha 1): **0.7985 / 0.8073 / 0.8140**.

> Ta đang tiền-huấn-luyện trên một tác vụ **0.57** để giúp một tác vụ **0.80**. Đây không phải
> "pretrain mạnh → finetune yếu" mà là chiều ngược lại. Chưa mục nào của dự án nêu thẳng điều này.

### Thứ tự chuyển giao ĐI ĐÚNG theo độ sạch của nhãn nguồn

Từ §40.5, n=15 mỗi ô, RecAdam λ=0.05, ρ=0:

| nguồn | val Pha 1 | Δ macro-F1 vs baseline | Δ ROC-AUC |
|---|---|---|---|
| `4cwe` | **0.70** | +0.0133 (11/15) | +0.0030 (8/15) |
| `com` | 0.59 | +0.0092 (10/15) | +0.0056 (9/15) |
| `full` | 0.57 | +0.0021 (8/15) | **−0.0094 (4/15)** |

**Nhiều dữ liệu nguồn hơn ⇒ chuyển giao KÉM hơn.** Đó là chữ ký của nhiễu nhãn, không phải của
thiếu dữ liệu. `full` gấp đôi `com` và âm trên ROC-AUC.

> **Nhiễu loạn phải nêu:** ba nguồn khác nhau không chỉ ở độ sạch mà còn ở **cỡ** (930/3744/7598),
> **tỉ lệ ngôn ngữ** (87% js / 63% ccpp / 80% ccpp) và **độ trùng CWE với đích** (`4cwe` đúng bằng
> bốn CWE của đích). Thứ tự này **nhất quán** với giả thuyết nhiễu nhưng **chưa tách** khỏi ba
> biến kia. Phép tách nằm ở đối chứng "xáo nhãn nhị phân nguồn" — **chưa chạy**.

### Tài liệu khớp

- **CleanVul** (arXiv:2411.17274) tự đo: bộ dữ liệu lỗ hổng mang **40–75% nhiễu nhãn**, vì mọi
  thay đổi trong một commit vá đều bị dán nhãn "liên quan lỗ hổng".
- **arXiv:2309.17002** (ICLR'24): nhiễu nhãn lúc tiền-huấn-luyện **có thể có lợi in-domain**
  nhưng **luôn làm hại out-of-domain**. Chuyển giao **xuyên ngôn ngữ chính là out-of-domain**.

### Cấu trúc CẶP là có thật — đã kiểm

`com`: **1755 cặp 2-phần tử hoàn chỉnh** / 1797 `pair_id` (93,8% số dòng).

| phép đo (mẫu 300 cặp) | giá trị |
|---|---|
| Jaccard 5-gram **trong cặp** | trung vị **0.802** (92/300 ≥ 0.9; 54/300 < 0.3) |
| Jaccard cặp **ngẫu nhiên** | trung vị **0.000** |
| chênh độ dài tương đối trong cặp | trung vị 0.086 |
| cùng `cwe_id` ở cả hai nửa | **1755/1755** |

Tập **đích cũng là dữ liệu cặp**: 380 vul / 380 fixed, **301/380** tìm lại được bạn đối nghịch
1-1 ở Jaccard ≥ 0.5 (trung vị 0.68). Tức "hàng gần trùng nhưng ngược nhãn" mà reviewer cố ý đưa
vào split ngẫu nhiên **chính là nửa còn lại của cặp**.

### Bẫy dữ liệu đi kèm

Trường `cwe` của **nguồn** là `CWE-89`, của **đích** là `CWE-089`; `cwe_class` của nguồn là
**pillar CWE-1000 (10 lớp)** còn của đích là **0–3 cho đúng 4 CWE**. Ghép hai bên **phải** chuẩn
hoá qua `cwe_id` dạng số. Sau khi chuẩn hoá, cả bốn CWE của đích đều có trong `com` ở **cả hai
ngôn ngữ**: CWE-89 46 dòng (6 ccpp/40 js), CWE-78 100 (36/64), CWE-22 92 (54/38), CWE-79 692 (22/670).

---

## §52 — `fus5060`: fusion vs baseline đo CÙNG MỘT MÁY, n=5 (14/09/2026)

Lỗ hổng đo lường còn lại của §47/§48: mọi con số `fusft` trước đây so với baseline chạy ở
**máy khác** (hai vast đã huỷ), mà sàn nhiễu giữa loại GPU là 0.028 — lớn hơn cả hiệu ứng.
Khối này chạy **cả ba nhánh trên cùng một vast RTX 5060 Ti**, cùng phiên.

codebert · nguồn `com` · λ=0.05 · `adamw` · `--sam_rho 0` · fold 1–5 · seed 42 · **15/15 ô**.
Pha 1 dùng lại (không huấn luyện lại): `n48/...com_l0p05` và `fus2/...com_l0p05_ad48`.

| so sánh | F1@0.5 | F1@val | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| `fusft` − baseline | **+0.0637 5/5** | +0.0611 4/5 | **+0.0445 5/5** | +0.0322 4/5 |
| chốt − baseline | +0.0240 4/5 | +0.0203 4/5 | +0.0150 4/5 | +0.0163 4/5 |
| `fusft` − chốt | **+0.0397 5/5** | **+0.0408 5/5** | **+0.0294 5/5** | +0.0160 4/5 |

Mạnh hơn ước lượng khác máy trước đây (`fus1`: `fusft`−chốt +0.0259 5/5 F1).

> **ĐÍNH CHÍNH — đọc §52.1 trước khi dùng bảng này.** Cùng ngày, khối `fus3` ở local
> chạy lại đúng ba nhánh này với **cùng checkpoint Pha 1** trên A4000: `fusft` − chốt
> tụt từ **+0.0397 5/5** xuống **−0.0005 3/5** ở F1. Nhưng **§52.2** tính đủ bốn khối cho thấy
> ROC/PR **dương ở 4/4 khối** — thứ rút lại là con số ở **ngưỡng**, không phải cả phát biểu.

> **KHÔNG hồi sinh AdapterFusion.** §48 vẫn đứng — một nửa lợi ích tái tạo được bằng adapter
> **ngẫu nhiên** (+0.0111 so với +0.0226); §50 vẫn đứng — trên nhóm rò rỉ `train`, adapter **đã
> học THUA** adapter ngẫu nhiên (−0.0595, 1/5). Phản biện "chỉ là sức chứa" chưa bị bác. Thêm:
> n=5 thì p=0.0625 là **sàn**, một backbone duy nhất, và t5p **không lặp** ở n=5 (+0.0036 3/5).
> Đây là một lỗ hổng đo lường được bịt, không phải một kết luận mới.

Kéo về `results_fus5060_ntat/`, đối chiếu 15/15 file **khớp tuyệt đối cả tên lẫn byte**.

---

## §53 — Pha 1 TẠO RA trục lỗ hổng; biểu diễn code gốc KHÔNG có (14/09/2026)

`tools/patch_direction_probe.py`, chạy trên CPU, **không tốn một giây GPU huấn luyện**.
1755 cặp `com` (1063 ccpp / 692 js), `d = h(vul) − h(fixed)`, pooling `cls`.
Hai mô hình, cùng một phép đo: **CodeBERT gốc** và **checkpoint Pha 1** (`com`, λ=0.05).

### Cấu trúc hướng-vá

| | CodeBERT **gốc** | sau **Pha 1** |
|---|---|---|
| cos, cùng CWE **khác** ngôn ngữ | +0.0081 | **+0.2850** |
| cos, khác CWE khác ngôn ngữ | +0.0130 | +0.2468 |
| hiệu (cùng − khác) | **−0.0049**, z = −1.76, p=0.065 | **+0.0383**, z = **+4.24**, p=0.000 |
| hiệu sau khi trừ trung bình toàn cục | −0.0137, z = −3.24 | +0.0885, z = +4.65 |
| `\|trung bình toàn cục\| / \|d\|` | **0.152** | **0.779** |

### `μ_c` ước từ `com` (ccpp+js) chấm với đặc trưng PYTHON — 0 bước huấn luyện

AUC phân biệt vul/fixed trên python, dùng hướng lấy **hoàn toàn từ ngôn ngữ khác**:

| CWE | n | gốc | **sau Pha 1** | dùng μ của CWE **khác** (Pha 1) | hướng ngẫu nhiên |
|---|---|---|---|---|---|
| CWE-22 | 66 | 0.5280 | 0.6474 | 0.6314 | 0.489 ± 0.086 |
| CWE-78 | 204 | 0.5013 | **0.8271** | 0.7941 | 0.558 ± 0.244 |
| CWE-79 | 82 | 0.5259 | **0.8132** | 0.7793 | 0.468 ± 0.196 |
| CWE-89 | 408 | 0.5296 | 0.5915 | 0.5620 | 0.511 ± 0.073 |
| **một hướng duy nhất**, cả 760 | | **0.5242** | **0.6477** | | |

### Ba điều đọc được

1. **Biểu diễn code gốc KHÔNG chứa "trục lỗ hổng".** Mọi AUC 0.50–0.53, không phân biệt được
   với hướng ngẫu nhiên (0.499 ± 0.03), và các hướng-vá gần như **trực giao** (cos ≈ 0.01).
   **Pha 1 tạo ra toàn bộ hiệu ứng.**
   > Hệ quả: `d = h(vul) − h(fixed)` sau Pha 1 **chính là trục quyết định của bộ phân loại Pha 1**.
   > Một loss ép căn hướng-vá chỉ dựng lại tường minh cái huấn luyện nhị phân đã dựng ngầm —
   > **không phải cơ chế mới**. Đề xuất A của `NEXT_CONTRIBUTION.md` §4 **đóng** tại đây.

2. **Phần "theo CWE" là thật nhưng nhỏ.** z = +4.24 rất chắc, nhưng dùng μ của **sai** CWE chỉ
   mất ~0.03 AUC. **78%** độ lớn hướng-vá nằm trên **một** trục duy nhất.

3. **Pha 1 NÉN SỤP không gian hướng-vá**: từ trực giao (0.152) thành gần một chiều (0.779).
   Sau huấn luyện, một bản vá SQL-injection và một bản vá path-traversal trỏ gần cùng hướng.
   Đây đúng là cơ chế arXiv:2309.17002 nêu là nguyên nhân nhiễu nhãn hại **out-of-domain** —
   mà chuyển giao xuyên ngôn ngữ chính là out-of-domain. **Giả thuyết mới, ngược với đề xuất
   ban đầu: vấn đề không phải hướng-vá chưa đủ căn, mà là Pha 1 căn QUÁ TAY.**

### Ràng buộc phải nêu

- So "gốc vs Pha 1" đổi **hai** thứ cùng lúc: hướng μ **và** không gian đặc trưng. Nó đủ để
  kết luận "Pha 1 tạo ra trục chuyển giao được", **không** đủ để tách phần nào do cái nào.
  Đã có đối chứng trong không gian Pha 1: hướng **ngẫu nhiên** cho ~0.50, nên trong cùng
  không gian đó hướng-từ-cặp hơn hẳn ngẫu nhiên.
- **Nguồn mỏng thì chuyển giao yếu**: CWE-89 chỉ có 46 dòng trong `com` (6 ccpp/40 js) và cho
  AUC 0.5915 — trong khi nó là **408/760 hàng của tập đích**. Đó là lý do con số gộp chỉ 0.6477
  dù CWE-78 và CWE-79 đạt 0.81–0.83.
- Trừ trung bình không ảnh hưởng AUC (chỉ tịnh tiến điểm), nên không có rò rỉ transductive.

### §52.1 — §52 KHÔNG LẶP LẠI trên card khác. Phép lặp sạch nhất dự án từng có (14/09)

Khối `fus3` ở local (A4000) chạy **đúng ba nhánh, đúng 5 fold, đúng seed 42** của §52, dùng
**cùng file checkpoint Pha 1** (`model/n48/phase1` và `model/fus2/phase1` — cùng đường dẫn,
đã đối chiếu từng byte khi đẩy lên vast), cùng mã, cùng dữ liệu. **Chỉ khác GPU.**

| nhánh | RTX 5060 Ti | A4000 | lệch |
|---|---|---|---|
| baseline | 0.7738 | 0.7631 | −0.011 |
| chốt | 0.7978 | **0.8220** | **+0.024** |
| `fusft` | **0.8375** | 0.8215 | −0.016 |

| phép so | 5060 Ti | A4000 |
|---|---|---|
| `fusft` − baseline | +0.0637 5/5 | +0.0584 5/5 |
| chốt − baseline | +0.0240 4/5 | **+0.0589 5/5** |
| **`fusft` − chốt, F1@0.5** | **+0.0397 5/5** | **−0.0005 3/5** |
| `fusft` − chốt, F1@val | +0.0408 5/5 | −0.0001 2/5 |
| `fusft` − chốt, ROC-AUC | +0.0294 5/5 | +0.0203 4/5 |
| `fusft` − chốt, PR-AUC | +0.0160 4/5 | **+0.0345 5/5** |

**Ba điều đọc được:**

1. **Con số +0.0397 5/5 của §52 BỊ RÚT LẠI — nhưng CHỈ ở chỉ số NGƯỠNG.** Một kết quả 5/5 fold
   biến mất sạch ở F1 khi chỉ đổi card. Nguyên nhân nằm gần trọn ở nhánh **chốt** (+0.024
   giữa hai máy) chứ không ở `fusft` (−0.016).

   > **ĐÍNH CHÍNH 14/09 chiều — bản đầu của dòng này viết "phát biểu `fusft` hơn chốt bị rút
   > lại", QUÁ RỘNG.** Tính đủ **bốn khối độc lập trên bốn máy** (**§52.2**) thì ROC-AUC và
   > PR-AUC **dương ở 4/4 khối**. Thứ bị rút lại là con số và đếm dấu ở **ngưỡng 0.5**, không
   > phải cả phát biểu.
2. **Sàn nhiễu 0.028 giữa hai loại GPU được xác nhận lần nữa — và lần này nó nuốt trọn một
   kết quả 5/5.** Đây là lần đầu dự án có phép lặp với **checkpoint Pha 1 giống hệt**, nên
   không thể đổ cho dao động Pha 1 (§48.2): toàn bộ chênh lệch sinh ra ở **Pha 2**.
3. **Ngưỡng sập, thứ hạng giữ.** F1@0.5 và F1@val về 0; ROC-AUC và PR-AUC dương ở **cả hai
   máy**. Đúng mẫu hình đã lặp năm lần với ASAM (mục 2b): cơ chế cải thiện **xếp hạng**, không
   cải thiện **quyết định ở ngưỡng 0.5**. In một chỉ số thì cùng bộ dữ liệu này cho ra hoặc
   "5/5 rất mạnh" hoặc "không có gì", tuỳ chỉ số chọn.

> **Luật rút ra, áp cho mọi khối sau:** n=5 trên **một máy** không đủ để phát biểu, kể cả khi
> 5/5 fold và cả bốn chỉ số cùng dấu. Phải có **một lần lặp trên phần cứng khác** trước khi
> viết bất cứ điều gì. Đây là lần thứ **sáu** một mẫu hình sạch ở quy mô nhỏ biến mất khi mở rộng.

---

## §54 — XÁO NHÃN NGUỒN: nhãn **có** mang tri thức, nhưng nó chỉ **ngăn hỏng** chứ không **cộng thêm** (14/09, 24 ô, vast `ntat` 5060 Ti)

Dự đoán ghi **trước khi đo**: `records/prediction_2026-09-14_xao_nhan_phoi_nhiem_hay_tri_thuc.md`.
Bậc 1 — kiểm chứng, n=3 fold, seed 42, **cả hai backbone**, nguồn `com`, nhánh `none`
(cô lập đúng nhãn nhị phân), `adamw`, `--sam_rho 0`, Pha 2 trên `sven_python_folds_norm`.

Ba nhánh Pha 1 **cùng seed, cùng split, cùng đúng 12 epoch** (`--save_last_epoch`, tắt dừng sớm),
chỉ khác cột `label`. Phép chia Pha 1 chia theo `pair_id` và **không đọc nhãn**, nên split trùng khít.

### Pha 1 — phép xáo làm đúng việc

| | codebert | t5p |
|---|---|---|
| nhãn thật | 0.5765 | 0.5890 |
| xáo toàn bộ | **0.4374** | **0.4938** |
| đổi chỗ trong cặp | **0.4525** | **0.5105** |

Cả bốn ô xáo đều **ở hoặc dưới mức đoán ngẫu nhiên** — tác vụ thành không học được, đúng thiết kế.

### Δ ghép cặp theo `(backbone, fold)` — 6 điểm mỗi ô

| | F1@0.5 | F1@val | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| **thật − xáo toàn bộ** | **+0.0619 5/6** | **+0.0768 6/6 p=0.031** | **+0.0578 5/6** | **+0.0531 5/6** |
| **xáo toàn bộ − baseline** | −0.0601 2/6 | **−0.0753 0/6 p=0.031** | **−0.0736 0/6 p=0.031** | **−0.0692 0/6 p=0.031** |
| thật − đổi chỗ trong cặp | +0.0387 3/6 | +0.0357 4/6 | +0.0398 5/6 | +0.0527 5/6 |
| đổi chỗ trong cặp − xáo toàn bộ | +0.0232 4/6 | +0.0411 4/6 | +0.0180 4/6 | +0.0004 4/6 |
| **thật − baseline** | **+0.0018 3/6** | +0.0015 2/6 | **−0.0158 3/6** | −0.0161 2/6 |
| đổi chỗ trong cặp − baseline | −0.0369 2/6 | −0.0342 2/6 | −0.0556 2/6 | −0.0688 1/6 |

### Tách theo backbone — `thật − nhãn bịa`

| | F1@0.5 | F1@val | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| codebert, thật − xáo toàn bộ | +0.0733 2/3 | +0.0794 3/3 | +0.0440 2/3 | +0.0406 2/3 |
| **t5p, thật − xáo toàn bộ** | **+0.0505 3/3** | **+0.0741 3/3** | **+0.0715 3/3** | **+0.0655 3/3** |
| codebert, thật − đổi chỗ cặp | +0.0742 2/3 | +0.0615 2/3 | +0.0567 2/3 | +0.0812 2/3 |
| t5p, thật − đổi chỗ cặp | **+0.0031 1/3** | +0.0098 2/3 | +0.0229 3/3 | +0.0241 3/3 |

### Ba kết luận, phân theo mức chắc chắn

**1. CHẮC — qua cổng 2, lặp trên cả hai backbone: nhãn nguồn MANG tri thức chuyển giao được.**
`thật − xáo toàn bộ` dương ở **cả bốn chỉ số**, 5/6 hoặc 6/6, biên độ 0.05–0.08 (gấp 5–8 lần sàn
nhiễu 0.010), và trên **t5p là 3/3 ở cả bốn chỉ số**. Nhánh xáo được nhìn **đúng từng ký tự**
cùng bộ code, cùng 12 epoch, cùng seed, cùng split — chỉ nhãn khác.
> **Cách hiểu "lợi ích Pha 1 chỉ là phơi nhiễm miền" BỊ BÁC.**

**2. CHẮC: tiền-huấn-luyện bằng nhãn sai CHỦ ĐỘNG GÂY HẠI.** `xáo toàn bộ − baseline` âm ở cả
bốn chỉ số, **0/6 ở ba chỉ số** (p=0.031 — sàn của kiểm định dấu ở n=6). Đúng dự đoán của
arXiv:2309.17002 cho chuyển giao **out-of-domain**, mà xuyên ngôn ngữ chính là out-of-domain.

**3. KHÔNG KẾT LUẬN — theo đúng ngưỡng đã ghi trước.** Biến quyết định tôi đăng ký là
`D = thật − đổi-chỗ-trong-cặp` trên F1@0.5, đòi **≥ +0.020 VÀ ≥ 5/6 fold**. Thực tế: **+0.0387
nhưng chỉ 3/6**. Biên độ đạt, đếm dấu **không** đạt ⇒ rơi vào vùng "không kết luận" mà tôi đã
vạch sẵn. Và hai backbone **nói ngược nhau** (codebert +0.0742 2/3, t5p +0.0031 1/3), đúng
trường hợp cổng 2 sinh ra để chặn.

### Phát hiện khó chịu nhất, và cách đọc nó

`thật − baseline` = **+0.0018 (3/6)** trên F1, **âm nhẹ** ở ROC và PR. Trong cấu hình này,
Pha 1 với nhãn thật **không mang lại gì** so với không có Pha 1.

Ghép với (1) và (2) thì cách đọc là:

> **Nhãn đúng không làm Pha 1 tốt lên — nó ngăn Pha 1 làm hỏng.** Vai trò của nhãn ở đây là
> **bảo vệ**, không phải **cộng thêm**. Nhãn sai hoàn toàn: −0.06. Nhãn thật (mà CleanVul tự đo
> là sai 40–75%): về mức hoà. Khoảng **0.06 đó chính là biên độ mà một cách huấn luyện chịu
> được nhãn nhiễu có thể giành lại.**

**Hạn chế do chính thiết kế tạo ra, phải nêu:** để khớp số bước gradient, cả ba nhánh bị ép
chạy đúng 12 epoch và lấy checkpoint **CUỐI**, trong khi mọi kết quả đã công bố lấy checkpoint
**tốt nhất theo val**. Phép so **giữa ba nhánh** vẫn sạch (cùng một quy tắc); nhưng dòng
`thật − baseline` **KHÔNG so được** với +0.0092 của §40.5.

**Quan sát phụ:** độ trải giữa các fold tăng đơn điệu theo mức phá nhãn — baseline 0.042,
thật 0.105, xáo toàn bộ 0.157, đổi chỗ cặp 0.209 (codebert). Phá nhãn không chỉ hạ trung bình,
nó **thổi phồng phương sai**.


### §52.2 — Bốn khối độc lập: `fusft` − chốt trên codebert (14/09, tổng hợp, 0 GPU)

Mỗi khối n=5, codebert, nguồn `com`, `adamw`, seed 42, **đối chứng cùng cây cùng máy cùng phiên**.
**Không gộp qua máy** (mục 4 cấm) — đây là đếm xem bao nhiêu khối độc lập đồng ý.

| khối | máy | F1@0.5 | F1@val | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| `fus1` | vast A, 5060 Ti | +0.0259 5/5 | +0.0222 4/5 | +0.0224 5/5 | +0.0192 3/5 |
| `fus2` | vast B, 5060 Ti | +0.0226 4/5 | +0.0214 4/5 | +0.0113 3/5 | +0.0056 4/5 |
| `fus5060` | vast C, 5060 Ti | +0.0397 5/5 | +0.0408 5/5 | +0.0294 5/5 | +0.0160 4/5 |
| `fus3` | local A4000 | **−0.0005 3/5** | −0.0001 2/5 | +0.0203 4/5 | +0.0345 5/5 |
| **số khối có TB dương** | | **3/4** | 3/4 | **4/4** | **4/4** |

**Đọc đúng:**

- Ở **thứ hạng** (ROC/PR), `fusft` ≥ chốt ở **cả bốn khối**, biên độ **+0.011…+0.029**. Phát biểu
  *"fusion không hơn chốt"* là **SAI**.
- Ở **ngưỡng 0.5**, ba khối dương, khối thứ tư **đúng bằng 0**. Con số cụ thể của §52 không lặp.
- Biên độ nằm **quanh sàn nhiễu cùng loại GPU (0.010)** và **dưới sàn giữa các loại GPU (0.028)**
  ở ba trên bốn khối.

**Nhưng điều này KHÔNG làm AdapterFusion thành đóng góp**, và lý do **không** nằm ở chuyện lặp lại:

| bằng chứng | số |
|---|---|
| §48 — adapter **ngẫu nhiên** khớp std tái tạo ~một nửa lợi ích | +0.0111 / +0.0226, cùng 4/5 fold; phần dư +0.0115 **3/5**, **âm ở PR-AUC** |
| §50 — trên nhóm `train` (đúng chỗ đo phân biệt lỗ hổng) | adapter **đã học THUA** ngẫu nhiên: **−0.0595, 1/5** |
| §47 — backbone thứ hai | t5p n=5: **+0.0036, 3/5** — không lặp, trượt cổng 2 |
| §48.1 — trọng số fusion | gán **cao hơn** cho adapter ngẫu nhiên (0.7065 vs 0.5566), 4/5 fold |

> **Phát biểu chính xác:** `fusft` cho điểm **thứ hạng** nhỉnh hơn chốt một chút và khá đều qua
> bốn khối, **nhưng phần nhỉnh đó không tách được khỏi hiệu ứng thêm tham số**, và nó **âm đúng
> ở nhóm đo khả năng phân biệt lỗ hổng**.

**Bẫy công cụ gặp khi dựng bảng này:** `results_fus2_ntat/` chứa **ba** cây (`fus2_codebert`,
`fus2dbg_codebert`, `fus2w_codebert`). Glob `*codebert*/*/seed_*/fold*.json` gộp cả ba và cho
`fus2` = +0.0298 5/5 thay vì +0.0226 4/5 — đúng loại **va chạm khoá** mục 13 CLAUDE.md cảnh báo.
Phải trỏ **đường dẫn cây tường minh**, không dùng glob lỏng.

### §54.1 — LEO LÊN n=5 + nhánh `realbest`: Pha 1 **CÓ** giá trị, nhưng chỉ ở NGƯỠNG (14/09, 50 ô)

Khối `shuf2` thêm 26 ô: fold 4–5 cho ba nhánh cũ (Pha 1 **dùng lại nguyên vẹn**), cộng nhánh
`realbest` — Pha 1 y hệt nhánh `real` nhưng **chọn-theo-val + dừng sớm bình thường** (15 epoch,
patience mặc định) thay vì ép 12 epoch lấy checkpoint cuối. Cây `shuf1` giờ **50/50 ô**,
n=5 × 2 backbone. Đối chiếu từng byte: khớp tuyệt đối.

### Điểm tuyệt đối (TB 5 fold)

| nhánh | codebert F1@0.5 | ROC | PR | t5p F1@0.5 | ROC | PR |
|---|---|---|---|---|---|---|
| **`realbest`** | **0.7777** | 0.8563 | 0.8536 | **0.8314** | **0.9159** | **0.9213** |
| `real` (ép 12 ep) | 0.7660 | 0.8482 | 0.8539 | 0.8043 | 0.9053 | 0.9094 |
| baseline | 0.7598 | **0.8710** | **0.8766** | 0.8013 | 0.8939 | 0.9022 |
| `shufpair` | 0.6884 | 0.7841 | 0.7830 | 0.7812 | 0.8684 | 0.8666 |
| `shufall` | 0.6797 | 0.7749 | 0.7747 | 0.7541 | 0.8270 | 0.8148 |

### Δ ghép cặp theo `(backbone, fold)` — **10 điểm**, sàn kiểm định dấu p=0.002

| | F1@0.5 | F1@val | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| **`realbest` − baseline** | **+0.0240 8/10 p=0.109** | **+0.0293 9/10 p=0.021** | +0.0037 5/10 | −0.0020 5/10 |
| `real` − baseline | +0.0046 5/10 | +0.0057 6/10 | −0.0057 6/10 | −0.0077 5/10 |
| **`real` − `shufall`** | **+0.0682 9/10 p=0.021** | **+0.0746 10/10 p=0.002** | **+0.0758 9/10 p=0.021** | **+0.0869 9/10 p=0.021** |
| `real` − `shufpair` | +0.0503 7/10 | +0.0432 8/10 | **+0.0505 9/10 p=0.021** | +0.0568 8/10 |
| `shufpair` − `shufall` | +0.0179 7/10 | +0.0314 7/10 | +0.0253 8/10 | +0.0301 8/10 |
| **`shufall` − baseline** | −0.0636 2/10 | **−0.0688 0/10 p=0.002** | **−0.0814 0/10 p=0.002** | **−0.0946 0/10 p=0.002** |
| `shufpair` − baseline | −0.0457 2/10 | −0.0375 2/10 | −0.0562 2/10 | −0.0646 1/10 p=0.021 |

### §54 ĐÃ SAI ở một điểm, sửa tại đây

§54 viết *"Pha 1 với nhãn thật KHÔNG mang lại gì"* dựa trên `real − baseline = +0.0018`. Sai —
đó là hậu quả của **chính quy tắc ép 12 epoch** mà tôi đặt ra để khớp số bước gradient. Với cách
chọn checkpoint **đã công bố**, con số là **+0.0240 (8/10)** trên F1@0.5 và **+0.0293 (9/10,
p=0.021)** trên F1@ngưỡng-val. Trên sàn nhiễu 0.010 gấp 2–3 lần, và **dương ở cả hai backbone**.

> **Nhưng chỉ ở NGƯỠNG.** Trên thứ hạng, gộp lại là ~0 (+0.0037 và −0.0020, đều 5/10), và
> **hai backbone nói ngược nhau**: codebert ROC −0.0147 (1/5), PR −0.0230 (1/5); t5p ROC
> +0.0220 (4/5), PR +0.0190 (4/5). Cổng 2 **không qua** cho phát biểu về thứ hạng.

Đây là mẫu hình **đối xứng gương** với ASAM (mục 2b): ASAM cải thiện **thứ hạng** không cải
thiện **ngưỡng**; Pha 1 `none` cải thiện **ngưỡng** không cải thiện **thứ hạng**. Hai cơ chế
trong cùng một phương pháp đẩy vào hai chỗ khác nhau, và điểm tổng che mất điều đó.

### Ba kết luận cuối của khối, phân theo mức chắc chắn

**1. CHẮC — nhãn nguồn mang tri thức chuyển giao được.** `real − shufall` dương **cả bốn chỉ số**,
9/10 và 10/10, p=0.021–0.002, lặp trên **cả hai backbone**. Cách hiểu "chỉ là phơi nhiễm miền"
**bị bác** dứt điểm.

**2. CHẮC — nhãn sai gây hại chủ động.** `shufall − baseline` âm cả bốn, **0/10 ở ba chỉ số**,
p=0.002. Đúng dự đoán arXiv:2309.17002 cho out-of-domain.

**3. CHẮC (ở ngưỡng) — Pha 1 có giá trị ròng ~+0.025…+0.029 F1**, nhưng **0 ở thứ hạng** và
hai backbone ngược nhau ở đó.

> **Con số then chốt cho hướng đi:** khoảng cách giữa *nhãn sai* và *nhãn thật* là **0.068–0.087**;
> khoảng cách giữa *nhãn thật* và *không có Pha 1* chỉ **0.024–0.029**. Nghĩa là **phần lớn giá
> trị của nhãn đang bị tiêu vào việc gỡ lại thiệt hại** mà chính việc tiền-huấn-luyện trên bộ
> dữ liệu này gây ra. Một cách huấn luyện chịu được nhãn nhiễu có **~0.06 biên độ** để giành lại,
> và đó là mục tiêu bằng số đầu tiên dự án có cho hướng đóng góp.

**Biến quyết định đăng ký trước vẫn KHÔNG KẾT LUẬN**: `real − shufpair` trên F1@0.5 đòi
≥+0.020 **và** đếm dấu ~5/6; thực tế **+0.0503 nhưng 7/10** (70% < 83%). Biên độ đạt, đếm dấu
không. ROC-AUC thì đạt (9/10, p=0.021). Áp đúng bảng, không nới sau khi thấy số.

---

## §55 — BỎ head `latent_bottleneck` khỏi Pha 1 xoá sạch lợi ích của fusion (14/09, 15 ô, vast `ntat` 5060 Ti)

Người dùng yêu cầu 14/09: thử bỏ head `latent_bottleneck` nhưng **giữ nguyên adapter + fusion**.
Hai nhánh khác **đúng một biến** (`aux_mode`), cây riêng có baseline của chính nó, cùng máy cùng
phiên. Giao thức Pha 1 đọc **thẳng từ `training_args`** của checkpoint nhánh có head — 15 epoch,
**patience 10**, lr 2e-5, `adapter_dim 48`, `adapter_lr 1e-4`, SAM tắt — không đoán.

### Pha 1: nhánh KHÔNG head **sập hoàn toàn**

| Pha 1 (đều có adapter) | val | epoch chốt | epoch đã chạy |
|---|---|---|---|
| **có** head `latent_bottleneck` | **0.5758** | 12 | 15 |
| **không** head (`none`) | **0.4430** | **3** | 13 (dừng sớm) |

Không phải "học kém" mà là **không học được gì**: train loss đứng ở 0.696–0.701 suốt 13 epoch,
trong khi `ln(2) = 0.6931` là mức đoán bừa. Val đi 0.3333 (đoán một lớp) → 0.4037 → 0.4430 rồi
đứng im 10 epoch liền.

> **`none` trên `com` KHÔNG có adapter thì bình thường** — checkpoint cũ `s42/phase1/codebert__none_com`
> đạt val **0.5518**. Nên nghi vấn là chính **adapter** làm Pha 1 mất ổn định khi không có head giữ.

### Pha 2 — n=5, codebert, seed 42

| nhánh | F1@0.5 | F1@val | ROC-AUC | PR-AUC | val Pha 1 |
|---|---|---|---|---|---|
| `fusft` **có head** | **0.8195** | **0.8154** | **0.9051** | **0.9105** | 0.5758 |
| baseline | 0.7693 | 0.7673 | 0.8701 | 0.8703 | — |
| `nonefus` **không head** | 0.7660 | 0.7698 | 0.8707 | 0.8651 | 0.4430 |

| Δ ghép cặp | F1@0.5 | F1@val | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| `fusft` − baseline | **+0.0502 5/5** | **+0.0481 5/5** | **+0.0350 5/5** | +0.0401 4/5 |
| **`nonefus` − baseline** | **−0.0033 2/5** | **+0.0025 3/5** | **+0.0005 3/5** | **−0.0052 3/5** |
| `fusft` − `nonefus` | **+0.0535 5/5** | **+0.0456 5/5** | +0.0344 4/5 | **+0.0454 5/5** |

`nonefus` rơi **đúng** về baseline: cả bốn chỉ số trong khoảng ±0.005, không chỉ số nào quá 3/5 fold.

> **ĐỌC §55.1 TRƯỚC KHI DÙNG MỤC NÀY.** Pha 1 nhánh không-head **không phải luôn sập**:
> ở seed 7 nó đạt 0.5787 bình thường. Con số `fusft − nonefus = +0.0535` dưới đây so với
> **một lần rút hỏng**, nên **không đo** giá trị của head phụ.

### Đọc cho đúng — đây KHÔNG phải "head đáng +0.053"

Phát biểu đúng là: *trong lần chạy này, bỏ head làm **Pha 1 sập**, và một Pha 1 đã sập thì không
mang gì sang Pha 2.* Đây là **lần thứ ba** hiện tượng "`none` sập, `latent_bottleneck` không sập"
được quan sát độc lập (hai lần trước ở `codebert × full`, hai λ khác nhau — §46), và là lần đầu
trên nguồn `com`, trong cấu hình adapter.

### Một tinh chỉnh đáng giá cho câu chuyện SỨC CHỨA (§48)

| Pha 1 sập | có adapter+fusion? | Δ vs baseline |
|---|---|---|
| `shufall` val 0.4374 (§54.1) | **không** | **−0.0636** |
| `nonefus` val 0.4430 (mục này) | **có** | **−0.0033** |

Hai Pha 1 sập tương đương nhau; cái có adapter+fusion về **đúng** baseline, cái không có thì **hại**
−0.064. ⇒ **Sức chứa thêm vào có tác dụng ĐỆM THIỆT HẠI, chứ không TẠO RA lợi ích.** Đây là cách
đọc chính xác hơn "một nửa lợi ích là sức chứa" của §48.

### Cảnh báo phương pháp: fold 1 lại lừa, lần thứ năm

Ở fold 1, `nonefus` cho **0.789** so với baseline **0.744** (+0.045) và tôi đã ghi nhận nó như
một quan sát đáng chú ý — **kèm chữ n=1, không kết luận**. Ở n=5 trung bình là **−0.0033**:
fold 2 cho 0.749 vs 0.783 và fold 3 cho 0.736 vs 0.756, đều âm.

### Câu còn mở, và phép kiểm rẻ nhất cho nó

Bỏ head thì Pha 1 **luôn** sập, hay lần này chỉ xui? §48.2 đã đo được hai lần rút cùng cấu hình
lệch rất xa, nên **một lần sập không đủ để gọi là tính chất**. Khối `p1seed` (đang chạy) chạy lại
**đúng Pha 1 đó ở seed 7 và 1234**, cộng đối chứng nhánh có head ở cùng seed — **không tốn một ô
Pha 2 nào**.

- Sập lại ở cả hai seed ⇒ *"adapter không có head phụ thì Pha 1 không ổn định"* là thật, và nó
  **cứu head phụ ở đúng một vai trò cụ thể: ổn định, không phải độ chính xác.**
- Không sập ⇒ lần trước chỉ là xui, và câu hỏi gốc *"fusion có cần head không"* **vẫn chưa được
  trả lời** — phải chạy lại khối `fusnone` với Pha 1 không sập.

### §55.1 — Khối `p1seed`: head phụ **ổn định hoá** Pha 1 (3/3 vs 1/3), và §55 phải đọc lại (14/09, 4 Pha 1, **0 ô Pha 2**)

§55 kết luận từ **một** lần rút rằng bỏ head làm Pha 1 sập. §48.2 đã cảnh báo một lần rút không
đủ. Khối này chạy lại **đúng Pha 1 đó** ở seed 7 và 1234, kèm đối chứng nhánh có head ở cùng seed.

Tiêu chí "có học không" là **train loss có rời khỏi `ln(2)=0.6931` không**, không phải val —
vì seed 1234 có val 0.5124 (trên mức ngẫu nhiên) nhưng train loss đứng im, tức cùng một kiểu hỏng.

| nhánh | seed | val | epoch chốt | train loss đầu → cuối | có học |
|---|---|---|---|---|---|
| **có head** | 7 | 0.5932 | 15 | 0.8137 → **0.4140** | ✅ |
| **có head** | 42 | 0.5758 | 12 | — | ✅ |
| **có head** | 1234 | 0.5372 | 14 | 0.8058 → **0.5683** | ✅ |
| **không head** | 7 | **0.5787** | 12 | 0.7080 → **0.4682** | ✅ |
| **không head** | 42 | 0.4430 | 3 | 0.7081 → **0.6963** | ❌ |
| **không head** | 1234 | 0.5124 | 2 | 0.7032 → **0.6963** | ❌ |

> **Có head 3/3 · không head 1/3.** *(§55.3: ở 6 seed là **6/6 vs 3/6**, Fisher p=0.18.)* Giá trị của head phụ là **ĐỘ ỔN ĐỊNH**, không phải **độ
> chính xác** — đúng điều §46 đã nêu là tính chất duy nhất còn sống sót, nay lần đầu được đo
> bằng thiết kế **lặp theo seed** thay vì giai thoại.

### §55 PHẢI ĐỌC LẠI — hai chỗ

**1. Suy đoán "adapter không có head thì Pha 1 LUÔN không ổn định" — RÚT.** Ở seed 7 nó học
bình thường và đạt **0.5787**, còn **cao hơn** nhánh có head ở seed 42 (0.5758). Không phải
"luôn", mà là **2 trên 3 lần**.

**2. Con số `fusft − nonefus = +0.0535 5/5` của §55 BỊ NHIỄU LOẠN.** Nó so nhánh có head với
**một lần rút hỏng**, không phải với "không có head". Nó **không đo** cái nó định đo, và
**không được** trích dẫn như giá trị của head phụ.

### Điều này đổi gì cho hướng đi

Khi **cả hai** nhánh cùng học được (seed 7), khoảng cách Pha 1 chỉ **0.0145** (0.5932 vs 0.5787)
— nằm trong dải dao động giữa các lần rút. Nên giả thuyết *"fusion cần head phụ để có lợi ích"*
**chưa có bằng chứng**; phải chạy lại khối `fusnone` ở **seed 7**, nơi cả hai Pha 1 đều lành.

**Giới hạn:** 3 seed mỗi nhánh. 3/3 so với 1/3 ở n=3 thì kiểm định Fisher cho p≈0.4 — **gợi ý
mạnh, chưa phải bằng chứng**. Nhưng cộng với hai lần `none` sập độc lập ở `codebert × full`
(§46, hai λ khác nhau), mẫu hình này đã xuất hiện ở **bốn bối cảnh độc lập**.

### §55.2 — Chạy lại ở seed 7 (cả hai Pha 1 LÀNH): giá trị của head co xuống **43%** so với §55 (15 ô)

Khối `fusnone7`: **đúng** thí nghiệm của §55 nhưng ở **seed 7**, nơi cả hai Pha 1 đều học được
(0.5932 và 0.5787, chênh 0.0145). Pha 1 dùng lại từ `p1seed`; cổng kiểm **nội dung** checkpoint
(`aux_mode` đúng, 48 khoá adapter, val > 0.53) chứ không chỉ kiểm file tồn tại. 15/15 ô, đối chiếu
byte khớp tuyệt đối.

| | seed 42 — Pha 1 không-head **HỎNG** | seed 7 — cả hai **LÀNH** |
|---|---|---|
| `fusft` − baseline, F1@0.5 | **+0.0502 5/5** | **+0.0553 5/5** |
| `nonefus` − baseline, F1@0.5 | **−0.0033 2/5** | **+0.0371 4/5** |
| **`fusft` − `nonefus`, F1@0.5** | **+0.0535 5/5** | **+0.0182 3/5** |
| `fusft` − `nonefus`, F1@val | +0.0456 5/5 | +0.0330 4/5 |
| `fusft` − `nonefus`, ROC-AUC | +0.0344 4/5 | **+0.0119 2/5** |
| `fusft` − `nonefus`, PR-AUC | +0.0454 5/5 | +0.0201 4/5 |

Ở seed 7, **cả bốn** chỉ số của `fusft − baseline` đều **5/5**.

> **§55.3 SỬA con số 57% dưới đây thành 49%** — nó tính từ tỉ lệ 1/3 ước trên 3 seed;
> với 6 seed tỉ lệ là **3/6**. Phần định tính không đổi, tỉ lệ thì đổi.

### Giá trị THẬT của head phụ, tách làm hai phần

Khi **cả hai** Pha 1 chạy được, head chỉ còn đáng **+0.0182 (3/5)** ở F1 và **+0.0119 (2/5)** ở
ROC — quanh sàn nhiễu, đếm dấu không nhất quán. Con số **+0.0535 5/5** của §55 phần lớn là do
nhánh không-head rút phải một Pha 1 hỏng.

Nhưng Pha 1 không-head hỏng **2/3 seed** (§55.1). Phép tính kỳ vọng theo seed — *minh hoạ từ
3 seed, KHÔNG phải phép đo*:

```
E[có head]    = +0.0528                                    (3/3 seed chạy được)
E[không head] = (1/3)(+0.0371) + (2/3)(−0.0033) = +0.0102  (1/3 seed chạy được)
chênh kỳ vọng = +0.0426
   trong đó "giỏi hơn khi cả hai chạy được" = +0.0182  (43%)
            "không hỏng"                     = +0.0244  (57%)
```

> **Hơn một nửa giá trị của head phụ là BẢO HIỂM, không phải ĐỘ CHÍNH XÁC.** Đây là phát biểu
> hẹp hơn hẳn "head là đòn bẩy độ chính xác" (đã đóng ở §46) nhưng nó **bảo vệ được** và có cơ
> chế đo trực tiếp: tỉ lệ Pha 1 học được.

**Giới hạn**: tỉ lệ hỏng ước từ **3 seed**. Khoảng tin cậy của 1/3 với n=3 rất rộng — con số 57%
là minh hoạ, không phải phép đo. Cần thêm seed để siết.

### Điều chắc nhất của cả nhánh `fusion`

`fusft` − baseline trên codebert, **bốn khối độc lập, ba máy, hai seed, hai bộ checkpoint Pha 1**:

| khối | seed | máy | ΔF1@0.5 |
|---|---|---|---|
| `fus5060` | 42 | vast C 5060 Ti | **+0.0637 5/5** |
| `fus3` | 42 | local A4000 | **+0.0584 5/5** |
| `fusnone` | 42 | vast ntat 5060 Ti | **+0.0502 5/5** |
| `fusnone7` | **7** | vast ntat 5060 Ti | **+0.0553 5/5** |

**5/5 fold ở cả bốn khối**, biên độ 0.050–0.064. Đây là con số ổn định nhất dự án có. Nhưng nó
là *"chốt + adapter + fusion hơn baseline"*, **không** phải *"fusion hơn chốt"* — phần gia tăng
so với chốt vẫn là chỗ §48/§50/§52.1 chặn lại.

> **ĐỌC §55.4 TRƯỚC.** Toàn bộ mục này chỉ đúng cho **codebert**. Trên t5p, cả hai nhánh
> học được **3/3** seed — không có bất ổn nào, nên phát biểu **trượt cổng 2**.

### §55.3 — SÁU seed trên codebert: có head **6/6**, không head **3/6**; và kết cục có tính **LƯỠNG CỰC** (đêm 14→15/09, 12 Pha 1, 0 ô Pha 2)

§55.1 ước tỉ lệ từ **3 seed** (3/3 vs 1/3) và tôi đã ghi rõ khoảng tin cậy quá rộng. Thêm seed
**1, 2026, 999** đưa lên **6 seed mỗi nhánh**. Tiêu chí "có học" = **train loss rời khỏi
`ln(2)=0.6931` ít nhất 0.05**, không dùng val.

| seed | CÓ head | học | KHÔNG head | học |
|---|---|---|---|---|
| 42 | 0.5758 | ✅ | **0.4430** | ❌ |
| 7 | 0.5932 | ✅ | 0.5787 | ✅ |
| 1234 | 0.5372 | ✅ | **0.5124** | ❌ |
| 1 | 0.5614 | ✅ | **0.4668** | ❌ |
| 2026 | 0.6057 | ✅ | 0.6002 | ✅ |
| 999 | 0.6009 | ✅ | 0.5814 | ✅ |
| | **6/6** | | **3/6** | |

### Phát hiện quan trọng hơn cả tỉ lệ: kết cục LƯỠNG CỰC

| | giá trị | trung bình |
|---|---|---|
| có head, mọi seed | 0.5372 … 0.6057 | **0.5790** (độ tản 0.0240) |
| không head, **khi học được** | 0.5787 / 0.5814 / 0.6002 | **0.5868** |
| không head, **khi hỏng** | 0.4430 / 0.4668 / 0.5124 | 0.4741 |

> **Bỏ head KHÔNG làm mô hình kém hơn — nó làm mô hình LƯỠNG CỰC.** Khi học được, nhánh không
> head cho **0.5868**, tức **ngang hoặc hơn** nhánh có head (0.5790). Khi hỏng, nó không học gì
> cả. Head phụ **không nâng đỉnh, nó xoá đuôi dưới.**

Đây là phát biểu chính xác hơn hẳn "head làm mô hình tốt hơn", và nó **kiểm chứng được trực
tiếp** bằng tỉ lệ Pha 1 học được — không cần đi vòng qua điểm số Pha 2.

### Phép tính kỳ vọng, SỬA LẠI từ §55.2

§55.2 dùng tỉ lệ 1/3 (từ 3 seed) và kết luận **57%** giá trị của head là "không hỏng".
Với 3/6:

```
E[có head]    = +0.0528
E[không head] = (3/6)(+0.0371) + (3/6)(−0.0033) = +0.0169
chênh         = +0.0359
   "giỏi hơn khi CẢ HAI chạy được" = +0.0182  →  51%
   "không hỏng"                    = +0.0177  →  49%
```

**Con số 57% của §55.2 sửa thành 49%.** Giá trị của head chia **gần đúng một nửa** giữa *giỏi
hơn khi cả hai chạy được* và *không hỏng*, chứ không nghiêng về bảo hiểm như §55.2 nói.

### Giới hạn — phải nêu

**Fisher hai phía cho 6/6 vs 3/6: p = 0.18.** Ở n=6 mỗi nhánh, khác biệt này **chưa có ý nghĩa
thống kê**. Nó là **mẫu hình nhất quán qua sáu lần rút độc lập** (và cộng thêm hai lần `none`
sập ở `codebert × full`, §46, là **tám** bối cảnh), nhưng con số p nói thẳng: cần nhiều seed hơn
để phát biểu chắc. **Không được viết vào bài như một kết quả có ý nghĩa.**

Khối `p1seed_t5p` (đang chạy) hỏi cùng câu trên **backbone thứ hai** — cổng 2.

### §55.4 — CỔNG 2 TRƯỢT: bất ổn của Pha 1 là **đặc thù codebert**, t5p không có (đêm 14→15/09, 6 Pha 1, 0 ô Pha 2)

`p1seed_t5p` hỏi **đúng** câu của §55.1/§55.3 trên backbone thứ hai (`codet5p-220m-bimodal`,
pooling `mean`), cùng ba seed, cùng giao thức.

| seed | **codebert** không head | có head | **t5p** không head | có head |
|---|---|---|---|---|
| 42 | **0.4430** ❌ | 0.5758 ✅ | 0.5939 ✅ | 0.6051 ✅ |
| 7 | 0.5787 ✅ | 0.5932 ✅ | **0.6137** ✅ | 0.6003 ✅ |
| 1234 | **0.5124** ❌ | 0.5372 ✅ | 0.5915 ✅ | 0.5878 ✅ |
| **tỉ lệ học được** | **1/3** | **3/3** | **3/3** | **3/3** |

Trên t5p **không có bất ổn nào**: cả hai nhánh học được ở cả ba seed, và nhánh **không head**
còn nhỉnh hơn (TB **0.5997** so với 0.5977).

> **Phát biểu "head phụ ổn định hoá Pha 1" TRƯỢT CỔNG 2.** Nó là hiện tượng **đặc thù codebert**,
> không lặp trên backbone thứ hai. Theo luật của `NEXT_CONTRIBUTION.md` §3, nó **không được**
> phát biểu như một tính chất của phương pháp.

Đây là lần thứ **bảy** trong dự án một cơ chế tách theo backbone — §38.2 đã ghi *"mọi cơ chế đã
thử đều tách theo backbone"*, và mục này là bằng chứng mới nhất, lần này với một thiết kế
**lặp theo seed** nên không thể đổ cho một lần rút xui.

**Cái còn lại sau khi trượt cổng 2:** head phụ có thể là **bảo hiểm cho riêng codebert**, nơi
Pha 1 `none` + adapter hỏng một nửa số lần (§55.3: 3/6). Đó là một lưu ý kỹ thuật đáng ghi cho
người dùng lại repo này, **không phải** một đóng góp học thuật.

---

## §56 — HEAD PHỤ KHÔNG ĐÓNG GÓP GÌ cho fusion khi Pha 1 lành: **+0.0053, 5/10** (15/09, 30 ô, hai backbone)

Trả lời trọn vẹn yêu cầu của người dùng 14/09 (*"thử bỏ latent bottleneck head"*). Hai khối,
mỗi khối n=5 fold, cây riêng có baseline của chính nó, **cùng máy cùng phiên**, Pha 1 dùng lại
từ `p1seed`/`p1seed_t5p` ở seed mà **cả hai nhánh đều học được**:

| | codebert seed 7 | t5p seed 42 |
|---|---|---|
| Pha 1 có head | 0.5932 | 0.6051 |
| Pha 1 không head | 0.5787 | 0.5939 |

t5p phải bật `--grad_checkpointing` cho **cả hai** nhánh (OOM thiếu đúng 24 MiB ở lần đầu, y hệt
§47/§48). `use_reentrant=False` nên gradient không đổi.

### Điểm tuyệt đối (TB 5 fold)

| nhánh | codebert F1 | ROC | PR | t5p F1 | ROC | PR |
|---|---|---|---|---|---|---|
| `fusft` **có head** | **0.8303** | **0.9112** | **0.9158** | 0.8405 | 0.9168 | 0.9210 |
| `nonefus` **không head** | 0.8120 | 0.8993 | 0.8957 | **0.8481** | **0.9321** | **0.9393** |
| baseline | 0.7749 | 0.8780 | 0.8830 | 0.7838 | 0.8882 | 0.8975 |

### Δ ghép cặp theo `(backbone, fold)` — **10 điểm**

| | F1@0.5 | F1@val | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| **`fusft` − `nonefus`** | **+0.0053 5/10** | +0.0134 6/10 | **−0.0017 3/10** | **+0.0009 5/10** |
| `fusft` − baseline | **+0.0560 9/10 p=0.021** | +0.0593 8/10 | **+0.0309 9/10 p=0.021** | **+0.0282 9/10 p=0.021** |
| `nonefus` − baseline | **+0.0507 9/10 p=0.021** | +0.0459 8/10 | +0.0326 8/10 | +0.0273 8/10 |

Tách theo backbone, `fusft − nonefus` **đổi dấu**:

| | F1@0.5 | ROC-AUC |
|---|---|---|
| codebert | +0.0182 3/5 | +0.0119 2/5 |
| t5p | **−0.0077 2/5** | **−0.0153 1/5** |

### Kết luận

> **Head phụ `latent_bottleneck` KHÔNG đóng góp gì** cho cấu hình adapter+fusion khi Pha 1 không
> sập: **+0.0053 trên F1@0.5 với 5/10 fold** — đúng bằng tung đồng xu — và **âm** trên ROC-AUC
> (−0.0017, 3/10). Hai backbone **đổi dấu** nhau, tức trượt cổng 2 theo cả hai hướng.
>
> Thứ tạo ra lợi ích là **Pha 1 + adapter + fusion**: cả hai nhánh hơn baseline **~+0.05 F1**
> và **~+0.03 ROC**, 9/10 fold, p=0.021, trên **cả hai** backbone.

**Hệ quả thực tiễn: có thể BỎ HẲN head phụ.** Phương pháp gọn đi một thành phần mà không mất gì
đo được — trừ rủi ro Pha 1 sập, và rủi ro đó **chỉ có ở codebert** (§55.4: t5p 3/3 seed đều lành).

Cộng với §46 (head hơn `none` chỉ +0.0005 trên 132 ô), §44 (nút thắt thua PCA-8), §45 (head giỏi
gấp ba không đổi gì): **mạch "head phụ là đóng góp" đóng hoàn toàn.** Điều còn lại của nó là một
lưu ý kỹ thuật cho riêng codebert, không phải đóng góp học thuật.

---

## §57 — BIÊN TRONG CẶP (Đề xuất B) ở β=0.5: **thua BCE**, và Pha 1 tốt hơn lại chuyển giao KÉM hơn (15/09, 18 ô)

Dự đoán ghi **trước khi đo**: `records/prediction_2026-09-15_bien_trong_cap.md`.
Bậc 1, n=3 fold, seed 42, **cả hai backbone**, nguồn `com`, Pha 2 **thuần** (không adapter,
không fusion, không head phụ — §56 cho thấy head không đóng góp gì).

`L = CE(nhãn) + β · mean_cặp softplus(m − (s_vul − s_fixed))`, `s = logit[1] − logit[0]`,
β=0.5, m=1.0. Sampler theo cặp cho **1576/1576 cặp dùng được mỗi epoch** (93.4% số dòng).

### Pha 1 — hai backbone rẽ hai hướng

| | val | epoch | loss biên-cặp |
|---|---|---|---|
| codebert `bce` | 0.5649 | 9 | — |
| codebert **`pairB`** | **0.4326** | 2 | 1.2285 → 1.2124 (**đứng im**) |
| t5p `bce` | 0.5892 | 15 | — |
| t5p **`pairB`** | **0.6002** | 6 | 1.2107 → **0.2386** (giảm mạnh) |

Trên t5p hàm mục tiêu **tối ưu được rất tốt** và cho val Pha 1 **cao nhất** dự án từng đo trên
`t5p × none × com`. Trên codebert nó sập — cùng kiểu bất ổn §55.3 đã đo (`none` sập 3/6 seed
trên codebert, t5p 3/3 lành), không phải khuyết tật của biên-cặp.

### Pha 2 — Δ ghép cặp theo `(backbone, fold)`, 6 điểm

| | F1@0.5 | F1@val | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| **`pairB` − `bce`** | **−0.0175 2/6** | **−0.0232 1/6** | **−0.0121 1/6** | **−0.0237 2/6** |
| `bce` − baseline | +0.0195 4/6 | +0.0139 4/6 | +0.0068 3/6 | +0.0139 5/6 |
| `pairB` − baseline | +0.0020 4/6 | −0.0093 4/6 | −0.0053 3/6 | −0.0097 3/6 |

Tách theo backbone, **t5p** (nơi Pha 1 LÀNH và còn cao hơn):

| t5p | F1@0.5 | F1@val | ROC-AUC | PR-AUC |
|---|---|---|---|---|
| `pairB` − `bce` | **−0.0223 1/3** | −0.0289 1/3 | +0.0011 1/3 | −0.0092 2/3 |

### Phát hiện đáng giữ: Pha 1 TỐT HƠN lại chuyển giao KÉM HƠN

t5p `pairB` có val Pha 1 **0.6002** (cao hơn `bce` 0.5892) và loss biên-cặp giảm 5× — tức mô
hình **rất giỏi** tác vụ *tương đối*: phân biệt một hàm với **chính bản vá của nó**. Nhưng Pha 2
lại kém hơn **−0.0223** F1.

> Kỹ năng "tách một hàm khỏi bản vá của chính nó" **không mang sang** bài toán phân loại
> **tuyệt đối** ở ngôn ngữ mới. Cách đọc hợp lý: tác vụ tương đối giải được bằng cách bắt
> **dấu vết chỉnh sửa** (độ dài, vị trí token đổi) chứ không cần hiểu ngữ nghĩa lỗ hổng.

Khớp với §54: chất lượng Pha 1 và khả năng chuyển giao **không gắn chặt**.

### Phán quyết theo đúng ngưỡng đã ghi trước

Biến đăng ký `D = Δ(pairB − bce)` trên F1@0.5: **−0.0175, 2/6**. Bảng ngưỡng nói
`≤ +0.005 ⇒ không phân biệt được với nhiễu ⇒ đóng Đề xuất B ở β này`; và hai chỉ số
(F1@val, ROC-AUC) đạt **5/6 ngược dấu**, tức chạm cả dòng "âm rõ ⇒ ghi lại và đóng".

**Giới hạn — phải nêu:**
- **Một giá trị β duy nhất.** Kết quả này bác β=0.5, **không** bác ý tưởng.
- **Nửa codebert bị nhiễu loạn** vì Pha 1 sập. Phần đọc được nằm ở t5p.
- **Nhiễu loạn sampler chưa tách.** Batch theo cặp gồm toàn hàm gần trùng nhau ⇒ gradient trong
  batch tương quan cao; tự nó đã có thể đổi kết quả mà không liên quan gì tới hàm mục tiêu.
  Đã ghi sẵn trong bản dự đoán; cờ `--pair_sampler_only` đã viết và thử khói ba chiều xong,
  **chưa chạy** (máy đã dừng).

---

## §58 — CỔNG 3 cho §56: lợi ích transfer tập trung ~4× ở nhóm GẦN-TRÙNG-NGƯỢC-NHÃN, và head phụ chỉ hại ở NGƯỠNG (15/09, **0 GPU**, đọc lại 30 ô của §56)

Khai báo trước: `records/prediction_2026-09-15_gate3_fusion_nhom_ro_ri.md`, viết **trước** khi
chạy, kèm ngưỡng và vùng "không kết luận". Công cụ: `tools/leak_groups_pair.py` (macro-F1, đã
có) + `tools/leak_groups_auc.py` (**mới**, in ROC-AUC và PR-AUC theo nhóm — vì công cụ cũ chỉ
in một chỉ số, trái luật §2b). Cổng hai chiều của công cụ mới: `A − A` cho **đúng 0.0000** ở
mọi nhóm, `A − baseline` khác 0.

Nhóm mỗi fold: `train` 23.4 hàng · `test` 7.6 · `none` 111.2 (trên 152).

### (1) Giá trị của head phụ, tách theo nhóm — 10 ô ghép cặp, 2 backbone

| nhóm | Δ macro-F1@0.5 | Δ ROC-AUC | Δ PR-AUC |
|---|---|---|---|
| **`train`** | **−0.0243 2/10** | **−0.0020 4/10** | +0.0149 5/10 |
| `none` | +0.0027 5/10 | −0.0077 5/10 | −0.0052 6/10 |
| TẤT CẢ | +0.0053 5/10 | −0.0016 3/10 | +0.0010 5/10 |

Trên `train`, F1@0.5 âm ở **cả hai** backbone (codebert −0.0319 **1/5**, t5p −0.0166 **1/5**).

**Phán quyết theo đúng ngưỡng đã ghi trước:** ô thứ ba của bảng Q1 (`Δ ≤ −0.020` **và** `≤3/10`)
**kích hoạt** ⇒ *"head làm hại phân biệt lỗ hổng"*. **NHƯNG** ràng buộc số 2 của chính khai báo
đó đã lường trước: bảng Q1 viết trên **một** chỉ số. Chỉ số **thứ hạng** cùng nhóm cho
**−0.0020, 4/10** — tức **null**, không âm.

> **Đọc đúng: đây là hiệu ứng NGƯỠNG, không phải hiệu ứng PHÂN BIỆT.** Head không làm mô hình
> xếp hạng kém đi ở nhóm khó; nó làm **điểm cắt 0.5** rơi sai chỗ trên đúng những hàng đó.
> Cùng mẫu hình ngưỡng-vs-thứ-hạng đã lặp bảy lần (§2b, §52.1, §54.1).
>
> Với §56 thì kết luận **mạnh thêm**: null +0.0053 **không** phải hai hiệu ứng thật triệt tiêu
> nhau — ở thứ hạng head null ở **mọi** nhóm. Bỏ head vẫn an toàn, và ở ngưỡng 0.5 bỏ head còn
> **tốt hơn** trên nhóm khó.

### (2) Lợi ích so với baseline, tách theo nhóm — ĐÂY MỚI LÀ PHÁT HIỆN

ROC-AUC, Δ ghép cặp theo fold, **từng backbone riêng**:

| nhánh | backbone | `train` (23 hàng) | `none` (111 hàng) | tỉ lệ |
|---|---|---|---|---|
| `fusft` **có head** | codebert | **+0.1166 5/5** | +0.0243 5/5 | 4.8× |
| `fusft` **có head** | t5p | **+0.1192 5/5** | +0.0109 3/5 | 10.9× |
| `nonefus` **không head** | codebert | **+0.1096 4/5** | +0.0187 5/5 | 5.9× |
| `nonefus` **không head** | t5p | **+0.1302 5/5** | +0.0319 5/5 | 4.1× |
| **gộp** `fusft` | 10 ô | **+0.1179 10/10 p=0.002** | +0.0176 8/10 | 6.7× |
| **gộp** `nonefus` | 10 ô | **+0.1199 9/10 p=0.021** | +0.0253 10/10 p=0.002 | 4.7× |

Điểm tuyệt đối (TB 5 fold) cho thấy vì sao:

| | codebert `train` ROC | `none` ROC | t5p `train` ROC | `none` ROC |
|---|---|---|---|---|
| baseline | **0.7109** | 0.9102 | **0.7282** | 0.9170 |
| `fusft` | 0.8275 | 0.9345 | 0.8474 | 0.9279 |
| `nonefus` | 0.8205 | 0.9289 | **0.8583** | **0.9488** |

**Baseline yếu hẳn ở đúng nhóm khó** (0.71–0.73 so với 0.91–0.92 ở nhóm sạch), và Pha 1 lấp
khoảng đó.

### Đối chứng TRẦN — phải nêu, nó cắt phát hiện trên xuống một nửa

Nhóm `none` đã ở 0.91 nên chỉ còn 0.09 dư địa; nhóm `train` ở 0.71 nên còn 0.29. Chuẩn hoá
theo dư địa `Δ / (1 − baseline)`:

| điều kiện | `train` | `none` |
|---|---|---|
| codebert `fusft` | 40.3% | 27.1% |
| codebert `nonefus` | 37.9% | 20.8% |
| t5p `fusft` | **43.9%** | 13.1% |
| t5p `nonefus` | 47.9% | 38.4% |

> Sau chuẩn hoá, `train` vẫn hơn `none` ở **4/4** điều kiện, nhưng tỉ lệ co từ **~5×** xuống
> **1.2–3.4×**. Con số đáng trích dẫn là **4/4 cùng chiều**, không phải "5 lần".

### Phán quyết Q2 theo đúng ngưỡng đã ghi trước: **KHÔNG KẾT LUẬN**

Điều kiện chống-H1 viết trên **macro-F1** và đòi `Δ_train − Δ_none ≥ 0.030` ở **cả hai** nhánh
**và cả hai** backbone. Thực tế trên macro-F1, `fusft` × codebert cho `train` **+0.0288** so với
`none` **+0.0544** — tức `train` **thấp hơn**. Điều kiện **không** đạt ⇒ theo luật đã chốt,
**không kết luận**, không được nới sau khi thấy số.

Mẫu hình 4/4 ở trên nằm trên chỉ số **thứ hạng**, **không** có trong khai báo trước ⇒ nó là
**quan sát hậu nghiệm**, phải lặp ở một khối độc lập rồi mới được phát biểu.

### Vì sao nó vẫn đáng theo — và nó khớp với cái gì

1. **Qua cổng 2**: cùng chiều trên **cả hai** backbone, với **và** không có head — bốn điều kiện
   độc lập. Rất hiếm trong dự án này (§38.2: *"mọi cơ chế đã thử đều tách theo backbone"*).
2. **Khớp §50** — khối hoàn toàn khác (chốt, **không** adapter, n=15, 3 seed): `train` +0.0591
   so với `none` +0.0459, cùng chiều tuy biên độ nhỏ hơn.
3. **Khớp §53**: Pha 1 **tạo ra** trục lỗ hổng (AUC 0.5242 → 0.6477 trên Python, 0 bước huấn
   luyện). Nhóm `train` chính là nơi cần trục đó — phân biệt một hàm với **chính bản vá của nó**.
4. **Khớp §54**: nhãn nguồn mang tri thức chuyển giao được (thật − xáo +0.068…+0.087).
5. **Không mâu thuẫn với §48/§50(3)**: thứ mang khả năng phân biệt là **huấn luyện Pha 1 của
   backbone**, không phải **nội dung adapter nguồn** — đó là lý do adapter đã học có thể thua
   adapter ngẫu nhiên (−0.0595) trong khi Pha 1 vẫn +0.12 ở cùng nhóm.

### Điều này đổi gì cho giả thuyết H1 (§0c của sổ bằng chứng)

H1 nói *"adapter+fusion không thêm tri thức, nó thêm đường vòng quanh một Pha 1 hỏng"*. Số ở
đây **không bác H1** — H1 nói về **phần gia tăng của fusion so với chốt**, còn mục này đo
**Pha 1 + adapter + fusion so với KHÔNG có Pha 1**. Hai câu khác nhau. Nhưng nó thu hẹp H1:
phần "đường vòng" giải thích được **sức chứa**, không giải thích được vì sao lợi ích **tập
trung** ở nhóm cần phân biệt gần-trùng-lặp.

### Ràng buộc

- Nhóm `train` chỉ **~23 hàng/fold** — phương sai lớn. Đọc đếm dấu (4–5/5 mỗi điều kiện), đừng
  đọc trung bình một mình.
- Nhóm `test` (7.6 hàng) **không đọc**: nó cho +0.1372 ở chỗ này và −0.0043 ở chỗ kia.
- **Một khối, một máy** (vast `ntat`), seed 7 cho codebert và 42 cho t5p. §50 là hậu thuẫn độc
  lập nhưng ở cấu hình khác.
- `tools/leak_groups_auc.py` bỏ ô có nhóm chỉ chứa **một lớp** (2/10 ô ở nhóm `test`); số ô
  thực dùng in ra ở mỗi dòng.

---

## §59 — CẮT 512 TOKEN: không xoá tín hiệu ở NGUỒN, nhưng ở ĐÍCH nó tạo ra **16.2% hàng bất khả thi** trong đúng nhóm khó (15/09, **0 GPU**, chỉ tokenizer)

Người dùng nêu 15/09: *"với length 512 thì rất dễ missing vì cpp hay js có code rất dài, lỗ hổng
rất có thể nằm ở phân đoạn cuối"*. Đo bằng tokenizer, không chạy mô hình.

### Khối lượng bị cắt

| tập | n | token TB | trung vị | **vượt 512** |
|---|---|---|---|---|
| nguồn `com` | 3 744 | 922.9 | 426 | **43.4%** |
| ↳ ccpp | 2 360 | 1 189.5 | — | **54.4%** |
| ↳ js | 1 384 | 468.3 | — | 24.6% |
| đích python | 760 | 430.6 | 252 | 28.9% |

Nguồn bị cắt **gấp rưỡi** đích ⇒ Pha 1 huấn luyện trên đầu vào cụt hơn hẳn Pha 2. Đây là một
dịch chuyển phân bố **cộng thêm** vào dịch chuyển ngôn ngữ.

### Ở NGUỒN, cắt 512 KHÔNG xoá tín hiệu nhãn — 1 755 cặp

| | |
|---|---|
| sau khi cắt, hai bản (vul, fixed) **giống hệt nhau** | **0 / 1 755 = 0.0%** |
| vùng sửa **bắt đầu** sau token 512 | **0.0%** |
| vùng sửa **nằm trọn** trong 512 | **78.2%** |
| vùng sửa **kết thúc** sau 512 (mất một phần bản vá) | **21.8%** — ccpp 28.0%, js 12.1% |
| vị trí token khác nhau **đầu tiên** | p50 = **94**, p90 = 345, p95 = 416 |

> **Bản vá gần như luôn nằm ở đầu hàm.** Mô hình luôn nhìn thấy **một phần** khác biệt; ở 21.8%
> cặp nó chỉ thấy một phần. Thứ bị mất nhiều là **ngữ cảnh**, không phải **tín hiệu phân biệt**.

### Ở ĐÍCH, chuyện khác hẳn — và nó rơi đúng vào nhóm của §58

| nhóm rò rỉ | n | token TB | trung vị | vượt 512 |
|---|---|---|---|---|
| **`train`** (gần trùng, ngược nhãn, trong train) | 117 | **704.3** | 590 | **58.1%** |
| `test` | 38 | 743.2 | 696 | 78.9% |
| `none` (73% số hàng) | 556 | 317.5 | 192 | 16.9% |

**Nhóm khó CHÍNH LÀ nhóm code dài.** Khớp từng hàng `train` với bản đối nghịch gần nhất trong
train (Jaccard 5-gram ≥ 0.75) rồi so chuỗi token đã cắt:

> ### **19 / 117 hàng = 16.2% trở thành GIỐNG HỆT một mẫu train NGƯỢC NHÃN sau khi cắt 512.**
>
> Với những hàng đó, mô hình nhận **đúng cùng một đầu vào** với một mẫu huấn luyện mang **nhãn
> ngược lại**. Không mô hình nào đúng được cả hai. Đây là **trần cứng**, không phải "học kém".

Thêm **14.3%** hàng mất một phần vùng khác biệt. Khác biệt đầu tiên: p50 = 142, p90 = 365,
**0 hàng** có khác biệt đầu tiên sau 512.

### Nó ăn mất bao nhiêu điểm — bỏ 19 hàng bất khả thi rồi tính lại nhóm `train`

| | codebert ROC giữ → bỏ | t5p ROC giữ → bỏ |
|---|---|---|
| baseline | 0.7109 → **0.7594** (+0.049) | 0.7282 → 0.7230 (−0.005) |
| `fusft` | 0.8275 → 0.8525 (+0.025) | 0.8474 → 0.8523 (+0.005) |
| `nonefus` | 0.8205 → 0.8344 (+0.014) | 0.8583 → 0.8644 (+0.006) |
| **Δ `fusft` − baseline** | **+0.117 → +0.093** | +0.119 → +0.129 |

> **§58 KHÔNG phải hiện vật của cắt chuỗi.** Bỏ hết hàng bất khả thi thì hiệu ứng vẫn +0.093
> (codebert) và +0.129 (t5p). Nhưng trên codebert biên độ **co ~20%**, nên con số +0.1179 phải
> được trích dẫn kèm ghi chú này.

### Độ dài KHÔNG mang thông tin nhãn — nên không có thiên lệch độ dài

AUC khi đoán nhãn **chỉ bằng số token**: nguồn **0.4793**, đích **0.4937** — đều dưới 0.5, tức
ngang ngẫu nhiên. Mọi cách gộp chunk (kể cả trung bình) sẽ **không** vô tình bơm thiên lệch
độ dài vào điểm số.

### Ràng buộc kiến trúc — quyết định cách sửa

| backbone | giới hạn vị trí | nới dài được không |
|---|---|---|
| `microsoft/codebert-base` (RoBERTa) | `max_position_embeddings = 514` | **KHÔNG** — phải chunk hoặc nội suy embedding vị trí |
| `Salesforce/codet5p-220m-bimodal` (T5) | vị trí **tương đối**, 32 bucket, không có trần | **CÓ — đổi `--max_length` là xong, 0 dòng mã** |

Đây là phép kiểm rẻ nhất cho toàn bộ giả thuyết: chạy t5p ở `max_length` 1024 so với 512, cùng
máy, **một biến duy nhất**. Nếu trục ngữ cảnh có độ dốc thì nó hiện ra ở đó trước, với chi phí
~2× và không phải viết chunking.

---

## §60 — CODE DÀI KHÓ HƠN **TRƯỚC KHI** bị cắt: −0.08 ROC trên dải 0→512 token, và val Pha 1 KHÔNG dự báo được Pha 2 (15/09, **0 GPU**)

Hai câu hỏi của người dùng 15/09: (a) *"512 đôi khi nó nén mất đặc trưng lỗ hổng khi bị dài?"*
(b) đồng nghiệp dùng multi-window (chunk → mean) đạt ROC 0.92, nhưng **val Pha 1 ở nguồn không
cao hơn**, chỉ đích mới tốt lên.

### (a) Tách "BỊ CẮT" khỏi "BỊ PHA LOÃNG"

Chỉ lấy nhóm rò rỉ **`none`** (không có bản đối nghịch — loại bỏ nhiễu loạn của §58), chia theo
độ dài token, gộp 5 fold. **Ba khoảng đầu KHÔNG bị cắt một token nào.**

| khoảng token | n | codebert baseline | `fusft` | `nonefus` | t5p baseline | `fusft` | `nonefus` |
|---|---|---|---|---|---|---|---|
| 0–128 | 155 | **0.9538** | 0.9464 | 0.9592 | **0.9277** | 0.9530 | 0.9803 |
| 128–256 | 199 | 0.8848 | 0.9300 | 0.8898 | 0.9041 | 0.9050 | 0.9469 |
| 256–512 | 107 | **0.8667** | 0.8996 | 0.8779 | **0.8477** | 0.8632 | 0.8895 |
| 512–1024 *(bị cắt)* | 69 | 0.8403 | 0.8151 | 0.7672 | 0.7714 | 0.8756 | 0.9286 |
| 1024+ *(n nhỏ, không đọc)* | 26 | 0.7679 | 0.9345 | 0.9226 | 0.9048 | 0.8333 | 0.8095 |

> **Trong vùng KHÔNG bị cắt, ROC đã tụt 0.087 (codebert) và 0.080 (t5p) từ 0–128 xuống 256–512.**
> Đơn điệu ở **mọi nhánh, cả hai backbone**. Tức *"code dài khó hơn"* tồn tại **độc lập** với
> việc cắt ở 512 — câu hỏi (a) là **CÓ**, và nó xuất hiện **trước** ngưỡng cắt.

**Hai nguyên nhân phép đo này KHÔNG tách được:** (i) pha loãng biểu diễn — một vector cho nhiều
token hơn; (ii) khó nội tại — hàm dài có lỗi tinh vi hơn.

> **Chính thí nghiệm multi-window là phép tách:** nếu MW nâng dải **256–512** (nơi không có gì
> bị cắt) thì nguyên nhân là **pha loãng**; nếu MW chỉ nâng dải **>512** thì nguyên nhân là **cắt**.
> Đây là dự đoán phản chứng được, và kiểm được trên dữ liệu MW **đã có**, không cần chạy lại.

### (b) val Pha 1 ở NGUỒN không dự báo được kết quả ở ĐÍCH

Năm cặp mà **chỉ Pha 1 khác nhau**, mọi thứ khác giữ nguyên (cùng cây, cùng máy, cùng fold):

| so sánh | val A | val B | val A > B | F1 A | F1 B | khớp | ROC A | ROC B | khớp |
|---|---|---|---|---|---|---|---|---|---|
| codebert s42 head vs none | 0.5758 | **0.4430** | có | 0.8195 | 0.7660 | ✔ | 0.9050 | 0.8707 | ✔ |
| codebert s7 head vs none | 0.5932 | 0.5787 | có | 0.8303 | 0.8120 | ✔ | 0.9114 | 0.8993 | ✔ |
| **t5p s42 head vs none** | 0.6051 | 0.5939 | có | 0.8405 | **0.8481** | ✘ | 0.9168 | **0.9321** | ✘ |
| codebert bce vs pairB | 0.5649 | **0.4326** | có | 0.7973 | 0.7846 | ✔ | 0.8826 | 0.8572 | ✔ |
| **t5p bce vs pairB** | 0.5892 | **0.6002** | không | 0.8244 | 0.8021 | ✘ | 0.8965 | 0.8975 | ✔ *(lệch 0.001 — hoà)* |
| | | | | | | **3/5** | | | **4/5** |

**Lọc tiếp — chỉ giữ cặp mà CẢ HAI Pha 1 đều lành** (bỏ hai dòng có val 0.4430 / 0.4326, nơi
quan hệ đúng một cách tầm thường):

> ### **F1 1/3 · ROC 2/3, trong đó một "khớp" là chênh 0.001.** Val Pha 1 **không** dự báo được Pha 2.

Chênh lệch val Pha 1 ở ba cặp đó đều **≤ 0.015** — nằm trong dao động giữa hai lần rút (§48.2).
Cộng với §57 (t5p: Pha 1 **tốt hơn** 0.6002 vs 0.5892 nhưng Pha 2 **kém hơn** −0.0223) và §54.1
(đổi **quy tắc chọn checkpoint** Pha 1 đáng +0.024 trong khi val nguồn gần như không đổi):

> **Quan sát của đồng nghiệp — "Pha 1 nguồn không cao hơn nhưng đích lại tốt hơn" — là chuyện
> BÌNH THƯỜNG trong dự án này, không phải dấu hiệu sai.** Val nguồn đo *"học được tác vụ nguồn
> nhiễu 40–75% tới đâu"*; nó không đo *"biểu diễn có chuyển giao được không"*. Đại lượng dự báo
> tốt hơn là probe hướng-vá của §53 (AUC trên Python, 0 bước huấn luyện) — **chưa** ai đo nó
> trên checkpoint MW.

### Ràng buộc

- Bảng (a) gộp fold nên **không** có đếm dấu ghép cặp; đọc là **mô tả**, không phải kiểm định.
  Dải 1024+ chỉ 26 hàng, không đơn điệu, **không đọc**.
- Bảng (b) chỉ n=5 (và n=3 sau khi lọc). Nó **bác** được phát biểu *"val Pha 1 dự báo Pha 2"*,
  không đủ để phát biểu điều ngược lại có cấu trúc gì.
