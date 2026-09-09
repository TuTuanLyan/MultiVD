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

## §27 — Nội suy TRỌNG SỐ: ô đầu tiên nói KHÔNG (09/09 06:00, n=1 — sơ bộ)

`run/wblend.sh` chạy thật lần đầu. Ô đầu tiên (`4cwe`, fold 1, t5p) — **101 tensor nội suy được,
4 tensor riêng của nhánh chuyển giao giữ nguyên**, đúng như cơ chế đã kiểm ở §25.5.

| α (không gian **trọng số**) | 0.0 | 0.2 | **0.5** | 0.8 | 1.0 |
|---|---|---|---|---|---|
| test ROC-AUC | 0.8762 | 0.8736 | **0.8668** | 0.8821 | **0.8911** |
| test F1@0.5 | 0.7696 | 0.7753 | **0.7456** | 0.7871 | **0.8211** |

**Đường α trong không gian trọng số LÕM XUỐNG GIỮA** — ngược hẳn với không gian xác suất (§25,
đỉnh nội tại ở α≈0.5). α=0.5 **kém cả hai đầu mút**. Và **α chọn trên VAL = 1.0** trên cả hai tiêu
chí, tức tập val tự nói *"chỉ dùng mô hình chuyển giao, nội suy trọng số không đóng góp gì"*.

Đối chiếu trên **cùng cặp**: trộn **xác suất** α=0.5 cho ROC **0.8910** — ngang với mô hình chuyển
giao một mình (0.8911) và hơn mọi mức nội suy trọng số ở giữa.

**Đọc**: hai mô hình **không nối tuyến tính** theo nghĩa WiSE-FT cần — có một *hàng rào* giữa
chúng. Phép thử ở §25.5 (trộn 50/50 hai bản codebert vẫn ra mô hình chạy được) đã gợi ý ngược lại,
nhưng đó là **cặp khác** (hai baseline khác fold), còn cặp thật (baseline ↔ chuyển giao qua Pha 1)
thì có hàng rào. Bài học: *"kiểm cơ chế trên một cặp thay thế"* rẻ và đáng làm, nhưng **không thay
được cặp thật**.

**Hệ quả cho phương pháp**: phép trộn của §25 vẫn phải giữ **hai mô hình khi suy luận**. Đó là một
giới hạn phải nêu trong bài, không phải thứ có thể lấp bằng nội suy trọng số.

**n=1 — sơ bộ.** Khối chạy 3 nguồn × 3 fold = 9 ô; sẽ cập nhật khi đủ. Nhưng cơ chế (val chọn
α=1.0, đường lõm) rõ ngay ở ô đầu.

---

## §28 — CẤU HÌNH CHỐT ở bậc 3: toàn bộ mức tăng nằm ở hai CWE hiếm (09/09/2026)

Cấu hình chốt sau §7 + §24: **`latent_bottleneck` (nút thắt 8 chiều) · λ=0.05 · Pha 2 dùng AdamW +
ASAM ρ=2.0**. Khối `asamaw` đã chạy đúng cấu hình này ở **n=15 (5 fold × 3 nguồn)**, đối chứng là
**baseline** (không có Pha 1) cùng máy cùng fold. Không cần chạy thêm gì — số đã có sẵn.

### Tổng thể

| máy | n | ΔF1@0.5 | ΔF1@val | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|---|---|
| **ntat** | **15** | **+0.0535 (12/15)** | **+0.0685 (12/15)** | **+0.0337 (14/15)** | **+0.0300 (13/15)** |
| ntat2 (độc lập) | 9 | +0.0319 (4/9) | +0.0280 (5/9) | +0.0264 (7/9) | +0.0204 (7/9) |

Cả bốn chỉ số dương trên cả hai máy. Trên ntat, **cả ba nguồn** dương trên cả bốn chỉ số
(4cwe ROC +0.0352 5/5 · com +0.0300 4/5 · full +0.0360 5/5).

**So với chính nó khi TẮT ASAM** (`aw_r0`, cùng head cùng λ cùng AdamW): ROC +0.0137 (11/15) và
PR **−0.0018 (6/15)**. Bật ASAM ρ=2.0 làm ROC hơn **gấp 2,5 lần** và lật PR từ null sang +0.0300.

### Theo CWE — đây mới là chỗ đáng viết

**ntat, n=15:**

| CWE | hàng test | ΔF1@0.5 | ΔROC-AUC |
|---|---|---|---|
| **022** path traversal | 13 | **+0.2050 (11/15, p=0.022)** | **+0.2238 (13/15, p=0.002)** |
| 078 OS command inj. | 40 | −0.0082 (7/15, p=1.00) | −0.0097 (6/15, p=0.79) |
| **079** XSS | 16 | **+0.2978 (14/15, p=0.001)** | **+0.3638 (15/15, p<0.001)** |
| 089 SQL injection | 81 | +0.0142 (7/15, p=0.55) | +0.0079 (11/15, p=0.12) |

**ntat2 độc lập, n=9:** CWE-079 **+0.2614 ROC với 9/9 fold (p=0.004)**; CWE-022 +0.1619 (6/9);
hai lớp thường null (078 +0.0173, 089 −0.0011).

**Đọc**: toàn bộ mức tăng của cấu hình chốt nằm ở **hai lớp hiếm**, và **CWE-079 đạt 15/15 fold
trên ntat, 9/9 trên ntat2** — cùng dấu tuyệt đối trên hai máy độc lập. Hai lớp thường (chiếm 121
trong 150 hàng test) **đúng bằng không**. Con số tổng +0.0337 nhỏ chỉ vì hai lớp thường áp đảo về
số hàng, không phải vì hiệu ứng yếu.

Biên độ ở đây (**+0.36** ROC cho CWE-079) **lớn hơn nhiều** §23 (+0.176) vì §23 gộp 507 ô của
**mọi** cấu hình, kể cả các cấu hình yếu và các mức ρ đã bị loại. Đây là cấu hình chốt, đo riêng.

**Cảnh báo phải in kèm**: CWE-079 chỉ có **16 hàng test mỗi fold**, CWE-022 **13**. Thứ làm con số
đáng tin **không phải biên độ** mà là **15/15 và 9/9 fold cùng dấu trên hai máy độc lập**. Trích
biên độ mà không trích đếm dấu là đọc sai theo đúng kiểu mục 2b đã cấm.
