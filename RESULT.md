# Kết quả thí nghiệm

Tài liệu này ghi lại các thí nghiệm đã chạy và số liệu cụ thể, để tra cứu về sau.
Số thô nằm ở `results_vast/`, log ở `logs_vast/` (không track trong git).

Mọi so sánh trong tài liệu này chỉ hợp lệ **trong cùng một bảng**. Xem mục
[Cảnh báo so sánh chéo](#cảnh-báo-so-sánh-chéo-máy) trước khi ghép số từ hai bảng khác nhau.

---

## 1. Thiết lập chung

| Hạng mục | Giá trị |
| --- | --- |
| Target | SVEN Python, 760 mẫu, 5 fold (456 train / 152 val / 152 test) |
| Source (Phase 1) | `data/train_ccpp_js.jsonl` — 1284 dòng, ccpp 146 + js 1138 |
| Phase 1 | Multitask: binary vulnerability + head phụ, AdamW, tối đa 15 epoch |
| Phase 2 | RecAdam trên Python, head phụ đóng băng, tối đa 30 epoch |
| Hyperparameter | batch 16, max_length 512, lr 2e-5, λ_aux 0.2, patience 5, min_epochs 3 |
| Truncation | `head_middle_tail` |
| Seed | 36 (một seed duy nhất — xem [Hạn chế](#7-hạn-chế-đã-biết)) |
| Chọn checkpoint | Val Macro-F1@0.5 của riêng task nhị phân |

Kiến trúc: CodeBERT → CLS (hoặc mean pooling với T5) → `Linear(768,2)` phán an toàn/lỗ hổng.
Song song là một **head phụ chỉ tồn tại trong Phase 1**, đóng băng và không dùng lúc inference.

---

## 2. Thí nghiệm A — Bốn chế độ head phụ trên CodeBERT

Cả bốn dùng chung source, folds, seed, hyperparameter. Khác biệt duy nhất là head phụ.

| Config | Head phụ | Cần nhãn CWE |
| --- | --- | --- |
| `baseline` | Không có Phase 1, train thẳng trên Python | Không |
| `cwe` | `Linear(768, 4)` — phương pháp gốc | Có |
| `latent_bottleneck` | `Linear(768, 8) → Linear(8, 4)` | Có |
| `latent_proto` | 8 prototype, gán cân bằng Sinkhorn | **Không** |
| `none` | Bỏ hẳn, λ=0 (ablation) | Không |

### Macro-F1 @0.5

| Config | fold1 | fold2 | fold3 | fold4 | fold5 | mean ± std | Δ baseline |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.7479 | 0.8023 | 0.8082 | 0.7958 | 0.7561 | 0.7821 ± 0.0280 | — |
| cwe | 0.7801 | 0.8158 | 0.8155 | 0.8349 | 0.8082 | 0.8109 ± 0.0199 | +0.0289 |
| latent_bottleneck | 0.8220 | 0.8289 | 0.7813 | 0.8220 | 0.8223 | 0.8153 ± 0.0192 | +0.0332 |
| latent_proto | 0.8408 | 0.8683 | 0.8355 | 0.8482 | 0.8352 | **0.8456 ± 0.0138** | +0.0635 |
| none | 0.8166 | 0.7431 | 0.7431 | 0.8214 | 0.7953 | 0.7839 ± 0.0385 | +0.0019 |

### ROC-AUC và PR-AUC (độc lập ngưỡng)

| Config | ROC-AUC mean ± std | Δ | PR-AUC mean ± std | Δ |
| --- | --- | --- | --- | --- |
| baseline | 0.8752 ± 0.0297 | — | 0.8693 ± 0.0303 | — |
| cwe | **0.9037 ± 0.0143** | +0.0285 | **0.9010 ± 0.0191** | +0.0317 |
| latent_bottleneck | 0.8974 ± 0.0176 | +0.0221 | 0.8878 ± 0.0279 | +0.0185 |
| latent_proto | 0.8924 ± 0.0089 | +0.0172 | 0.8543 ± 0.0326 | **−0.0149** |
| none | 0.8536 ± 0.0332 | −0.0216 | 0.8305 ± 0.0549 | −0.0388 |

### Metric tại ngưỡng 0.5 (trung bình 5 fold)

| Config | Precision | Recall | Positive-F1 | Accuracy | Macro-F1 @valcal |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.7705 | 0.8128 | 0.7891 | 0.7829 | 0.7856 |
| cwe | 0.8044 | 0.8267 | 0.8147 | 0.8118 | 0.8136 |
| latent_bottleneck | 0.8256 | 0.8002 | 0.8118 | 0.8158 | 0.8101 |
| latent_proto | **0.8412** | **0.8558** | **0.8473** | **0.8461** | **0.8327** |
| none | 0.7621 | 0.8305 | 0.7937 | 0.7855 | 0.7839 |

### Kết luận thí nghiệm A

**Task phụ tạo ra khả năng phân biệt.** `none` (λ=0) ngang baseline về Macro-F1 (+0.002)
nhưng âm rõ trên cả hai metric xếp hạng: ROC-AUC −0.022, PR-AUC −0.039. Nghĩa là source
pretraining + RecAdam **tự thân không cải thiện gì**; head phụ mới là thứ có tác dụng.
Ablation này chưa xuất hiện trong y văn.

**`cwe` thắng về xếp hạng.** Là cấu hình duy nhất dương trên cả ROC-AUC lẫn PR-AUC.

**`latent_proto` thắng về điểm vận hành nhưng hỏng về xếp hạng.** Cao nhất ở Macro-F1@0.5,
precision, recall — nhưng PR-AUC âm, và threshold do validation chọn dao động 0.63–0.92,
áp lên test làm Macro-F1 tụt từ 0.8456 xuống 0.8327. Lợi thế nằm ở chỗ ngưỡng 0.5 tình cờ
đặt đúng, không phải ở khả năng phân biệt. Nguyên nhân và cách sửa: xem [mục 5](#5-sửa-lỗi-của-latent_proto).

**`latent_bottleneck` là lựa chọn thực dụng.** Bám sát `cwe`, dương trên 5/6 metric, và
gỡ được ràng buộc taxonomy — điều kiện cần để dùng source như PrimeVul (140 CWE).

---

## 3. Thí nghiệm B — Per-CWE: lợi ích đến từ đâu

Macro-F1 tại ngưỡng val-calibrated, trung bình 5 fold. Số trong ngoặc là tổng mẫu test.

| Config | CWE-022 (66) | CWE-079 (82) | CWE-078 (204) | CWE-089 (408) |
| --- | --- | --- | --- | --- |
| baseline | 0.3478 | 0.4052 | 0.7397 | **0.9429** |
| cwe | 0.4631 | 0.5878 | 0.7399 | 0.9412 |
| latent_bottleneck | 0.4932 | 0.7421 | 0.7469 | 0.9078 |
| latent_proto | **0.6298** | **0.7608** | 0.7436 | 0.9176 |
| none | 0.4030 | 0.6575 | 0.7351 | 0.8870 |

**Toàn bộ lợi ích nằm ở hai lớp hiếm.** `latent_proto` hơn baseline +0.282 trên CWE-022 và
+0.356 trên CWE-079. Trên CWE-089 — chiếm 54% test — **baseline tốt nhất** và mọi biến thể
transfer đều kém hơn. CWE-078 thì cả năm ngang nhau (0.735–0.747).

Đây là hình dạng thuận lợi cho bài báo: phương pháp bù đúng vào chỗ dữ liệu target mỏng nhất.

---

## 4. Thí nghiệm C — Ba backbone

Cùng phương pháp (`cwe`), cùng folds và seed. Mỗi backbone so với **baseline của chính nó**.
T5-family chỉ nạp encoder và dùng mean pooling (không có CLS token).

| Backbone | Metric | Baseline | Transfer | Δ |
| --- | --- | --- | --- | --- |
| CodeBERT | Macro-F1 | 0.7821 ± 0.0280 | 0.8109 ± 0.0199 | **+0.0289** |
| CodeBERT | ROC-AUC | 0.8752 ± 0.0297 | 0.9037 ± 0.0143 | **+0.0285** |
| CodeBERT | PR-AUC | 0.8693 ± 0.0303 | 0.9010 ± 0.0191 | **+0.0317** |
| CodeT5 | Macro-F1 | 0.8315 ± 0.0235 | 0.8045 ± 0.0285 | **−0.0270** |
| CodeT5 | ROC-AUC | 0.9119 ± 0.0242 | 0.8810 ± 0.0205 | **−0.0309** |
| CodeT5 | PR-AUC | 0.9195 ± 0.0146 | 0.8744 ± 0.0443 | **−0.0451** |
| CodeT5+ | Macro-F1 | 0.8419 ± 0.0158 | 0.8059 ± 0.0692 | **−0.0361** |
| CodeT5+ | ROC-AUC | 0.9235 ± 0.0097 | 0.8844 ± 0.0577 | **−0.0391** |
| CodeT5+ | PR-AUC | 0.9265 ± 0.0196 | 0.8797 ± 0.0489 | **−0.0468** |

### Quy luật đơn điệu

Baseline càng mạnh, transfer càng có hại:

| Baseline Macro-F1 | 0.7821 | 0.8315 | 0.8419 |
| --- | --- | --- | --- |
| **Δ transfer** | **+0.029** | **−0.027** | **−0.036** |

Transfer còn làm mất ổn định: std giữa các fold của CodeT5+ tăng từ 0.0158 lên 0.0692.

**Con số đối chiếu quan trọng nhất:** CodeT5+ không transfer gì đạt ROC-AUC **0.9235**,
so với **0.9037** của CodeBERT cấu hình tốt nhất. Đổi checkpoint pretrained ăn đứt toàn bộ
pipeline transfer 0.020 ROC-AUC, với chi phí thấp hơn nhiều.

Đây là lý do nhiệm vụ **làm phương pháp không phụ thuộc pretrained model** trở thành mục tiêu chính.

---

## 5. Sửa lỗi của `latent_proto`

Đối chiếu với thiết kế tham chiếu SwAV (Caron et al., NeurIPS 2020) phát hiện hai sai lệch:

**Dùng chung trunk.** `latent_proj` đọc thẳng vector đã pool — cũng là vector `vul_head` đọc.
SwAV tính prototype trên **projection head riêng**. Dùng chung nên mục tiêu clustering bóp
méo trực tiếp hình học đặc trưng mà bộ phân loại phụ thuộc. Đây đúng cơ chế Müller, Kornblith
& Hinton (NeurIPS 2019) mô tả cho label smoothing: các mẫu cùng lớp co cụm chặt lại, cải thiện
độ chính xác tại một điểm vận hành cố định nhưng phá thông tin quan hệ mà metric xếp hạng cần.

**Prototype học ngay từ bước 1.** SwAV giữ cố định một số bước đầu (`freeze_prototypes_niters`).

Ghi chú quan trọng: **không thể sửa bằng hiệu chuẩn hậu kỳ.** Guo et al. (ICML 2017) chứng minh
temperature scaling là đơn điệu nên không đổi AUC. PR-AUC tụt phản ánh thứ hạng thật sự kém đi.

Đã sửa trong code: projection head hai lớp riêng, `--freeze_prototypes_steps`, và
`--selection_metric {macro_f1,pr_auc,roc_auc}` để chọn checkpoint theo metric độc lập ngưỡng.
Chạy lại chưa hoàn tất (mất máy giữa chừng, xem [Nhật ký sự cố](#8-nhật-ký-sự-cố)).

---

## 6. Vấn đề dữ liệu: rò rỉ cặp và bộ folds twin

SVEN Python gồm 380 **cặp** hàm: bản có lỗi và bản đã vá. Bộ folds gốc
(`data/sven_python_folds_norm`) chia **theo dòng**, nên rất nhiều cặp bị tách đôi giữa
train và test.

Ví dụ thật trong fold 1, độ tương đồng 0.9928 — khác đúng một ký tự:

```python
# TRAIN, nhãn 0 (an toàn) — truy vấn tham số hóa
self.cursor.execute("SELECT ... where id = %s and money >= %s", (user_id, money))

# TEST, nhãn 1 (lỗ hổng) — nối chuỗi trực tiếp
self.cursor.execute("SELECT ... where id = %s and money >= %s" % (user_id, money))
```

Đo trên cả 5 fold: **61–67 trên 152 mẫu test mỗi fold** có bản gần trùng trong train,
trên 90% là bản đối nghịch nhãn của chính nó. Tức 40–44% "đề thi đã lộ".

### Bộ folds twin

`src/build_folds.py` gom các dòng thành cụm near-duplicate rồi chia **theo cụm**.
Ngưỡng 0.75 cho 384 cụm, khớp gần chính xác con số 380 cặp mà SVEN công bố.

| | Folds gốc | Folds twin |
| --- | --- | --- |
| Rò rỉ >0.90 trong test | 40–44% | **0.1%** (1/760) |
| Kích thước split | 456/152/152 | ~456/152/152 (giữ nguyên) |
| Phân bố CWE | — | giữ nguyên |
| Trường bổ sung | — | `pair_id` |

`pair_id` là mục đầu trong `GROUP_FIELDS` của loader, nên chỉ cần có nó là việc chia
source ở Phase 1 tự động thành group-aware, không phải sửa gì thêm.

Sinh lại bằng:

```bash
python src/build_folds.py --output_dir data/sven_python_twin --threshold 0.75
```

**Mọi số ở mục 2, 3, 4 đều đo trên folds gốc, nên đều bị thổi phồng.** So sánh giữa các
cấu hình vẫn công bằng vì dùng chung folds, nhưng giá trị tuyệt đối không phản ánh khả năng
khái quát hóa. Thí nghiệm chạy lại trên folds twin đang tiến hành.

---

## 7. Hạn chế đã biết

**Một seed.** Chạy lại đúng cấu hình trên GPU và bản PyTorch khác làm dịch Macro-F1
**0.028** — tương đương toàn bộ hiệu ứng đang đo. Độ lệch chuẩn giữa các fold lên tới 0.06.
Y văn (Reimers & Gurevych 2017; Dodge et al. 2020) khuyến nghị tối thiểu 5 seed, mục tiêu 10.

**Chọn checkpoint theo Macro-F1@0.5.** Chính điều này khiến một model chỉ thắng tại ngưỡng
0.5 trông như tốt nhất. Đã bổ sung `--selection_metric` nhưng số ở trên chưa dùng nó.

**Thiếu regularizer đối chứng.** Match-Tuning (IJCAI 2022) benchmark RecAdam và thấy nó
chỉ +0.55, **thua Mixout, R3F và cả hai biến thể Child-Tuning**. Reviewer sẽ hỏi tại sao vắng.

**Trùng tên với một bài đã công bố.** Curto et al., "MultiVD: A Transformer-based Multitask
Approach for Software Vulnerability Detection", SECRYPT 2024, DOI `10.5220/0012719400003767`
— cũng CodeBERT/LineVul đa nhiệm binary + CWE. RQ3 của họ quét trọng số auxiliary loss và
kết luận "no significant performance improvements".

**λ_aux chưa quét.** Cố định 0.2 xuyên suốt.

---

## 8. Nhật ký sự cố

Ghi lại để không lặp lại và để biết số nào tin được.

| Sự cố | Ảnh hưởng | Xử lý |
| --- | --- | --- |
| `train_baseline.py` không có `--output_dir` | 4 fold baseline train xong nhưng chết ở inference | Bỏ flag; chạy bù inference từ checkpoint đã có |
| Đĩa 20GB đầy (41 checkpoint × ~440MB) | CodeT5 dừng ở fold 3, CodeT5+ chết ở phase1 | Tự xóa checkpoint fold sau khi có result JSON |
| Sync rsync thất bại im lặng (shell kẹt sai thư mục) | Máy chạy script cũ, ghi đè source checkpoint CodeT5 | Xóa và chạy lại CodeT5 từ đầu |
| Hai tiến trình chain trùng nhau | Suýt ghi đè kết quả lẫn nhau | Kill trước khi ghi; dùng file sentinel thay `pgrep` |
| `${MODES:-default}` coi chuỗi rỗng là chưa set | Chạy lại nhầm cấu hình đã xong | Đổi sang `${MODES-default}` |
| Instance bị thu hồi sau khi stop | Mất toàn bộ checkpoint trên đĩa | Kết quả đã kéo về local trước đó; checkpoint không cần cho phân tích |

---

## Cảnh báo so sánh chéo máy

Thí nghiệm chạy trên nhiều GPU khác nhau (RTX A4000, RTX 4070 Ti SUPER, RTX 5070 Ti) và
nhiều bản PyTorch. Đo được: **cùng cấu hình, cùng seed, chỉ đổi máy → Macro-F1 lệch 0.028**,
và phân kỳ bắt đầu ngay từ Phase 1 (val Macro-F1 0.6589 so với 0.5301).

Vì vậy **chỉ so số trong cùng một bảng**. Mỗi bảng ở trên đều được sinh từ một lượt chạy
trên một máy duy nhất, và mỗi backbone luôn có baseline riêng chạy cùng lượt.

---

## 10. Thí nghiệm trên folds twin (không rò rỉ)

Toàn bộ mục 2–4 đo trên folds gốc có rò rỉ 40–44%. Mục này chạy lại trên
`data/sven_python_twin` (rò rỉ 0.1%), cùng seed 36, cùng máy cho mỗi bảng.

### 10.1 CodeBERT — bốn chế độ head phụ, 5 fold

| Config | Macro-F1 @0.5 | Δ | ROC-AUC | Δ | PR-AUC | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.8281 | — | 0.9153 | — | 0.9121 | — |
| cwe | 0.8698 | **+0.0417** | 0.9387 | **+0.0234** | 0.9407 | **+0.0286** |
| latent_bottleneck | 0.8549 | +0.0268 | 0.9333 | +0.0181 | 0.9316 | +0.0195 |
| latent_proto | 0.8536 | +0.0254 | 0.9195 | +0.0042 | 0.8989 | −0.0132 |
| none | 0.8257 | −0.0025 | 0.8992 | −0.0160 | 0.8778 | −0.0343 |

**Hiệu ứng transfer sống sót khi hết rò rỉ.** `cwe` giữ +0.042 Macro-F1 (folds cũ: +0.029).
Đây là kết quả quan trọng nhất của mục này: lợi ích **không phải do rò rỉ tạo ra**.

**`latent_proto` mất phần lớn ưu thế: +0.0635 → +0.0254.** Trên folds cũ nó cao gần gấp đôi
các mode khác; trên folds sạch nó ngang bằng, và PR-AUC vẫn âm (−0.013). Kết luận: ưu thế
trước đây **chủ yếu là ảo**, do nó khai thác được cấu trúc cặp bị lộ. Nếu không dựng folds
twin thì ta đã đi tiếp với một kết quả sai.

**`none` vẫn âm trên metric xếp hạng** (ROC −0.016, PR −0.034) dù Macro-F1 gần bằng baseline.
Kết luận "task phụ tạo ra khả năng phân biệt" giữ nguyên.


### 10.3 Thống kê ghép cặp trên folds twin (CodeBERT, 5 fold)

Trung bình trần không đủ khi độ lệch chuẩn giữa các fold (tới 0.09) lớn hơn hiệu ứng cần đo
(0.02–0.04). Bảng dưới dùng các kiểm định mà y văn phương pháp luận khuyến nghị cho mẫu nhỏ.

| Cấu hình | Δ mean | Δ std | **A12** | Wilcoxon p | t hiệu chỉnh |
| --- | --- | --- | --- | --- | --- |
| `cwe` | +0.0417 | 0.0311 | **0.88** | 0.0625 | +1.83 |
| `latent_bottleneck` | +0.0268 | **0.0196** | **0.84** | 0.0625 | **+1.87** |
| `latent_proto` | +0.0254 | 0.0380 | 0.72 | 0.1250 | +0.92 |
| `none` | −0.0025 | 0.0249 | 0.48 | 1.0000 | −0.14 |

**A12** = xác suất cấu hình thắng baseline trên một fold bốc ngẫu nhiên. Ngưỡng 0.71 trở lên
là hiệu ứng lớn (Vargha & Delaney). `cwe` 0.88 và `latent_bottleneck` 0.84 đều vượt; `none`
đạt 0.48 tức **đúng bằng tung đồng xu**.

`latent_bottleneck` có **độ lệch chuẩn thấp nhất** và **t hiệu chỉnh cao nhất** dù Δ mean thấp
hơn `cwe` — hiệu ứng nhỏ hơn nhưng nhất quán hơn.

### Giới hạn phải nêu rõ

**Ở n=5, giá trị Wilcoxon p nhỏ nhất có thể đạt là 0.0625.** Không cấu hình nào có thể xuống
dưới 0.05 với một seed, dù hiệu ứng sạch đến đâu — đây là tính chất của cỡ mẫu, không phải
của dữ liệu.

Hệ quả: mọi kết quả một-seed trong tài liệu này là **sàng lọc hướng đi**, không phải bằng chứng
công bố được. Từ ước tính công suất ở `RESEARCH_2026-08-20_0959.md` §6 (σ≈0.03, δ≈0.02 → cần
~18 cặp quan sát), cấu hình nào sống sót cần **4 seed × 5 fold**.

Công cụ: `src/paired_stats.py`, đã kiểm chứng bằng các ca biết trước đáp án (toàn dấu dương →
đúng 0.0625; dấu lẫn lộn → 0.8125; A12 = 1.00 khi thắng tuyệt đối, 0.50 khi giống hệt).

### 10.2 CodeT5+ — negative transfer tái hiện, và LP-FT không cứu được

| Config | Macro-F1 @0.5 | Δ | ROC-AUC | Δ |
| --- | --- | --- | --- | --- |
| baseline | 0.8879 | — | 0.9574 | — |
| cwe | 0.8696 | −0.0184 | 0.9455 | −0.0119 |
| cwe + LP-FT | 0.8682 | −0.0198 | 0.9358 | **−0.0216** |

Negative transfer tái hiện trên folds sạch (−0.018 so với −0.027 trên folds cũ), nên đây là
hiện tượng thật. **LP-FT không những không cứu mà còn làm tệ hơn** trên metric xếp hạng.

Đã kiểm chứng linear probe thật sự chạy trước khi kết luận: log có dòng `Linear probe`,
`lp_epochs=3`, `lp_lr=1e-3`, hai nhánh dừng ở epoch khác nhau (29 vs 13) với val khác nhau.

**Khoảng cách backbone lớn hơn mọi can thiệp.** Baseline CodeT5+ đạt 0.8879 Macro-F1 và
0.9574 ROC-AUC; CodeBERT với cấu hình transfer tốt nhất chỉ đạt 0.8698 và 0.9387. Đổi
checkpoint pretrained vẫn ăn đứt toàn bộ pipeline transfer.

---

## 11. Kho dữ liệu source

Bốn biến thể PrimeVul đã có sẵn (không tự xử lý lại — có quy tắc lọc CWE nội bộ), tất cả
đều `lang='ccpp'`, nhãn cân bằng chính xác 50/50, schema `code, cwe, cwe_id, label, lang`:

| File | Dòng | Số CWE | Ghi chú |
| --- | --- | --- | --- |
| `ccpp_primevul_paired_4cwe` | 178 | 4 | quá nhỏ, hợp làm đối chứng "source thiếu dữ liệu" |
| `ccpp_primevul_paired_common` | 2975 | 73 | CWE giao giữa hai phía |
| `ccpp_primevul_paired_full` | 9408 | **121** | phép thử cực đoan cho khái quát hóa taxonomy |
| `ccpp_primevul_paired_ignored` | 9042 | 115 | bỏ các negative lọc tay |

Cộng với `train_ccpp_js.jsonl` (1284 dòng, ccpp 146 + js 1138, **chỉ 4 CWE**). Phần js chỉ
có 4 CWE nên hai chiều "ngôn ngữ" và "số CWE" không giao nhau tự do được.

**Nút thắt đã gỡ.** `resolve_cwe_class()` hard-map `{22,78,79,89}` và ném `-100` cho phần
còn lại, nên `full` chỉ dùng được 4 trong 121 CWE. `--cwe_vocab source` xây từ điển từ chính
dữ liệu: `full` → **120 lớp phụ, chỉ bỏ 1 dòng** thay vì bỏ 116 loại.

---

## 12. Nhiệm vụ pretrained-agnostic: tình trạng các can thiệp

Tiêu chí thành công đặt trước: phải **đồng thời** giữ được mức tăng trên CodeBERT và xóa
được mức giảm trên CodeT5+. Chỉ làm transfer trở nên vô hại ở mọi backbone là thất bại trá hình.

| Can thiệp | Cơ chế | Trạng thái |
| --- | --- | --- |
| **LP-FT** | Fit head trước, rồi mở khóa backbone | **Thất bại** (ROC −0.022 vs −0.012) |
| **Nội suy trọng số** | `θ = α·θ_phase1 + (1−α)·θ_pretrained` | **Thất bại**, mọi α (xem đường cong dưới) |
| Tỉ lệ dữ liệu target | 114 / 228 / 456 dòng mỗi fold | đang chạy |
| LoRA Phase 1 | Đóng băng backbone, chỉ cập nhật hạng r | đang chạy, r ∈ {8, 32} |
| Task arithmetic / merging | — | **Loại**: sập khi trộn qua kiến trúc khác họ |

### Đường cong nội suy: không bao giờ cắt trục 0

| α | 1.00 | 0.75 | 0.50 | 0.25 | 0.00 |
| --- | --- | --- | --- | --- | --- |
| Δ Macro-F1 | −0.0184 | **−0.0371** | −0.0284 | −0.0066 | 0.0000 (baseline) |

Lõm xuống ở α=0.75 rồi hồi lên đơn điệu về baseline. **Không có tỉ lệ pha trộn nào làm
transfer có lợi.** Ở α=0.25 gần chạm baseline nhưng vẫn âm, và đó là vì chỉ còn 25% Phase 1
— tức gần như không transfer nữa. Nội suy chỉ làm *mờ dần* transfer, không tạo ra điểm tốt.

### Vì sao nội suy hỏng

Trộn 25% backbone gốc vào lại làm tệ **gấp đôi** so với không trộn. Nếu nội suy chỉ đơn thuần
kéo mô hình về phía baseline thì α=0.75 phải nằm giữa α=1.0 (−0.018) và α=0 (baseline, 0.000).
Nó không nằm giữa — đường đi vòng qua một vùng xấu.

Đã loại trừ khả năng lỗi code: kiểm chứng phép trộn trên backbone thật cho sai số **0.000e+00**
so với công thức, trọng số lai đúng tỉ lệ 0.75.

Giải thích khả dĩ nhất: **chỉ backbone được nội suy, còn `vul_head` giữ nguyên từ Phase 1.**
Head đó học để đọc đặc trưng của backbone Phase 1, giờ phải đọc một backbone lai chưa từng
thấy. WiSE-FT gốc nội suy toàn bộ mô hình, nhưng ở đây backbone gốc không có head tương ứng
để trộn vì lúc đó head còn ngẫu nhiên. Nặng hơn nữa, RecAdam neo θ* vào chính trạng thái lai
đó, nên giữ chặt mô hình quanh một điểm đã hỏng.

Nếu cả ba can thiệp còn lại đều thất bại thì giả thuyết "Phase 1 làm méo đặc trưng" sai, và
kết luận trung thực nhất từ dữ liệu sẽ là: phương pháp có giá trị **trong chế độ backbone yếu
hoặc target ít dữ liệu**, chứ không phải ở mọi chế độ. Bằng chứng cho hướng đó đã có trong
mục 3: lợi ích dồn hết vào hai CWE hiếm, còn CWE-089 với 408 mẫu thì baseline thắng.

---

## 13. Thí nghiệm theo tỉ lệ dữ liệu target

CodeBERT, folds twin, mỗi mức có **baseline riêng cũng bị cắt cùng mức** (`--max_train_samples`
chỉ áp cho Phase 2/test/baseline, **không** áp cho Phase 1 — nếu áp sẽ cắt nhầm corpus source).

| Dòng train/fold | baseline | transfer_cwe | Δ |
| --- | --- | --- | --- |
| 456 (100%) | 0.8281 | 0.8698 | **+0.0417** |
| 114 (25%) | 0.6835 | 0.5781 | **−0.1054** (n=3) |

Đảo dấu hoàn toàn. Ở dữ liệu đầy đủ transfer giúp; ở 25% nó **hại nặng**.

### Nguyên nhân: RecAdam, không phải transfer

Soi fold 1 ở mức 114 dòng:

| | confusion | xác suất |
| --- | --- | --- |
| transfer | [[39, 38], [31, 45]] | min 0.3172 max 0.8302 **std 0.1158** |
| baseline | [[45, 32], [18, 58]] | min 0.0022 max 0.9985 **std 0.4312** |

Dự đoán **cân đối**, không suy biến. Nhưng phân bố xác suất của transfer bị nén vào [0.32, 0.83],
độ lệch chuẩn hẹp hơn baseline gần **4 lần**. Đây là model **chưa được huấn luyện đủ**, không
phải model sập.

Cơ chế: với 114 dòng thì 8 batch/epoch × 30 epoch = **240 bước**, và best checkpoint rơi vào
epoch 7 ≈ **bước 56**. RecAdam nhân gradient task đích với λ(t) và λ được hiệu chỉnh theo
`total_steps`; ở bước 56 λ vẫn còn nhỏ nên model bị **giữ chặt tại nghiệm source**. Với 456
dòng có 870 bước nên model kịp thoát ra.

→ Kết quả âm ở chế độ ít dữ liệu là phát hiện về **optimizer**, không phải về transfer.
Đang chạy `--phase2_optimizer adamw` để kiểm chứng: nếu AdamW lấy lại delta dương thì lịch
annealing là thủ phạm.

Liên hệ với quan sát gốc ở `RESEARCH_2026-08-20_0959.md` §3: λ(t) bóp learning rate hiệu dụng
giai đoạn đầu, và không tài liệu nào nêu điều này. Đây là ca cụ thể đo được tác hại.

---

## 14. Nhật ký sự cố, phần 2

| Sự cố | Ảnh hưởng | Xử lý |
| --- | --- | --- |
| LoRA inject sau `.to(device)` | Tham số LoRA ở CPU, model ở GPU → crash forward. Cả rank 8 và 32 chết, không sinh fold nào | Gọi `model.to(device)` sau inject. Smoke test cục bộ chạy CPU nên **không bắt được** — kiểm thử placement bắt buộc phải trên GPU thật |
| `num_cwes` không truyền sang Phase 2 | Phase 1 xây 120 lớp, Phase 2 dựng lại với mặc định 4 → `size mismatch` ở **cả 10 fold**. Log trông bình thường, chỉ phát hiện vì bảng kết quả trống | Đọc `num_cwes` **từ checkpoint** thay vì từ cờ dòng lệnh |
| Sentinel sót lại gây chạy song song | Lần PrimeVul hỏng vẫn chạy tới cuối và tạo `PV_DONE`; chain kế tiếp thấy file đó nên khởi động ngay, hai job giành GPU, còn 1113 MiB trống | Điều kiện chờ kiểm tra **cả sentinel lẫn `pgrep` tiến trình** |

**Mẫu hình chung của ba lỗi đầu tiên** (`--output_dir`, `${MODES:-}`, `num_cwes`): một tham số
được set ở một chỗ nhưng phase khác không thấy, script **chạy trót lọt và log trông bình thường**
nhưng không sinh dữ liệu. Cách phát hiện duy nhất là đối chiếu "log nói gì" với "có bao nhiêu
file kết quả thật".

---

## 15. Bốn giả thuyết đã bị bác bỏ

Giá trị lớn nhất của phiên này là thu hẹp không gian tìm kiếm. Bốn giải thích hiển nhiên nhất
cho hiện tượng đã được kiểm chứng và loại bỏ.

| Giả thuyết | Cách kiểm chứng | Kết quả |
| --- | --- | --- |
| Rò rỉ cặp tạo ra hiệu ứng transfer | dựng lại folds group-aware | **Sai** — hiệu ứng còn tăng: +0.029 → +0.042 |
| `latent_proto` thật sự vượt trội | đo lại trên folds sạch | **Sai** — +0.064 → +0.025, PR-AUC vẫn âm |
| Phase 1 làm méo đặc trưng backbone mạnh | LP-FT, nội suy α, LoRA | **Sai** — cả ba độc lập đều thất bại |
| Bottleneck cứu được taxonomy lớn | PrimeVul 121 CWE, 5 fold | **Sai** — `cwe` +0.004, `bottleneck` −0.017 |
| RecAdam gây hại khi target ít dữ liệu | ablation AdamW, cùng source ckpt | **Sai phần lớn** — chỉ giải thích 20% |

### Chi tiết ablation optimizer (114 dòng, n=3)

| | fold1 | fold2 | fold3 | mean | Δ |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.6818 | 0.6536 | 0.7150 | 0.6835 | — |
| transfer + RecAdam | 0.5260 | 0.5163 | 0.6920 | 0.5781 | −0.1054 |
| transfer + AdamW | 0.6455 | 0.5162 | 0.6371 | 0.5996 | −0.0839 |

Bỏ hẳn cơ chế neo của RecAdam chỉ lấy lại **0.021 trên 0.105**. Bốn phần năm thiệt hại vẫn còn,
nên **transfer tự nó có hại ở chế độ ít dữ liệu**, không phải do optimizer.

Bằng chứng gián tiếp trước đó (xác suất bị nén vào [0.32, 0.83], std 0.116 so với 0.431) trông
rất thuyết phục nhưng chỉ mô tả **triệu chứng**, không xác định được **nguyên nhân**. Đây là bài
học đáng ghi: một cơ chế nghe hợp lý và khớp với triệu chứng vẫn cần ablation trực tiếp.

### Ba can thiệp pretrained-agnostic (CodeT5+, folds twin)

| Can thiệp | Cách tác động | Δ | n |
| --- | --- | --- | --- |
| *(không can thiệp)* | — | −0.0176 | 3 |
| LP-FT | sửa thứ tự huấn luyện Phase 2 | −0.0198 | 5 |
| Nội suy α | sửa trọng số sau Phase 1 | −0.028 … −0.037 | 3 |
| LoRA r=8 | đóng băng 99.73% backbone ở Phase 1 | −0.0175 | 3 |

LoRA là phép thử sạch nhất — Phase 1 gần như không được chạm vào backbone — và cho kết quả
**trùng khít** với không can thiệp. LoRA r=32 đã **cắt** vì cùng cơ chế, chỉ khác cường độ.

---

## 16. Điều còn đứng vững

1. **Task phụ tạo ra khả năng phân biệt.** `none` có A12 = 0.48 — đúng bằng tung đồng xu.
   Ablation này chưa từng xuất hiện trong y văn.
2. **Hiệu ứng đổi dấu theo backbone, không phải suy giảm dần.** Baseline 0.7821 → +0.029;
   0.8315 → −0.027; 0.8419 → −0.036. Không can thiệp nào đảo ngược được. Xem §18 — cách
   phát biểu "đơn điệu theo độ mạnh" ở các bản trước là **sai bản chất**.
3. **Chọn source quan trọng hơn quy mô source.** Cùng backbone, cùng folds, cùng baseline:
   ccpp+js 1284 dòng cho **+0.042**; PrimeVul 9408 dòng cho **+0.004**. Gấp bảy lần dữ liệu
   nhưng lệch miền thì vô ích, và không kiến trúc head nào bù được.

Điểm 3 gợi hướng đi khác hẳn câu hỏi ban đầu: thay vì tinh chỉnh head phụ, câu hỏi đáng theo
đuổi là **chọn source thế nào cho khớp target**. Y văn đã có công cụ — task embedding của
Vu et al. (EMNLP 2020) và ước lượng transferability của Poth et al. (EMNLP 2021), xem
`RESEARCH_2026-08-20_0959.md` §2.

---

## 17. Kỷ luật thống kê rút ra từ phiên

Bốn lần trong phiên tôi diễn giải một tín hiệu ở n≤2 và bốn lần phải rút lại:

| Tín hiệu ở n nhỏ | Sự thật ở n≥3 |
| --- | --- |
| `latent_proto` +0.0635 | +0.0254 khi hết rò rỉ |
| PrimeVul: bottleneck hơn cwe 0.065 (n=1) | khoảng cách biến mất |
| PrimeVul: bottleneck dương cả 2 fold (n=2) | âm ở n=4 |
| LoRA −0.0032 (n=2) | −0.0175 ở n=3, bằng không can thiệp |

Với độ lệch chuẩn giữa các fold tới **0.09** trên tập test 152 mẫu, một hoặc hai fold gần như
không mang thông tin. Kèm cảnh báo về cỡ mẫu **không cứu được** một kết luận sai.

**Quy tắc áp dụng từ giờ: không phát biểu nhận định nào dưới n=3.**

---

## 18. Đọc lại quan hệ với backbone: đổi dấu, không phải suy giảm

### 18.1 Giả thuyết trần bị bác

Cách phát biểu cũ — "lợi ích giảm dần khi backbone mạnh lên" — hàm ý một **hiệu ứng trần**:
backbone mạnh đã giải quyết gần hết bài toán nên task phụ không còn gì để thêm. Nếu đúng vậy,
chuẩn hóa lợi ích theo phần lỗi còn lại (`Δ / (1 − baseline)`) phải làm các backbone hội tụ về
nhau. Đo bằng `src/backbone_headroom.py` trên toàn bộ kết quả đã có:

| Backbone | n | baseline | transfer | Δ | dư địa | % lỗi còn lại được lấy đi |
| --- | --- | --- | --- | --- | --- | --- |
| CodeBERT (folds gốc) | 5 | 0.7821 | 0.8109 | **+0.0289** | 0.2179 | **+12.8%** |
| CodeBERT (folds twin) | 5 | 0.8281 | 0.8698 | **+0.0417** | 0.1719 | **+24.6%** |
| CodeT5 | 5 | 0.8315 | 0.8045 | **−0.0270** | 0.1685 | **−17.6%** |
| CodeT5+ | 5 | 0.8419 | 0.8059 | **−0.0361** | 0.1581 | **−25.4%** |
| CodeT5+ (folds twin) | 5 | 0.8879 | 0.8696 | **−0.0184** | 0.1121 | **−17.1%** |

Chuẩn hóa làm **spread rộng ra**, không thu hẹp: 0.0777 → 0.4994. Giả thuyết trần bị bác.

Quan trọng hơn, bảng này cho thấy điều mà cách phát biểu cũ che mất: lợi ích **không suy giảm
dần về 0** mà **đổi dấu**. Trên T5 và T5+, transfer không phải là vô ích — nó **có hại**, ổn
định ở cả ba lần đo, trên cả folds gốc lẫn folds twin. Một hiệu ứng trần không tạo ra dấu âm.
Đây là hai hiện tượng khác nhau và tôi đã gộp nhầm chúng suốt nhiều mục trước.

### 18.2 Biến gây nhiễu tôi tự tạo ra

CodeBERT và họ T5 **chưa bao giờ được chạy cùng một cách đọc biểu diễn**. CodeBERT pool từ CLS;
T5 không có token cấp câu ở vị trí 0 nên các lần chạy đó pool bằng trung bình có mask. Đó là
một lựa chọn trong repo này, không phải thuộc tính của checkpoint — và nó **biến thiên cùng
lúc** với backbone trong mọi lần chạy đã làm. Không lần nào tách được hai biến.

Cơ chế hợp lý: bằng chứng của một lỗ hổng nằm ở vài dòng, còn trung bình có mask chia đều tín
hiệu đó cho độ dài hàm. Một task phụ định hình backbone sẽ được CLS giữ lại nhưng bị trung bình
làm loãng.

`run/pooling.sh` chạy CodeBERT với `POOLING=mean`, giữ nguyên mọi thứ khác. `--pooling` nằm
trong khối `SHARED` của driver nên baseline cũng được huấn luyện lại cùng kiểu pooling.

- Δ vẫn dương → pooling không phải cơ chế, khác biệt thật sự nằm ở backbone.
- Δ đổi dấu âm → **cách đọc biểu diễn mới là cơ chế**, và hướng sửa cho T5 là cấp cho nó một
  slot tổng hợp, chứ không phải bảo vệ trọng số — hướng đó đã thất bại ba lần.

### 18.3 Truncation: cơ chế được đo, không được giả định

PrimeVul transfer kém hơn một corpus nhỏ hơn bảy lần. Một giải thích là 70% hàm vượt giới hạn
512 token, làm mất đúng những dòng phân biệt bản lỗi với bản vá, khiến hai nửa của một cặp
tokenize gần như giống hệt nhau dưới hai nhãn ngược nhau.

Đó là một cơ chế, và cơ chế thì **đo được**. `src/pair_collapse.py` tokenize từng cặp đúng
chiến lược đang dùng (`head_middle_tail`, 512) rồi đếm số cặp trùng khít:

| Corpus | số cặp | trùng khít | tỉ lệ |
| --- | --- | --- | --- |
| `train_ccpp_js` | 596 | 8 | **1.3%** |
| `ccpp_primevul_paired_common` | 600 | 11 | **1.8%** |
| `ccpp_primevul_fit512` | 600 | 0 | **0.0%** |

**1.8% không giải thích được** một Phase 1 gần như đoán ngẫu nhiên (ma trận nhầm lẫn
`[[131, 19], [127, 23]]`, std xác suất 0.0338). `head_middle_tail` giữ ba cửa sổ đầu–giữa–cuối
nên hầu như luôn còn khác biệt nào đó sót lại. **Dạng mạnh của giả thuyết truncation bị bác.**

`fit512` vẫn chạy nhưng để kiểm chứng dạng yếu hơn: dù hai bản còn khác nhau sau cắt, dòng
**thật sự quyết định** có thể đã mất trong khi một khác biệt vô nghĩa còn lại — model thấy khác
biệt nhưng không phải khác biệt đúng. Vì thế `fit512` bị xếp **sau** `pooling.sh` trong hàng đợi.

Ghi lại vì đây là điểm khác biệt về quy trình: lần này giả thuyết bị bác **trước** khi tiêu GPU,
nhờ hỏi "cơ chế tôi hình dung có thật sự xảy ra không?" thay vì chạy thẳng thí nghiệm. Bốn lần
trước trong §15 đều phải trả giá bằng nhiều giờ GPU mới biết mình sai.

---

## 19. Phase 1 có tái lập được không? — nghi vấn làm lung lay mọi con số phía trên

### 19.1 Bằng chứng

Đọc metadata của mọi checkpoint Phase 1 (`best_val_macro_f1` + vân tay MD5 của tensor đầu tiên)
cho ra một bảng không thể bỏ qua:

| run | nhánh | val Macro-F1 nguồn | best epoch | vân tay trọng số |
| --- | --- | --- | --- | --- |
| `frac114_codebert-base` | `transfer_cwe` | **0.5450** | 3 | `8bd16812` |
| `frac228_codebert-base` | `transfer_cwe` | **0.6654** | 13 | `73f40a0a` |
| `twin_ccppjs` | `transfer_cwe` (s36) | 0.6322 | 13 | — |
| `t5p_twin` | `cwe`, `+a025`, `+a050`, `+a075`, `+lpft` | 0.6815 | 7 | `e8b0c7c9` |
| `t5p_twin` | `+lora8`, `+lora32` | **0.5281** | 5 | `0addf22b` |

Hai dòng đầu có **tham số Phase 1 giống hệt nhau**: cùng `data/train_ccpp_js.jsonl`, cùng seed 36,
15 epoch, lr 2e-5, `lora_rank=0`, `lp_epochs=0`, `source_interpolation=1.0`,
`max_train_samples=None`. Chỉ khác `run_name`, thứ không đi vào huấn luyện. Chúng cho **hai bộ
trọng số khác nhau**, lệch **0.12** trên val nguồn.

`src/train_transfer.py` đã đặt `manual_seed`, `cuda.manual_seed_all`,
`cudnn.deterministic=True`, `cudnn.benchmark=False`, `num_workers=0`, seed cả `random`, `numpy`,
generator của DataLoader, và `train_test_split` cũng nhận `random_state=seed`. Về nguyên tắc
phải tái lập.

### 19.2 Nhưng bằng chứng ngược mạnh hơn — và nó bác lại chính tôi

Phản xạ đầu tiên của tôi là kết luận "Phase 1 không tái lập". Đó lại đúng là kiểu suy luận ở
n=1 mà §17 đã cam kết không lặp lại. Xem toàn bộ 8 checkpoint Phase 1 của CodeBERT trên cùng
source `train_ccpp_js`:

| run | val Macro-F1 nguồn |
| --- | --- |
| `frac228` cwe s36 | 0.6654 |
| `twin_ccppjs` latent_bottleneck s36 | 0.6439 |
| `twin_ccppjs` none s36 | 0.6423 |
| `twin_ccppjs` none s12 | 0.6360 |
| `twin_ccppjs` cwe s36 | 0.6322 |
| `twin_ccppjs` cwe s12 | 0.6277 |
| `twin_ccppjs` latent_proto s36 | 0.5986 |
| `frac114` cwe s36 | **0.5450** |

sd toàn bộ = 0.0369. **Bỏ riêng `frac114` thì sd tụt còn 0.0202**, dải 0.5986–0.6654.

Hai phép so sánh quyết định:

| | lệch |
| --- | --- |
| cùng config, **khác seed** (cwe s36 vs s12) | **0.0045** |
| cùng config, **khác seed** (none s36 vs s12) | **0.0063** |
| cùng config, **cùng seed** (frac228 vs frac114) | **0.1204** |

Đây là hình dạng **ngược** với thứ nhiễu ngẫu nhiên tạo ra. Nếu Phase 1 thật sự bất định, đổi
seed phải làm nó dịch chuyển ít nhất bằng khi giữ nguyên seed. Thực tế đổi seed gần như **không
làm gì** (0.0045 và 0.0063), trong khi đúng một cặp cùng seed lệch 0.12.

Cách đọc đúng: **`frac114` là một lần chạy hỏng**, không phải bằng chứng về bất định chung.
Ứng viên khả dĩ nhất là sự cố tranh chấp GPU đã ghi ở §14 — có lúc GPU chỉ còn 1113 MiB trống vì
hai tiến trình chạy đè lên nhau.

Hệ quả kéo theo cũng phải sửa: vì đổi seed hầu như không dịch chuyển Phase 1, **đa seed chủ yếu
lấy mẫu lại Phase 2 và cách chia fold, chứ không lấy mẫu lại model nguồn.** Lập luận "đa seed là
cách duy nhất để lấy mẫu lại Phase 1" ở bản nháp trước là **sai**.

### 19.3 Vấn đề thiết kế vẫn còn, và nó độc lập với tính tái lập

Kể cả khi `frac114` chỉ là một lần chạy hỏng, một vấn đề thiết kế **vẫn còn nguyên** và nó
không liên quan gì đến tính tái lập: cả 5 fold **dùng chung đúng một checkpoint Phase 1**. Nghĩa
là 5 fold là **5 phép đo trên cùng một model nguồn**, không phải 5 mẫu độc lập của phương pháp.
Kiểm định ghép cặp theo fold ở §17 vì thế chỉ đo nhiễu của **cách chia fold**, và §19.2 vừa cho
thấy đa seed cũng không lấy mẫu lại được model nguồn.

Điều này không làm các Δ đã đo sai đi, nhưng nó giới hạn phát biểu được phép rút ra: chúng là
"với **model nguồn này**, phương pháp cho +0.042 trên 5 fold", chưa phải "phương pháp cho +0.042".
Muốn phát biểu thứ hai thì phải lấy trung bình trên **nhiều model nguồn độc lập** — điều chưa
thí nghiệm nào trong tài liệu này làm.

Còn dải 0.5986–0.6654 (sd 0.0202) giữa các Phase 1 hợp lệ là con số cần đối chiếu: nó **cùng
bậc** với hiệu ứng 0.042 đang đo.

### 19.4 Một kết luận cũ phải sửa

Bảng vân tay cho thấy `+a025/a050/a075/+lpft` **dùng chung** đúng checkpoint `e8b0c7c9` với nhánh
`cwe` gốc. Ba so sánh đó **sạch**.

Nhưng `+lora8` và `+lora32` huấn luyện Phase 1 **riêng** và rơi vào một checkpoint kém hơn hẳn:
val nguồn 0.5281 so với 0.6815. §15 kết luận "LoRA là phép thử sạch nhất — Phase 1 gần như không
chạm vào backbone — và cho kết quả trùng khít với không can thiệp". Câu đó **nói quá**. Ở nhánh
LoRA có **hai thứ đổi cùng lúc**: ràng buộc lên Phase 1, *và* chất lượng model nguồn thu được.
LoRA đánh đổi "ít làm hỏng backbone" lấy "model nguồn tệ hơn", và kết quả ròng bằng không can
thiệp. Kết luận "bảo vệ trọng số không cứu được T5" vẫn đứng, nhưng nó dựa vào LP-FT và nội suy —
hai nhánh thật sự chia sẻ Phase 1 — chứ **không** dựa vào LoRA.

### 19.5 Phép thử

`run/determinism.sh` trả lời **hai** câu hỏi tách biệt, và câu thứ hai mới là câu quan trọng.

**Ba lần chạy ở cùng seed 36** — một con số đã báo cáo có lấy lại được không:

- vân tay giống nhau → Phase 1 tái lập được, `frac114` đúng là lần chạy hỏng như §19.2 dự đoán,
  và mọi bảng phía trên giữ nguyên hiệu lực.
- vân tay khác nhau → seed không ghim được Phase 1, và mọi nhánh bắt buộc phải dùng chung một
  checkpoint Phase 1 thay vì tự huấn luyện.

**Một lần chạy ở mỗi seed 7, 12, 18** — các model nguồn độc lập cách nhau bao xa. Đây là con số
§19.3 cần: nếu độ lệch chuẩn giữa các lần rút cùng bậc với 0.042 thì hiệu ứng đang báo cáo nằm
trong nhiễu của việc chọn model nguồn, và mọi kết luận phải phát biểu kèm điều kiện đó.

Hai bằng chứng hiện có mâu thuẫn nhau về câu thứ hai: hai cặp cùng-config-khác-seed chỉ lệch
0.0045 và 0.0063 (rất ổn định), nhưng toàn bộ 8 checkpoint hợp lệ trải trên sd 0.0202 (cùng bậc
với hiệu ứng). Ba lần rút có chủ đích sẽ phân xử.
