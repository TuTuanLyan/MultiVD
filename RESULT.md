# Kết quả thí nghiệm

Tài liệu này ghi lại các thí nghiệm đã chạy và số liệu cụ thể, để tra cứu về sau.
Số thô nằm ở `results_vast/`, log ở `logs_vast/` (không track trong git).

Mọi so sánh trong tài liệu này chỉ hợp lệ **trong cùng một bảng**. Xem mục
[Cảnh báo so sánh chéo](#cảnh-báo-so-sánh-chéo-máy) trước khi ghép số từ hai bảng khác nhau.

---

## 0. Trạng thái bằng chứng — đọc mục này trước

Tài liệu dài 26 mục vì nó ghi cả những thứ đã bị bác. Bảng này là bản đồ.

### Đã đứng vững

| Phát biểu | Bằng chứng | Mục |
| --- | --- | --- |
| **Head phụ latent tốt hơn head CWE tường minh** | `latent_bottleneck` là nhánh **duy nhất** vượt 0.05 trên cả hai kiểm định, và vá được metric xếp hạng mà `cwe` trượt | **§27** |
| Transfer thắng baseline ở **điểm vận hành 0.5** | 10/10 fold ghép cặp, 2 seed, Wilcoxon p = 0.0020 | §20 |
| Lợi ích **không phụ thuộc lần rút Phase 1** | 5/5 lần rút dương, sd 0.0085 so với sd 0.0521 của val nguồn | §26 |
| Lợi ích **tập trung ở lớp CWE hiếm** | CWE-022 +0.214 và CWE-079 +0.179, mỗi lớp 9/10 fold | §23 |
| Head phụ **gỡ bỏ thiệt hại** mà pretrain trần gây ra | λ=0 âm trên lớp phổ biến (−0.026, 1/9 fold dương), `cwe` đưa về +0.004 | §23.2 |
| **Quy mô source không mua được gì** | 9408 dòng → +0.004; 1284 dòng → +0.042 | §21 |
| **Phase 1 không tái lập** kể cả khi cố định seed | 3 lần chạy cùng seed → 3 trọng số khác nhau, sd 0.0521 ≈ sd khi đổi seed | §19.3 |

### Chưa xác lập

| Câu hỏi | Tình trạng | Mục |
| --- | --- | --- |
| **Độ lớn trên metric xếp hạng** | với `cwe`: ROC-AUC và PR-AUC đều p = 0.084. `latent_bottleneck` vượt được ROC-AUC (p = 0.0059) nhưng PR-AUC vẫn 0.0645 | §20.2, §27 |
| Lọc CWE có cứu được PrimeVul không | hướng nhất quán ~+0.026 qua hai head phụ, nhưng p = 0.31–0.44 | §21 |
| Phương pháp có tốt nhất trên backbone mạnh không | CodeT5+ dưới cls đạt 0.8800, vẫn thua baseline mean-pool 0.8823 | §22.2 |
| `latent_bottleneck` có bền trước nhiễu lần rút Phase 1 không | `cwe` đã kiểm chứng 5/5 lần rút (§26); nhánh latent **chưa** — `run/latent-draws.sh` đang trong hàng đợi | §27.3 |

### Đã bị bác — chín giả thuyết

Rò rỉ cặp tạo ra hiệu ứng (§15) · Phase 1 làm méo backbone mạnh (§15) · Bottleneck cứu taxonomy
lớn (§15) · RecAdam gây hại khi ít dữ liệu (§15) · Truncation làm hỏng cặp, đo được chỉ 1.8%
(§18.3) · Hiệu ứng trần giải thích quan hệ backbone (§18.1) · `common` không phân tách được gì
(§21.4) · `common` hơn `full` "gấp mười lần" (§21.1) · Chọn lần rút Phase 1 theo val nguồn (§26).

**Năm trong chín là dự đoán của chính tôi**, và ba lần tôi phải rút lại phát biểu đã viết vào tài
liệu này. Hai giả thuyết được ghi dự đoán bằng số **trước** khi chạy; một đã kiểm tra và trượt
(§26), một đang chờ (§22.3).

### Nguyên tắc đang áp dụng

1. Không phát biểu nhận định nào dưới **n=3**.
2. Hai nhánh dùng chung baseline thì **ghép cặp theo fold**, không so hai Δ trung bình.
3. Hai kiểm định bất đồng thì **báo cáo cả hai**; nhiều metric thì báo cáo hết, không chọn theo
   kết quả.
4. Trước khi tiêu GPU cho một giả thuyết, **đo thẳng cơ chế** mà nó giả định.
5. Khi có thể, **ghi dự đoán bằng số trước** khi chạy.

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
các lần chạy T5 pool bằng trung bình có mask. Lý do ghi trong `src/model.py` là "T5 không có
token cấp câu ở vị trí 0".

**Lý do đó sai.** Kiểm tra thẳng tokenizer:

| Model | lớp tokenizer | token ở vị trí 0 |
| --- | --- | --- |
| `microsoft/codebert-base` | `RobertaTokenizerFast` | `<s>` (id 0) |
| `Salesforce/codet5p-220m` | `RobertaTokenizerFast` | `<s>` (id 1) |
| `Salesforce/codet5-base` | `RobertaTokenizerFast` | `<s>` (id 1) |

Cả ba dùng **cùng một lớp tokenizer** và cùng phát ra `<s>` ở vị trí 0. Không model nào trong số
đó pretrain vị trí này thành biểu diễn cấp câu — RoBERTa cũng không — và ở cả hai họ thì chính
giai đoạn finetune mới dạy nó gộp thông tin. Nghĩa là pooling là **lựa chọn tự do cho cả hai
họ**, không phải thuộc tính của checkpoint.

Hậu quả: pooling **biến thiên cùng lúc** với backbone trong mọi lần chạy đã làm, nên chưa lần nào
tách được "CodeT5 transfer kém hơn" khỏi "mean pooling transfer kém hơn". Bảng 2×2 giữa
{CodeBERT, CodeT5+} × {cls, mean} mới chỉ có **đường chéo**.

Cơ chế hợp lý: bằng chứng của một lỗ hổng nằm ở vài dòng, còn trung bình có mask chia đều tín
hiệu đó cho độ dài hàm. Một task phụ định hình backbone sẽ được CLS giữ lại nhưng bị trung bình
làm loãng.

Hai script điền nốt hai ô còn lại. `run/pooling.sh` chạy CodeBERT với `POOLING=mean`;
`run/t5-cls.sh` chạy CodeT5+ với `POOLING=cls`. Mỗi script chỉ đổi đúng pooling, và vì
`--pooling` nằm trong khối `SHARED` của driver nên baseline cũng được huấn luyện lại cùng kiểu
pooling — so sánh vẫn nằm trong cùng điều kiện.

| | `cls` | `mean` |
| --- | --- | --- |
| **CodeBERT** | +0.0417 (đã có) | `run/pooling.sh` |
| **CodeT5+** | `run/t5-cls.sh` | −0.0184 (đã có) |

- Dấu bám theo **cột** → cách đọc biểu diễn là cơ chế, và phương pháp **không** phụ thuộc
  pretrained model. Cách sửa cho T5 chỉ là một dòng cấu hình, chứ không phải bảo vệ trọng số —
  hướng đó đã thất bại ba lần ở §15.
- Dấu bám theo **hàng** → backbone thật sự là biến quyết định, và ba can thiệp đã thử vẫn là
  toàn bộ những gì đã loại trừ được.

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

### 19.2 Một suy luận trung gian đã sai, ghi lại vì cách sai mới là bài học

Phản xạ đầu tiên của tôi là kết luận ngay "Phase 1 không tái lập". Đó đúng là kiểu suy luận ở
n=1 mà §17 cam kết không lặp lại, nên tôi đi tìm bằng chứng ngược và tưởng đã tìm được:

| | lệch |
| --- | --- |
| cùng config, **khác seed** (cwe s36 vs s12) | 0.0045 |
| cùng config, **khác seed** (none s36 vs s12) | 0.0063 |
| cùng config, **cùng seed** (frac228 vs frac114) | 0.1204 |

Tôi đọc bảng này là "đổi seed gần như không làm gì, nên `frac114` phải là một lần chạy hỏng" —
hình dạng ngược với thứ nhiễu ngẫu nhiên tạo ra. Lập luận nghe chặt, và **nó sai**.

Sai ở chỗ: hai con số 0.0045 và 0.0063 là **hai lần rút từ một phân phối rộng tình cờ rơi gần
nhau**. Tôi lại đọc chúng như bằng chứng phân phối đó hẹp. Đúng lỗi n nhỏ mà §17 liệt kê, chỉ
khác là lần này nó xảy ra **trong lúc tôi đang sửa một lỗi n nhỏ khác**.

### 19.3 Phép thử trực tiếp phân xử

`run/determinism.sh` chạy đúng một cấu hình Phase 1 ba lần ở seed 36, rồi một lần ở mỗi seed
7/12/18. Cùng máy, cùng file dữ liệu, cùng mọi tham số.

| lần chạy | best epoch | val Macro-F1 nguồn | vân tay trọng số |
| --- | --- | --- | --- |
| `same36_rep1` | **3** | **0.5722** | `b466d1bc` |
| `same36_rep2` | 13 | 0.6591 | `9aab83d6` |
| `same36_rep3` | 11 | 0.6654 | `b07829a7` |
| `draw_seed7` | 14 | 0.6495 | `e8f352d2` |
| `draw_seed12` | 15 | 0.5968 | `13f50b5f` |
| `draw_seed18` | 13 | 0.6959 | `18c3c847` |

**Ba lần chạy cùng seed cho ba bộ trọng số khác nhau.** Và con số quyết định:

| | mean | sd | dải |
| --- | --- | --- | --- |
| cùng seed 36, 3 lần chạy | 0.6322 | **0.0521** | 0.0932 |
| 3 seed khác nhau | 0.6474 | **0.0496** | 0.0991 |

Tỉ lệ sd = **1.05**. Chạy lại cùng một seed dao động **bằng đúng** đổi seed. Seed **không kiểm
soát được gì** ở Phase 1, dù `src/train_transfer.py` đã đặt `manual_seed`, `cuda.manual_seed_all`,
`cudnn.deterministic=True`, `cudnn.benchmark=False`, `num_workers=0`, seed cả `random`, `numpy`,
generator của DataLoader và `random_state` của `train_test_split`.

Kết luận ban đầu ở §19.1 đúng; phần rút lại ở §19.2 sai. Phép thử trực tiếp phân xử được thứ mà
khảo cổ metadata không phân xử nổi.

`frac114` cũng không phải ngoại lệ: `same36_rep1` dừng ở **epoch 3** với val 0.5722, gần như trùng
khít `frac114` (epoch 3, val 0.5450). Đây là **một chế độ hỏng lặp lại được**, không phải sự cố
một lần. Nghi phạm rõ nhất là early stopping khuếch đại sai số số học rất nhỏ: `patience=5`,
`min_epochs=3`, nên một chênh lệch cỡ 1e-7 ở epoch đầu đủ để một lần chạy dừng ở epoch 3 trong
khi lần khác đi tiếp tới epoch 13. Thiếu `torch.use_deterministic_algorithms(True)` và
`CUBLAS_WORKSPACE_CONFIG` là nguồn sai số số học khả dĩ nhất.

### 19.4 Hệ quả

**Hai nhánh nào tự huấn luyện Phase 1 riêng thì không so sánh trực tiếp được.** Khác biệt giữa
chúng có thể hoàn toàn là khác biệt giữa hai lần rút, sd 0.052. Áp dụng ngay cho §19.5.

**5 fold dùng chung một checkpoint Phase 1**, nên fold không hề lấy mẫu biến thiên này. Và §19.3
vừa cho thấy **đa seed cũng không** — đổi seed không lấy mẫu rộng hơn chạy lại cùng seed.

Điều này không làm các Δ đã đo sai, nhưng giới hạn phát biểu được phép rút ra: chúng là "với
**model nguồn này**, phương pháp cho +0.042 trên 5 fold". Riêng §20 khá hơn một bậc — seed 36 và
seed 12 có **hai lần rút Phase 1 khác nhau** và cả hai đều dương (+0.0417 và +0.0370), nên kết
quả 10/10 đã bắc qua 2 lần rút. Nhưng 2 vẫn là n nhỏ, và `run/source-draws.sh` nâng lên 4 lần rút
trải từ val 0.5722 đến 0.6959 — gần trọn dải quan sát được.

### 19.5 Một kết luận cũ phải sửa

Vân tay trọng số cho thấy `+a025`, `+a050`, `+a075` và `+lpft` **dùng chung** đúng checkpoint
`e8b0c7c9` với nhánh `cwe` gốc. Ba so sánh đó **sạch** — chúng chỉ khác nhau ở Phase 2.

Nhưng `+lora8` và `+lora32` huấn luyện Phase 1 **riêng** và rơi vào checkpoint kém hơn hẳn: val
nguồn 0.5281 so với 0.6815. §15 kết luận "LoRA là phép thử sạch nhất — Phase 1 gần như không chạm
vào backbone — và cho kết quả trùng khít với không can thiệp". Câu đó **nói quá**. Ở nhánh LoRA có
hai thứ đổi cùng lúc: ràng buộc lên Phase 1, *và* lần rút Phase 1 thu được.

Và §19.3 làm nó nặng hơn nữa: khoảng cách 0.5281 → 0.6815 là **1.5 lần sd của nhiễu lần rút**
(0.052), nên hoàn toàn có thể chỉ là một lần rút xấu chứ không phải hệ quả của ràng buộc LoRA.
Kết luận "bảo vệ trọng số không cứu được T5" vẫn đứng, nhưng nó dựa vào LP-FT và nội suy — hai
nhánh thật sự chia sẻ Phase 1 — chứ **không** dựa vào LoRA. Muốn dùng LoRA làm bằng chứng thì
phải chạy lại nó nhiều lần rút.

---

## 20. Gộp hai seed: lần đầu vượt ngưỡng, và điều ngưỡng đó **không** nói

Seed 12 đã xong cả 5 fold, gộp với 5 fold của seed 36 thành **10 quan sát ghép cặp**. `paired_stats.py`
ghép mỗi fold với baseline **của chính seed đó**, nên dù seed 36 chạy trên `ntat` và seed 12 trên
`ntat2`, từng Δ vẫn là so sánh trong cùng một máy.

| Nhánh | n | Δ mean | Δ sd | A12 | Wilcoxon p | t (Nadeau–Bengio) |
| --- | --- | --- | --- | --- | --- | --- |
| `transfer_cwe` | **10** | **+0.0393** | 0.0286 | **0.86** | **0.0020** | +2.09 |
| `transfer_latent_bottleneck` | 6 | +0.0245 | 0.0184 | 0.83 | 0.0312 | +1.88 |
| `transfer_latent_proto` | 6 | +0.0298 | 0.0356 | 0.78 | 0.0625 | +1.18 |
| `transfer_none` | 10 | +0.0080 | 0.0269 | 0.62 | 0.3750 | +0.45 |

Seed 12 mới xong một vài fold cho hai nhánh latent, nên n của chúng chủ yếu vẫn là seed 36 cộng
thêm. Chưa đủ để phát biểu; `run/seed12-latent.sh` đang chạy nốt.

**Chốt chặn tính hợp lệ của phép gộp.** Trước khi gộp hai seed cho nhánh latent, tôi đối chiếu
`training_args` trong checkpoint Phase 1 của cả hai. Mọi giá trị thực chất trùng khít:
`num_latent=8`, `latent_temperature=0.1`, `freeze_prototypes_steps=0`, `lambda_cwe=0.2`,
`pooling=cls`, `lr=2e-5`, 15 epoch, `patience=5`, `min_epochs=3`, batch 16, `max_length=512`,
`head_middle_tail`, `selection_metric=macro_f1`. Chỗ khác biệt duy nhất là seed 36 ghi `None` ở
`cwe_vocab`, `phase2_optimizer`, `lora_rank`, `source_interpolation`, `lp_epochs` — vì các cờ đó
**chưa tồn tại** khi nó chạy — trong khi seed 12 ghi đúng giá trị mặc định tương đương
(`fixed4`, `recadam`, `0`, `1.0`, `0`). Hành vi như nhau, chỉ khác việc có được ghi lại hay không.

Ghi lại vì `None` trong metadata rất dễ bị đọc nhầm thành "cấu hình khác", và vì kiểm tra này mất
hai phút trong khi phát hiện muộn sẽ làm hỏng cả một bảng kết quả — đúng như đã xảy ra với LoRA ở
§19.5.

Lưu ý thêm, đúng theo bài học §19.2: val Phase 1 của hai seed rất gần nhau (`cwe` 0.6322 và
0.6277; `latent_proto` 0.5986 và 0.5974). **Không được** đọc đó là bằng chứng Phase 1 ổn định —
§19.3 đã cho thấy chạy lại cùng seed lệch tới 0.09. Đây chỉ là hai lần rút tình cờ rơi gần nhau,
đúng cái bẫy đã sập một lần.

Δ từng fold, seed 36 rồi seed 12:

```
transfer_cwe   +0.025 +0.000 +0.046 +0.034 +0.080  +0.059 +0.027 +0.007 +0.028 +0.087  -> 10/10 duong
transfer_none  +0.032 +0.033 -0.020 +0.001 +0.026  -0.033 -0.006 -0.019 +0.020         ->  5/9  duong
```

Riêng mỗi seed: seed 36 cho +0.0417, seed 12 cho +0.0370. Hai máy khác nhau, hai lần rút Phase 1
khác nhau, lệch nhau 0.0047.

### 20.1 Điều đã đạt được

Sàn Wilcoxon tụt từ 0.0625 (n=5, không bao giờ với tới 0.05) xuống 0.0020, và `transfer_cwe`
**chạm đúng sàn đó**: p = 0.0020 là giá trị nhỏ nhất n=10 có thể cho, đạt được khi và chỉ khi
phương pháp thắng ở **mọi** quan sát. `transfer_cwe` dương ở cả **10/10 fold**, trên 2 seed và
2 máy; `transfer_none` chỉ 5/9 và Δ gần 0.

Đây là bằng chứng mạnh nhất cho phát biểu đã đứng vững lâu nhất trong dự án: **head phụ mới là
thứ tạo ra lợi ích, không phải bản thân việc pretrain trên source rồi RecAdam.** Ablation λ=0
giờ đã có n=9 và vẫn phẳng.

### 20.2 Trên metric độc lập ngưỡng thì yếu hơn hẳn

Macro-F1@0.5 là đại lượng **rời rạc** trên tập test 152 mẫu. Điều này không trừu tượng: hai model
Phase-2 khác hẳn nhau — ROC-AUC 0.9430 so với 0.9548, precision 0.843 so với 0.901, recall 0.909
so với 0.831 — vẫn cho **đúng cùng một** Macro-F1@0.5 = 0.8699, vì hai kiểu đánh đổi lỗi tình cờ
bù trừ về cùng một ma trận nhầm lẫn. Một metric có thể trùng khít giữa hai model khác nhau như vậy
thì phải kiểm tra kết luận trên metric liên tục.

Cùng 10 quan sát ghép cặp, ba metric:

| Metric | Δ `transfer_cwe` | A12 | Wilcoxon p | Δ `transfer_none` |
| --- | --- | --- | --- | --- |
| Macro-F1@0.5 | **+0.0393** | 0.86 | **0.0020** | +0.0080 |
| ROC-AUC | +0.0203 | 0.65 | **0.0840** | −0.0072 |
| PR-AUC | +0.0213 | 0.64 | **0.0840** | −0.0191 |

Một điều đáng theo dõi, **chưa phải kết luận**: trên ROC-AUC, `latent_bottleneck` cho Δ +0.0185 —
gần bằng `cwe` (+0.0203) — nhưng độ lệch chuẩn chỉ **0.0086** so với 0.0286, tức **nhỏ hơn ba lần**.
Kéo theo t hiệu chỉnh **+3.02** (p ≈ 0.029), giá trị cao nhất từng thấy trong dự án, và Wilcoxon
0.0312. Nếu điều này còn đứng vững khi seed 12 xong đủ 5 fold thì nó đáng chú ý: nhánh **không bị
khoá vào 4 CWE** lại là nhánh **ổn định nhất**. Nhưng n hiện tại là 6, trong đó 5 đến từ một seed,
nên đúng theo §17 thì chưa được phát biểu gì.

**Trên metric xếp hạng, hiệu ứng chỉ bằng khoảng một nửa và không vượt 0.05.**

Điều này không xoá kết quả, nhưng thu hẹp nó đáng kể. Phát biểu được phép đưa ra:

- Ở **điểm vận hành 0.5**, phương pháp thắng ở 10/10 fold — dấu rất chắc.
- Về **khả năng phân biệt tổng thể**, lợi ích khoảng +0.020 và **chưa đạt ý nghĩa thống kê** ở n=10.
- Ablation λ=0 **âm** trên cả hai metric xếp hạng, khớp với kết luận §2: head phụ là thứ tạo ra
  khả năng phân biệt, còn pretrain + RecAdam trần thì làm hỏng nó.

Chỉ báo cáo dòng Macro-F1 mà giấu hai dòng dưới sẽ là **chọn metric theo kết quả** — đúng lỗi mà
§20.3 đã cảnh báo với việc chọn kiểm định, chỉ khác trục.

### 20.3 Điều ngưỡng đó không nói

**Kiểm định t hiệu chỉnh Nadeau–Bengio cho p = 0.066, vẫn chưa vượt 0.05.** Hai kiểm định không
mâu thuẫn — chúng trả lời hai câu khác nhau:

- Wilcoxon/sign test hỏi *"phương pháp có gần như luôn thắng không?"* → có, 10/10, p = 0.0020.
- t hiệu chỉnh hỏi *"độ lớn của lợi ích có được xác định chắc chắn không, sau khi tính đến việc
  các fold dùng chung dữ liệu huấn luyện?"* → chưa, p = 0.066.

Thêm dữ liệu kéo p của t từ 0.109 xuống 0.066, nên khoảng cách đang thu hẹp chứ không phải bế
tắc. Nhưng báo cáo p = 0.0020 mà giấu p = 0.066 sẽ là chọn kiểm định theo kết quả. Phát biểu
đúng là: **dấu của hiệu ứng đã chắc chắn, độ lớn thì chưa.**

**Và cả 10 quan sát vẫn chỉ đến từ 2 model nguồn.** Theo §19.3, fold lấy mẫu lại cách chia target,
seed lấy mẫu lại Phase 2 — không cái nào lấy mẫu lại model nguồn. `run/source-draws.sh` giữ cố
định mọi thứ ở hạ nguồn và chỉ đổi checkpoint Phase 1; nó mới là thứ quyết định con số 0.0393
thuộc về **phương pháp** hay thuộc về **hai lần rút may mắn**.

---

## 21. Lớp CWE nào của source mới chuyển giao được — và tôi lại đoán sai

### 21.1 Con số, và vì sao cách đọc đầu tiên của tôi sai

Cùng corpus gốc PrimeVul, cùng CodeBERT, cùng folds twin, cùng baseline. Khác biệt duy nhất là
**bộ CWE nào được giữ lại**:

| Source | dòng | CWE | trùng CWE với target | Δ `transfer_cwe` |
| --- | --- | --- | --- | --- |
| `ccpp_primevul_paired_full` | 9408 | 121 | 4 CWE, 178 dòng (2%) | +0.0037 (n=5) |
| `ccpp_primevul_paired_common` | **2975** | 73 | 4 CWE, 178 dòng (6%) | +0.0307 (n=5) |
| `train_ccpp_js` | 1284 | 4 | 4 CWE, 1284 dòng (100%) | +0.0417 (n=5) |

Đọc bảng này rồi kết luận "xóa 6433 dòng làm transfer tốt lên gấp mười lần" là **sai phương pháp**,
và tôi đã viết đúng câu đó ở bản trước. Nó so **hai trung bình ở n khác nhau** (khi đó là n=3 với
n=5) trong khi `full` và `common` **dùng chung y hệt một baseline**, nên phép so ghép cặp theo
từng fold luôn sẵn có và chặt hơn hẳn.

Ghép cặp theo fold, đủ 5 fold:

| fold | baseline | `full` cwe | `common` cwe | common − full |
| --- | --- | --- | --- | --- |
| 1 | 0.8113 | **0.7532** | 0.8700 | **+0.1168** |
| 2 | 0.8157 | 0.8431 | 0.8348 | −0.0083 |
| 3 | 0.8150 | 0.8289 | 0.8486 | +0.0197 |
| 4 | 0.8654 | 0.8871 | 0.8542 | −0.0329 |
| 5 | 0.8333 | 0.8467 | 0.8865 | +0.0398 |

| nhánh phụ | 5 hiệu ghép cặp | dương | mean | sd | Wilcoxon p |
| --- | --- | --- | --- | --- | --- |
| `cwe` | +0.1168, −0.0083, +0.0197, −0.0329, +0.0398 | 3/5 | +0.0270 | 0.0573 | **0.4375** |
| `latent_bottleneck` | −0.0260, +0.0329, +0.0798, +0.0525, −0.0132 | 3/5 | +0.0252 | 0.0444 | **0.3125** |

Ba điều cùng lúc, không được bỏ điều nào:

**Hướng nhất quán.** Cả hai head phụ độc lập đều cho `common` hơn `full` khoảng **+0.026**, và
dấu sống sót qua phép bỏ-một-fold ở cả hai (`cwe`: +0.0046 … +0.0420; `latent_bottleneck`:
+0.0115 … +0.0380). Hai phép đo bán độc lập trùng hướng và trùng độ lớn.

**Không đạt ý nghĩa thống kê.** p = 0.4375 và 0.3125, trong khi sàn ở n=5 là 0.0625. Còn rất xa.

**Nhánh `cwe` dựa nhiều vào fold 1.** Bỏ fold 1 thì nó chỉ còn +0.0046. `latent_bottleneck` thì
không có điểm tựa đơn lẻ nào như vậy, nên nó mới là bằng chứng tốt hơn cho cùng một hướng.

Kết luận: khác biệt giữa hai bộ source **vẫn chưa được xác lập**, nhưng cũng không phải không có
gì — nó là một hướng nhất quán ở cỡ hiệu ứng khoảng +0.026 mà n=5 không đủ sức phân giải. Điều
đứng vững chắc chắn vẫn là: `full` với 9408 dòng cho +0.0037, còn `train_ccpp_js` với 1284 dòng
cho +0.0417 — **quy mô source không mua được gì**. Muốn xác lập phần còn lại thì cần nhiều lần
rút Phase 1, vì §19.3 cho thấy nhiễu lần rút có sd 0.052 — **lớn gấp đôi hiệu ứng đang tranh luận**.

### 21.2 Ba giải thích cho khoảng cách — nếu khoảng cách là thật

**Không phải quy mô.** Quy mô đi ngược chiều — corpus nhỏ hơn lại là corpus tốt hơn.

**Không phải chất lượng model nguồn.** Val Macro-F1 Phase 1 của `full` là 0.5329, của `common` là
0.5197 — gần như bằng nhau. Hai model nguồn học được ngang nhau trên task của chính chúng, nhưng
chuyển giao lệch nhau gấp mười lần. **Độ chính xác của Phase 1 không dự báo được giá trị transfer.**

**Cũng không phải trùng nhãn với target.** `common` chỉ trùng **6%** số dòng với 4 CWE của target,
gần bằng 2% của `full`, mà vẫn gần chạm nguồn trùng 100%. Đây là giả thuyết đầu tiên tôi định
đưa ra và số liệu bác ngay.

### 21.3 Điều thật sự phân biệt hai bộ

48 CWE bị loại khỏi `full` để thành `common`, xếp theo số dòng:

```
CWE-119 buffer overflow          1097     CWE-399 resource management       311
CWE-125 out-of-bounds read       1006     CWE-264 permissions               289
CWE-787 out-of-bounds write       818     CWE-189 numeric errors            269
CWE-476 NULL dereference          603     CWE-369 divide by zero            206
CWE-416 use-after-free            460     CWE-401/772 memory & resource leak 353
```

Gần như toàn bộ là **lớp bộ nhớ và quản lý tài nguyên thủ công — những thứ không tồn tại trong
Python.** Không có con trỏ, không có buffer, không có `free()`, có GC.

73 CWE được giữ lại:

```
CWE-020 input validation          745     CWE-022 path traversal             80
CWE-200 information exposure      482     CWE-078 command injection          50
CWE-703 unchecked exceptions      419     CWE-079 XSS                        40
CWE-190 integer overflow          341     CWE-059 link following             36
CWE-835 infinite loop             115     CWE-770 alloc without limits       32
```

Toàn bộ là **lớp logic và kiểm tra đầu vào, độc lập ngôn ngữ**, và đều tồn tại trong Python.

Giả thuyết tương ứng: điều quyết định không phải nhãn CWE có trùng target hay không, mà **kiểu
hỏng mà lớp CWE đó mô tả có tồn tại được trong ngôn ngữ target hay không**. CWE bộ nhớ dạy model
một khái niệm "thế nào là lỗ hổng" mà Python không thể biểu đạt.

Đây là một giả thuyết **hấp dẫn và chưa được kiểm chứng**. §21.1 cho thấy dữ liệu hiện có chưa đủ
sức phân biệt nó với nhiễu. Ghi lại ở đây vì nó gợi ra một thí nghiệm cụ thể — lọc source theo
tính khả chuyển khái niệm rồi đo trên nhiều lần rút Phase 1 — chứ không phải vì nó đã được chứng
minh.

### 21.4 Dự đoán sai thứ năm, và một lỗi phương pháp thứ sáu

Tôi đã báo rằng thí nghiệm `common` "nhiều khả năng **không phân tách được** hai giả thuyết"
vì Phase 1 của nó gần như đoán ngẫu nhiên (ma trận nhầm lẫn `[[131, 19], [127, 23]]`, std xác
suất 0.0338), và lý do đưa ra là "không có gì để chuyển giao". Nó lại cho hiệu ứng **lớn thứ hai
trong toàn dự án**.

Sai lầm nằm ở chỗ ngầm giả định **độ chính xác nhị phân của Phase 1 đo được lượng thứ có thể
chuyển giao**. §21.2 cho thấy nó không đo. Một model nguồn gần như đoán bừa trên task nhị phân
của nó vẫn có thể định hình biểu diễn theo cách hữu ích cho target — thứ được chuyển đi là hình
học đặc trưng do head phụ tạo ra, không phải năng lực phân loại của Phase 1.

Lỗi thứ sáu là của chính bản ghi này: tôi so hai trung bình ở n khác nhau khi phép so ghép cặp
đã sẵn có, rồi viết "gấp mười lần" vào tài liệu. Bài học lặp lại lần nữa — **khi hai nhánh dùng
chung baseline thì luôn ghép cặp theo fold trước, đừng bao giờ so hai Δ trung bình**, nhất là khi
n của hai bên khác nhau.

`run/common45.sh` đã chạy xong fold 5; số liệu ở §21.1 là bản đầy đủ.

---

## 22. Cách đọc biểu diễn, không phải backbone — nhưng chỉ đúng một nửa

### 22.1 Số liệu

CodeT5+ chạy lại với `POOLING=cls`, đọc từ token `<s>` ở vị trí 0 mà §18.2 chứng minh là **có tồn
tại** trong họ T5. Mọi thứ khác giữ nguyên; `--pooling` nằm trong `SHARED` nên baseline cũng được
huấn luyện lại cùng kiểu pooling.

| fold | base cls | transfer cls | Δ_cls | base mean | transfer mean | Δ_mean | Δ_cls − Δ_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.8896 | 0.8961 | +0.0065 | 0.8961 | 0.8766 | −0.0195 | **+0.0260** |
| 2 | 0.8823 | 0.8822 | −0.0001 | 0.8889 | 0.8625 | −0.0264 | **+0.0263** |
| 3 | 0.8421 | 0.8617 | +0.0196 | 0.8618 | 0.8549 | −0.0069 | **+0.0265** |

**Δ_mean = −0.0176 → Δ_cls = +0.0087. Dấu đổi.**

Hiệu Δ_cls − Δ_mean là **+0.0260, +0.0263, +0.0265** — sd **0.00025**, trên một bài toán mà độ lệch
chuẩn giữa các fold là 0.09. Hiệu của hai hiệu triệt tiêu độ khó của từng fold, đó chính là lý do
thiết kế ghép cặp có tác dụng, nhưng mức trùng khớp này thì hoặc là một hiệu ứng rất sạch, hoặc là
trùng hợp mà hai fold nữa sẽ phá vỡ. `run/t5cls45.sh` đã xếp hàng để đưa lên n=5.

### 22.2 Tầng thứ ba, làm dịu hai tầng trên

Nhìn giá trị **tuyệt đối** trên đúng ba fold đó:

| cấu hình | Macro-F1 |
| --- | --- |
| **baseline, mean pooling** | **0.8823** |
| transfer, cls pooling | 0.8800 |
| baseline, cls pooling | 0.8713 |
| transfer, mean pooling | 0.8647 |

Cấu hình tốt nhất **vẫn là không transfer gì, dùng mean pooling**. Đổi cách đọc làm transfer thôi
gây hại và bắt đầu có lợi **so với baseline của chính nó**, nhưng **không** làm nó vượt được
baseline tốt nhất hiện có cho CodeT5+ — còn kém 0.0023, tức hòa.

Phát biểu đúng cho mục tiêu pretrained-agnostic: **cách đọc biểu diễn giải thích được dấu âm, chứ
chưa biến phương pháp thành lựa chọn tốt nhất trên backbone mạnh.**

### 22.3 Cơ chế, và một dự đoán phát biểu **trước** khi chạy

Bóc tách cho thấy hai chiều ngược nhau: cls cho baseline **kém hơn** (0.8713 so với 0.8823) nhưng
transfer **tốt hơn** (0.8800 so với 0.8647). Nghĩa là **mean pooling tốt hơn cho finetune thường,
cls tốt hơn cho phương pháp transfer.**

Cơ chế khớp với thiết kế: head phụ đọc đúng vector đã pool. Với cls, nó có một **slot riêng** để
định hình. Với mean, muốn định hình vector pool thì phải định hình lại phân bố của **mọi token**,
và việc đó xung đột trực tiếp với task nhị phân.

Nếu cơ chế này đúng thì nó là tính chất của **cách đọc**, không phải của checkpoint, nên nó phải
lặp lại trên CodeBERT. Dự đoán cụ thể, ghi trước khi `run/pooling.sh` chạy:

> CodeBERT dưới mean pooling sẽ **mất khoảng 0.026** so với chính nó dưới cls, tức Δ rơi từ
> **+0.0417 xuống khoảng +0.016**, và **vẫn dương**.

- Rơi khoảng 0.026 → cơ chế "head phụ cần một slot riêng" đứng vững trên cả hai backbone.
- Rơi ít hơn nhiều, hoặc không rơi → hiệu ứng là đặc thù của CodeT5+, và §22.1 chỉ là trùng hợp
  ở n=3.
- Rơi tới mức âm → mean pooling phá phương pháp mạnh hơn nhiều so với ước lượng từ CodeT5+.

Ghi dự đoán trước là cách duy nhất để lần này không rơi vào đúng cái bẫy đã sập sáu lần: nhìn số
rồi mới dựng câu chuyện khớp với nó.

---

## 23. Lợi ích nằm ở đâu — và ablation λ=0 **không** phẳng như tôi vẫn nói

Kết quả per-CWE ở §3 dựa trên một seed và folds cũ. Giờ có đủ dữ liệu để làm lại ở **n=10**, ghép
cặp theo `(seed, fold)`, gộp seed 36 và 12 trên folds twin.

| CWE | mẫu test | baseline | `cwe` | Δ | số fold dương |
| --- | --- | --- | --- | --- | --- |
| **CWE-022** path traversal | 66 | 0.5019 | 0.7159 | **+0.2139** | **9/10** |
| **CWE-079** XSS | 82 | 0.5999 | 0.7791 | **+0.1793** | **9/10** |
| CWE-078 command injection | 204 | 0.7784 | 0.7856 | +0.0072 | 5/10 |
| CWE-089 SQL injection | 408 | 0.9447 | 0.9484 | +0.0037 | 5/10 |

**Toàn bộ lợi ích nằm ở hai lớp hiếm**, và ở đó nó rất lớn: +0.21 và +0.18, dương ở 9/10 fold cho
cả hai. Hai lớp này chỉ chiếm 148 trên 760 mẫu, nên chúng bị pha loãng thành +0.039 khi gộp — đó
là lý do con số tổng hợp khiêm tốn hơn nhiều so với tác dụng thật.

### 23.1 Ablation λ=0 không phẳng, nó **triệt tiêu**

Đây là phần buộc phải sửa. Suốt tài liệu này tôi mô tả `transfer_none` là "phẳng" vì Δ tổng hợp
+0.0037. Bóc theo lớp thì nó không phẳng chút nào:

| CWE | Δ `none` | số fold dương | Δ `cwe` |
| --- | --- | --- | --- |
| CWE-022 | **+0.1286** | 7/9 | +0.2139 |
| CWE-079 | **+0.1369** | 8/9 | +0.1793 |
| CWE-078 | **−0.0240** | 3/9 | +0.0072 |
| CWE-089 | **−0.0261** | **1/9** | +0.0037 |

`none` **có** cải thiện lớp hiếm (+0.13, +0.14) và **có** làm hỏng lớp phổ biến (−0.024, −0.026,
với CWE-089 chỉ dương ở 1/9 fold). Hai chiều triệt tiêu nhau nên tổng hợp ra gần 0. "Phẳng" là
một **artefact của phép gộp**, không phải mô tả đúng hành vi.

### 23.2 Phát biểu đúng về vai trò của head phụ

Bản cũ: *"head phụ tạo ra toàn bộ lợi ích, pretrain + RecAdam tự thân không làm gì"*. Bản đúng có
hai vế, và vế thứ hai mới là phần đáng kể:

1. **Pretrain trên source tự nó đã giúp lớp hiếm** — `none` cho +0.13 và +0.14 mà không cần bất kỳ
   nhãn CWE nào. Rất có thể chỉ là hiệu ứng thêm dữ liệu và khởi tạo tốt hơn.
2. **Head phụ làm hai việc mà pretrain trần không làm được**: nó cộng thêm vào lớp hiếm
   (+0.13 → +0.21 và +0.14 → +0.18), và quan trọng hơn, nó **gỡ bỏ thiệt hại trên lớp phổ biến**
   (−0.024 → +0.007 và −0.026 → +0.004).

Vế thứ hai giải thích luôn vì sao `none` âm trên ROC-AUC và PR-AUC ở §20.2 trong khi vẫn xấp xỉ 0
trên Macro-F1: nó phá thứ hạng ở đúng lớp chiếm 54% tập test.

Nói cách khác, đóng góp thật của task phụ không phải "tạo ra lợi ích" mà là **giữ cho việc chuyển
giao không phải trả giá bằng các lớp mà target vốn đã học tốt**. Đây là phát biểu chặt hơn, và nó
được ủng hộ bởi 10 quan sát chứ không phải 5.

---

## 24. Thí nghiệm target thứ hai: C/C++ → JavaScript

Mục tiêu "tiến tới các source, target khác" cần ít nhất một target thứ hai. `run/js-target.sh`
dùng JS làm target với ba nhánh `cwe`, `latent_bottleneck`, `none` trên 3 fold.

| Hạng mục | Giá trị |
| --- | --- |
| Target | `data/js_twin`, 1138 mẫu, 5 fold (679 / 229 / 230), nhãn cân bằng |
| CWE target | CWE-079 (478), CWE-078 (84), CWE-089 (75), CWE-022 (42) |
| Source | `data/ccpp_primevul_paired_common.jsonl` — **2975 dòng, 100% C/C++** |
| `TARGET_LANG` | `js`, khớp trường `lang` trong dữ liệu |

### 24.1 Kiểm tra rò rỉ, làm **trước** khi chạy

Điều này không hiển nhiên và suýt sai. `data/train_ccpp_js.jsonl` — source mặc định của mọi thí
nghiệm Python — chứa 1138 dòng JS, và `js_twin` cũng có **đúng 1138** mẫu. So khớp mã nguồn sau
chuẩn hoá khoảng trắng:

| Cặp | Số hàm trùng |
| --- | --- |
| `train_ccpp_js` (phần JS) vs `js_twin` | **1138 / 1138 — trùng hoàn toàn** |
| `ccpp_primevul_paired_common` vs `js_twin` | **0 / 1138** |

Nghĩa là dùng source mặc định cho target JS sẽ là **rò rỉ toàn phần**: Phase 1 huấn luyện trên
đúng tập test của Phase 2. Script đã dùng source C/C++ thuần nên sạch, nhưng khoảng cách giữa
"sạch" và "rò rỉ 100%" ở đây chỉ là một biến môi trường.

Ghi lại vì đây đúng là loại lỗi mà §6 đã tốn nhiều công để phát hiện một lần rồi, và vì bất kỳ ai
sau này đổi `PHASE1_DATA_PATH` cho target JS đều sẽ vô tình tạo lại nó.

---

## 25. Chất lượng lần rút Phase 1 có dự báo được lợi ích transfer không?

§19.3 cho thấy Phase 1 không tái lập: các lần rút trải sd 0.052 trên val nguồn. §19.4 coi đó
thuần tuý là nhiễu phải lấy trung bình. `run/source-draws.sh` — vốn dựng ra để đo biên độ nhiễu đó
— lại lộ ra một thứ khác.

Ba lần rút đã xong, Phase 2 cố định ở seed 36, cùng 3 fold, cùng baseline. Biến duy nhất là
checkpoint Phase 1:

| Lần rút | val Macro-F1 nguồn | Δ Macro-F1 | Δ ROC-AUC |
| --- | --- | --- | --- |
| `twin_ccppjs` (gốc) | 0.6322 | +0.0311 | +0.0153 |
| `draw_seed7` | 0.6495 | +0.0399 | +0.0277 |
| `draw_seed18` | 0.6959 | +0.0529 | +0.0319 |

**Thứ tự trùng khít trên cả hai metric.** Pearson r = +0.989 và +0.860.

### 25.1 Vì sao r **không phải** bằng chứng

Với ba điểm, hầu như mọi quan hệ đơn điệu đều cho r gần 1 — ba điểm gần như xác định một đường
thẳng. Báo cáo r = 0.989 như một phát hiện sẽ là lỗi n nhỏ, lần thứ tám.

Bằng chứng phải là **dự đoán ngoài mẫu**. Lần rút thứ tư trong hàng đợi là `same36_rep1`, val
**0.5722** — thấp hơn cả ba lần trên, và đúng là lần rút "hỏng" dừng ở epoch 3 ở §19.3. Hồi quy
tuyến tính trên ba điểm cho:

> **Dự đoán, ghi trước khi chạy:** `same36_rep1` sẽ cho Δ Macro-F1 ≈ **+0.013** và
> Δ ROC-AUC ≈ **+0.005** — thấp hơn rõ rệt cả ba lần rút kia, và là lần rút duy nhất mà transfer
> gần như không còn tác dụng.

- Trúng → val nguồn dự báo được lợi ích transfer **trong cùng một corpus**, và nhiễu Phase 1 chuyển
  từ vấn đề thành công cụ: chạy Phase 1 vài lần, giữ lần tốt nhất theo val nguồn, rồi mới transfer.
  Đây là một công thức rẻ và dùng được ngay, vì val nguồn **không đụng đến dữ liệu target**.
- Trượt → tương quan chỉ là artefact của n=3, và §19.4 giữ nguyên: nhiễu lần rút là nhiễu, phải
  lấy trung bình chứ không chọn lọc được.

### 25.2 Điểm thứ tư vào, và tương quan yếu đi đúng như đã cảnh báo

`draw_seed12` xong với val nguồn **0.5968** — thấp nhất trong bốn — nhưng cho Δ **+0.0400**, cao
hơn lần rút gốc (val 0.6322, +0.0311). **Quan hệ đơn điệu bị phá.**

| Lần rút | val nguồn | Δ Macro-F1 | Δ ROC-AUC |
| --- | --- | --- | --- |
| `draw_seed12` | 0.5968 | +0.0400 | +0.0187 |
| `twin_ccppjs` (gốc) | 0.6322 | +0.0311 | +0.0153 |
| `draw_seed7` | 0.6495 | +0.0399 | +0.0277 |
| `draw_seed18` | 0.6959 | +0.0529 | +0.0319 |

Pearson r rơi từ **+0.989 (3 điểm) xuống +0.699 (4 điểm)**. Đúng như §25.1 đã cảnh báo: ba điểm
gần như luôn khớp một đường thẳng. Giả thuyết "chọn lần rút theo val nguồn" **yếu đi rõ rệt**;
lần rút thứ năm `same36_rep1` sẽ quyết định, và dự đoán +0.013 của tôi giờ trông khó trúng, vì
`draw_seed12` ở val chỉ cao hơn một chút đã cho +0.0400.

### 25.3 Nhưng câu hỏi quan trọng hơn thì đã có câu trả lời

§19.4 nêu vấn đề nghiêm trọng nhất còn lại: cả 10 quan sát của §20 đều đến từ 2 model nguồn, nên
+0.0393 có thể thuộc về **hai lần rút may mắn** chứ không phải phương pháp. Bốn lần rút độc lập
trả lời được:

| | Δ Macro-F1 | Δ ROC-AUC |
| --- | --- | --- |
| dải | +0.0311 … +0.0529 | +0.0153 … +0.0319 |
| mean | **+0.0410** | +0.0234 |
| sd | **0.0090** | 0.0075 |
| số lần rút dương | **4/4** | **4/4** |

**Cả bốn đều dương trên cả hai metric**, và độ lệch chuẩn giữa các lần rút chỉ **0.0090** — nhỏ hơn
bốn lần so với chính hiệu ứng (+0.0410).

Đối chiếu quan trọng: nhiễu lần rút Phase 1 đo trên **val nguồn** có sd **0.052** (§19.3), nhưng
nhiễu đó truyền xuống Δ transfer chỉ còn sd **0.0090**. Nghĩa là **chất lượng model nguồn dao động
rất mạnh, còn lợi ích transfer thì bền vững trước dao động đó.**

Đây là câu trả lời cho lo ngại lớn nhất của §19.4, và nó tích cực: +0.0393 **không** thuộc về một
lần rút may mắn. Lần rút thứ năm sẽ kiểm tra thêm, đặc biệt vì nó là lần rút "hỏng" dừng ở epoch 3.

### 25.3.1 Toàn bộ lưới (lần rút × fold), thống kê mô tả

Trải phẳng mọi ô đã có — mỗi ô là một cặp (lần rút Phase 1, fold), so với cùng một baseline:

| Metric | ô dương | mean | sd | min | max |
| --- | --- | --- | --- | --- | --- |
| Macro-F1@0.5 | **14/14** | +0.0399 | 0.0190 | +0.0073 | +0.0847 |
| ROC-AUC | **14/14** | +0.0212 | 0.0158 | +0.0021 | +0.0435 |
| PR-AUC | 13/14 | +0.0321 | 0.0227 | −0.0016 | +0.0636 |

**Không một tổ hợp (lần rút, fold) nào cho ROC-AUC âm.** Đây là góc nhìn khác về điểm yếu ở
§20.2: ở đó Wilcoxon trên 10 fold cho ROC-AUC p = 0.0840 vì độ lệch chuẩn **giữa các fold** (0.0286)
lớn so với hiệu ứng; ở đây, khi cố định fold và đổi lần rút, mọi ô đều dương.

**Cảnh báo bắt buộc:** 14 ô này **không phải 14 quan sát độc lập** — chúng dùng lại đúng 3 fold,
nên các ô cùng fold tương quan mạnh với nhau. Vì thế đây là **thống kê mô tả**, không phải kiểm
định, và tuyệt đối không được đem chạy Wilcoxon trên 14 ô rồi báo cáo p. Điều duy nhất bảng này
nói là: trong không gian (lần rút × fold) đã khảo sát, chưa gặp ô nào âm trên hai metric đầu.

### 25.4 Quan hệ với §21.2

§21.2 kết luận "độ chính xác Phase 1 không dự báo được giá trị transfer", dựa trên `full`
(val 0.5329 → +0.0037) so với `common` (val 0.5197 → +0.0307). Hai phát biểu **không mâu thuẫn**
vì chúng hỏi hai câu khác nhau:

- **Giữa các corpus khác nhau**: val nguồn không so sánh được, vì mỗi corpus có tập validation
  riêng và độ khó riêng. §21.2 vẫn đúng.
- **Trong cùng một corpus, giữa các lần rút**: mọi lần rút chia sẻ đúng một tập validation, nên
  val nguồn là thước đo so sánh được. Đây mới là câu §25 hỏi.

Với r đã tụt xuống +0.699 ở §25.2, phát biểu gộp tạm thời là: **val nguồn không dùng để chọn
corpus, và cũng chưa chứng minh được là dùng để chọn lần rút.** Điều đã chắc chắn là §25.3 —
lợi ích transfer bền vững trước việc lần rút nào được dùng.

---

## 26. Phép thử dự đoán: tôi ghi trước, và tôi trượt

### 26.1 Kết quả

§25.1 ghi một dự đoán **trước khi chạy**: lần rút `same36_rep1` (val nguồn 0.5722, thấp nhất, và
là lần rút "hỏng" dừng ở epoch 3 ở §19.3) sẽ cho Δ Macro-F1 ≈ **+0.013**.

| | |
| --- | --- |
| Dự đoán ghi trước | **+0.0126** |
| Thực tế | **+0.0335** |
| Lệch | **+0.0209** |

**Trượt.** Không phải trượt nhỏ — thực tế cao gấp 2.7 lần dự đoán, và nằm giữa dải chứ không phải
dưới đáy.

Bảng đầy đủ năm lần rút:

| Lần rút | val nguồn | Δ Macro-F1 |
| --- | --- | --- |
| `same36_rep1` (dừng ở epoch 3) | 0.5722 | +0.0335 |
| `draw_seed12` | 0.5968 | +0.0400 |
| `twin_ccppjs` (gốc) | 0.6322 | +0.0311 |
| `draw_seed7` | 0.6495 | +0.0399 |
| `draw_seed18` | 0.6959 | +0.0529 |

Pearson r theo số điểm: **+0.989 (3) → +0.699 (4) → +0.742 (5)**. Giả thuyết "chọn lần rút Phase 1
theo val nguồn" **bị bác**. Val nguồn thấp nhất lại không cho Δ thấp nhất; Δ thấp nhất thuộc về
lần rút có val **hạng ba**.

### 26.2 Nhưng phép thử trượt lại củng cố một điều khác, mạnh hơn

`same36_rep1` là lần rút **tệ nhất** có thể lấy được: nó dừng ở epoch 3 vì early stopping, và val
nguồn của nó thấp hơn lần rút tốt nhất tới 0.124. Nó vẫn cho **+0.0335**.

Năm lần rút độc lập, trải gần trọn dải val quan sát được:

| | giá trị |
| --- | --- |
| dải Δ | +0.0311 … +0.0529 |
| mean | **+0.0395** |
| sd | **0.0085** |
| số lần rút dương | **5/5** |

Đối chiếu quyết định: **sd của val nguồn giữa các lần rút là 0.0521; sd của Δ transfer chỉ 0.0085**
— nhỏ hơn sáu lần. Chất lượng model nguồn dao động dữ dội và **lợi ích transfer gần như không đi
theo**.

Đây là kết quả tốt hơn nhiều so với thứ tôi định chứng minh. Tôi đi tìm một công thức chọn lọc
("chạy Phase 1 vài lần, giữ lần tốt nhất") và thay vào đó tìm ra rằng **không cần chọn lọc**:
phương pháp hoạt động kể cả khi Phase 1 hỏng. Với một phương pháp muốn dùng được ở nơi khác, "không
cần may mắn" đáng giá hơn "biết cách chọn lần may mắn".

### 26.3 Ghi chú về quy trình

Đây là giả thuyết thứ chín bị bác trong dự án, và là lần đầu tiên **tôi ghi dự đoán bằng số trước
khi có dữ liệu**. Chênh lệch +0.0209 là một con số cụ thể tôi không thể diễn giải lại theo hướng
có lợi.

Đối chiếu với §21.4, nơi tôi nhìn số rồi mới dựng câu chuyện "gấp mười lần" và phải rút lại: cùng
một sai lầm về bản chất, nhưng ghi dự đoán trước khiến việc phát hiện mất **ba mươi giây thay vì
hai giờ**, và không có lúc nào một khẳng định sai được ghi vào tài liệu như thể nó đúng.

Dự đoán ghi trước còn lại đang chờ: §22.3 dự báo CodeBERT dưới mean pooling rơi khoảng 0.026 xuống
≈ +0.016. `run/pooling.sh` là công việc tiếp theo trên ntat.

---

## 27. Head phụ latent **tốt hơn** head CWE tường minh

Cả hai nhánh latent giờ có đủ n=10 như `cwe`. Bảng đầy đủ: bốn nhánh × ba metric × hai kiểm định,
gộp seed 36 và 12, ghép cặp theo `(seed, fold)`.

| Nhánh | Metric | Δ | sd | A12 | Wilcoxon p | t hiệu chỉnh | p của t | vượt 0.05 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `cwe` | Macro-F1 | +0.0393 | 0.0286 | 0.86 | **0.0020** | +2.09 | 0.066 | Wilcoxon |
| `cwe` | ROC-AUC | +0.0203 | 0.0286 | 0.65 | 0.0840 | +1.08 | 0.308 | không |
| `cwe` | PR-AUC | +0.0213 | 0.0339 | 0.64 | 0.0840 | +0.96 | 0.362 | không |
| **`latent_bottleneck`** | Macro-F1 | +0.0294 | **0.0169** | 0.83 | **0.0020** | **+2.64** | **0.027** | **cả hai** |
| **`latent_bottleneck`** | ROC-AUC | **+0.0218** | 0.0210 | **0.73** | **0.0059** | +1.58 | 0.149 | Wilcoxon |
| `latent_bottleneck` | PR-AUC | +0.0201 | 0.0360 | 0.68 | 0.0645 | +0.85 | 0.417 | không |
| `latent_proto` | Macro-F1 | +0.0279 | 0.0334 | 0.72 | **0.0195** | +1.27 | 0.236 | Wilcoxon |
| `latent_proto` | ROC-AUC | +0.0023 | 0.0226 | 0.56 | 0.4316 | +0.15 | 0.884 | không |
| `latent_proto` | PR-AUC | −0.0164 | 0.0405 | 0.44 | 0.5566 | −0.62 | 0.551 | không |
| `none` | Macro-F1 | +0.0080 | 0.0269 | 0.62 | 0.3750 | +0.45 | 0.663 | không |
| `none` | ROC-AUC | −0.0072 | 0.0220 | 0.43 | 0.5703 | −0.49 | 0.636 | không |
| `none` | PR-AUC | −0.0188 | 0.0366 | 0.43 | 0.1934 | −0.78 | 0.455 | không |

### 27.1 Hai điều bảng này nói

**`latent_bottleneck` là nhánh duy nhất vượt 0.05 trên *cả hai* kiểm định** (Macro-F1: Wilcoxon
0.0020 và t hiệu chỉnh 0.027). `cwe` không đạt — t hiệu chỉnh của nó dừng ở 0.066.

**Và nó vá đúng điểm yếu ở §20.2.** Trên ROC-AUC, `cwe` cho p = 0.084 còn `latent_bottleneck`
cho **p = 0.0059**, với Δ nhỉnh hơn (+0.0218 so với +0.0203) và A12 cao hơn (0.73 so với 0.65).

Nguyên nhân không phải hiệu ứng lớn hơn mà là **ổn định hơn**: sd 0.0169 so với 0.0286 trên
Macro-F1. `cwe` có trung bình cao hơn (+0.0393 so với +0.0294) nhưng phân tán gấp rưỡi, nên nó
thắng ở con số quảng cáo và thua ở con số kiểm định được.

### 27.2 Điều này trả lời câu hỏi ban đầu của dự án

Ý tưởng gốc là biến head CWE tường minh thành latent để phương pháp **không bị khoá vào đúng 4
CWE**. Câu hỏi đặt ra khi đó là "làm thế có ổn không?". Câu trả lời ở n=10:

**Không chỉ ổn — nó tốt hơn.** `latent_bottleneck` vừa gỡ được ràng buộc taxonomy (điều kiện cần
để dùng source như PrimeVul với 121 CWE), vừa cho bằng chứng thống kê mạnh hơn head tường minh.

`latent_proto` — biến thể **không cần nhãn CWE nào** — thì chỉ vượt được ở Macro-F1 (p = 0.0195)
và **âm trên PR-AUC** (−0.0164). Đây đúng triệu chứng §5 đã chẩn đoán: mục tiêu phân cụm bằng
prototype cải thiện điểm vận hành nhưng phá hình học thứ hạng. Bỏ hẳn nhãn thì phải trả giá; giữ
nhãn nhưng ép qua nút thắt K chiều thì không.

### 27.3 Việc còn treo: chưa đối chiếu y văn

Phát hiện ở §27.1 — **head phụ đi qua nút thắt K chiều cho phương sai giữa các fold thấp hơn hẳn
head phân loại trực tiếp** (sd 0.0169 so với 0.0286) — cần đối chiếu với y văn trước khi coi là
đóng góp. Cơ chế nghe hợp lý và có họ hàng với chính quy hoá bằng nút thắt thông tin, nên khả năng
nó đã được mô tả ở đâu đó là **không nhỏ**.

Tôi **không kiểm tra được** trong phiên này: hạn mức tìm kiếm web đã dùng hết (500/500). Muốn làm
thì cần nâng `CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION`, hoặc để phiên sau.

Câu hỏi cụ thể cần tra: (a) đã có ai báo cáo head phụ dạng bottleneck ổn định hơn head phẳng trong
multi-task transfer chưa; (b) có kết quả lý thuyết nào nối chiều nút thắt với phương sai của
gradient task phụ không; (c) trong phát hiện lỗ hổng, đã có ai dùng CWE như task phụ latent chưa.
Kho trích dẫn đã kiểm chứng nằm ở `RESEARCH_2026-08-20_0959.md`.

### 27.4 Giới hạn

Tất cả vẫn trong **một backbone (CodeBERT), một source (ccpp+js), một target (Python), hai seed**.
`latent_bottleneck` chưa được chạy qua nhiều lần rút Phase 1 như `cwe` đã làm ở §26, nên độ bền
của nó trước nhiễu lần rút **chưa được kiểm chứng**. Và PR-AUC của nó vẫn chưa vượt ngưỡng
(p = 0.0645), nên "tốt hơn" đúng cho hai trên ba metric chứ không phải cả ba.

---

## 28. Bảng 2×2 hoàn chỉnh: **dấu bám theo cách đọc, không theo backbone**

`run/pooling.sh` xong. Cả bốn ô của bảng {CodeBERT, CodeT5+} × {cls, mean} giờ đã có, cùng 3 fold,
mỗi ô có baseline riêng huấn luyện đúng kiểu pooling của nó.

| backbone | pooling | baseline | transfer | Δ | Δ từng fold |
| --- | --- | --- | --- | --- | --- |
| CodeBERT | cls | 0.8140 | 0.8451 | **+0.0311** | +0.0586, +0.0274, +0.0073 |
| CodeBERT | mean | 0.8338 | 0.8296 | **−0.0042** | +0.0131, −0.0450, +0.0194 |
| CodeT5+ | cls | 0.8713 | 0.8800 | **+0.0087** | +0.0065, −0.0001, +0.0196 |
| CodeT5+ | mean | 0.8823 | 0.8647 | **−0.0176** | −0.0195, −0.0264, −0.0069 |

| Δ | cls | mean | cls − mean |
| --- | --- | --- | --- |
| **CodeBERT** | +0.0311 | −0.0042 | **+0.0353** |
| **CodeT5+** | +0.0087 | −0.0176 | **+0.0263** |

### 28.1 Phép thử đã ghi trước ở §18.2

§18.2 ghi trước hai khả năng: *"Dấu bám theo **cột** → cách đọc biểu diễn là cơ chế, và phương pháp
không phụ thuộc pretrained model. Dấu bám theo **hàng** → backbone thật sự là biến quyết định."*

**Dấu bám theo cột.** Dưới `cls`, cả hai backbone **dương** (+0.0311 và +0.0087). Dưới `mean`, cả
hai **âm** (−0.0042 và −0.0176). Hiệu cls − mean cùng chiều và cùng bậc trên cả hai backbone
(+0.0353 và +0.0263).

Nghĩa là **hiện tượng "transfer làm hại backbone mạnh" phần lớn là artefact của cách đọc biểu
diễn**, do một khẳng định sai trong `src/model.py` rằng họ T5 không có token ở vị trí 0 (§18.2).
Đây là kết quả trực tiếp cho mục tiêu **không phụ thuộc pretrained**: dùng `cls` thì phương pháp
dương trên cả hai họ backbone.

### 28.2 Dự đoán số của tôi thì trượt

§22.3 ghi: *"CodeBERT dưới mean pooling sẽ mất khoảng 0.026, tức Δ rơi từ +0.0417 xuống khoảng
+0.016, và **vẫn dương**."*

| | |
| --- | --- |
| Hướng | **đúng** — mean pooling làm giảm Δ |
| Độ lớn | **ước lượng thấp** — thực tế rơi 0.0353 chứ không phải 0.026 |
| Dấu kết quả | **sai** — tôi nói vẫn dương, thực tế ra **−0.0042** |

Hai trong ba thành phần sai. Ước lượng 0.026 lấy từ CodeT5+, và giả định ngầm là hiệu ứng pooling
có cùng độ lớn trên mọi backbone. Nó không: CodeBERT mất **nhiều hơn** (0.0353).

Thêm một chi tiết đáng chú ý: trên CodeT5+, hiệu cls − mean cực kỳ ổn định (+0.0260, +0.0263,
+0.0265, sd 0.00025). Trên CodeBERT thì **không hề**: −0.0455, −0.0724, +0.0121. Sự trùng khớp
kinh ngạc ở §22.1 vì thế **không phải tính chất chung của phép đo** mà là đặc thù của CodeT5+ —
đúng như §22.1 đã ngờ và ghi lại.

### 28.3 Nhưng "dương" không đồng nghĩa "tốt nhất"

| backbone | cấu hình tốt nhất trong bốn ô | giá trị |
| --- | --- | --- |
| **CodeBERT** | **transfer + cls** | **0.8451** |
| **CodeT5+** | baseline + mean | 0.8823 |

Trên CodeBERT, phương pháp **là** lựa chọn tốt nhất — nó vượt cả baseline mean-pool (0.8338).
Trên CodeT5+, nó **không**: transfer+cls đạt 0.8800, vẫn thua baseline mean-pool 0.8823.

Phát biểu chặt nhất cho mục tiêu pretrained-agnostic: **cách đọc biểu diễn giải thích được dấu, và
sửa nó làm phương pháp dương trên cả hai backbone. Nhưng trên backbone mạnh, "dương so với baseline
cùng cách đọc" vẫn chưa bằng "không transfer gì và pool bằng trung bình".** Khoảng cách còn lại là
0.0023 — nhỏ, nhưng có dấu.
