# Kết quả thí nghiệm

Tài liệu này ghi lại các thí nghiệm đã chạy và số liệu cụ thể, để tra cứu về sau.
Số thô nằm ở `results_vast/` (**có** track trong git — các file `summary.json` và `fold*.json`
đi kèm repo để đối chiếu lại được). Log ở `logs_vast/` thì không track vì quá nặng.

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
| Phương pháp có mang lại gì trên backbone mạnh không | CodeT5+ dưới cls: +0.0039 (n=5), không phân biệt được với baseline mean-pool (hiệu +0.0014, sd 0.0057) | §28.3 |
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

*Cập nhật lên n=13 sau khi seed 7 xong 3 fold:*

| CWE | mẫu test | baseline | `cwe` | Δ | fold dương | `latent_bottleneck` Δ | fold dương |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **CWE-022** path traversal | 66 | 0.4767 | 0.7057 | **+0.2290** | **12/13** | **+0.1594** | 11/13 |
| **CWE-079** XSS | 82 | 0.5750 | 0.7873 | **+0.2124** | **12/13** | **+0.1785** | 12/13 |
| CWE-078 command injection | 204 | 0.7746 | 0.7785 | +0.0039 | 7/13 | −0.0070 | 7/13 |
| CWE-089 SQL injection | 408 | 0.9471 | 0.9434 | −0.0037 | 5/13 | −0.0009 | 7/13 |

**Toàn bộ lợi ích nằm ở hai lớp hiếm**, và ở đó nó rất lớn: +0.21 và +0.18, dương ở 9/10 fold cho
cả hai. Hai lớp này chỉ chiếm 148 trên 760 mẫu, nên chúng bị pha loãng thành +0.039 khi gộp — đó
là lý do con số tổng hợp khiêm tốn hơn nhiều so với tác dụng thật.

### 23.1 Ablation λ=0 không phẳng, nó **triệt tiêu**

Đây là phần buộc phải sửa. Suốt tài liệu này tôi mô tả `transfer_none` là "phẳng" vì Δ tổng hợp
+0.0037. Bóc theo lớp thì nó không phẳng chút nào:

| CWE | Δ `none` (n=13) | Δ `cwe` (n=13) |
| --- | --- | --- |
| CWE-022 | **+0.1129** | +0.2290 |
| CWE-079 | **+0.1510** | +0.2124 |
| CWE-078 | **−0.0244** | +0.0039 |
| CWE-089 | **−0.0171** | −0.0037 |

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

*Bảng dưới là bản đủ **5 lần rút** (bản trước ở 4 lần rút ghi ROC-AUC mean +0.0234, sd 0.0075):*

| | Δ Macro-F1 | Δ ROC-AUC |
| --- | --- | --- |
| dải | +0.0311 … +0.0529 | +0.0109 … +0.0319 |
| mean | **+0.0395** | **+0.0209** |
| sd | **0.0085** | **0.0087** |
| số lần rút dương | **5/5** | **5/5** |

**Cả năm đều dương trên cả hai metric**, và độ lệch chuẩn giữa các lần rút chỉ **0.0085** — nhỏ hơn
gần năm lần so với chính hiệu ứng (+0.0395).

Đối chiếu quan trọng: nhiễu lần rút Phase 1 đo trên **val nguồn** có sd **0.052** (§19.3), nhưng
nhiễu đó truyền xuống Δ transfer chỉ còn sd **0.0085**. Nghĩa là **chất lượng model nguồn dao động
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

### 28.3 CodeT5+ ở đủ 5 fold: kết luận ở n=3 bị đảo

`run/t5cls45.sh` xong. Ở n=5, CodeT5+ dưới `cls` cho **+0.0039** (so với +0.0087 ở n=3), dưới
`mean` giữ nguyên **−0.0184**. Dấu vẫn bám theo cột.

Nhưng thứ hạng tuyệt đối thì đảo:

| cấu hình | n=3 | **n=5** |
| --- | --- | --- |
| transfer + cls | 0.8800 | **0.8893** ← cao nhất |
| baseline + mean | **0.8823** ← cao nhất | 0.8879 |
| baseline + cls | 0.8713 | 0.8854 |
| transfer + mean | 0.8647 | 0.8696 |

Ở n=3 tôi kết luận "trên CodeT5+ phương pháp **không** phải lựa chọn tốt nhất". Ở n=5 nó vươn lên
đầu. Ghép cặp trực tiếp `transfer+cls` với `baseline+mean` theo từng fold:

```
+0.0000, -0.0067, -0.0001, +0.0073, +0.0066   ->  mean +0.0014, sd 0.0057, 2/5 duong
```

Trung bình **nhỏ hơn độ lệch chuẩn**, hai fold gần như hoà tuyệt đối. Phát biểu đúng không phải
"phương pháp thắng" cũng không phải "phương pháp thua" mà là: **trên CodeT5+, phương pháp với `cls`
và baseline với `mean` không phân biệt được.**

Và độ ổn định kinh ngạc ở §22.1 cũng tan ở n=5: hiệu cls − mean từng fold là +0.0260, +0.0263,
+0.0265, **+0.0056**, +0.0270 — sd nhảy từ **0.00025 lên 0.0093**. Fold 4 phá vỡ nó. Ba fold đầu
trùng khớp tới bốn chữ số thập phân **là trùng hợp**, đúng như §22.1 đã ngờ.

### 28.4 Phát biểu cuối cho mục tiêu pretrained-agnostic

**Cách đọc biểu diễn giải thích được dấu.** Sửa nó làm phương pháp dương trên cả hai họ backbone,
và điều đó gỡ bỏ phần lớn hiện tượng "phụ thuộc pretrained" đã tiêu ba can thiệp thất bại để truy.

**Nhưng lợi ích thì co lại theo độ mạnh backbone.** CodeBERT +0.0311; CodeT5+ +0.0039 — nhỏ hơn
tám lần, và không phân biệt được với việc đơn giản đổi cách pool. Phương pháp **không còn gây hại**
trên backbone mạnh, nhưng cũng **chưa mang lại gì đáng kể** ở đó.

---

## 29. Seed thứ ba: metric xếp hạng vượt ngưỡng, và một khẳng định của tôi yếu đi

`run/seed7-core.sh` xong 3 fold. Δ Macro-F1 theo từng seed:

| Nhánh | seed 36 | seed 12 | seed 7 |
| --- | --- | --- | --- |
| `cwe` | +0.0417 (n=5) | +0.0370 (n=5) | +0.0337 (n=3) |
| `latent_bottleneck` | +0.0268 (n=5) | +0.0319 (n=5) | +0.0272 (n=3) |
| `none` | −0.0025 (n=5) | +0.0184 (n=5) | +0.0072 (n=3) |

Ba seed độc lập, ba lần rút Phase 1 khác nhau, hai máy khác nhau — và ba cột **khớp nhau chặt**.

Gộp thành **n=13**:

| Nhánh | Metric | Δ | sd | A12 | Wilcoxon p | t hiệu chỉnh | p của t |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cwe` | Macro-F1 | +0.0380 | 0.0286 | 0.85 | **0.0007** | +2.08 | 0.060 |
| `cwe` | ROC-AUC | +0.0219 | 0.0304 | 0.67 | **0.0327** | +1.13 | 0.281 |
| `latent_bottleneck` | Macro-F1 | +0.0289 | 0.0248 | 0.80 | **0.0034** | +1.82 | 0.094 |
| `latent_bottleneck` | ROC-AUC | +0.0208 | 0.0261 | 0.69 | **0.0215** | +1.25 | 0.235 |
| `none` | Macro-F1 | +0.0078 | 0.0262 | 0.60 | 0.2734 | +0.47 | 0.647 |
| `none` | ROC-AUC | −0.0031 | 0.0241 | 0.45 | 0.9097 | −0.20 | 0.845 |

### 29.1 Điểm yếu chính đã được giải quyết

§20.2 nêu vấn đề lớn nhất còn lại: trên metric **độc lập ngưỡng**, hiệu ứng chỉ bằng nửa và
không vượt 0.05 (ROC-AUC p = 0.0840 ở n=10).

Ở n=13, **ROC-AUC vượt ngưỡng cho cả hai nhánh**: `cwe` p = **0.0327**, `latent_bottleneck`
p = **0.0215**. Đây không phải hiệu ứng lớn lên (Δ vẫn ~+0.021) mà là **thêm quan sát thật**, đúng
như §20.2 đã nói là cách hợp lệ duy nhất. Ablation λ=0 vẫn phẳng và **âm** trên ROC-AUC (−0.0031,
p = 0.91), giữ nguyên kết luận về vai trò head phụ.

### 29.2 Nhưng §27 đã nói quá, và giờ phải sửa

§27 kết luận **"`latent_bottleneck` là nhánh duy nhất vượt 0.05 trên cả hai kiểm định"**, dựa trên
n=10 nơi nó có sd 0.0169 và t hiệu chỉnh +2.64 (p = 0.027).

Thêm seed 7 thì lợi thế phương sai đó **co lại**:

| | n=10 | n=13 |
| --- | --- | --- |
| sd | 0.0169 | **0.0248** |
| t hiệu chỉnh | +2.64 | **+1.82** |
| p của t | **0.027** (vượt) | **0.094** (không vượt) |

Ở n=13, **không nhánh nào vượt được kiểm định t hiệu chỉnh** — kể cả `cwe` (p = 0.060). Cả hai chỉ
vượt Wilcoxon.

Phát biểu đúng bây giờ: **`cwe` và `latent_bottleneck` gần như tương đương.** `cwe` có trung bình
cao hơn (+0.0380 so với +0.0289) và A12 cao hơn (0.85 so với 0.80); `latent_bottleneck` có sd nhỏ
hơn chút (0.0248 so với 0.0286) và p tốt hơn trên ROC-AUC (0.0215 so với 0.0327). Không đủ căn cứ
để nói nhánh nào hơn.

Điều **vẫn đứng vững** và mới là điểm đáng kể: `latent_bottleneck` **ngang** head CWE tường minh
trong khi **không bị khoá vào taxonomy nguồn**. Với mục tiêu tổng quát hóa thì "ngang mà không
ràng buộc" đã đủ; không cần nó phải hơn.

Đây là lần thứ tám một tín hiệu của tôi co lại khi có thêm dữ liệu, và lần thứ tư tôi phải sửa một
phát biểu đã viết vào tài liệu này. Lần này khoảng cách chỉ là **hai giờ** — §27 viết ở n=10, sửa
ở n=13.

---

## 30. Quy trình sàng lọc: một seed trước, nhiều seed sau

Từ đây trở đi, mọi ý tưởng mới đi theo quy trình này thay vì chạy nhiều seed ngay:

| Cổng | Làm gì | Điều kiện đi tiếp |
| --- | --- | --- |
| **1** | Seed 42, **fold 1–3**. Mỗi fold: baseline → các phương pháp → các pretrained cùng phương pháp | xu hướng mạnh |
| **2** | Chạy nốt **fold 4–5**, cùng seed | mean cao ở **mọi** fold, và biên độ **vượt nhiễu** |
| **3** | Lặp lại trên **3–5 seed** khác để loại trừ may rủi | vẫn giữ |
| **4** | Mới đến ablation: bỏ RecAdam, đổi dữ liệu, đổi source/target, xác định lượng dữ liệu cần, kiểm định thống kê | — |

**Chỉ hai thứ quyết định: phương pháp và pretrained.** Cổng 1–3 kiểm đúng hai thứ đó và không gì
khác. Mọi câu hỏi còn lại — RecAdam có cần không, cần bao nhiêu dữ liệu, đổi source thì sao — đều
**vô nghĩa nếu phương pháp không thắng ngay từ đầu**, nên chúng bị đẩy hết xuống Cổng 4.

Chạy nhiều seed ngay từ đầu là lãng phí khi chưa biết có gì đáng xác nhận. Ba seed cho một ý tưởng
hỏng tốn gấp ba lần một seed cho cùng kết luận đó.

### 30.0 Thứ tự trong từng fold, và luật cùng máy

`run/gated.sh` chạy đúng thứ tự này trong **mỗi** fold:

1. **baseline của từng backbone** — chạy trước mọi thứ, vì mọi Δ đều quy về nó;
2. **từng phương pháp**, và với mỗi phương pháp thì **mọi backbone chạy liền nhau**.

Vòng ngoài là phương pháp chứ không phải backbone, nên "phương pháp → pretrained cùng phương pháp"
nằm sát nhau trong thời gian, dễ đọc xu hướng.

**Luật cùng máy là luật mềm, phạm vi hẹp.** Một *nhóm so sánh* — baseline cùng các phương pháp của
nó, cùng seed, cùng fold — nên nằm trên một máy, vì chênh lệch phần cứng đo được là **0.028
Macro-F1**, lớn hơn chính hiệu ứng. Ngoài phạm vi đó thì tách máy thoải mái: **fold khác nhau hoặc
seed khác nhau chạy song song hai máy đều hợp lệ** và nhanh gấp đôi. Điều duy nhất cấm là lấy
baseline máy này ghép với phương pháp máy kia trong cùng một phép so.

### 30.0.1 "Vượt nhiễu" nghĩa là gì

Ở Cổng 2, Δ cỡ **0.00x** không tính là thắng — nó nằm gọn trong nhiễu. Ngưỡng này không phải quy
ước chung mà lấy từ chính dự án: sd giữa các fold tới **0.09** trên tập test 152 mẫu.

Ví dụ có thật: CodeT5+ dưới `cls` cho Δ = **+0.0039**. `src/report_gate.py` tự xếp nó vào nhóm
**"ngang baseline"** chứ không phải "thắng" — và đó là cách đọc đúng, vì ghép cặp với baseline
mean-pool cho hiệu +0.0014 với sd 0.0057, tức trung bình nhỏ hơn chính độ lệch chuẩn của nó.

### 30.1 Vì sao đủ 5 fold, không phải 3

Tài liệu này có bốn lần một tín hiệu ở n=3 co lại hoặc **đảo dấu** ở n=5:

| Tín hiệu ở n=3 | Sự thật ở n=5 |
| --- | --- |
| `common` hơn `full` "gấp mười lần" | 3/5 fold, p = 0.4375 |
| CodeT5+ `cls` tốt hơn baseline mean-pool | đảo ngược — hoà, hiệu +0.0014 |
| hiệu cls − mean sd 0.00025 (cực ổn định) | sd 0.0093, fold 4 phá vỡ |
| `latent_bottleneck` vượt cả hai kiểm định | ở n lớn hơn thì không |

Với sd giữa các fold tới 0.09 trên tập test 152 mẫu, ba fold **không đủ** để thấy xu hướng. Năm
fold là mức tối thiểu, và đó là lý do bước 1 chạy hết chứ không dừng ở 3.

### 30.2 Vì sao sàng lọc một seed lại đáng tin ở đây

Có một phản biện hiển nhiên: §19.3 cho thấy **Phase 1 không tái lập** — chạy lại cùng seed vẫn cho
val nguồn lệch sd 0.0521. Vậy một seed thì ghim được gì?

§26 trả lời được, và câu trả lời thuận lợi: nhiễu đó **không truyền xuống** Δ transfer.

| Đại lượng | sd giữa các lần rút |
| --- | --- |
| val nguồn của Phase 1 | **0.0521** |
| **Δ transfer** | **0.0085** |

Nhỏ hơn sáu lần, và 5/5 lần rút đều dương kể cả lần rút hỏng dừng ở epoch 3. Nghĩa là **Δ ổn định
hơn nhiều so với checkpoint sinh ra nó**, nên sàng lọc trên Δ ở một seed là hợp lệ.

Điều này chỉ được biết vì `run/source-draws.sh` đã đo. Trước khi có §26, quy trình một-seed là một
canh bạc; sau đó thì nó có cơ sở.

### 30.3 Áp dụng đầu tiên

`run/t5-seed42.sh` — head latent trên backbone T5, seed 42, đủ 5 fold.

Đây là **khoảng trống thật**, không phải chạy lại: mọi lần chạy trên họ T5 từ trước tới nay đều chỉ
dùng nhánh `cwe`. `latent_bottleneck` chưa từng chạy trên backbone nào ngoài CodeBERT, trong khi
hai nhánh **không** hành xử giống nhau (§27: `cwe` có Δ lớn hơn, `latent_bottleneck` có p tốt hơn
trên ROC-AUC). Câu "phương pháp không phụ thuộc pretrained" vì thế hiện mới chỉ được kiểm cho `cwe`.

---

## 31. Nhánh latent cũng bền trước xổ số Phase 1

§27.4 để lại một giới hạn: `latent_bottleneck` chưa từng được kiểm qua nhiều lần rút Phase 1 như
`cwe` đã làm ở §26. `run/latent-draws.sh` đóng khoảng trống đó — 4 checkpoint Phase 1 độc lập,
Phase 2 cố định seed 36, cùng 3 fold, cùng baseline.

| Lần rút | val nguồn | vân tay trọng số | Δ Macro-F1 | Δ ROC-AUC |
| --- | --- | --- | --- | --- |
| `latentdraw_12` | 0.6109 | `c9a3161c` | +0.0334 | +0.0124 |
| `latentdraw_7` | 0.6212 | `3d990b3c` | +0.0377 | +0.0248 |
| `latentdraw_36` | 0.6438 | `fecc1498` | +0.0225 | +0.0145 |
| `latentdraw_18` | 0.6969 | `9a27567d` | +0.0377 | +0.0155 |

**4/4 dương trên cả hai metric.** Đối chiếu trực tiếp với `cwe` (§26, 5 lần rút):

| Nhánh | Metric | mean | **sd giữa các lần rút** | dải | dương |
| --- | --- | --- | --- | --- | --- |
| `cwe` | Macro-F1 | +0.0395 | 0.0085 | +0.0311 … +0.0529 | 5/5 |
| `latent_bottleneck` | Macro-F1 | +0.0328 | **0.0072** | +0.0225 … +0.0377 | 4/4 |
| `cwe` | ROC-AUC | +0.0209 | 0.0087 | +0.0109 … +0.0319 | 5/5 |
| `latent_bottleneck` | ROC-AUC | +0.0168 | **0.0055** | +0.0124 … +0.0248 | 4/4 |

Cùng hình dạng như trên trục fold: `cwe` có **trung bình cao hơn**, `latent_bottleneck` có **phân
tán nhỏ hơn** — lần này trên ROC-AUC thì sd chỉ bằng **0.63 lần** (0.0055 so với 0.0087).

### 31.1 Nhưng đừng lặp lại sai lầm §27

§27 đã dùng đúng lập luận "phương sai nhỏ hơn" để kết luận nhánh latent **hơn** `cwe`, rồi §29 phải
rút lại khi seed 7 làm sd nở từ 0.0169 lên 0.0248. Lần này là **trục khác** (nhiễu lần rút, không
phải nhiễu fold) và **n=4**, nên nó chưa mạnh hơn lần trước là bao.

Phát biểu được phép: **`latent_bottleneck` bền trước nhiễu lần rút Phase 1 ngang `cwe`** — 4/4 dương,
sd cùng bậc hoặc nhỏ hơn. Đó là điều §27.4 cần và giờ đã có. Còn "latent ổn định **hơn**" thì vẫn
**chưa xác lập**, và sẽ chỉ xác lập được nếu nó sống sót qua nhiều lần rút hơn.

---

## 32. Thêm seed **không** gỡ được kiểm định t hiệu chỉnh — và vì sao

Trước khi cho máy rảnh chạy thêm một seed nữa, tôi tính xem nó có gỡ được điểm yếu cuối không.
Câu trả lời là **gần như không**, và lý do nằm ngay trong công thức.

Sai số chuẩn hiệu chỉnh Nadeau–Bengio:

```
SE = sqrt( sd² × ( 1/n  +  n_test/n_train ) )
                   ↑         ↑
              giam theo n   HANG SO = 152/456 = 0.3333
```

Số hạng thứ hai **không phụ thuộc n**. Với tỉ lệ chia hiện tại nó bằng 0.3333, áp đảo hoàn toàn
`1/n` (ở n=15 chỉ là 0.067). Nên thêm bao nhiêu fold hay seed cũng chỉ gặm được phần nhỏ:

| Nhánh · metric | n=15 | n=20 (+1 seed) | n=25 (+2 seed) |
| --- | --- | --- | --- |
| `cwe` Macro-F1 | p = 0.066 | p = 0.056 | p = **0.050** |
| `cwe` ROC-AUC | p = 0.199 | p = 0.184 | p = 0.175 |
| `latent_bottleneck` Macro-F1 | p = 0.131 | p = 0.118 | p = 0.110 |
| `latent_bottleneck` ROC-AUC | p = 0.163 | p = 0.149 | p = 0.141 |

Phải chạy **thêm hai seed** mới đưa được đúng một ô chạm vạch 0.050, còn các metric xếp hạng thì
**không bao giờ tới**. Từ n=20 trở đi gần như bão hoà.

### 32.1 Hệ quả cho kế hoạch

**Chạy thêm seed để đuổi theo t hiệu chỉnh là khoản đầu tư tồi.** Nó chỉ cải thiện được khi đổi
**tỉ lệ chia dữ liệu** (train lớn hơn so với test), chứ không phải khi lặp lại nhiều lần hơn.

Điều này cũng nói rõ giới hạn của thiết kế hiện tại: với 760 mẫu target chia 456/152/152, kiểm định
t hiệu chỉnh **bị chặn trên bởi chính tỉ lệ chia**, không phải bởi công sức bỏ ra. Muốn vượt thì
phải có target lớn hơn, không phải chạy lâu hơn.

Vì thế máy rảnh được dùng cho **chiều đang mỏng nhất** thay vì làm dày chiều đã đủ: `cwe` và
`latent_bottleneck` trên CodeBERT đã có n=15, trong khi **CodeT5+ mới chỉ có n=5 ở đúng một seed**.
`run/gated.sh` với `SEED=18` chạy cả hai backbone trên cùng một máy để bổ sung đúng chỗ đó.

---

## 33. Hai run cổng chặn hoàn tất — bức tranh 5 seed, và một kết quả âm quan trọng

`gate1` (seed 42) và `gate18` (seed 18) chạy xong đủ 5 fold, mỗi run có baseline riêng cho từng
backbone trên chính máy đó. Gộp với ba seed cũ:

### 33.1 CodeBERT · cls — 5 seed, n=25

| Nhánh | seed 36 | seed 12 | seed 7 | seed 42 | seed 18 | **gộp n=25** | **sd giữa các seed** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `cwe` | +0.0417 | +0.0370 | +0.0442 | **+0.0157** | +0.0422 | **+0.0362** | 0.0117 |
| `latent_bottleneck` | +0.0268 | +0.0319 | +0.0203 | +0.0234 | +0.0266 | **+0.0258** | **0.0043** |
| `none` | −0.0025 | +0.0184 | +0.0079 | +0.0090 | **+0.0264** | +0.0119 | 0.0110 |

**`latent_bottleneck` ổn định gấp 2.7 lần `cwe` giữa các seed** — sd 0.0043 so với 0.0117, dải chỉ
+0.0203…+0.0319 trong khi `cwe` trải +0.0157…+0.0442. Đây là phiên bản mạnh hơn nhiều của tuyên bố
đã bị rút ở §29: lần đó dựa trên 2 seed và trục fold, lần này 5 seed và trục seed.

**Nhưng `none` cũng phải sửa.** Ở n=13 nó là +0.0078 và được mô tả là phẳng. Ở n=25 nó là **+0.0119**
và dải trải từ −0.0025 tới **+0.0264** — seed 18 cho `none` gần bằng `latent_bottleneck`. Ablation
λ=0 **không null sạch** như các mục trước nói; khoảng cách `cwe` − `none` thu hẹp còn +0.024.

### 33.2 CodeT5+ · cls — kết quả âm

| Nhánh | seed 36 | seed 42 | seed 18 | gộp | sd giữa các seed |
| --- | --- | --- | --- | --- | --- |
| `cwe` | +0.0039 | +0.0279 | **−0.0132** | +0.0062 (n=15) | **0.0206** |
| `latent_bottleneck` | — | **−0.0079** | **−0.0094** | **−0.0086** (n=10) | 0.0010 |
| `none` | — | +0.0081 | −0.0144 | −0.0032 (n=10) | 0.0160 |

**Trên CodeT5+, nhánh latent âm ở cả hai seed** (−0.0079 và −0.0094, cực kỳ nhất quán về phía âm).
`cwe` thì dao động hoang dã: −0.0132 đến +0.0279, sd giữa seed **0.0206** — lớn hơn cả hiệu ứng
trung bình +0.0062.

Đây là **kết quả âm và phải nói thẳng**: §28 kết luận sửa cách đọc biểu diễn làm phương pháp dương
trên cả hai họ backbone, dựa trên **một seed** của CodeT5+ (+0.0039, vốn đã nằm trong nhiễu). Thêm
hai seed cho thấy con số đó không lặp lại, và nhánh latent — nhánh tốt nhất trên CodeBERT — **âm
đều** trên CodeT5+.

### 33.3 Phát biểu đúng sau tất cả

| | CodeBERT (5 seed, n=25) | CodeT5+ (2–3 seed) |
| --- | --- | --- |
| `cwe` | +0.0362, sd seed 0.0117 | +0.0062, sd seed **0.0206** — không lặp lại |
| `latent_bottleneck` | **+0.0258, sd seed 0.0043** | **−0.0086** — âm cả hai seed |

**Phương pháp có tác dụng rõ và ổn định trên CodeBERT, và không chuyển được sang CodeT5+.** Mục
tiêu "không phụ thuộc pretrained" **chưa đạt**. Việc sửa pooling ở §28 gỡ được phần lớn thiệt hại
nhưng không tạo ra lợi ích trên backbone mạnh — và ở nhánh latent thì vẫn còn hại nhẹ.

Điều đứng vững nhất còn lại: trên backbone mà phương pháp hoạt động, **nhánh latent vừa gỡ được
ràng buộc taxonomy vừa là nhánh ổn định nhất qua 5 seed**.

---

## 34. Vì sao phương pháp hỏng trên CodeT5+ — chẩn đoán từ dữ liệu đã có

Máy đã hủy nên không chạy thêm được, nhưng câu hỏi "vì sao" trả lời được **không cần GPU** từ chính
kết quả per-CWE đã tải về.

### 34.1 Dư địa: CodeT5+ đã giải sẵn đúng chỗ CodeBERT thu lợi

| CWE | mẫu | baseline CodeBERT | baseline CodeT5+ | chênh | dư địa còn lại |
| --- | --- | --- | --- | --- | --- |
| **CWE-022** | 66 | 0.4876 | **0.6586** | +0.1711 | CodeT5+ ít hơn **33%** |
| **CWE-079** | 82 | 0.5793 | **0.6756** | +0.0962 | CodeT5+ ít hơn **23%** |
| CWE-078 | 204 | 0.7775 | 0.8321 | +0.0545 | — |
| CWE-089 | 408 | 0.9489 | 0.9763 | +0.0274 | — |

§23 cho thấy **toàn bộ** lợi ích của phương pháp nằm ở CWE-022 và CWE-079. Đó lại đúng là hai lớp
CodeT5+ vượt CodeBERT nhiều nhất (+0.17 và +0.10). Dư địa mà phương pháp sống nhờ đã bị backbone
mạnh ăn mất phần lớn.

### 34.2 Nhưng không phải hỏng hoàn toàn — Δ theo từng lớp

| CWE | CodeBERT `none` → `cwe` | CodeT5+ `none` → `cwe` |
| --- | --- | --- |
| CWE-022 | +0.1193 → **+0.1954** | +0.0338 → **+0.0201** |
| CWE-079 | +0.1305 → **+0.2102** | +0.1079 → **+0.1202** |
| CWE-078 | −0.0038 → **+0.0157** | −0.0312 → **−0.0256** |
| CWE-089 | −0.0153 → −0.0065 | −0.0135 → −0.0000 |

Trên CodeT5+, **CWE-079 vẫn được +0.1202** — phương pháp *vẫn* giúp lớp hiếm đó rất nhiều. Cái hỏng
là hai chỗ khác:

1. **CWE-022 sụp từ +0.195 xuống +0.020** — đúng lớp mà CodeT5+ đã tự giải tốt (0.6586).
2. **CWE-078 âm −0.0256 và không được cứu.** Lớp này có **204 mẫu**, gấp 2.5 lần CWE-079, nên nó
   kéo tổng hợp xuống nhiều hơn phần CWE-079 kéo lên.

### 34.3 Cơ chế "gỡ thiệt hại" ngừng hoạt động

§23.2 xác định đóng góp thật của head phụ là **gỡ bỏ thiệt hại mà pretrain trần gây ra trên lớp phổ
biến**. Đo trực tiếp giá trị gia tăng đó, `Δ(cwe) − Δ(none)`:

| CWE | CodeBERT | CodeT5+ |
| --- | --- | --- |
| CWE-022 | **+0.0761** | **−0.0137** |
| CWE-079 | **+0.0797** | +0.0124 |
| CWE-078 | +0.0195 | +0.0056 |
| CWE-089 | +0.0089 | +0.0134 |

Trên CodeBERT, head phụ cộng thêm **+0.076 và +0.080** ở hai lớp hiếm. Trên CodeT5+ nó cộng
**−0.014 và +0.012** — gần như không còn giá trị gia tăng, thậm chí âm ở CWE-022.

**Kết luận cơ chế:** head phụ chỉ có tác dụng khi backbone **chưa** biểu diễn tốt lớp hiếm. Khi
backbone đã giải sẵn (CodeT5+ đạt 0.6586 trên CWE-022 so với 0.4876), tín hiệu CWE mà head phụ áp
vào là **thông tin backbone đã có**, nên nó không thêm gì mà chỉ nhiễu thêm.

### 34.4 Việc nên làm khi có máy trở lại

Chẩn đoán này thu hẹp không gian tìm kiếm rất nhiều — nó nói **đừng** làm gì:

| Đừng làm | Vì sao |
| --- | --- |
| Thêm seed cho CodeT5+ | vấn đề là dư địa, không phải nhiễu |
| Sửa cách đọc biểu diễn nữa | §28 đã làm, và §33 cho thấy không lặp lại được |
| Bảo vệ trọng số (LP-FT, LoRA, nội suy) | §15 đã bác cả ba |
| Thêm seed đuổi theo t hiệu chỉnh | §32: bị chặn bởi tỉ lệ chia, không phải công sức |

Hướng còn lại có cơ sở từ §34.3: **cần một tín hiệu phụ mà backbone mạnh CHƯA có**, thay vì lặp lại
nhãn CWE mà nó đã biểu diễn được. Hai ứng viên cụ thể, chạy được theo quy trình bốn cổng ở §30:

1. **Target nhỏ hơn.** Nếu cơ chế là dư địa thì giảm dữ liệu target sẽ mở lại dư địa cho CodeT5+.
   Kiểm được ngay và bác được ngay: §13 cho thấy ở 114 dòng thì transfer **có hại** trên CodeBERT,
   nên dự đoán là nó cũng không cứu được — đây là phép thử rẻ và có khả năng bác cao.
2. **Đổi tín hiệu phụ, không đổi kiến trúc.** Head phụ hiện dự đoán nhãn CWE. Trên backbone đã biết
   CWE rồi thì tín hiệu đó thừa. Một task phụ mang thông tin khác — ví dụ dự đoán vị trí dòng sửa
   lỗi, hoặc khoảng cách sửa đổi giữa hai bản của cặp — có thể còn giá trị gia tăng.

Ứng viên 2 là hướng đáng đầu tư hơn, nhưng nó **thay đổi phương pháp** chứ không phải tinh chỉnh,
nên cần bắt đầu lại từ Cổng 1.

---

## 35. Tín hiệu phụ thay thế: đã kiểm chứng khả thi, sẵn sàng chạy Cổng 1

§34.3 chỉ ra hướng duy nhất còn cơ sở: **cần tín hiệu phụ mà backbone mạnh chưa có**. Máy đã hủy
nên không chạy được, nhưng toàn bộ phần kiểm chứng khả thi làm được **không cần GPU** — và nó có
thể bác hướng đi này trước khi tốn một giờ GPU nào.

### 35.1 Dữ liệu có đủ không

| | Kết quả |
| --- | --- |
| Source `train_ccpp_js.jsonl` có `pair_id`? | **Không** |
| Nhưng cặp liền kề suy ra được? | **638 cặp, phủ 1276/1284 dòng = 99%** |
| Fold target có `pair_id`? | Có, nhưng chỉ 159/234 cặp đủ hai nửa trong `train` |

Chỉ **source** mới quan trọng, vì head phụ chỉ chạy ở Phase 1 rồi bị đóng băng. Source phủ 99% nên
tín hiệu dựa trên diff là **khả thi**.

Đáng ghi: các dòng trong fold target **không** nằm cạnh nhau theo cặp (tôi đã giả định sai lúc đầu
và phải kiểm lại) — chúng phải nhóm qua `pair_id`. Riêng source thì liền kề thật.

### 35.2 Tín hiệu có học được không, hay suy biến

Đo hai ứng viên trên 638 cặp:

| Tín hiệu | Phân bố | Lớp lớn nhất |
| --- | --- | --- |
| **Số dòng bị sửa** (4 nhóm) | 30% / 22% / 27% / 21% | **30%** — cân bằng |
| Vị trí sửa đổi đầu tiên (3 nhóm) | 54% / 31% / 16% | 54% — lệch hơn |

Ranh giới nhóm lấy từ **tứ phân vị đo được** (q1=1, median=3, q3=8), không phải số tròn tự nghĩ.

**Số dòng bị sửa cân bằng hơn hẳn cả phân bố CWE gốc** (nơi CWE-089 chiếm 54% tập test). Và nó
tình cờ cũng **4 lớp** như head CWE gốc.

### 35.3 Chạy được mà không sửa một dòng code model

`src/build_edit_labels.py` chỉ **ghi đè trường `cwe_class`** bằng nhóm kích thước sửa đổi.
`resolve_cwe_class` vốn ưu tiên `cwe_class` khi có sẵn, nên chạy với `AUX_MODE=cwe` là head phụ học
tín hiệu mới. Đã sinh `data/train_ccpp_js_editsize.jsonl` và kiểm chứng:

```
so dong giu nguyen : True      code khong doi  : True
truong giu nguyen  : True      label khong doi : True
chi cwe_class doi  : True      gia tri moi     : [-100, 0, 1, 2, 3]
```

8 dòng không ghép cặp được gán **−100**, đúng `ignore_index` của `cross_entropy`, nên chúng không
vào loss thay vì âm thầm thành một lớp giả.

Nhờ tối giản như vậy, phép so **"cùng kiến trúc, khác tín hiệu"** là sạch tuyệt đối: chỉ đúng một
biến thay đổi giữa hai nhánh.

### 35.4 Cách chạy khi có máy trở lại

```bash
BACKBONES="codebert=microsoft/codebert-base:cls t5p=Salesforce/codet5p-220m:cls" \
MODES="cwe none" SEED=42 RUN_NAME=gate_edit \
PHASE1_DATA_PATH=data/train_ccpp_js_editsize.jsonl \
  bash run/gated.sh
```

Đối chiếu `gate_edit_*` với `gate1_*` (cùng seed 42, cùng backbone, cùng kiến trúc) là đo trực tiếp
**tín hiệu phụ nào tốt hơn**. Điều cần nhìn không phải Δ tuyệt đối mà là **`Δ(edit) − Δ(none)` trên
CodeT5+** — §34.3 cho thấy đại lượng đó ở tín hiệu CWE là −0.0137 và +0.0124, tức gần như bằng
không. Nếu tín hiệu mới cũng ra gần không thì hướng này bị bác và nên bỏ.

### 35.5 Tín hiệu mới có thật sự mới không

Đây là phép kiểm có thể **giết hướng này miễn phí**, và tôi suýt bỏ qua: nếu nhóm kích thước sửa
đổi chỉ là CWE trá hình thì nó chẳng mang thông tin gì backbone chưa có, và toàn bộ §35 vô nghĩa.

Đo thông tin tương hỗ chuẩn hoá trên 1276 dòng có nhãn:

| Cặp đại lượng | NMI |
| --- | --- |
| `edit_class` vs **CWE gốc** | **0.0286** |
| `edit_class` vs **nhãn nhị phân** | **0.0000** |
| CWE gốc vs nhãn nhị phân | 0.0000 |

**NMI với CWE gần như bằng không** — hai tín hiệu gần như độc lập hoàn toàn. Bảng chéo xác nhận:
mỗi CWE trải đều qua cả bốn nhóm, không lớp nào bị một nhóm chiếm quá 36%.

| CWE | 1 dòng | 2-3 | 4-10 | >10 |
| --- | --- | --- | --- | --- |
| CWE-079 (824) | 296 | 208 | 198 | 122 |
| CWE-078 (188) | 42 | 36 | 60 | 50 |
| CWE-022 (132) | 22 | 16 | 46 | 48 |
| CWE-089 (132) | 28 | 20 | 40 | 44 |

**NMI với nhãn nhị phân bằng đúng 0** cũng quan trọng không kém, và nó đúng theo thiết kế: hai nửa
của một cặp nhận **cùng** nhãn kích thước sửa đổi, nên tín hiệu này **không thể rò rỉ** task chính.
Head phụ định hình biểu diễn mà không đi tắt.

Đây là hồ sơ lý tưởng cho một task phụ: **mới so với CWE, và mù với nhãn cần dự đoán**. Hướng §35
sống sót phép kiểm này, và giờ đã được kiểm chứng trên ba trục — dữ liệu có đủ, phân bố không suy
biến, và trực giao với các tín hiệu đã có.

### 35.6 Rủi ro đã đo trước

**26% số cặp có bản lỗi dài hơn 60 dòng.** Ở đó chỗ sửa có thể nằm ngoài cửa sổ 512 token, khiến
nhãn trở thành thứ model không nhìn thấy được. Đây là **giả thuyết cần kiểm chứ không phải khiếm
khuyết đã biết** — §18.3 từng cho thấy một giả thuyết truncation nghe rất hợp lý mà đo ra chỉ 1.8%.
Cách kiểm rẻ: so Δ trên nhóm cặp ngắn với nhóm cặp dài.

---

## 36. Mục tiêu có thể đang đặt sai — và điều đó cần bạn quyết

§34 cho thấy cơ chế: phương pháp có tác dụng nhờ **lấp dư địa ở lớp CWE hiếm**, và giá trị gia
tăng của head phụ tắt hẳn khi backbone đã tự biểu diễn được lớp đó.

Nếu cơ chế này đúng thì nó kéo theo một hệ quả khó chịu về chính mục tiêu:

> Một phương pháp **hoạt động bằng cách lấp dư địa** thì về bản chất **không thể** cho lợi ích như
> nhau trên mọi backbone, vì backbone mạnh hơn để lại ít dư địa hơn. Yêu cầu "không phụ thuộc
> pretrained" theo nghĩa *lợi ích không đổi theo backbone* có thể là **yêu cầu bất khả** đối với
> họ phương pháp này, chứ không phải một khiếm khuyết cần sửa.

### 36.1 Nhưng tôi **không** chứng minh được điều đó

Chỉ có **hai** cấu hình sạch để kiểm (cùng `cls`, cùng folds twin):

| Cấu hình | baseline | Δ | Δ / dư địa |
| --- | --- | --- | --- |
| CodeBERT cls, 5 seed | 0.8275 | +0.0362 | 0.210 |
| CodeT5+ cls, 3 seed | 0.8806 | +0.0062 | 0.052 |

**Hai điểm không fit được đường nào.** Kết luận "lợi ích tỉ lệ với dư địa" từ hai điểm chính là lỗi
n nhỏ mà tài liệu này cảnh báo suốt — §25.1 đã cho thấy ba điểm cho Pearson r = 0.989 rồi sụp còn
0.699 ở điểm thứ tư. Nên đây là **giả thuyết**, không phải quy luật.

Muốn kiểm cần **ít nhất 4–5 backbone** trải rộng độ mạnh (ví dụ thêm GraphCodeBERT, UniXcoder,
CodeT5-large), mỗi cái chạy theo Cổng 1 ở §30.

### 36.2 Hai cách phát biểu mục tiêu, và hệ quả khác nhau

| Cách phát biểu | Trạng thái hiện tại | Kiểm bằng gì |
| --- | --- | --- |
| **A.** Lợi ích **không đổi** theo backbone | **chưa đạt**, và có thể bất khả nếu §36 đúng | cần 4–5 backbone |
| **B.** Phương pháp **không gây hại** trên mọi backbone, và **có lợi khi còn dư địa** | **gần đạt**: `cwe` +0.0062 trên CodeT5+ (trong nhiễu), nhánh latent −0.0086 (hại nhẹ) | §35 kiểm được |

Cách **B** là thứ dữ liệu hiện có gần chạm tới, và nó vẫn là một đóng góp dùng được: biết trước khi
nào nên dùng phương pháp và khi nào không. Cách **A** thì cần chứng minh cơ chế sai, hoặc tìm được
tín hiệu phụ mang thông tin **backbone mạnh vẫn thiếu** — đúng thứ §35 đang chuẩn bị kiểm.

**Đây là quyết định của bạn, không phải của tôi.** Tôi nêu ra vì nếu mục tiêu giữ nguyên cách A mà
cơ chế §34 đúng thì mọi công sức tiếp theo sẽ đổ vào một đích không tới được — và điều đó đáng biết
trước khi thuê máy trở lại.

---

## 37. Tín hiệu kích thước sửa đổi **bị bác** — và vì sao lập luận §35 thiếu một vế

`run/edit-gate.sh` chạy ba nhánh trên cùng một baseline, hai backbone, hai máy, seed 42.
Cổng chặn phán **dừng** ở cả hai.

### 37.1 Số liệu

Giá trị gia tăng của head phụ, `Δ − Δ(none)`, trên fold ghép cặp:

**Bảng này là bản n=5. Bản n=3 tôi viết trước đó kết luận sai — xem §37.1.1.**

| Backbone | Metric | `cwe` | `edit` | dương | bỏ fold tốt nhất |
| --- | --- | --- | --- | --- | --- |
| CodeBERT | Macro-F1 | **+0.0477** | **+0.0288** | 3/5 | **+0.0191** |
| CodeBERT | ROC-AUC | **+0.0371** | **+0.0293** | — | — |
| CodeT5+ | Macro-F1 | −0.0013 | +0.0011 | 2/5 | −0.0116 |
| CodeT5+ | ROC-AUC | −0.0017 | −0.0073 | — | — |

Δ từng fold của `edit`:

```
CodeBERT  -0.0204  -0.0086  +0.0593  +0.0460  +0.0678
CodeT5+   +0.0520  -0.0065  -0.0133  -0.0331  +0.0064
```

**Trên CodeBERT, `edit` là một tín hiệu THẬT.** +0.0288 trên Macro-F1 và +0.0293 trên ROC-AUC, dương
3/5 fold, và **sống sót phép bỏ fold tốt nhất** (+0.0191). Nó chỉ **yếu hơn** `cwe` — bằng 60% trên
Macro-F1 và 79% trên ROC-AUC — chứ không phải nhiễu.

**Trên CodeT5+, cả hai tín hiệu đều bằng không.** `cwe` −0.0013, `edit` +0.0011, và `edit` âm trên
ROC-AUC (−0.0073).

### 37.1.1 Kết luận ở n=3 của tôi sai, và sai theo hướng nào

Ở n=3 tôi viết: *"`edit` kém `cwe` gần năm lần (+0.0101 so với +0.0481)"* và *"bỏ fold tốt nhất thì
cả hai đều âm"*. Ở n=5:

| | n=3 | n=5 |
| --- | --- | --- |
| `edit` trên CodeBERT | +0.0101 | **+0.0288** |
| dương | 1/3 | **3/5** |
| bỏ fold tốt nhất | **−0.0145** | **+0.0191** |

Fold 4 (+0.0460) và fold 5 (+0.0678) đều mạnh, và chúng đảo hẳn bức tranh. Đây là **lần thứ năm**
trong tài liệu này một tín hiệu ở n=3 bị đọc sai và n=5 sửa lại — lần này theo hướng **tôi đánh giá
thấp** thay vì đánh giá cao.

Bài học tương ứng: quy tắc "n=3 chỉ đủ để DỪNG, không đủ để KẾT LUẬN" ở §30 phải áp dụng **cả hai
chiều**. Tôi đã dùng nó để tránh kết luận quá sớm rằng một thứ *hoạt động*, nhưng lại quên nó khi
kết luận sớm rằng một thứ *không hoạt động*.

### 37.1.2 Điều bức tranh n=5 thật sự nói

Hai tín hiệu phụ **trực giao nhau** (NMI 0.029) đều:

- **hoạt động rõ trên CodeBERT** — +0.0477 và +0.0288
- **bằng không trên CodeT5+** — −0.0013 và +0.0011

Đây là bằng chứng **mạnh hơn** cho cơ chế §34 so với trước. Nếu chỉ có `cwe` thất bại trên CodeT5+
thì còn có thể đổ tại "nhãn CWE tình cờ dư thừa với backbone đó". Nhưng một tín hiệu **hoàn toàn
khác** cũng thất bại y hệt, trong khi cả hai đều hoạt động trên CodeBERT, thì lời giải thích hợp lý
không nằm ở **tín hiệu** mà nằm ở **backbone**: CodeT5+ không còn dư địa để bất kỳ head phụ nào lấp.

Điều đó cũng có nghĩa: **đi tìm tín hiệu phụ thứ ba là hướng sai.** Vấn đề không phải chọn sai tín
hiệu.

### 37.2 Lập luận §35 thiếu vế nào

§35 kiểm tín hiệu mới trên ba trục: dữ liệu có đủ, phân bố không suy biến, và **trực giao với CWE**
(NMI 0.029). Cả ba đều đạt, nên tôi kết luận hướng này đáng chạy.

Vế thiếu là **tính liên quan**. Trực giao với CWE nghĩa là tín hiệu mang *thông tin khác*, nhưng
không bảo đảm thông tin đó **liên quan đến việc phát hiện lỗ hổng**. Ở n=3 tôi kết luận vế thiếu là **tính liên quan** — rằng số dòng cần sửa không dạy model gì về việc
nhận ra lỗ hổng. **Số liệu n=5 bác luôn cách đọc đó**: trên CodeBERT tín hiệu này cho +0.0288, tức
nó **có** liên quan, chỉ là kém hơn nhãn CWE.

Vế thiếu thật sự nằm chỗ khác, và §37.1.2 chỉ ra: tôi giả định rằng vì `cwe` dư thừa trên backbone
mạnh nên một tín hiệu trực giao sẽ không dư thừa. Giả định đó sai — trên CodeT5+ **cả hai** đều
bằng không. Vấn đề không phải tín hiệu dư thừa mà là **không còn dư địa cho bất kỳ tín hiệu nào**.

Đây là bài học có thể kiểm được trước khi tiêu GPU lần sau: một tín hiệu phụ ứng viên phải qua
**cả hai** cửa — trực giao với thứ backbone đã biết, **và** gắn với ngữ nghĩa lỗ hổng.

### 37.3 Điều thí nghiệm này vẫn xác lập

Không phải công cốc. Nhánh `cwe` chạy song song trên cùng máy, cùng baseline, và cho:

| Backbone | `cwe` cộng thêm so với `none` |
| --- | --- |
| CodeBERT | **+0.0481** (Macro-F1), **+0.0313** (ROC-AUC) |
| CodeT5+ | −0.0049 (Macro-F1), −0.0017 (ROC-AUC) |

Đây là lần thứ tư đại lượng này được đo trên CodeT5+, qua ba GPU khác nhau, và **luôn quanh 0**.
Cơ chế ở §34.3 — head phụ tắt tác dụng khi backbone đã tự biểu diễn được lớp hiếm — giờ đã được
tái lập trên phần cứng mới, với baseline mới, ở n=5.

### 37.4 Một giả thuyết tôi kiểm trước khi nêu, và nó sai

Nhìn số seed 42 tôi thấy một mẫu hình hấp dẫn: trên CodeBERT `Δ(none)` là **−0.0133** (pretrain
trần **có hại**, head phụ cứu lại), còn trên CodeT5+ là **+0.0149** (pretrain trần **đã tốt**, không
còn gì để cứu). Nếu đúng thì nó giải thích gọn ghẽ mọi thứ: head phụ là **cơ chế sửa chữa**, và
CodeT5+ không cần sửa.

Kiểm trên toàn bộ dữ liệu đã có thì mẫu hình đó **không tồn tại**:

| | Δ(none) |
| --- | --- |
| CodeBERT s36 / s12 / s7 / s42 / s18 | −0.0025, +0.0184, +0.0079, +0.0090, **+0.0264** |
| CodeT5+ s42 / s18 | +0.0081, **−0.0144** |

Dấu **không ổn định theo backbone** — CodeBERT phần lớn dương, CodeT5+ có cả hai dấu. Mẫu hình tôi
thấy chỉ là đặc thù của một seed trên một máy.

Đáng ghi thêm: `Δ(none)` của CodeBERT seed 42 là **+0.0090 trên máy cũ** nhưng **−0.0133 trên máy
mới** — cùng seed, cùng cấu hình, khác máy, **đảo dấu**. Đúng bằng chênh lệch phần cứng 0.028 đã đo,
và là lý do luật "baseline chạy lại trên mỗi máy" tồn tại.

### 37.5 Hai lỗi trong công cụ của chính tôi

`src/report_edit_gate.py` bản đầu phán **"edit TỐT HƠN cwe — đáng chạy tiếp"** cho cấu hình chỉ
dương 1/3 fold, vì nó chỉ nhìn trung bình. Bản sau lại **trừ hai trung bình tính trên số fold khác
nhau** (`edit` n=3 với `none` n=5).

Cả hai đúng là lỗi đã khiến khẳng định "gấp mười lần" ở §21 phải rút lại — lần này nằm trong code
thay vì trong câu chữ. Đã sửa: phán quyết dựa trên fold ghép cặp, và kiểm cả số fold dương lẫn kết
quả sau khi bỏ fold tốt nhất.
