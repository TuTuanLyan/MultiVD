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
