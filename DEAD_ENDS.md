# Ngõ cụt — đã kiểm và đã bác

Danh sách này chỉ có một mục đích: **không chạy lại thứ đã trả lời rồi**. Nó không nói phương
pháp nên đi hướng nào; phần đó thuộc về số liệu trong `FACTS.md` và về phán đoán của người đọc.

Mỗi mục ghi ba thứ: giả thuyết, phép đo đã bác nó, và điều kiện nào sẽ khiến nó đáng mở lại.
Bản gốc dài của các lập luận nằm ở `archive/RESULT_2026-08-23.md` — đọc nó khi cần truy nguồn
một con số, **không** đọc khi đang thiết kế thí nghiệm mới.

---

## A. Về nguyên nhân của hiệu ứng transfer

| # | Giả thuyết | Phép đo bác nó | Mở lại khi nào |
| --- | --- | --- | --- |
| 1 | Rò rỉ cặp vulnerable/fixed tạo ra hiệu ứng | Dựng lại fold group-aware. Hiệu ứng **tăng** +0.029 → +0.042 | Không. Đã kiểm trên ba bộ fold. |
| 2 | Truncation làm hai bản của một cặp thành cùng token | Tokenise mọi cặp: chỉ **1.8%** trùng khít id | Nếu đổi `max_length` xuống dưới 256 |
| 3 | Phase 1 làm méo đặc trưng của backbone mạnh | Ba can thiệp độc lập — LP-FT, nội suy α, LoRA r=8 — đều **không** cứu được. LoRA đóng băng 99.73% backbone mà kết quả trùng khít với không can thiệp | Nếu tìm được cách đo "méo" trực tiếp thay vì suy từ kết quả |
| 4 | Cực tiểu Phase 1 của backbone hỏng thì **nhọn** hơn | Đo độ nhọn đủ **5 backbone**: thứ tự độ nhọn không khớp thứ tự bất kỳ đại lượng nào. Hai backbone phẳng nhất (t5pe 86%, t5p 101%) cho kết quả transfer **ngược dấu** nhau dù encoder giống hệt | Không. Đây là phép đo trực tiếp trên chính đại lượng đó. |

## B. Về thiết kế head phụ và source

| # | Giả thuyết | Phép đo bác nó | Mở lại khi nào |
| --- | --- | --- | --- |
| 5 | Bottleneck cứu được taxonomy CWE lớn | PrimeVul 121 CWE, 5 fold: `cwe` +0.004, `bottleneck` −0.017 | Nếu head phụ chuyển sang dạng khác hẳn (không phải bottleneck tuyến tính) |
| 6 | Quy mô source mua được hiệu quả | Cùng backbone/fold/baseline: 1284 dòng → **+0.042**; 9408 dòng → **+0.004** | Không. Gấp bảy lần dữ liệu, kém mười lần. |
| 7 | `common` hơn `full` "gấp mười lần" | Ghép cặp theo fold: thắng **2/4 fold**, sd 0.0656, bỏ một fold thì đảo dấu. Hai trung bình gốc tính trên n=3 và n=5 | Đây là lỗi phương pháp, không phải giả thuyết. Đã có `src/report_paired.py` để chặn. |
| 8 | Chọn lần rút Phase 1 theo val nguồn thì được checkpoint tốt hơn | val nguồn sd **0.0521** giữa các lần rút, nhưng Δ transfer sd chỉ **0.0085** và dương 5/5 lần rút kể cả lần hỏng. Val nguồn **không** dự báo Δ | Không |
| 9 | Kích thước bản vá là tín hiệu phụ thay thế được nhãn CWE | Chạy đủ, kết quả âm | Nếu tìm được tín hiệu phụ khác có thông tin backbone mạnh còn thiếu |

## C. Về RecAdam

| # | Giả thuyết | Phép đo bác nó | Mở lại khi nào |
| --- | --- | --- | --- |
| 10 | RecAdam gây hại khi target ít dữ liệu | Ablation AdamW cùng checkpoint nguồn, 114 dòng: bỏ RecAdam chỉ lấy lại **0.021 trên 0.105** thiệt hại | Đã mở lại — xem #11 |
| 11 | Bỏ RecAdam tốt hơn ở 7/8 nhánh | **Sai vì cách tính.** Ghép cặp theo fold trên đủ 5 fold: 6/8 nhánh nằm trong nhiễu, toàn bộ biên độ đến từ **một** nhánh (`latent_proto` trên CodeT5-base, +0.2702, 5/5 fold) — và đó là cú sập chứ không phải cải thiện chung | Đang mở. Ma trận reset chạy cả hai optimizer cạnh nhau trong cùng fold. |
| 12 | RecAdam ở pipeline này chỉ là một lịch warmup nên bỏ đi thay đổi rất ít | Phép tính tốc độ thả neo (`lr × pretrain_cof = 0.1`, còn 9.1% sau 1 epoch) **đúng**, nhưng suy luận từ nó sai: một epoch bị ghìm đủ đổi hẳn quỹ đạo | Không. Bài học: "thả nhanh" không đồng nghĩa "ảnh hưởng ít". |

---

## D. Lỗi phương pháp đã tìm ra trong chính tài liệu cũ

Không phải giả thuyết, mà là **cách đo sai** — đáng ghi vì cả bốn cùng một dạng: một quy ước
gộp hoặc loại trừ được chọn **sau khi** nhìn số.

| Ở đâu | Sai gì | Đã chặn bằng |
| --- | --- | --- |
| §48.1–48.2 | So hai trung bình giữa hai đợt chạy; 4/8 ô không dựng lại được từ file thô | `src/report_paired.py` — chỉ dùng fold có ở cả hai nhánh, in n |
| Quy ước `*` bỏ fold Macro-F1 < 0.55 | Biến −0.2345 (n=5) thành −0.1156 (n=2), che đúng hiện tượng cần thấy | `report_paired.py` in **cả hai** bản cạnh nhau, không cho chọn sau |
| §39 (target JS) | Bằng chứng trung tâm cho cơ chế "dư địa" nằm trọn trong vùng ROC-AUC 0.498–0.650, tức mọi nhánh gần như không phân biệt được vul/non-vul | `report_fold.py` đánh dấu `~` cho AUC < 0.65 |
| `sam-gate.sh:50` | Tìm Phase 1 ở `model/fam1_emb/` trong khi run tên `emb1_emb`; `cp` trượt im lặng, nhánh tự huấn luyện Phase 1 khác. Phép so "chỉ đổi SAM" thực ra đổi hai biến | `run/matrix.sh` dùng kho Phase 1 khoá theo `(backbone, nhánh, seed)`, không sao chép giữa thư mục |
| `check_phase1.py` | Chỉ bắt `best_epoch ≤ 1`, nên bỏ lọt một Phase 1 dừng ở epoch 2 với val **0.5109** (ngang ngẫu nhiên) | chưa chặn — xem `FACTS.md` mục việc còn dở |

---

## E. Ràng buộc thống kê cứng, không phải giả thuyết

- **Với n=5 fold, Wilcoxon signed-rank không thể trả về p < 0.0625.** Một dòng 5 fold vì thế
  không bao giờ đạt p < 0.05 dù hiệu ứng lớn đến đâu; `p = 0.0625` ở n=5 chỉ có nghĩa "cùng dấu
  ở cả 5 fold". Muốn xuống dưới 0.05 phải có ≥ 6 quan sát ghép cặp, tức thêm seed hoặc thêm fold.
- **sd giữa các fold tới 0.09** trên tập test 152 mẫu, nên Δ cỡ 0.00x không phân biệt được với
  nhiễu, và **một fold có thể mang cả kết luận**.
- **Chênh lệch phần cứng đo được là 0.028 Macro-F1** — lớn hơn phần lớn hiệu ứng đang đo. Một
  nhóm so sánh (baseline + các nhánh của nó, cùng seed, cùng fold) phải nằm trên một máy.
- **Phase 1 không tái lập** kể cả khi cố định seed: ba lần chạy cùng seed cho val nguồn sd
  **0.0521**. Nhưng Δ transfer sd chỉ **0.0085**, nên sàng lọc trên Δ ở một seed là hợp lệ.
- **n=3 đã bốn lần đổi dấu hoặc co lại ở n=5** trong dự án này. Ba fold đủ để **dừng**, không đủ
  để **kết luận**.
