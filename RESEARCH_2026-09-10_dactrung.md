# Đêm 10→11/09 — Cái gì THẬT SỰ chuyển giao, và ba can thiệp suy ra từ đó

> **Đọc mục 1 là đủ nắm.** Mục 2–4 là chi tiết từng khối: phương pháp, setting đầy đủ, n, kết quả.
> Mục 5 là tra cứu tài liệu. Mục 6 là đề xuất bước tiếp, **chưa chạy, chờ anh duyệt**.
>
> Không thuê vast đêm nay. Toàn bộ chạy trên 161 + 158.

---

## 1. Tóm tắt

Đêm nay có **một phát hiện cơ chế** và **một khối thực nghiệm âm tính có ích**.

**Phát hiện (§36).** Lần đầu đo THẲNG cái mà cả dự án từ trước tới nay chỉ SUY RA:

| | đặc trưng có chuyển giao không? | hàm quyết định có chuyển giao không? |
|---|---|---|
| **codebert** | **CÓ — linear probe +0.1125 ROC-AUC, 5/5 fold** | **KHÔNG — 0.537 F1 / 0.645 ROC zero-shot** |
| t5p | gần như không (+0.0202 ROC, 3/5) | KHÔNG — 0.503 F1 / 0.545 ROC |

Nghĩa là: Pha 1 (chỉ thấy C/C++ và JavaScript) làm không gian đặc trưng của **Python** tách được
tuyến tính tốt hơn hẳn, dù nó chưa từng thấy một dòng Python nào. Nhưng cái đầu ra nhị phân nó học
được thì vô dụng trên Python — gần đúng mức đoán bừa.

Đây là câu trả lời trực tiếp cho phản biện *"chỉ là fine-tune hai lần"*: nếu Pha 1 chỉ là "huấn
luyện thêm" thì cả hai thứ phải cùng chuyển giao. Đằng này **một thứ chuyển, một thứ không** — và
điều đó nói cho ta biết chính xác nên can thiệp ở đâu.

**Khối âm tính (§37, `bridge3`, n=3).** Thử đưa dữ liệu nguồn thẳng vào Pha 2 (replay) và cho head
CWE học tiếp trên nhãn CWE của Python. Kết quả: cho head CWE học trên đích **nâng F1@0.5 nhưng hạ
AUC** trên cả hai backbone — đó là hiệu ứng ngưỡng, không phải cải thiện thật. Replay thuần thì
dương cả bốn chỉ số nhưng **chỉ trên t5p**, codebert null ⇒ chưa lặp qua backbone ⇒ mới là giả
thuyết.

**Đang chạy (`feat3`, n=3, xong khoảng 19:00–19:30 UTC).** Hai can thiệp suy trực tiếp từ phát hiện
trên, cả hai **chưa từng chạy trong dự án**: neo trong **không gian đặc trưng**, và **khởi tạo lại**
cái head vô dụng kia.

---

## 2. §36 — Probe: đặc trưng chuyển giao, hàm quyết định không

### Phương pháp

`tools/feature_probe.py`. **Chạy CPU, 0 GPU** — không đụng vào máy đang chạy khối khác.

1. Lấy **760 dòng Python** (toàn bộ `sven_python_folds_norm`, gộp mọi fold, khử trùng theo nội dung code).
2. Trích đặc trưng pooled từ **hai mô hình đóng băng hoàn toàn**, không fine-tune gì:
   - **(a)** backbone pretrained nguyên bản — `microsoft/codebert-base` / `Salesforce/codet5p-220m-bimodal`, chưa hề thấy dữ liệu lỗ hổng
   - **(b)** checkpoint Pha 1 — `latent_bottleneck`, nguồn `4cwe`, λ=0.05, seed 42
   - lấy **trước dropout**: CLS cho codebert, mean-pool cho t5p (đúng `pooling` của từng backbone)
3. Mỗi fold: chuẩn hoá theo train, fit **logistic regression** trên 456 dòng train, chọn `C` trên
   val theo ROC-AUC (lưới 0.01…3.0), chấm test.
4. Δ = (b) − (a), **ghép cặp theo fold**.
5. Cột thứ ba: **zero-shot** — dùng thẳng `vul_head` của Pha 1 trên đặc trưng (b), không fit gì.

### Setting

| | |
|---|---|
| **n** | **5 fold**, seed 42 (đây là phép đo trên checkpoint đã có, không phải khối huấn luyện) |
| dữ liệu | `data/sven_python_folds_norm`, 760 dòng, 456/152/152 mỗi fold |
| checkpoint | `model/n48/phase1/{codebert,t5p}__latent_bottleneck_4cwe_l0p05/seed_42/best.pt` |
| val Pha 1 | codebert 0.6532 (ep 6) · t5p 0.6976 (ep 6) |
| max_length | 512, truncation `head_middle_tail` |
| chi phí | ~45 phút CPU/backbone, 3 luồng, `nice -n 19` |

### Kết quả

**Δ (Pha 1 LP − pretrained LP), ghép cặp theo fold, n=5:**

| | ΔF1@0.5 | ΔROC-AUC | ΔPR-AUC |
|---|---|---|---|
| **codebert** | **+0.0849 5/5** | **+0.1125 5/5** | **+0.1010 5/5** |
| t5p | +0.0198 4/5 | +0.0202 3/5 | +0.0097 3/5 |

codebert: cả ba chỉ số dương ở **cả năm fold**, biên độ gấp 4–11 lần sàn nhiễu 0.010.

**Zero-shot của `vul_head` Pha 1 trên Python** (dải qua 5 fold):

| | F1@0.5 | ROC-AUC |
|---|---|---|
| codebert | 0.537 (0.488–0.591) | 0.645 |
| t5p | 0.503 (0.474–0.542) | 0.545 |

t5p ROC 0.545 ≈ ngẫu nhiên, khớp với khối INT1 cũ (α=0 cho 0.5205).

**Theo CWE — hai backbone học hai thứ khác nhau** (ΔROC-AUC của linear probe):

| | 022 | 078 | 079 | 089 |
|---|---|---|---|---|
| codebert | +0.169 **5/5** | +0.199 **5/5** | +0.159 **5/5** | +0.046 4/5 |
| t5p | −0.023 2/5 | +0.027 2/5 | **+0.173 5/5** | −0.017 2/5 |

Trên **t5p, thứ duy nhất chuyển giao là CWE-079** — đúng lớp chiếm **74%** nguồn `4cwe` (692/930
dòng là CWE-79). Trên codebert thì cả bốn lớp đều lên.

### Một nghịch lý đáng chú ý

codebert được **+0.199 ROC 5/5** cho CWE-078 trong không gian đặc trưng, nhưng end-to-end
(§34/§35) CWE-078 **null ở mọi phép đo**. Tức là fine-tune Pha 2 **không dùng** phần đặc trưng đã
tốt lên đó. Giả thuyết: 456 dòng train đủ để mô hình tự tìm lời giải riêng cho hai lớp lớn (078 có
40 hàng test, 089 có 82), nên lợi thế nguồn chỉ còn sống ở hai lớp hiếm. Chưa kiểm.

### Cảnh báo khi trích số

Bộ `norm` chia theo dòng nên ~40% hàng test có bản gần trùng trong train. **Cả hai vế của Δ đều
chịu chung**, nên Δ ghép cặp vẫn hợp lệ, nhưng **con số tuyệt đối là lạc quan**. Linear probe tuyệt
đối thấp hơn fine-tune đầy đủ nhiều (codebert LP ROC 0.73–0.82 so với FT ~0.88) — đúng như kỳ vọng.

Số đầy đủ: `results/probe/codebert_4cwe.json`, `results/probe/t5p_4cwe.json`.

---

## 3. Khối `bridge3` — replay nguồn + cầu CWE ở Pha 2 (**xong**, n=3)

### Ý tưởng

Bốn khối cũ (OPT1 / RET1 / SPD1 / INT1) đã đóng hẳn trục **không gian trọng số**: RecAdam ở mọi γ,
SPD, Fisher, WiSE-FT — tất cả null. Lý do cơ chế: cái neo chỉ **ràng buộc** đường đi, nó không mang
thêm một bit thông tin nào từ dữ liệu nguồn. Đường duy nhất chưa thử: cho **gradient của dữ liệu
nguồn** nặn trực tiếp nghiệm đích.

Thêm một ý: nguồn `4cwe` và đích Python dùng **chung đúng bảng 4 lớp** CWE-022/078/079/089, nên head
phụ của Pha 1 có thể học tiếp trên **cả hai ngôn ngữ** — gọi là "cầu CWE".

### Bốn nhánh

| tag | cờ Pha 2 | đo cái gì |
|---|---|---|
| `plain` | — | **đối chứng**: fine-tune hai lần thuần |
| `cwe05` | `--phase2_lambda_cwe 0.05` | head CWE học tiếp trên nhãn CWE **của Python** |
| `rp50` | `--replay_data data/phase1_4cwe.jsonl --replay_mu 0.5 --replay_epochs 6 --replay_stratify` | mỗi bước thêm 1 batch nguồn, μ giảm tuyến tính 0.5→0 sau 6 epoch, cân tầng (nhãn, CWE) |
| `rpc` | cả hai + `--replay_lambda_cwe 0.05` | cầu CWE đầy đủ |

### Setting

| | |
|---|---|
| **n** | **3 fold** (1, 2, 3), seed 42 — **bậc 1, chỉ để sàng lọc** |
| quy mô | 4 nhánh × 3 fold × 2 backbone + baseline = **30 ô** |
| máy | codebert trên **161** (A4000, dùng chung với `cuongtm`), t5p trên **158** (A4000, trống) |
| Pha 1 | **dùng lại**, không huấn luyện lại cái nào |
| optimizer Pha 2 | AdamW trần, `--sam_rho 0` (cấu hình chốt sau khi rút ASAM ở §35.2) |
| epochs / batch / lr | 30 / 16 / 2e-5, `min_epochs` 3, patience 5, chọn checkpoint theo macro-F1 val |
| đối chứng | `baseline` (không Pha 1) **cùng máy cùng fold** |
| thời gian | 2 h 17 mỗi máy |

Chi tiết μ: `0.500 0.417 0.333 0.250 0.167 0.083 0.000 …` — về 0 sau epoch 6, nên **hàm quyết định
cuối cùng vẫn thuần đích**; nguồn chỉ nặn đặc trưng giai đoạn đầu. Nguồn dùng 834/930 dòng (phần
train của Pha 1, giữ nguyên tập val Pha 1 làm held-out).

### Kết quả — Δ so với `plain`, ghép cặp cùng fold, n=3

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
   F1@0.5 mà **hạ** ROC và PR (0–1/3 fold). Cho head phụ học nhãn CWE của đích không làm mô hình
   xếp hạng tốt hơn — nó chỉ dịch ngưỡng. Nếu chỉ nhìn F1 thì sẽ tưởng nhầm là thắng.
2. **`rpc` (ghép cả hai) HẠI trên AUC ở cả hai backbone** — ROC 0/3 và PR 0/3. Phần `cwe05` kéo
   xuống nhiều hơn phần replay kéo lên.
3. **`rp50` (replay thuần) dương cả bốn chỉ số — nhưng chỉ trên t5p.** codebert null (1/3 ba lần).
   **Không lặp qua backbone ⇒ giả thuyết, không phải phát hiện.** Đáng lên n=5 nếu anh muốn.

**Per-CWE:** so với `plain`, trên codebert cả hai nhánh có replay đều nâng CWE-022 **3/3 fold**
(`rp50` F1 +0.125 / ROC +0.113; `rpc` +0.118 / +0.157). t5p không lặp. CWE-022 chỉ **14 hàng
test/fold** nên biên độ không đáng tin; thứ đáng chú ý là 3/3 cùng dấu.

Đọc lại bằng: `python3 tools/bridge_report.py results/bridge3_codebert` (và `_t5p`).

---

## 4. Khối `feat3` — ĐANG CHẠY, xong khoảng 19:00–19:30 UTC (02:00–02:30 sáng VN)

Hai can thiệp **suy trực tiếp từ §36**, cả hai chưa từng chạy trong dự án.

### 4.1 Neo trong KHÔNG GIAN ĐẶC TRƯNG (`fd`)

Cộng vào loss Pha 2: **β · (1 − cos(f_θ(x), f_θ*(x)))** trên **input đích**, với f_θ* là đặc trưng
của chính mô hình Pha 1.

Vì sao khác hẳn RecAdam/L2-SP/SPD/Fisher (bốn khối đã bác):

- Neo **trọng số** ràng buộc θ, mà θ thì **không biết gì về dữ liệu đích**. Nó chỉ làm chậm việc
  rời khỏi θ*.
- Neo **đặc trưng** ràng buộc *hàm* trên **chính các dòng Python đang huấn luyện**. Và nó giữ đúng
  đại lượng mà §36 vừa đo được là **có giá trị** — chứ không phải giữ bừa.

Dùng **cosine** chứ không phải L2: cosine chỉ giữ **hướng**, bất biến thang đo — đúng thứ mà linear
probe dùng (probe chuẩn hoá trước khi fit). L2 sẽ ghim cả độ dài vector, mà độ dài thì trôi mạnh
khi fine-tune.

**Chi phí: gần bằng 0.** Vì thầy đóng băng và input đích cố định (tokenise tất định, không tăng
cường), đặc trưng thầy **không bao giờ đổi** — nên tính sẵn **một lần** vào bộ đệm 456×768 (1.4 MB),
tra theo chỉ số hàng. Không cần mô hình thứ hai, **0 VRAM thêm, 0 giây thêm mỗi bước**.

### 4.2 Khởi tạo lại `vul_head` (`rh`)

§36 đo được `vul_head` của Pha 1 gần như đoán bừa trên Python (0.537 / 0.503 F1). Nhưng đường chạy
mặc định **nạp nguyên xi** cái head đó (`strict=True`) rồi bắt Pha 2 gỡ một hàm quyết định **sai
một cách tự tin**. Câu hỏi: bỏ nó đi có hơn không? Backbone Pha 1 giữ nguyên, chỉ head là mới.
Chi phí: 0.

### Setting

| | |
|---|---|
| **n** | **3 fold** (1, 2, 3), seed 42 — **bậc 1, sàng lọc** |
| nhánh | `fd1` (β=1) · `fd10` (β=10) · `rh` · `fd10rh` (cả hai) |
| quy mô | 4 nhánh × 3 fold × 2 backbone = **24 ô mới** |
| đối chứng | **dùng lại `plain` + `baseline` của `bridge3`** — cùng máy, cùng fold, cùng ngày, cùng mã (mọi cờ mới mặc định tắt ⇒ đường chạy cũ không đổi một byte). Tiết kiệm 6 ô GPU và vẫn đúng CLAUDE.md mục 4 |
| còn lại | y hệt `bridge3`: nguồn 4cwe, λ=0.05, AdamW trần, Pha 1 dùng lại |

Trục β hai điểm vì đo được d(f, f*) chỉ ~0.05 trong lúc huấn luyện: β=1 chỉ đóng góp ~7% loss (ràng
buộc nhẹ), β=10 đóng góp ~0.5, ngang cross-entropy (ràng buộc thật).

### Kiểm trước khi chạy

`tests/test_feat_anchor.py`, 4 phép **hai chiều**. Phép đắt nhất là **địa chỉ hoá**: bộ đệm tra theo
`batch["index"]`, nếu chỉ số đó không phải chỉ số hàng thì mỗi hàng bị ghép với đặc trưng của hàng
**khác** — loss vẫn chạy, vẫn giảm, vẫn ra số đẹp, và phép neo âm thầm biến thành **nhiễu** mà không
cổng nào báo. Đã kiểm bằng loader **xáo trộn**, đối chiếu với bảng tính thẳng từng hàng: 0/37 hàng
sai. Cộng thêm smoke GPU cả bốn đường (fd, rh, cả hai + ASAM, và đường cũ).

---

## 5. Tra cứu tài liệu

*(mục này được bổ sung khi phần tra cứu chạy xong — xem cuối file)*

---

## 6. Đề xuất bước tiếp — CHƯA CHẠY, chờ anh duyệt

Xếp theo giá trị trên mỗi giờ GPU:

1. **Lên n=5 rồi n=15 cho nhánh nào sống sót ở `feat3`.** Nếu neo đặc trưng thắng ở cả hai backbone
   thì đó là đóng góp phương pháp thật, và nó có một lập luận cơ chế đi kèm mà §36 đã đo — dạng
   "đo cơ chế, rồi thiết kế can thiệp cho đúng cơ chế đó", chính là dạng reviewer khó bắt bẻ.
2. **Probe trên `com`/`full` và trên nhiều seed.** §36 mới đo đúng một nguồn (`4cwe`) và một seed.
   Chi phí: 0 GPU, ~45 phút CPU mỗi ô. Nếu "+0.11 ROC 5/5" lặp qua nguồn thì phát biểu mạnh hẳn.
3. **Probe theo LỚP của backbone** — đặc trưng lớp 6, 9, 12 thay vì chỉ lớp cuối. Trả lời "Pha 1
   đổi gì, ở đâu", và nếu chỉ tầng trên đổi thì có thể chỉ cần fine-tune tầng trên. 0 GPU.
4. `rp50` lên n=5 để xem chuyện "chỉ thắng trên t5p" là thật hay may.
5. Ba việc cũ trong HANDOFF §0.-3 (đối chứng âm nhãn CWE xáo trộn; kiểm rò rỉ near-dup riêng cho
   CWE-022/079; chồng lấn phân phối nguồn/đích) — vẫn chưa duyệt.

**Không tự chạy cái nào trong số này** (CLAUDE.md mục 9).

---

## 7. Mã mới đêm nay

| file | việc |
|---|---|
| `tools/feature_probe.py` | probe §36, chạy CPU |
| `tools/bridge_report.py` | đọc khối, in **cả bốn** chỉ số + per-CWE, ghép cặp theo (cây, seed, fold), ngưỡng hoà 1e-12 |
| `src/replay.py` | replay nguồn ở Pha 2, μ(e), lấy mẫu cân tầng |
| `src/train.py` | vòng Pha 2 tách `_phase2_target_loss` / `_phase2_replay_loss`; hai backward nối tiếp (không tràn VRAM); SAM lượt 2 tính lại **đúng** mục tiêu |
| `src/train_transfer.py` | cờ `--replay_*`, `--phase2_lambda_cwe`, `--feat_distill_beta/mode`, `--phase2_reinit_head`; bộ đệm đặc trưng; chặn chồng hai không gian nhãn |
| `src/model.py` | `forward(return_features=True)` trả thêm `pooled` |
| `run/bridge3.sh`, `run/feat3.sh` | hai khối |
| `tests/test_replay.py`, `tests/test_feat_anchor.py` | 8 phép kiểm hai chiều |

Mọi cờ mới **mặc định tắt** ⇒ đường chạy cũ không đổi một byte (đã xác nhận bằng smoke: nhánh
`plain` vẫn qua đúng các cổng an toàn cũ).
