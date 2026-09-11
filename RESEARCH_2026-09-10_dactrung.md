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

## 4. Khối `feat3` + `lpft3` — **ĐÃ XONG** (66 ô, n=3, xong 20:39 UTC)

> Kết quả đầy đủ ở FACTS §38 / §38.1 / §38.2 / §38.3. Tóm tắt ở đây.

### 4.0 Kết quả — KHÔNG nhánh nào đủ điều kiện leo n=5

9 nhánh × 2 backbone × 3 fold. Đánh ✓ khi **dương cả bốn** chỉ số so với `plain` cùng fold:

| nhánh | loại can thiệp | codebert | t5p |
|---|---|---|---|
| `lp3` (`--lp_epochs 3`) | **giữ** — fit head trên backbone đóng băng | **✓** +0.0283 F1 3/3, +0.0204 ROC 3/3 | ✗ |
| `fd1` (neo đặc trưng β=1) | **giữ** | **✓** +0.0110 / +0.0066 | ✗ (−0.0351 ROC 0/3) |
| `rp50` (replay nguồn) | **thêm** | ✗ | **✓** +0.0217 / +0.0122 |
| `rhlp3` (reinit head + probe) | **thay** | ✗ | **✓** +0.0021 / +0.0147 ROC 3/3 |
| `rh`, `cwe05`, `rpc`, `fd10`, `fd10rh` | — | ✗ | ✗ |

**Không nhánh nào ✓ ở cả hai backbone**, nên theo luật leo bậc đêm nay không cái nào lên n=5.

### 4.1 Hai điều đáng giữ

1. **Biến thể thắng trên codebert KHÔNG phải LP-FT sách giáo khoa.** `lp3` giữ head Pha 1 rồi
   tinh chỉnh; `rhlp3` khởi tạo lại head rồi probe — đúng công thức Kumar et al. ICLR 2022 — và nó
   **null/âm** trên codebert. Head Pha 1, dù chỉ 0.537 F1 zero-shot, vẫn hơn ngẫu nhiên làm điểm
   xuất phát cho bước probe. Đây là chỗ khác bài gốc, phải nêu rõ nếu viết.
2. **Nhánh thắng tách theo LOẠI can thiệp, và §36 dự báo đúng chiều** (đo **trước** khi chạy):
   codebert có đặc trưng chuyển giao mạnh ⇒ loại **giữ** thắng; t5p thì không ⇒ loại **thêm/thay**
   thắng. Neo quá chặt (β=10) hại ở **cả hai**.

### 4.2 Nhưng phép kiểm khai báo trước đã LÀM YẾU điều (2) — §38.3

Đo Δ probe trên **cả 6 ô** (2 backbone × 3 nguồn), ngưỡng 0.06 **viết ra file trước khi đo**:

| | 4cwe | com | full |
|---|---|---|---|
| codebert | +0.1125 5/5 | +0.0849 5/5 | +0.0849 5/5 |
| t5p | +0.0202 3/5 | +0.0434 4/5 | +0.0122 3/5 |

Ngưỡng tách **sạch** — nhưng tách theo **backbone**, không theo **nguồn**. Nghĩa là Δ probe gần
như là **thuộc tính của backbone**, nên *"probe chọn can thiệp"* rút gọn thành *"backbone chọn can
thiệp"*, mà ta chỉ có **hai** backbone. Hai điểm dữ liệu không dựng được quy tắc, và phép kiểm GPU
mà §38.2 đề xuất **không chạy được như thiết kế**.

Dự đoán của tôi cũng **sai một nửa**: `com`/`full` thấp hơn `4cwe` đúng trên codebert, sai trên t5p
(`com` cao gấp đôi `4cwe`). **Val Pha 1 không dự báo được Δ probe.**

### 4.3 Thiết kế 2×2 đã chạy — setting đầy đủ

| | |
|---|---|
| **n** | 3 fold (1,2,3), seed 42, bậc 1 |
| nhánh mới | `fd1` β=1 · `fd10` β=10 · `rh` · `fd10rh` · `lp3` · `rhlp3` |
| quy mô | 6 nhánh × 3 fold × 2 backbone = 36 ô mới (+30 ô của `bridge3`) = **66 ô** |
| đối chứng | `plain` + `baseline` dùng lại của `bridge3`, **cùng máy cùng fold cùng ngày cùng mã** |
| máy | codebert trên 161, t5p trên 158; **0 vast suốt đêm** |
| thời gian | 161: 17:01→19:33 · 158: 17:01→20:39 (158 mất 25 phút chờ VRAM khi bị chiếm) |

### 4.4 Neo đặc trưng — cách cài (giữ lại vì mã đã có)

Cộng **β·(1 − cos(f_θ(x), f_θ*(x)))** trên **input đích**, f_θ* là đặc trưng của chính mô hình
Pha 1. Thầy đóng băng và input đích cố định ⇒ đặc trưng thầy không bao giờ đổi ⇒ tính sẵn một lần
vào bộ đệm 456×768 (1.4 MB), tra theo chỉ số hàng: **0 VRAM thêm, 0 giây thêm mỗi bước**. Dùng
cosine chứ không phải L2 vì cosine chỉ giữ **hướng** — đúng thứ linear probe dùng.

Kiểm: `tests/test_feat_anchor.py`, 4 phép hai chiều. Phép đắt nhất là **địa chỉ hoá** bộ đệm —
ghép sai hàng thì loss vẫn giảm, số vẫn đẹp, và phép neo âm thầm thành nhiễu. Đã kiểm bằng loader
xáo trộn đối chiếu bảng tính thẳng: 0/37 hàng sai.

## 4bis. (mục cũ, giữ để tra cứu) Khối `feat3` khi mới phóng

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

## 5. Tra cứu tài liệu — 11/09

> Mọi mục dưới đây đã được **fetch trực tiếp** từ arXiv / ACL Anthology / OpenReview / DOI.
> Mục nào không xác minh được nằm ở §5.6 và **không được trích dẫn**.

### 5.1 Kết luận thẳng: neo đặc trưng KHÔNG mới, chẩn đoán MỚI

Phải nói thẳng để khỏi viết sai vào bài:

| thành phần | tình trạng |
|---|---|
| **Neo đặc trưng về mô hình khởi tạo, trên input đích** | **ĐÃ CÓ.** Đúng đối tượng của DELTA (ICLR 2019) và LDIFS (TMLR 2024) |
| **Probe đặc trưng trước/sau fine-tune** | **ĐÃ CÓ.** Chuẩn trong NLP (Merchant 2020), đã nhập vào code (Troshin & Chirkova 2022) |
| **"Đặc trưng còn, hàm quyết định hỏng"** | **ĐÃ CÓ.** Mai et al. NeurIPS 2024 nói đúng câu này ở bối cảnh khác |
| **Probe đích + zero-shot head nguồn thành MỘT chẩn đoán** | **KHÔNG TÌM THẤY Ở ĐÂU**, cả NLP lẫn code |
| **Dùng chẩn đoán đó để CHỌN can thiệp** | có đúng **một** tiền lệ, n=4 dataset, không kiểm định |
| **Chế độ 760 mẫu nhị phân** | **ngoài vùng bằng chứng của mọi bài** trong họ này |

Nói gọn: đóng góp bán được là **"chuyển giao phương pháp luận + chế độ dữ liệu mới"**, không phải
"ý tưởng mới". Cụ thể: *mượn phép probe từ interpretability của NLP, bổ sung phép đo zero-shot của
head nguồn, rồi dùng nó để chọn regularizer cho Pha 2 ở cỡ 760 mẫu mà chưa bài nào chạm tới.*

### 5.2 BA CẢNH BÁO phải xử lý trước khi viết

**(a) LP-FT kê đơn NGƯỢC với nhánh `rh` của mình.** Kumar, Raghunathan, Jones, Ma, Liang —
**ICLR 2022 Oral** (arXiv:2202.10054). Nguyên văn: *"the OOD error of fine-tuning is high when we
initialize with a fixed or random head … the lower layers change simultaneously and distort the
pretrained features."* Số: FT so với LP = **+2% ID / −7% OOD**; **LP-FT so với FT = +1% ID / +10%
OOD** trên 10 bộ dịch chuyển phân phối.

Tức là: LP-FT nói **head ngẫu nhiên làm MÉO đặc trưng tốt**, nên phải fit head trước rồi mới mở
backbone. Nhánh `rh` của mình đang làm đúng cái LP-FT bảo đừng làm. **Một reviewer biết LP-FT sẽ
hỏi ngay: sao không fit head trước?** Phải trả lời bằng SỐ, tức phải chạy LP-FT làm một nhánh.
Cờ `--lp_epochs` **đã có sẵn** trong `train_transfer.py`, chưa dùng trong khối nào gần đây.

**(b) Mai et al., NeurIPS 2024 — "Fine-Tuning is Fine, if Calibrated"** (arXiv:2409.16223) đã nêu
đúng kết luận §36 ở bối cảnh khác. Nguyên văn: *"the fine-tuned model neither forgets … nor
degrades the features … Instead, the fine-tuned model often produces more discriminative
features"* và *"what really hurts the accuracy is the discrepant logit scales."*
Cách chữa của họ là **hiệu chỉnh hậu kiểm**, không phải neo. Đáng chú ý: dự án mình **đã có sẵn**
`val_calibrated_threshold` và cột F1@val — tức là đã có một dạng hiệu chỉnh. Nên đọc lại §37 dưới
góc này: `cwe05` nâng F1@0.5 mà hạ AUC, còn F1@val cũng lên — rất khớp với "vấn đề là thang logit".

**(c) TRÙNG TÊN: "MultiVD" đã tồn tại, và nó gần thiết kế của mình.** Curto, Giordano, Palazzo,
Indelicato — **SECRYPT 2024**, doi:10.5220/0012719400003767: CodeBERT + head nhị phân + **head CWE
15 lớp**, BigVul, chỉ C/C++, 126 313 hàm; multitask F1 **95.51** so với LineVul 91.79.
**Không bottleneck, không transfer, tập đích khổng lồ.** Cần đổi tên dự án khi viết, và phải trích
bài này như tiền lệ gần nhất của thiết kế head phụ.

### 5.3 Neo trong không gian đặc trưng — ai đã làm gì

**DELTA** (Li, Xiong, Wang, Rao, Liu, Chen, Huan — ICLR 2019, arXiv:1901.09229). Phạt
`Σ_j W_j·‖FM_j(ω,x) − FM_j(ω*,x)‖²` trên feature map, **chạy trên input đích**, trọng số `W_j` là
softmax của mức tăng loss khi bỏ filter nguồn thứ *j*. ResNet-101, L2 / L2-SP / DELTA: Indoors
83.7/85.1/**85.5**, Dogs 83.3/88.3/**88.7**, CUB 78.4/79.5/**80.5**, Food-101 85.3/**86.4**/86.3
(DELTA **thua**). Bỏ attention mất 0.9–4.6 điểm.

> **Sửa một giả định của tôi:** tôi từng nghĩ DELTA có khảo sát theo cỡ dữ liệu. **Không có.**
> DELTA không hề có nghiên cứu 15/30/50/100% và không hề tuyên bố lợi ích tăng khi đích nhỏ đi.

**LDIFS** (Mukhoti, Gal, Torr, Dokania — **TMLR 2024**, arXiv:2308.13320) là bản hiện đại **gần
nhất với cái tôi vừa cài**: `L_CE + λ·(1/N)Σ‖Φ_θ(x) − Φ_θ₀(x)‖²`, đặc trưng **nhiều tầng nối lại**,
mốc là **chính điểm khởi tạo**, tính trên **tập train đích**. Chỉ dùng tầng cuối thì **kém hơn** —
đáng chú ý vì tôi đang chỉ dùng tầng cuối. ΔLP dương ở 8/10 tác vụ (EuroSAT +1.32 so với L2-SP
−0.85). λ phải chọn bằng cross-validation từng tác vụ.

**L2-SP** (Li, Grandvalet, Davoine — ICML 2018). Nguyên văn: *"when less training data are
available for the target problem, the improvement of L2-SP … are more important"* (Caltech-30 +2.0
so với Caltech-60 +1.1). Chi phí **<1% FLOPs**. **Fisher không mua thêm gì cho độ chính xác đích** —
khớp với kết quả null của khối Fisher bên mình. DELTA chính là L2-SP chuyển từ không gian trọng số
sang không gian kích hoạt, và lấy L2-SP làm đối chứng.

**LwF** (Li & Hoiem — ECCV 2016 / TPAMI 2018). Chưng cất **xác suất đầu ra của head cũ** (T=2), ghi
từ mô hình khởi tạo, trên ảnh của tác vụ mới. **Là logit, KHÔNG phải đặc trưng.** Đây chính là
**cái KHÔNG nên làm** ở dự án mình: head Pha 1 chỉ 0.54 zero-shot, chưng cất logit của nó là truyền
đi một mặt phân cách mình vừa đo được là vô nghĩa. Điểm này biện hộ thẳng cho lựa chọn dùng đặc
trưng thay vì logit.

**Co-Tuning** (NeurIPS 2020) là tiền lệ sắc nhất cho phát hiện zero-shot-head: nó **dùng lại head
nguồn** qua một ánh xạ nhãn nguồn↔đích **học được**, thay vì vứt đi hay chưng cất.

### 5.4 "Neo có lợi hơn khi đặc trưng nguồn vốn đã tốt?" — có, nhưng bằng chứng mỏng

Đây đúng là câu §36 gợi ra. Tài liệu trả lời **hai lần, đều mỏng**:

- **BSS** (Chen et al., NeurIPS 2019) là bài **duy nhất** có quét cỡ dữ liệu (15/30/50/100%).
  Nguyên văn: *"L2-SP penalty worsens the model's performance … especially when the amount of
  training data is limited"* — ở 15%: CUB 45.25→45.08, Cars 36.77→36.10, Aircraft 39.57→39.27,
  **đều tệ đi**. Nhưng trên Stanford **Dogs** (gần ImageNet nhất) thì neo vẫn trụ, và họ quy cho
  *"the transferability of pre-trained knowledge across these datasets."* Cơ chế được nêu, **chưa
  bao giờ thành quy tắc**.
- **Plested, Shen, Gedeon — ICONIP 2021** (arXiv:2107.08585) biến nó thành quy tắc đúng **một
  lần**: tín hiệu quyết định là **độ chính xác probe đặc trưng đóng băng trừ độ chính xác huấn
  luyện từ đầu** → âm thì dùng L2-SP + reinit nhiều tầng hơn; dương thì L2 thường. Caltech −16.2
  Có, DTD −7.8 Có, Cars +28.5 Không, Aircraft +28.9 Không. **n=4 dataset, một backbone, không hệ số
  tương quan, không kiểm định.** Bài tổng quan 2025 của chính tác giả (arXiv:2205.09904) nhắc lại và
  **nói rõ là cần thêm nghiên cứu**.
- **LEEP** (ICML 2020) và **LogME** (ICML 2021) ước lượng khả năng chuyển giao nhưng chỉ dùng để
  **chọn mô hình**, chưa ai dùng để **chọn regularizer**. Khoảng trống này là thật, và chạy LogME
  trên đặc trưng Pha 1 là cách gần như miễn phí để hình thức hoá đúng cái probe mình đã chạy.

> **Cảnh báo chế độ dữ liệu phải nêu trong bài:** không một bài nào trong họ này chạy ở **760 mẫu
> nhị phân**. Đích nhỏ nhất xác minh được là CUB@15%, khoảng 900 ảnh trên 200 lớp.

### 5.5 Khởi tạo lại head — ai đã đo gì

**Zhang, Wu, Katiyar, Weinberger, Artzi — ICLR 2021** (arXiv:2006.05987). Khởi tạo lại pooler +
L tầng trên cùng theo N(0, 0.02²), L∈{1..6}, BERT-Large, **20 seed**, chọn L trên val. Cỡ tập:
RTE 2.5k, MRPC 3.7k, STS-B 5.8k, CoLA 8.6k. Chuẩn → Re-init (3 epoch): RTE 69.5±2.5→**72.6±1.6**,
MRPC 90.8±1.3→**91.4±0.8**, STS-B 89.0±0.6→**89.4±0.2**, CoLA 63.0±1.5→63.9±1.9.
**Hạ xuống 1k mẫu**: RTE 62.5→65.6, **MRPC 80.5±3.3→84.6±1.6 (hiệu ứng lớn nhất)**. Lợi ích **co
lại khi huấn luyện lâu hơn** (CoLA còn đảo dấu). Họ ghi: *"we already see improvements when only
the pooler layer is re-initialized"* — tức nhánh `rh` chỉ đổi head là biến thể nhẹ nhất của họ, và
là biến thể có bằng chứng.

**Về việc mang sang một head nguồn gần-ngẫu-nhiên: KHÔNG BÀI NÀO chạy thí nghiệm này.** Cái gần nhất:

- **STILTs** (Phang, Févry, Bowman, arXiv:1811.01088) **vứt** head trung gian **không hề có
  ablation**: *"we add only a single task-specific, randomly initialized output layer."*
  **Vứt head là mặc định chưa ai kiểm trong chính bài kinh điển của transfer hai pha** — phép đo
  zero-shot của mình chính là lời biện minh mà họ chưa từng đưa ra.
- **Pruksachatkun et al., ACL 2020**: 110 cặp tác vụ trung gian→đích, 25 tác vụ probe. Kết quả
  **âm** phải trích trung thực: *"we fail to observe more granular correlations between probing and
  target task performance."*
- **Vu et al., EMNLP 2020**: transfer có lợi nhất khi **dữ liệu đích khan hiếm** — ủng hộ chế độ
  của mình.
- **Mosbach et al., ICLR 2021**: bất ổn là do **tối ưu hoá**, không phải quên; họ **không** khuyến
  nghị re-init.

### 5.6 Chuyển giao xuyên NGÔN NGỮ cho phát hiện lỗ hổng — hiện trạng thật

Cần biết để phát biểu tính mới cho đúng. Hiện trường chia làm ba nhánh, và **nhánh của mình gần
như trống**:

**Chuyển giao trọng số xuyên ngôn ngữ (đúng dạng của mình) — chỉ một bài xác minh được:**
**DSHGT** (arXiv:2306.01376, **preprint, chưa xác minh venue**) đóng băng encoder HGT trên CPG, chỉ
fine-tune MLP 3 tầng; C/C++ → **Java** và **PHP** trên SARD, CWE-78/79/89; 84% acc C→Java, 88%
C→PHP. **Không nêu cỡ tập đích, và KHÔNG có đối chứng chỉ-đích** — mọi baseline đều đã transfer.
Là GNN, không phải mô hình ngôn ngữ tiền huấn luyện. Ngoài ra Hanifi et al. (**ENASE 2023**,
arXiv:2303.06177): CNN, C → Java, recall trung bình 72%, không nêu cỡ đích lẫn đối chứng.

**Huấn luyện chung đa ngôn ngữ (KHÁC transfer):** **MVD** (arXiv:2412.06166) 6 ngôn ngữ, **có** đối
chứng LineVul từng ngôn ngữ, PR-AUC +83.7–193.6%. **Yu et al., ISSTA 2025** (arXiv:2505.07376) 7
ngôn ngữ, **CodeT5P thắng cả các LLM**, tự mô tả là *"an initial step toward cross-language
vulnerability detection"*. **IRC-CLVul** (Electronics 12(14):3067) hợp nhất qua LLVM IR, +12% F1.

**Zero-shot xuyên ngôn ngữ:** Chen et al. 2026 (arXiv:2604.27714) C/C++ Juliet → Java/Python:
fine-tune đẩy **FPR 0.763→1.000** trong khi F1 *"deceptively stable"* 0.637–0.688 — một minh hoạ
mạnh cho việc phải đọc nhiều hơn một chỉ số, đúng mục 2b của mình.

**Chốt về tính mới:** không bài nào dùng tập đích **nhỏ hơn hàng nghìn**; trường chia đôi giữa
zero-shot và huấn luyện chung đa ngôn ngữ; **DSHGT là chuyển giao trọng số xuyên ngôn ngữ duy nhất
xác minh được, và nó không có đối chứng chỉ-đích.** Mình có đối chứng `baseline` ở mọi ô.

**Về bộ dữ liệu, phải nêu thẳng trong bài:** **SVEN** (He & Vechev, **CCS 2023**) là phương pháp
**làm cứng sinh mã**, không phải benchmark phát hiện; 1 606 chương trình = 803 cặp, chia
**Python 760 / C-C++ 846**. **760 dòng của mình chính là toàn bộ nửa Python của SVEN.** Nên nói rõ
là mình tái dụng một kho làm-cứng-sinh-mã làm tập đích phát hiện. **PrimeVul** (ICSE 2025):
235 768 hàm, 140 CWE, **chỉ C/C++**; độ chính xác nhãn 92.0%/86.0% so với **SVEN 94.0% (người xác
minh)** — chính con số này biện hộ cho việc chọn SVEN; trùng lặp BigVul 12.7% so với PrimeVul 0.0%;
một mô hình 7B rơi từ **68.26% F1 trên BigVul xuống 3.09% trên PrimeVul**.

### 5.7 Probe như một chẩn đoán — trong code đã có ai làm

**Troshin & Chirkova, BlackboxNLP 2022, "Probing Pretrained Models of Source Codes"**
(aclanthology 2022.blackboxnlp-1.31) là **tổ tiên phương pháp luận trực tiếp, BẮT BUỘC trích**.
Probe tuyến tính trên biểu diễn đóng băng từng tầng; §5.5 so **chỉ-tiền-huấn-luyện vs đã-fine-tune
(5 tác vụ, có Defect Prediction) vs huấn-luyện-từ-đầu**. Nguyên văn: *"Models finetuned for
discriminative tasks exhibit the highest information loss … which may indicate that models trained
on these tasks rely on some spurious features"* và *"finetuning may deteriorate the model's
understanding of code properties, especially in classification downstream tasks … especially if
multi-stage finetuning is used."*

> **Đây vừa là tổ tiên vừa là ĐỐI TRỌNG tốt:** họ thấy fine-tune phân biệt làm **MẤT** thông tin;
> mình đo được Pha 1 **THÊM** thông tin liên quan tới đích (+0.1125 ROC, 5/5). Hai kết quả ngược
> nhau, và đó là chỗ để lập luận.

Thêm: **Shi et al., ISSTA 2023** (arXiv:2304.05216, Telly) probe từng tầng trước/sau fine-tune —
tầng dưới và giữa được giữ, *"the representations of the top two layers change most"*. Nền quy ước:
**Alain & Bengio 2016**; **Hewitt & Liang, EMNLP 2019** (control task + selectivity) — **nên trích
phòng thủ trước** phản biện *"probe của anh học chính tác vụ chứ không phải đo đặc trưng"*.

Probe trong phát hiện lỗ hổng có tồn tại nhưng làm **bộ phát hiện**, không phải **chẩn đoán**:
LPASS (arXiv:2505.24451) dùng probe để chọn điểm cắt tầng; "Probing the Prefill" (arXiv:2608.16970)
huấn luyện probe MLP trên kích hoạt LLM đóng băng. **Không bài nào so đặc trưng đã-fine-tune với
tiền-huấn-luyện, và không bài nào đo head nguồn.**

### 5.8 Việc phải làm, suy ra từ tra cứu

| # | việc | vì sao | chi phí |
|---|---|---|---|
| 1 | **Thêm nhánh LP-FT** (`--lp_epochs`, cờ đã có) | Cảnh báo (a): reviewer biết LP-FT sẽ hỏi ngay, và LP-FT kê đơn **ngược** với `rh`. Phải trả lời bằng số | 3 ô/backbone, n=3 |
| 2 | Neo đặc trưng **nhiều tầng** thay vì chỉ tầng cuối | LDIFS đo được chỉ-tầng-cuối **kém hơn** — mình đang làm đúng cái kém hơn | sửa code + 3 ô/backbone |
| 3 | Đọc lại §37 dưới góc "thang logit" của Mai et al. | `cwe05` nâng F1@0.5 lẫn F1@val mà hạ AUC — rất khớp | **0 GPU**, số đã có |
| 4 | Chạy **LogME** trên đặc trưng Pha 1 | hình thức hoá probe bằng một đại lượng có bài trích | **0 GPU** |
| 5 | Đổi tên dự án khi viết | trùng **MultiVD, SECRYPT 2024** | 0 |
| 6 | Trích Hewitt & Liang, thêm **control task** cho probe | chặn trước phản biện "probe học tác vụ" | ~1 h CPU |

### 5.9 KHÔNG xác minh được — đừng trích

CLMDA (không có bản ghi nào); số của MSVD và VDMAF (ScienceDirect 403); con số "12%/32%" của
AdvFusion; F1 0.91 tuyệt đối của LineVul; F1 cụ thể của VulBERTa; venue của CleanVul (dòng "JACM"
chỉ là placeholder của acmart); chi tiết thí nghiệm của Ren et al. ICLR 2023; số cặp chính xác của
PrimeVul. **"mAdapter" cho code KHÔNG TỒN TẠI** — bài thật là Wang et al., **ICSE 2023**, *"One
Adapter for All Programming Languages?"* (arXiv:2303.15822), và nó **không** làm phát hiện lỗ hổng.
**AdvFusion** (SANER 2025, arXiv:2307.07854) là **tóm tắt mã + đoán tên hàm**, **không** phát hiện
lỗ hổng.

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
