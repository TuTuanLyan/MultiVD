# HÀNG ĐỢI — 22/09/2026, vast 51144271 (máy DUY NHẤT)

> **Nguồn sự thật cho cron.** Đọc mục "ĐANG CHẠY" trước khi phóng bất kỳ driver nào.
> **Không bao giờ** phóng lại một driver chỉ vì log chưa có dòng kết thúc.

Bậc: **n=5 fold, seed 42**. Chưa có n=15 nào. Mọi ô trên **một máy duy nhất**.

---

## 04/10 21:1x - KHỐI `mwonly5` (06/10 02:45: KHÔNG còn job chạy; mọi máy vast đã destroy; 161 + 158 rảnh) - MW-only chạy lại trên mã refactor, bậc 2 (n=5, seed 42, TF32, tất định)

### 🐍U 07/10 15:2x PHA 1 TRỘN JS:PY 4:1 NGẪU NHIÊN, VAL PHA 1 CHỈ JS - UNCOMMON CHẶT (người dùng 07/10 15:0x: "tab chung, chạy thêm uncommon 4:1, val Pha 1 chỉ JS. (phương pháp này chốt là SOTA, nguồn pha 1 là tuỳ chọn)"; hỏi lại định nghĩa ⇒ chọn bản CHẶT) - 161 f1/f3/f5 + 158 f2/f4, bậc 2 (n=5, seed 42, TF32), tab Kết quả GỐC

- **Người dùng chốt:** phương pháp Pha 1 trộn JS:Py 4:1 ngẫu nhiên + val Pha 1 chỉ JS + Pha 2 RecAdam + ASAM là **SOTA**; nguồn Pha 1 (common / 4CWE /
  full / uncommon) là **tuỳ chọn**.
- **Dữ liệu.** Pool `data/mwonly5_sources/js_uncommon_strict/` (`tools/build_js_uncommon_strict_pool.py`): hàng JS của HF `merged_undersampled/full`
  (giữ thứ tự HF) có mã thuộc bộ chặt 01/10 `data/final_experiment_data/phase1_uncommon_jsonly_strict.jsonl` (luật `--uncommon_strict_only`: JS full bỏ
  JS common, bỏ mẫu có nhãn CWE thuộc common hoặc nhãn không phải mã CWE) ⇒ 120 hàm = 60 cặp (CWE-1321 56, CWE-843 4), 120/120 khớp, 0 hàm thuộc JS
  common; chia như mọi pool mwonly5 (Random(42).shuffle, val = round(15 %)): train 102 (nhãn 1: 50) / val 18 (nhãn 1: 10), 14 cặp bị tách train/val.
  Trộn `js_py41jsval_uncommon_strict/` (`tools/build_p1_mixpy.py --val js`): train 128 = 102 JS + 26 Python (13/13, ⊂ SVEN train fold k, 0 hàm trùng
  test), val = test = CHỈ 18 JS. Dựng lại trùng byte (pool 3/3, trộn 15/15); md5 dữ liệu trùng 158 (20 file). (Bản không chặt = 264 hàm, không chạy.)
- **Run** (runs.json md5 3c7235bc, trùng 158; chỉ THÊM 53 dòng): `p1_jspy41jsval_uncommon_strict` → `rasam_jspy41jsval_uncommon_strict` (init_same_fold).
  Plan chỉ khác run common tương ứng ở tên / đường dẫn / `--data_root` (diff cả train lẫn test). Không có uncommon chỉ JS trong khối mwonly5 ⇒ so với
  baseline và với các nguồn khác cùng phương pháp (common 0,9443, 4CWE 0,9414, full 0,9445).
- **Trước khi phóng:** 161 + 158 đều 0 tiến trình của mình, lock tự do, VRAM trống (158: 15,6 GB trống, có 1 tiến trình 264 MiB không phải của mình),
  skip.txt không chặn, plan in đúng checkpoint, preflight 3/3 + 2/2.
- **Dự đoán (ghi TRƯỚC):** U1 Pha 1 chọn checkpoint sớm (best epoch ≤ 3) ở ≥ 3/5 fold (val 18 JS, 14 cặp bị tách - như lần chặt trước chọn epoch 1);
  U2 ROC TB của `rasam_jspy41jsval_uncommon_strict` nằm trong [0,900; 0,935]: hơn baseline (0,8966) ở ≥ 4/5 fold nhưng thua 4:1 jsval common (0,9443)
  ở ≥ 4/5 fold; U3 0/5 ô sập (ROC < 0,75).

### 🐍P 06/10 21:1x - XONG 23:07 (5/5 rc 0) PHA 1 TRỘN JS + PYTHON RÚT THEO CẶP, VAL PHA 1 CHỈ JS - COMMON (người dùng: "chạy thêm 1 bản JS common + 1/4 số lượng của JS hàm SVEN train_k (rút theo cặp). pha 1 dùng best roc js. Ưu tiên trước khi asam và recadam only") - paper_mw4 + 161 + 158, bậc 2 (n=5, seed 42, TF32)

- **Dữ liệu** `data/mwonly5_sources/js_py41pair_common/` (`tools/build_p1_mixpy_pairs.py`, mới): train = 842 JS common + **105 cặp SVEN** (hàm lỗi +
  bản vá, cả hai nửa nằm ở SVEN train fold k; 842 / 4 / 2 = 105,25 → 105 cặp = 210 hàm, nhãn tự cân), `Random(42)` mới mỗi fold; **val = CHỈ 148 JS
  val** (đúng val của `p1_common_jsonly`) ⇒ checkpoint Pha 1 chọn theo ROC trên JS, không nhìn val SVEN. Cặp SVEN ghép lại trên `data.jsonl` (760 hàm,
  mã không trùng) theo (tên `def`, cwe): 336 nhóm 1-1, 18 nhóm lớn hơn ghép tham lam theo SequenceMatcher (tỉ số nhỏ nhất 0,355 - bản vá viết lại
  nhiều, đã xem tay). Cặp đủ hai nửa ở train: 130 / 140 / 142 / 140 / 134. Dựng lại trùng byte 15/15; 0 hàm trùng test; md5 dữ liệu trùng 3 máy.
- **Run:** `p1_jspy41pair_common` → `rasam_jspy41pair_common` (cột chính, init_same_fold). Chạy khô: Pha 1 chỉ khác `p1_common_jsonly` / `p1_jspy41_common`
  ở `--data_root`; Pha 2 chỉ khác `rasam_jspy41_common` ở checkpoint nạp. `runs.json` md5 3ef1ef90 (161 / 158 / paper_mw4).
- **Máy:** paper_mw4 f1 → f3 (PHÓNG 21:10, dòng đầu nhận đủ 4 mục, lock giữ); 161 f2 → f4 (`queue_after` sau driver 4CWE PID 3348387); 158 f5
  (`queue_after` PID 2748200 sau driver 4CWE PID 2319841). ~45 phút/fold ⇒ xong ~23:00. Trang bản 44: dòng mới trong nhóm "Pha 1 trộn · common".
- **Sau khối này:** nhánh val chỉ JS ngẫu nhiên (🐍V, người dùng 22:0x), rồi chỉ ASAM / chỉ RecAdam trên các Pha 1 trộn (phạm vi chưa chốt).
- **KẾT QUẢ 5/5 (23:07):** ROC 0,943 / 0,941 / 0,943 / 0,952 / 0,942. Ghép cặp (ROC / PR / F1@0,5 / F1@val): - chỉ JS common +0,010 (+4/-0,
  5/5 không âm; theo fold +0,000 / +0,006 / +0,009 / +0,025 / +0,011) / +0,009 (+4/-0) / +0,024 (+4/-1) / -0,003 (+2/-3); - 4:1 cũ +0,001 (+3/-1) /
  +0,002 (+2/-1) / +0,020 (+4/-1) / -0,002 (+1/-3). Val ROC Pha 1 (chỉ JS): 0,644 / 0,619 / 0,594 / 0,645 / 0,592.
  **Chấm:** P1 ĐÚNG (min 0,941), P2 SAI (3/5 trong [0,60; 0,70]; f3 0,594, f5 0,592 hụt nhẹ), P3 ĐÚNG, P4 ĐÚNG. Đọc: bỏ val SVEN khỏi việc chọn
  checkpoint Pha 1 (và rút theo cặp) KHÔNG làm mất lợi ROC so với chỉ JS (+0,010, ngang 4:1 cũ +0,009); F1@ngưỡng val không theo. Hai thay đổi cùng
  lúc nên chưa tách được riêng phần val - chờ 🐍V (cùng train 4:1 cũ, chỉ đổi val).
- **Dự đoán (ghi TRƯỚC):** P1 0/5 ô sập; P2 val ROC (chỉ JS) của checkpoint Pha 1 trong [0,60; 0,70] ở ≥ 4/5 fold (`p1_common_jsonly` 0,635);
  P3 theo cặp - chỉ JS common: ROC TB ≥ 0 và ≥ 4/5 fold không âm (hoà 0,001); P4 theo cặp - 4:1 common: |ROC TB| ≤ 0,010, không 5/5 cùng dấu.
  Đọc: P3 đúng ⇒ lợi của trộn không cần chọn checkpoint trên val SVEN; P3 sai ⇒ nghi lợi đến từ val SVEN (hai thay đổi cùng lúc: cặp + val, nên P4
  chỉ đọc kèm, không tách được).

### 📄 07/10 01:5x - XONG 11:56 (60/60 ô đủ json + probs, seed hp khớp 12/12 run; 161 / 158 / paper_night hết hàng đợi) TAB "KẾT QUẢ PAPER" - BẬC 3 (n = 15 = 5 fold × seed 42 / 1234 / 7): baseline + common / 4CWE / full · JS:Py 4:1 ngẫu nhiên, val Pha 1 chỉ JS (người dùng 01:4x: "tạo thêm 1 tab kết quả nữa, gồm baseline + common/4cwe/full · JS:Py 4:1 ... val Pha 1 chỉ JS chạy lên n = 15. Các seed gồm 42 đã có, 1234 và [7] ... không cập nhật kết quả vào tab kết quả gốc, tab kết quả mới tên "Kết quả paper". Chạy nhánh chính trước") - 161 + 158

- **Làm rõ với người dùng (01:5x):** rút Python NGẪU NHIÊN cân nhãn (không theo cặp - "tôi nhầm"), theo seed của run: `Random(seed)` (seed 42 cũng vậy - bản
  val chỉ JS seed 42 đã rút bằng Random(42)); seed thứ ba là **7** (giữ bộ 42 / 1234 / 7 của CLAUDE.md, không dùng 2026).
- **Ô dùng lại (seed 42, đã có ở tab gốc):** `baseline`, `rasam_jspy41jsval_common`, `rasam_jspy41jsval_full` (đang chạy 🐍VF). **Ô mới (40 Pha 2 + 35 Pha 1):**
  `p1/rasam_jspy41jsval_4cwe` (seed 42), `p1/rasam_jspy41jsval_{common,4cwe,full}_s{1234,7}`, `baseline_s{1234,7}` - mọi run mới có `"tab": "paper"` +
  `"seed"` trong runs.json (md5 66d77196).
- **Hạ tầng:** `fpe.py` nhận khoá `seed` theo run (`--seed`, thư mục `seed_<s>`, Pha 2 nạp Pha 1 cùng seed); kiểm: 760/760 lệnh plan cũ TRÙNG BYTE trước/sau,
  chiều lệch đúng (`--seed 1234` thay 42, không lặp). `artifact_push.py` đẩy run `tab: paper` vào collection RIÊNG `paper_folds` (run cũ vẫn `folds`; thử
  8 doc hai chiều). Dữ liệu `data/mwonly5_sources/js_py41jsval_{4cwe, common_s1234, 4cwe_s1234, full_s1234, common_s7, 4cwe_s7, full_s7}` (`build_p1_mixpy.py
  --val js --seed s`): dựng lại trùng byte 7/7; JS + val trùng bản seed 42, Python khác theo seed (trùng 24-155 / 125-267 hàm); md5 trùng 158.
- **Lịch (fold vòng ngoài, nhánh chính trước, baseline cuối; chia tham lam theo thời gian ~35 / 43 / 53 phút cho 4CWE / common / full):** 161 33 mục
  (`queue_after` sau driver 🐍VF PID 3783806, `state/queue_161_paper.out`), 158 47 mục (`queue_after` sau PID 3359241, `state/queue_158_paper.out`);
  preflight 18 / 27 mục đạt. Bắt đầu ~03:15 (158) / ~03:50 (161), xong ~16:30 07/10.
- **07/10 03:4x THÊM vast paper_night** (người dùng: "tôi đã thuê thêm vast paper_night, phân chia việc nhé. nhắc tôi tắt sau khi chạy xong việc";
  "để ý phiên bản môi trường nhé"): 54537908, RTX A4000, driver 595.71.05, 4 vCPU, đĩa 23 GB, 0,091 $/h, Nhật (cùng máy chủ paper_mw3 cũ), SSH
  202.122.49.242:42446, hostname 353f204c13e9. Môi trường `setup_papermw3.sh` (wheel ghim); ĐỐI CHIẾU 3 máy: python 3.11.14, torch '2.9.1+cu128' cùng
  git 5811a8d7da87 + libtorch_cuda.so cùng md5, CUDA 12.8, cuDNN 9.10.2.21 (91002), transformers 4.57.1, numpy 2.3.4, sklearn 1.7.2, scipy 1.16.3,
  tokenizers 0.22.1, safetensors 0.6.2, triton 3.5.1, cublas 12.8.4.1 - trùng hết. CodeBERT md5 cây thư mục trùng 161. **Kiểm máy:** `chk_p1_jspy41_4cwe`
  f4 3 epoch đầu TRÙNG từng chữ số log 161 (cổng thử chiều lệch bắt được); dừng (kill đúng driver 1775 + trainer 1804), xoá checkpoint dở + /workspace/wheels
  (16 GB trống). **Chia lại (tham lam, 158 chỉ giữ mục của danh sách nó):** 158 31 mục (14 mục dời đi qua `skip.txt`, kiểm hai chiều), 161 25 mục (queue
  cũ PID 3836037 dừng khi CHƯA exec, phóng lại sau driver 3783806), paper_night 22 mục (PHÓNG 03:40, nhận đủ 22, lock giữ, `--seed 1234`). Ước xong
  ~12:10-12:40 (thay ~16:30). Monitor `mon_fpe_v11.sh` (+paper_night, MON_ST mới); vòng sync/live + HOSTS trang thêm paper_night. **Nhắc người dùng
  tắt paper_night khi driver in XONG + kéo về đối chiếu byte.**
- **07/10 10:49 DỒN MÁY:** 158 xong hàng đợi 10:48 (sớm) ⇒ dời `p1/rasam_jspy41jsval_4cwe_s7:5`, `baseline_s7:3/4/5` từ 161 sang 158 (161 `skip.txt` thêm 5 dòng,
  158 gỡ 2 dòng skip cũ của 4cwe_s7 f5; plan kiểm hai chiều cả hai máy; 158 0 tiến trình, lock tự do, preflight 4/4; PHÓNG `run.sh 158` 5 mục, lock giữ,
  `--seed 7`). 161 còn common_s1234 f5; paper_night còn full_s1234 f5 + full_s7 f5 (~12:30).
- **07/10 11:10 DỒN MÁY lần 2:** 161 xong 11:09 ⇒ dời `p1/rasam_jspy41jsval_full_s7:5` từ paper_night sang 161 (paper_night `skip.txt` +2 dòng, plan kiểm
  hai chiều; 161 0 tiến trình, lock tự do, preflight đạt; PHÓNG 11:10, `--seed 7`). paper_night chỉ còn full_s1234 f5 (Pha 1 đang kẹt) ⇒ xong sớm hơn.
- **07/10 11:29 paper_night XONG** (driver in `FPE paper_night XONG 2026-10-07 11:29:32`, 0 tiến trình, GPU trống). Kéo về: kết quả + log fold
  cuối, log `chk_p1_jspy41_4cwe` f4, state → `state/remote_logs/paper_night_54537908/state_full/`; checkpoint Pha 1 10 đơn vị (20 file .pt,
  5 luồng song song ~7-9 MB/s, 11:34-12:00) → `model/mwonly5/`. **Đối chiếu 12:01:** 20/20 ô có kết quả ở local (10 Pha 1 + 10 Pha 2, json + probs);
  file còn trên staging + state khớp md5 6/6; checkpoint **khớp md5 + đường dẫn 20/20** (byte 10 × 536 494 010, 7 × 536 510 137, 3 × 536 510 201),
  0 file tạm; chiều lệch (sửa 1 ký tự md5) bị bắt. **ĐÃ NHẮC người dùng tắt 54537908 (12:0x)**; còn: bỏ paper_night khỏi `artifact_tick.sh`,
  `live_all.sh`, HOSTS trang khi máy tắt.
- **Trang bản 49:** tab "Kết quả paper" (bảng TB ± SD trên ô đã xong + Δ ghép cặp (seed, fold) với baseline, lưới seed × fold); đọc `paper_folds` + 3 run seed 42
  ở `folds`; tab Kết quả gốc KHÔNG đọc `paper_folds` (thẻ máy chỉ hiện ô paper đang chạy). 80 doc gieo sẵn (`meta/seed_0710_paper/`).
  **Bản 50 (02:1x, người dùng: "bỏ F1@ngưỡng val ra..., thay vì đó là CWE"):** bảng chính còn ROC / PR / F1@0,5; thêm bảng theo CWE (022 / 078 / 079 / 089:
  ROC + F1@0,5, TB ± SD, Δ ghép cặp với baseline).
- **Dự đoán (ghi TRƯỚC):** Y1 0/40 ô mới sập; Y2 mỗi cấu hình - baseline: ROC TB ≥ +0,030 và ≥ 14/15 cặp dương (seed 42: common +0,048, 4:1 4CWE +0,044,
  4:1 full +0,050, 5/5); Y3 chênh giữa TB ROC 3 seed của một cấu hình ≤ 0,015; Y4 TB ROC baseline seed 1234 / 7 trong ±0,015 của seed 42 (0,897).
- **KẾT QUẢ (bậc 3, n = 15; `meta/insights/paper_n15/build_paper_n15.py`, chạy bằng vdenv vì cần scipy).** Hyperparameters trùng 15/15 trong mỗi cấu
  hình (ngoài đường dẫn / seed / fold), `init_ckpt` mọi ô Pha 2 trỏ đúng `seed_<s>/fold<k>`. TB ± SD:

  | cấu hình | ROC | PR | F1@0,5 | F1@val |
  |---|---|---|---|---|
  | baseline | 0,8775 ± 0,1067 | 0,8840 ± 0,0909 | 0,7718 ± 0,1227 | 0,7741 ± 0,0860 |
  | common | 0,9378 ± 0,0157 | 0,9449 ± 0,0124 | 0,8549 ± 0,0253 | 0,8367 ± 0,0321 |
  | 4CWE | 0,9396 ± 0,0142 | 0,9456 ± 0,0122 | 0,8560 ± 0,0242 | 0,8458 ± 0,0215 |
  | full | 0,9158 ± 0,0969 | 0,9221 ± 0,0944 | 0,8308 ± 0,1362 | 0,8270 ± 0,1119 |

  Δ ghép cặp (seed, fold) với baseline, +k/-k trên 15 cặp (hoà 1e-3), fold dương = TB 3 seed mỗi fold (p Wilcoxon theo fold, sàn 0,0625):
  common ROC +0,0603 (+15/-0, 5/5) · PR +0,0609 (+15/-0) · F1@0,5 +0,0831 (+13/-1) · F1@val +0,0627 (+13/-2);
  4CWE ROC +0,0621 (+15/-0, 5/5) · PR +0,0615 (+15/-0) · F1@0,5 +0,0842 (+15/-0) · F1@val +0,0718 (+13/-2);
  full ROC +0,0383 (+13/-2, 4/5, p 0,44) · PR +0,0381 (+13/-2) · F1@0,5 +0,0590 (+13/-2) · F1@val +0,0529 (+13/-2).
  **Hai ô sập gánh trung bình:** baseline s7 f1 (0,502) thổi Δ cặp đó lên ~0,43-0,45; full s1234 f5 (0,570) cho Δ -0,312. Trung vị Δ ROC: common
  +0,037, 4CWE +0,037, full +0,035. **Bỏ hai cặp (7, f1) + (1234, f5):** common +0,0337 (+13/-0, 5/5), 4CWE +0,0328 (+13/-0, 5/5), full +0,0344
  (+12/-1, 5/5). TB ROC theo seed (42 / 1234 / 7): baseline 0,8966 / 0,9050 / 0,8311 (bỏ f1: 0,9058 / 0,9086 / 0,9132), common 0,9443 / 0,9340 /
  0,9351, 4CWE 0,9414 / 0,9398 / 0,9376, full 0,9445 / 0,8555 / 0,9476.
- **Chấm dự đoán:** Y1 **SAI** (2/40 ô mới sập: baseline s7 f1, full s1234 f5). Y2 common **ĐÚNG** (+0,060, 15/15), 4CWE **ĐÚNG** (+0,062, 15/15),
  full **SAI** (+0,038 nhưng 13/15). Y3 common **ĐÚNG** (0,010), 4CWE **ĐÚNG** (0,004), full **SAI** (0,092 - do ô sập). Y4 s1234 **ĐÚNG** (+0,008),
  s7 **SAI** (-0,066 - do ô sập; bỏ f1 thì cả ba seed trong 0,0074).

### 🐍VF 07/10 01:2x - XONG 04:09 (5/5 rc 0) PHA 1 TRỘN 4:1 NGẪU NHIÊN, VAL PHA 1 CHỈ JS - FULL (người dùng: "thêm nhánh JS:Py 4:1 ngẫu nhiên, val Pha 1 chỉ JS nhưng bộ JS full") - 161 + 158, bậc 2 (n=5, seed 42, TF32)

- **Dữ liệu** `data/mwonly5_sources/js_py41jsval_full/` (`tools/build_p1_mixpy.py --pool data/mwonly5_sources/js_full --val js`): **train TRÙNG BYTE
  `js_py41_full`** (1 066 JS + 267 Python cân nhãn, Random(42)) ở cả 5 fold; val = test = CHỈ 188 JS full val (= val `p1_full_jsonly`; chiều lệch: khác
  val 4:1 full cũ). Dựng lại trùng byte 15/15; md5 dữ liệu trùng 161 / 158.
- **Run:** `p1_jspy41jsval_full` → `rasam_jspy41jsval_full` (init_same_fold). Chạy khô: chỉ khác `p1_jspy41_full` / `rasam_jspy41_full` ở `--data_root` /
  checkpoint nạp. `runs.json` md5 62bbee5c (161 / 158).
- **Máy:** 161 f1 → f3 → f5 (`queue_after` sau driver chỉ RecAdam PID 3704367, ~01:28); 158 f2 → f4 (`queue_after` PID 3359241 sau PID 3246770,
  ~01:42). ~47 phút/fold (Pha 1 full ~2 phút/epoch) ⇒ 158 ~03:20, 161 ~03:50. Trang bản 48: dòng mới trong nhóm "Pha 1 trộn · full".
- **CƠ CHẾ (02:2x, f1):** train trùng byte + tất định ⇒ quỹ đạo Pha 1 GIỐNG HỆT 4:1 cũ (train loss trùng từng epoch); val chỉ đổi EPOCH được chọn.
  f1 full: cả hai chọn ep6 ⇒ Pha 2 trùng 16 chữ số (0,9325334724395757; md5 best.pt khác vì file lưu kèm số val). Ở common: val chỉ JS chọn SỚM hơn
  ở 5/5 fold (ep 5 / 7 / 7 / 5 / 5 so với 7 / 9 / 9 / 8 / 9) và ΔROC Pha 2 -0,005 / +0,009 / -0,012 / +0,011 / +0,000 ⇒ phép so "chỉ khác val" thực chất là
  "checkpoint Pha 1 sớm hơn 2-4 epoch"; Δ = 0 đúng ở fold chọn cùng epoch.
- **KẾT QUẢ 5/5 (04:09):** ROC 0,933 / 0,944 / 0,949 / 0,965 / 0,933; val ROC Pha 1 (chỉ JS) 0,621 / 0,629 / 0,609 / 0,630 / 0,640; epoch chọn
  6 / 7 / 5 / 5 / 7 (4:1 full cũ 6 / 7 / 6 / 9 / 7 ⇒ f1, f2, f5 TRÙNG 16 chữ số). Ghép cặp ROC (PR / F1@0,5 / F1@val): - chỉ JS full +0,012 (+5/-0)
  (+0,012 / +0,025 / +0,009); - 4:1 full cũ -0,002 (+0/-2, 3 hoà) (-0,003 / -0,004 / -0,007); - baseline +0,048 (+5/-0). **Chấm:** VF1 ĐÚNG (min 0,933),
  VF2 ĐÚNG (5/5), VF3 ĐÚNG, VF4 ĐÚNG. Đọc: trên full, chọn checkpoint Pha 1 chỉ theo JS giữ nguyên lợi so với chỉ JS (+0,012, 5/5) - val SVEN không phải
  nguồn lợi; mục 9 doc `jspy41` (bản 7).
- **Dự đoán (ghi TRƯỚC):** VF1 0/5 ô sập; VF2 val ROC (chỉ JS full) của checkpoint Pha 1 trong [0,55; 0,70] ở ≥ 4/5 fold (phần JS của val 4:1 full
  cũ 0,57-0,63); VF3 val chỉ JS - 4:1 full cũ (CHỈ khác val): |ROC TB| ≤ 0,010, không 5/5 cùng dấu; VF4 val chỉ JS - chỉ JS full: ROC TB ≥ 0 và
  ≥ 4/5 fold không âm (4:1 full cũ +0,014, +5/-0).

### 🐍R 07/10 00:3x - XONG 01:36 (10/10 rc 0) CHỈ RECADAM TRÊN PHA 1 TRỘN 4:1 (common + full) - phần CHUNG của mọi phương án phạm vi "chỉ ASAM / chỉ RecAdam" (người dùng 06/10 20:0x: "thử thêm các nhánh ASAM only và recadam only của nhánh trộn pha 1"; 21:0x: ưu tiên theo cặp / val chỉ JS trước) - 158 + 161, bậc 2 (n=5, seed 42, TF32)

- **Vì sao chạy trước khi chốt phạm vi:** 158 rảnh 00:36; phương án nào cũng gồm chỉ RecAdam 4:1 common/full (chỉ ASAM 4:1 đã có, 🐍A). Phần còn
  lại (3:1, 4CWE, theo cặp, val chỉ JS) CHỜ người dùng chốt.
- **Run:** `raonly_jspy41_{common,full}` = `raonly_<mức>_jsonly` nhưng nạp `p1_jspy41_<mức>` cùng fold (init_same_fold). Chạy khô: chỉ khác đối chứng ở
  checkpoint Pha 1; so với `asamonly_jspy41_<mức>` chỉ khác cờ optimizer (RecAdam: pretrain_cof 500, sigmoid, k 0,05, t0 0,01; ASAM tắt).
  `runs.json` md5 64fb1a9c (161 / 158).
- **Máy theo nơi có checkpoint Pha 1:** 158 common f2/f4 + full f2/f4/f5 (PHÓNG 00:36, 0 tiến trình, lock tự do, preflight 5/5, dòng đầu nhận đủ
  5 mục, lock giữ); 161 common f1/f3/f5 + full f1/f3 (`queue_after` sau driver val chỉ JS PID 3482037). ~15-18 phút/ô ⇒ ~02:00. Trang bản 47
  (cột "Chỉ RecAdam" trên dòng 4:1 common / full).
- **KẾT QUẢ 10/10 (01:36):** ROC common 0,946 / 0,936 / 0,915 / 0,946 / 0,907; full 0,931 / 0,954 / 0,928 / 0,944 / 0,921. Ghép cặp
  (ROC / PR / F1@0,5 / F1@val): - chỉ RecAdam chỉ JS: common +0,014 (+3/-1) / +0,017 / +0,018 / +0,017; full **+0,033 (+5/-0)** / +0,040 (+5/-0) /
  +0,041 (+5/-0) / +0,037 (+5/-0); - cột chính trộn: common -0,013 (+1/-4) / -0,014 (+0/-4) / -0,011 / -0,009; full -0,011 (+1/-4) / -0,015 / -0,015 /
  -0,023; - chỉ ASAM trộn: common -0,010 (+1/-4), full -0,007 (+1/-4). **Chấm:** R41a ĐÚNG (min 0,907), R41b ĐÚNG (common 4/5, full 5/5 không âm),
  R41c SAI (cả hai mức thấp hơn cột chính trộn > 0,010). Đọc: Pha 1 trộn nâng chỉ RecAdam rõ nhất trong mọi optimizer (full +0,033 5/5 trên CẢ 4 chỉ
  số - đối chứng chỉ RecAdam chỉ JS full yếu, TB 0,903), nhưng vẫn kém cột chính trộn ~0,01 - thứ tự trên Pha 1 trộn: cột chính > chỉ ASAM > chỉ RecAdam.
- **Dự đoán (ghi TRƯỚC):** R41a 0/10 ô sập; R41b chỉ RecAdam trộn - chỉ RecAdam chỉ JS: ROC TB ≥ 0 và ≥ 4/5 fold không âm ở MỖI mức;
  R41c chỉ RecAdam trộn - cột chính trộn: |ROC TB| ≤ 0,010, không 5/5 cùng dấu ở mỗi mức.

### 🐍V 06/10 22:0x - XONG 00:41 07/10 (5/5 rc 0) PHA 1 TRỘN 4:1 NGẪU NHIÊN, VAL PHA 1 CHỈ JS - COMMON (người dùng: "thêm 1 nhánh pha 1 trộn ngẫu nhiên 4:1 lấy val js nữa") - 161 + 158, bậc 2 (n=5, seed 42, TF32)

- **Dữ liệu** `data/mwonly5_sources/js_py41jsval_common/` (`tools/build_p1_mixpy.py --val js`, cờ mới, mặc định giữ hành vi cũ - dựng lại js_py41_common
  trùng byte 15/15): **train TRÙNG BYTE `js_py41_common`** (842 JS + 211 Python cân nhãn, Random(42)) ở cả 5 fold; val = test = CHỈ 148 JS val (trùng
  byte val của `p1_common_jsonly`; chiều lệch: khác val 4:1 cũ). Dựng lại trùng byte 15/15; md5 dữ liệu trùng 161 / 158.
- **Phép so tách được:** - 4:1 cũ = CHỈ khác val chọn checkpoint Pha 1 (JS vs JS + SVEN val); - theo cặp (🐍P) = CHỈ khác cách rút Python (ngẫu nhiên
  cân nhãn vs theo cặp), cùng val chỉ JS; - chỉ JS common = có / không Python trong train Pha 1.
- **Run:** `p1_jspy41jsval_common` → `rasam_jspy41jsval_common` (init_same_fold). Chạy khô: chỉ khác `p1_jspy41_common` / `p1_jspy41pair_common` ở
  `--data_root`. `runs.json` md5 536526ee (161 / 158).
- **paper_mw4 54468293 DESTROY 23:00** (lời dặn "hủy vast sau khi dùng xong" + nghỉ sau 23:00): driver XONG 22:31, 0 tiến trình / 0 app GPU;
  results + logs còn lại 4/4 về 161 md5 khớp (phần trước đã chuyển qua sync); `state/` 6/6 tên + byte (`state/remote_logs/paper_mw4_54468293/state_full/`);
  checkpoint Pha 1 CHỈ có trên máy đó đã kéo về: `p1_jspy31_full` f1-f3 + `p1_jspy41pair_common` f1/f3, `best.pt` 5/5 và `best_train_loss.pt` 5/5 md5
  khớp (cổng thử chiều lệch bắt được); bỏ lại `chk_p1_jspy41_full` f1 (lượt kiểm máy dừng ep3). `vastai show instances` = 0; gỡ khỏi
  `artifact_tick.sh`, `live_all.sh`, HOSTS (trang bản 46).
- **Máy** (paper_mw4 KHÔNG nhận: nghỉ sau 23:00, máy thuê phải huỷ): 161 f1 → f3 → f5 (`queue_after` sau driver theo cặp PID 3420541); 158 f2 → f4
  (`queue_after` PID 2860196 sau PID 2748200 = theo cặp f5). ~42 phút/fold ⇒ 158 xong ~00:20, 161 ~00:45. Trang bản 45.
- **KẾT QUẢ 5/5 (00:41):** ROC 0,939 / 0,947 / 0,930 / 0,970 / 0,936; val ROC Pha 1 (chỉ JS) 0,605 / 0,601 / 0,585 / 0,634 / 0,652. Ghép cặp
  ROC (PR / F1@0,5 / F1@val): - 4:1 cũ (CHỈ khác val) +0,001 (+2/-2; theo fold -0,005 / +0,009 / -0,012 / +0,011 / +0,000) (PR +0,003, F1@0,5 +0,012,
  F1@val -0,006); - chỉ JS common +0,010 (+3/-2; -0,004 / +0,012 / -0,005 / +0,044 / +0,005) (PR +0,011 +4/-1, F1@0,5 +0,016 +4/-1, F1@val -0,008);
  - theo cặp (CHỈ khác cách rút) +0,000 (+2/-3). **Chấm:** V1 ĐÚNG (min 0,930), V2 ĐÚNG (4/5), V3 ĐÚNG, V4 SAI (3/5 không âm; TB +0,010 do f4
  +0,044), V5 ĐÚNG. Đọc: đổi val chọn checkpoint Pha 1 (JS + SVEN → chỉ JS) với CÙNG train KHÔNG đổi ROC TB (+0,001) ⇒ val SVEN không phải nguồn
  của lợi TB; nhưng so với chỉ JS, dấu theo fold kém ổn định hơn 4:1 cũ (+4/-0) và theo cặp (+4/-0) - mọi chênh lệch giữa ba bản đều dưới /
  sát sàn nhiễu 0,010. Câu hỏi còn mở: lợi đến từ CHUYỂN GIAO hay chỉ từ việc Pha 1 thấy dữ liệu TRAIN đích (phép tách Python-only Pha 1, chưa chạy).
- **Dự đoán (ghi TRƯỚC):** V1 0/5 ô sập; V2 val ROC (chỉ JS) của checkpoint Pha 1 trong [0,60; 0,70] ở ≥ 4/5 fold; V3 val chỉ JS - 4:1 cũ:
  |ROC TB| ≤ 0,010, không 5/5 cùng dấu (chọn checkpoint trên val SVEN không phải nguồn của lợi); V4 val chỉ JS - chỉ JS common: ROC TB ≥ 0 và ≥ 4/5
  fold không âm; V5 val chỉ JS - theo cặp: |ROC TB| ≤ 0,010, không 5/5 cùng dấu.

### 🅰️➕ 06/10 13:0x - XONG 14:17 (5/5 rc 0; vast 54437777 ĐÃ DESTROY 14:2x) HƯỚNG A - MASK TRÊN FULL JS + C/C++ ĐỦ 5 FOLD (người dùng: "tôi đã thêm paper_mw3 trên vast 'Hướng A - chỉ ASAM, bias/LN mask - full - JS + C/C++ · fold 4' trông có vẻ ổn, chạy thử đủ 5 fold xem") - vast paper_mw3 (54437777), bậc 2 theo yêu cầu (n=5, seed 42, TF32)

- **Máy:** vast 54437777 nhãn `paper_mw3` (người dùng thuê; RTX A4000, driver 595.71.05, 0,091 $/h, Nhật), SSH 202.122.49.242:42920. Môi trường
  `setup_papermw3.sh` (wheel ghim): python 3.11.14, torch 2.9.1+cu128, cuDNN 91002, transformers 4.57.1, numpy 2.3.4, sklearn 1.7.2 - trùng vdenv 161.
  Checkpoint Pha 1 `p1_full_jscpp` + CodeBERT chép theo mảnh 40 MB, so md5. `runs.json`: host mới, `asamAmask_full_jscpp` folds [4] → [1..5].
- **PHÓNG 13:01** (0 tiến trình, lock tự do, preflight đạt 5/5, chạy khô chỉ khác 161 ở đường dẫn python, VRAM trống, đĩa còn 12 GB; dòng đầu
  log nhận đủ 5 mục; trang bản 36). Lưu ý: `fpe.py` thêm `old_hostnames` (paper_mw3 cũ `31c24a2f1c4b`) để ô cũ không mất nhãn máy.
- **Thứ tự:** f4 CHẠY LẠI trước (kiểm trùng bit với 161: 0,9439497118910425; bản 161 sao lưu ở `state/backup_0610_maskfull_f4_161/`) → f1 → f2 → f3 → f5.
  ~20 phút/ô ⇒ ~1 h 40.
- Ô gốc chỉ ASAM full JS + C/C++: thoát ep9 / 11 / 10 / 15 / 14, ROC 0,877 / 0,923 / 0,915 / 0,942 / 0,852 (không fold nào sập); AdamW 0,906 / 0,913 /
  0,909 / 0,948 / 0,901; chính 0,899 / 0,921 / 0,916 / 0,946 / 0,886.
- **KẾT QUẢ (14:2x):** thoát gốc ep9 / 11 / 10 / 15 / 14 → mask ep5 / 5 / 4 / 7 / 4; ROC mask 0,906 / 0,911 / 0,894 / 0,944 / 0,891.
  Ghép cặp (ROC / PR / F1@0,5 / F1@val): - chỉ ASAM gốc +0,007 (+3/-2) / +0,006 / +0,001 / +0,001; - AdamW -0,006 (+0/-4) / -0,003 / -0,017 (+0/-4)
  / -0,009; - chính -0,004 (+2/-3) / +0,001 / -0,006 / -0,027 (+0/-5). **Chấm:** D1 ĐÚNG (f4 0,9439497118910425 trùng 161), D2 ĐÚNG, D3 ĐÚNG, D4 ĐÚNG.
  Đọc: trên nguồn không chọn lọc, mask rút bình nguyên 5/5 fold nhưng ROC ngang ô gốc và không hơn AdamW ở fold nào - cùng kết luận bước 1.
  Nhận định: mục 7 của doc `dirA_step1` (tab Nhận định), `meta/insights/dirA/maskfull5.py`. f4 local giờ là bản paper_mw3 (trùng bit), bản 161
  ở `state/backup_0610_maskfull_f4_161/`.
- **paper_mw3 54437777 DESTROY 14:2x** (theo lời dặn 02:3x "hủy vast sau khi dùng xong"): driver XONG 14:17, 0 tiến trình, 0 file results/logs còn trên
  máy, 15/15 file kết quả về 161, `state/` 3/3 khớp tên + byte (`state/remote_logs/paper_mw3_54437777/state_full/`, cổng thử chiều lệch báo 1),
  checkpoint trên máy chỉ có `p1_full_jscpp` chép từ 161 (md5 khớp); `vastai show instances` = 0; bỏ khỏi `artifact_tick.sh`, `live_all.sh`, HOSTS.
- **Dự đoán (ghi TRƯỚC):** D1 f4 trên paper_mw3 trùng bit 161 (test ROC đủ 16 chữ số); D2 mask thoát (train loss < 0,65) ở ≤ ep7 ở 5/5 fold;
  D3 mask - chỉ ASAM gốc ROC TB trong [-0,01; +0,03], không 5/5 cùng dấu; D4 mask - AdamW không 5/5 cùng dấu và mask thấp hơn cột chính ở ≥ 3/5 fold.

### 🐍3F 06/10 17:5x - XONG 20:26 (5/5 rc 0) PHA 1 TRỘN JS:PY 3:1 - FULL trên vast paper_mw4 (+ f4 161, f5 158) (người dùng: "tôi đã thuê thêm paper_mw4 trên vast chạy 3:1 trên full nhé") - bậc 2 (n=5, seed 42, TF32)

- **Máy:** vast 54468293 `paper_mw4` (người dùng thuê; RTX A4000, driver 580.82.09, 0,086 $/h, Quebec), SSH 172.97.225.87:40995, hostname 53d6b312584e.
  Môi trường `setup_papermw3.sh` (wheel ghim, như paper_mw3). CodeBERT chép theo mảnh từ 161 (HF main không có model.safetensors). Monitor `mon_fpe_v10.sh`.
- **Dữ liệu:** `data/mwonly5_sources/js_py31_full/` (`--ratio 3`): 1 066 JS + 355 Python (Random(42), cân nhãn 178/177); val = JS val 188 + SVEN val
  fold k. Dựng lại trùng byte 15/15; Python ⊂ SVEN train, 0 trùng test.
- **Run:** `p1_jspy31_full` → `rasam_jspy31_full` (init_same_fold). Chạy khô: chỉ khác bản 4:1 full ở `--data_root`.
- **Kiểm máy mới trước:** `chk_p1_jspy41_full` f1 (= `p1_jspy41_full` f1, tên riêng để không đè) chạy vài epoch, so dòng Epoch với log 161; trùng thì dừng
  và phóng thật (f1 → f5, fold-major). ~1 h/fold ⇒ ~5 h.
- **17:53 kiểm máy:** `chk_p1_jspy41_full` f1 trên paper_mw4 - 3 epoch đầu TRÙNG từng chữ số log 161 ⇒ **F1 ĐÚNG**; dừng run kiểm (kill đúng PID
  driver 2290 + trainer 2318, còn 0 tiến trình, lock tự do). **PHÓNG 18:00** `run.sh paper_mw4` 3:1 full f1 → f5 (10 mục, dòng đầu log nhận đủ,
  lock giữ). Ước xong ~23:00. Bản ghi `chk_p1_jspy41_full__f1` trên trang treo "đang chạy" (không hiện ở bảng).
- **18:2x dời f4 → 161, f5 → 158** (người dùng hỏi máy vast chậm: KHÔNG chậm - epoch ~2 phút như 161/158, GPU 100 % chạm trần 100 W, CPU ~1,1 lõi;
  chậm là do 5 fold × ~1 h trên một máy): paper_mw4 `skip.txt` thêm f4, f5 (plan: f3 chạy, f4/f5 bỏ qua); `queue_after` 161 sau driver chỉ ASAM
  (PID 3141408) chạy `p1_jspy31_full:4 rasam_jspy31_full:4`, 158 sau PID 2217452 chạy f5. Dữ liệu js_py31_full trên 158 md5 khớp. Ước xong
  cả khối ~21:00 thay vì ~23:00.
- **Tiến độ 20:1x (3/5):** ROC f1 0,949 / f2 0,950 / f4 0,944. Ghép cặp ROC: - 4:1 full +0,017 / +0,006 / -0,028; - chỉ JS full +0,023 / +0,011 /
  -0,002. f3 trên paper_mw4 (Pha 1), f5 trên 158 chạy Pha 2 từ checkpoint Pha 1 KẸT (val 0,540, xem ⚠ CẢNH BÁO).
- **KẾT QUẢ 5/5 (20:26):** ROC 0,949 / 0,950 / 0,926 / 0,944 / 0,921. Ghép cặp (ROC / PR / F1@0,5 / F1@val): - 4:1 full -0,009 (+2/-3) / -0,012
  (+2/-3) / -0,019 (+1/-3) / -0,005 (+2/-3); - chỉ JS full +0,006 (+3/-2) / +0,002 (+3/-2) / +0,010 (+4/-1) / +0,011 (+3/-1). Pha 1 tách ngôn ngữ:
  JS 0,57-0,63, Python 0,92-0,95 (f5 KẸT: JS 0,509, Py 0,579 - Pha 2 vẫn 0,921, chỉ JS -0,005). **Chấm:** F1 ĐÚNG, F2 ĐÚNG (-0,009, 2/3), F3 SAI
  (3/5 không âm), F4 ĐÚNG (min 0,921). Đọc: 3:1 full không hơn 4:1 full, hơn chỉ JS full nhỏ và không ổn định - như 3:1 common (K1-K4).
- **Dự đoán (ghi TRƯỚC):** F1 chk trùng 161 ở mọi dòng Epoch đã chạy; F2 ROC 3:1 full - 4:1 full TB trong ±0,010, không 5/5 cùng dấu;
  F3 ROC 3:1 full - chỉ JS full TB ≥ 0, ≥ 4/5 fold không âm; F4 0/5 ô sập.

### 🐍C 06/10 18:3x - XONG 22:16 (5/5 rc 0) PHA 1 TRỘN JS:PY 4:1 - 4CWE trên 158 (f4-f5 dời 161) (người dùng: "158 chạy 4CWE pha 1 trộn đi") - bậc 2 (n=5, seed 42, TF32)

- **Tỷ lệ 4:1 như khối trộn chính** (người dùng không nêu tỷ lệ), Pha 2 cột chính. Dữ liệu `data/mwonly5_sources/js_py41_4cwe/`: 500 JS + 125 Python
  (SVEN train fold k, Random(42), cân nhãn 63/62); val = 88 JS val + SVEN val fold k 152. Dựng lại trùng byte 15/15; Python ⊂ SVEN train, 0 trùng test.
- **Run:** `p1_jspy41_4cwe` → `rasam_jspy41_4cwe` (init_same_fold). Chạy khô: chỉ khác `p1_4cwe_jsonly` / `rasam_4cwe_jsonly` ở dữ liệu / checkpoint.
  Đối chứng `rasam_4cwe_jsonly` cùng fold. 158 cuối chuỗi: chỉ ASAM 4:1 → 3:1 full f5 → 4CWE f1 → f5 (`queue_after` sau PID 2307257). ~30 phút/fold.
- **20:08 DỜI f4-f5 sang 161** (161 rảnh sau 3:1 full f4): 158 `skip.txt` thêm 4 dòng (plan 158 kiểm hai chiều: f3 chạy, f4/f5 bỏ qua); 161 0 tiến
  trình, lock tự do, GPU 1 MiB, preflight đạt, lệnh train sau khi chuẩn hoá đường dẫn trùng 158 (84 / 96 token), md5 dữ liệu f4-f5 trùng; phóng
  `run.sh 161 'p1_jspy41_4cwe:4 rasam_jspy41_4cwe:4 p1_jspy41_4cwe:5 rasam_jspy41_4cwe:5'` (PID 3348387, log `state/driver_161_jspy41_4cwe.out`),
  dòng đầu nhận đủ 4 mục, lock bị giữ. 158 còn f1-f3.
- **KẾT QUẢ 5/5 (22:16):** ROC 0,930 / 0,951 / 0,944 / 0,958 / 0,922. Ghép cặp với 4CWE chỉ JS (ROC / PR / F1@0,5 / F1@val): +0,015 (+4/-1, min
  -0,009 f5, max +0,037) / +0,014 (+4/-1) / +0,012 (+4/-1) / +0,012 (+2/-2). Pha 1 tách ngôn ngữ: JS (88) 0,52-0,57, Python (152) 0,81-0,90.
  **Chấm:** G1 ĐÚNG (min 0,922), G2 ĐÚNG (+0,015, 4/5 không âm), G3 ĐÚNG (Python 5/5 ≥ 0,81), G4 ĐÚNG (JS 5/5 < 0,57). Lợi trên 4CWE (+0,015)
  lớn hơn common (+0,009) và gần full (+0,014). Giả thuyết CHƯA kiểm: JS val 4CWE chỉ 88 hàm nên chọn checkpoint gần như theo phần Python (152).
- **Dự đoán (ghi TRƯỚC):** G1 0/5 ô sập; G2 trộn - chỉ JS ROC TB ≥ 0 và ≥ 4/5 fold không âm (hoà 0,001); G3 phần Python của val Pha 1 ROC ≥ 0,75 ở ≥ 4/5
  checkpoint; G4 phần JS của val Pha 1 ROC < 0,70 ở ≥ 4/5 (88 hàm JS 4CWE, như common/full).

### 🐍A 06/10 17:5x - XONG 19:29 (10/10, rc 0) CHỈ ASAM TRÊN PHA 1 TRỘN 4:1 (người dùng: "khi 2 máy dưới xong thì chạy thêm 4:1 chỉ ASAM") - 161 + 158 xếp sau khối 3:1, bậc 2 (n=5, seed 42, TF32)

- **Run:** `asamonly_jspy41_{common,full}` = `asamonly_<mức>_jsonly` (chỉ ASAM) nhưng nạp `p1_jspy41_<mức>` cùng fold (init_same_fold). Chạy khô: chỉ khác
  đối chứng ở checkpoint Pha 1. Hiểu "4:1" = cả common lẫn full như khối 4:1 (10 ô).
- **Máy theo nơi có checkpoint Pha 1:** 161 common f1/f3/f5 + full f1/f3; 158 common f2/f4 + full f2/f4/f5 (`queue_after` sau driver 3:1:
  161 PID 3021148, 158 PID 1981187). ~15-18 phút/ô ⇒ ~1,5 h mỗi máy sau khi khối 3:1 xong (~18:00).
- **Đối chứng:** `asamonly_<mức>_jsonly` (chỉ ASAM, Pha 1 chỉ JS) và `rasam_jspy41_<mức>` (cột chính, cùng Pha 1 trộn) cùng fold.
- **KẾT QUẢ 10/10 (19:3x; mục 6 doc `jspy41`):** chỉ ASAM trộn - chỉ ASAM chỉ JS (ROC / PR / F1@0,5 / F1@val): common +0,004 (+3/-2) / +0,002 /
  ±0,000 / +0,002; full +0,014 (+4/-1) / +0,021 (+4/-1) / +0,008 / +0,015. - cột chính trộn: common -0,003 (+1/-3), full -0,005 (+1/-3).
  **Chấm:** A41a ĐÚNG (min 0,926), A41b SAI (common 3/5 không âm), A41c ĐÚNG. Đọc: với chỉ ASAM, lợi của trộn chỉ thấy ở full; common đổi dấu.
- **Dự đoán (ghi TRƯỚC):** A41a 0/10 ô sập; A41b chỉ ASAM trộn - chỉ ASAM chỉ JS: ROC TB ≥ 0 và ≥ 4/5 fold không âm ở mỗi mức;
  A41c chỉ ASAM trộn - cột chính trộn: |ROC TB| ≤ 0,010 và không 5/5 cùng dấu ở mỗi mức.

### 🐍3 06/10 16:0x - XONG 18:07 (5/5, rc 0) PHA 1 TRỘN JS:PY 3:1 - CHỈ COMMON (người dùng: "thử với common, train tỷ lệ 3:1 để kiểm tra tác động tỷ lệ nguồn và đích cho hướng này") - 161 + 158, bậc 2 (n=5, seed 42, TF32)

- **Dữ liệu:** `data/mwonly5_sources/js_py31_common/` (`tools/build_p1_mixpy.py --ratio 3`): train = 842 JS + 281 Python (SVEN train fold k, Random(42),
  cân nhãn 141/140); val = JS val 148 + SVEN val fold k 152 (như 4:1). Dựng lại trùng byte 15/15; Python ⊂ SVEN train, 0 trùng test. Mẫu Python KHÔNG
  lồng vào mẫu 4:1 (chung 184-191/211 hàm) - mỗi tỉ lệ một lần rút Random(42).
- **Run:** `p1_jspy31_common` → `rasam_jspy31_common` (init_same_fold). Chạy khô: chỉ khác bản 4:1 ở tên / dữ liệu / checkpoint.
- **Đối chứng:** `rasam_jspy41_common` (4:1) và `rasam_common_jsonly` (chỉ JS) cùng fold. Chọn n=5 (không phải bậc 1 n=3) để ghép đủ 5 fold với 4:1.
- **Lịch:** 161 f1, f3, f5 (phóng ngay); 158 f2, f4 sau full f5 của khối 4:1 (`queue_after`). ~40 phút/fold.
- **KẾT QUẢ 5/5 (18:1x; mục 5 của doc `jspy41` tab Nhận định):** ROC 0,942 / 0,938 / 0,953 / 0,944 / 0,928. Ghép cặp (ROC / PR / F1@0,5 / F1@val):
  3:1 - 4:1 -0,002 (+1/-3) / ±0,000 (+1/-2) / +0,008 (+2/-2) / +0,008 (+2/-2); 3:1 - chỉ JS +0,007 (+3/-1) / +0,008 (+4/-1) / +0,012 (+2/-0) /
  +0,007 (+2/-2). **Chấm:** K1 ĐÚNG, K2 ĐÚNG (4/5 không âm, hoà 0,001), K3 ĐÚNG sát (3/5, f4 +0,002), K4 ĐÚNG. Đọc: thêm 70 hàm Python (4:1 → 3:1)
  không đổi thứ hạng theo hướng nào; lợi so với chỉ JS vẫn nhỏ, dưới sàn nhiễu ở 3/5 fold.
- **Dự đoán (ghi TRƯỚC):** K1 ROC 3:1 - 4:1 cùng fold TB trong ±0,010, không 5/5 cùng dấu; K2 ROC 3:1 - chỉ JS TB ≥ 0, ≥ 4/5 fold không âm;
  K3 phần Python của val Pha 1 ROC cao hơn 4:1 ở ≥ 3/5 fold; K4 0/5 ô Pha 2 sập.

### 🐍 06/10 11:58 PHÓNG - XONG 16:21 (10/10 ô, rc 0) - PHA 1 TRỘN JS:PY 4:1 THEO FOLD (người dùng 11:5x: "trộn phase 1 cho model transfer tỷ lệ js:py là 4:1 với cả bộ full và common, val sẽ bằng cả js 15% là python của fold đó. Chạy thử 5 fold") - 161 + 158, bậc 2 theo yêu cầu (n=5, seed 42)

- **Chọn (hỏi 12:0x):** Python lấy mẫu cân nhãn từ SVEN train fold k; val Pha 1 = JS val 15 % + SVEN val fold k; Pha 2 chỉ cột chính (RecAdam + ASAM).
- **Dữ liệu:** `data/mwonly5_sources/js_py41_{common,full}/fold1-5/` (`tools/build_p1_mixpy.py`, `MANIFEST.json`): train = toàn bộ JS train pool
  (842 / 1 066, nguyên văn) + 211 / 267 hàm SVEN train fold k (random.Random(42) ở mọi fold, cân nhãn 106/105, 134/133); val = JS val (148 / 188) + SVEN val
  fold k (152); test = bản sao val. Dựng lại hai lần trùng byte 30/30; phần Python ⊂ SVEN train fold k, 0 hàm trùng test fold k. Trainer MW chỉ đọc
  code / label / lang (không dùng cwe_class, nên lệch đánh số CWE giữa JS và SVEN không ảnh hưởng).
- **Run:** `p1_jspy41_{common,full}` (= `p1_<mức>_jsonly`, chỉ khác `--data_root`, folds 1-5, mỗi fold một checkpoint) →
  `rasam_jspy41_{common,full}` (= `rasam_<mức>_jsonly`, `init_same_fold`: fold k nạp Pha 1 fold k). `fpe.py` thêm `init_same_fold` (mặc định vẫn
  fold1; chạy khô hai chiều: run cũ vẫn nạp fold1, run mới chỉ khác run gốc ở tên / dữ liệu / checkpoint). Đối chứng: `rasam_<mức>_jsonly` cùng fold.
- **Seed / TF32 (người dùng 12:2x: "Bật tf32 và seed 42 đầy đủ nhé"):** lấy mẫu Python seed 42 (đã đổi từ 42 + k, dựng lại, trùng byte 30/30);
  lệnh hiệu lực của cả 4 run: `PYTHONHASHSEED=42 FPE_TF32=1`, `--seed 42 --deterministic 1 --tf32 1` (= đối chứng `rasam_<mức>_jsonly`).
- **Lịch (fold trọn vẹn, fold-major):** mỗi fold: p1 common → rasam common → p1 full → rasam full (~1,6 h/fold). 161: f1, f3, f5 (~4,9 h);
  158: f2, f4 (~3,3 h). A4000 hai máy trùng bit đã kiểm nhiều lần.
- **15:20 dời full f5 sang 158** (158 xong f2, f4 lúc 15:19): 161 `state/skip.txt` thêm `p1_jspy41_full:5`, `rasam_jspy41_full:5` (plan: bỏ qua; common
  f5 vẫn chạy); 158 0 tiến trình, lock tự do, preflight đạt ⇒ `run.sh 158 'p1_jspy41_full:5 rasam_jspy41_full:5'`, dòng đầu log nhận đủ. Ước xong
  ~16:15 (161 common f5 ~15:55).
- **Dự đoán (ghi TRƯỚC khi phóng):**
  - **KẾT QUẢ 10/10 (16:2x; nhận định doc `jspy41` tab Nhận định, `meta/insights/jspy41/jspy41.md`):** cột chính Pha 1 trộn - Pha 1 chỉ JS, ghép theo fold
    (ROC / PR / F1@0,5 / F1@val): common +0,009 (+4/-0) / +0,008 (+4/-1) / +0,004 (+3/-1) / -0,001 (+2/-2); full +0,014 (+5/-0) / +0,015 (+3/-1) /
    +0,029 (+5/-0) / +0,016 (+4/-1). ROC theo fold common +0,001 / +0,003 / +0,007 / +0,032 / +0,005; full +0,006 / +0,005 / +0,029 / +0,026 / +0,006.
    Pha 1 tách val: phần JS 0,54-0,64, phần Python (SVEN val) 0,89-0,95. **Chấm:** J1 ĐÚNG 10/10, J2 ĐÚNG 10/10, J3 SAI (full 5/5 cùng dấu), J4 ĐÚNG.
    Đọc: lợi nhỏ, không fold nào âm; nhưng Pha 1 đã thấy 211-267 hàm SVEN train và chọn checkpoint trên SVEN val (= val Pha 2) ⇒ giả thuyết
    CHƯA tách là lợi đến từ dữ liệu / val ĐÍCH chứ không phải chuyển giao liên ngôn ngữ. Một seed, 5/5 ở n=5 là sàn Wilcoxon (p 0,0625).
  - J1: 10/10 checkpoint Pha 1 có val ROC (val trộn) ≥ 0,70.
  - J2: trên phần Python của val Pha 1 (152 hàm SVEN val, tách từ probs) ROC ≥ 0,75 ở ≥ 8/10 checkpoint.
  - J3: Pha 2 so với `rasam_<mức>_jsonly` cùng fold: ROC TB trong ±0,015 và không 5/5 cùng dấu, ở cả hai mức (null); vùng không kết luận |Δ| < 0,010.
  - J4: 0/10 ô Pha 2 sập (test ROC ≥ 0,75).

### 🅰️ 05/10 23:5x - XONG 06/10 02:25 (15/15 ô, rc 0; thêm f2/f5 tự chạy xong 02:45): HƯỚNG A - bước 0 + bước 1 (người dùng: "Tham khảo multiBABEL/audits/HUONG_DI_LN_VA_TRAN_JS_0510.md triển hướng A xem sao") - 158 + paper_mw3 + 161, BẬC 1 (kiểm cơ chế, một seed)

- **KẾT QUẢ + NHẬN ĐỊNH (06/10 02:5x):** `_FinalPaperExperiment/meta/insights/dirA/dirA.md` (dựng lại: `build_dirA.py`), tab Nhận định doc `dirA_step1`.
  mask rút bình nguyên từ >17 / 17 / >17 / 16 / 15 xuống thoát ở ep2 / 3 / 4 / 5 / 7 (5/5 ô), cứu 2/2 ô sập (0,607 → 0,898; 0,523 → 0,869);
  ROC mask - AdamW -0,006 (+0/-4), mask - chính -0,021 (+0/-5), exclude ≈ mask (+0,005). **Chấm:** A0 ĐÚNG (5/5 trùng bit), A1 SAI sát (3/5 thoát
  ≤ ep4; ≤ ep5 thì 4/5), A2 SAI (2/5), A3 ĐÚNG (bias + LN 85-94 % ‖g·T‖² ep1), A4 ĐÚNG. Thêm f2/f5 common: C1-C4 ĐÚNG (mask thoát ep3-5 ở 5/5 fold,
  lợi ROC chỉ ở fold sập f3). Đọc: nhiễu bias/LN là nguyên nhân bình nguyên, KHÔNG phải đòn bẩy độ chính xác; bước 2 (30 lượt) không đề xuất.
  **Trang (bản 34):** ẩn khỏi bảng chính 8 dòng (3 full chạy lại, 3 exclude, asamblnex/asamblnsh common chỉ JS), vẫn ở tab So sánh; giữ 3 dòng mask.
- **paper_mw3 (vast 54294215) ĐÃ DESTROY 06/10 02:4x** (người dùng 02:3x: "hủy vast sau khi dùng xong"): driver in XONG 02:44, 0 tiến trình, 0 file
  results/logs còn trên máy, `state/` 10/10 khớp tên + byte ở `state/remote_logs/paper_mw3/state_full/` (cổng thử chiều lệch: báo 1), checkpoint Pha 1
  `p1_sven_python` (best + best_train_loss) + `p1_common_jscpp` khớp md5 3/3 với 161; `vastai show instances` = 0. Đã bỏ khỏi `artifact_tick.sh`,
  `live_all.sh`, HOSTS của trang. **Hiện KHÔNG có job nào chạy (161, 158 rảnh).**


- **Bước 0 (mã, xong 23:5x, 161 + 158 trùng md5):** `src_mwonly/mwg/optim.py` SAMStep thêm `bias_ln="mask"` (chuẩn ‖T·g‖ trên MỌI tham số như full,
  ε = 0 trên bias/LN ⇒ ε weight trùng bit full; đồ chơi: ‖T⁻¹ε‖ = 0,72ρ) + `log_share` (cộng dồn ‖g·T_full‖² theo nhóm weight / bias / LN, ghi mỗi epoch
  dòng `SAM share ep k | weight % | bias % | LN %`); `is_bias_or_ln` nhận cả tên MWG (`graph.ln0/ln.N`, `norm1/2`, `bias_ih/hh`, `in_proj_bias`).
  `train.py`: `--sam_bias_ln mask`, `--sam_log_share 1`. Thử CPU hai chiều: full/exclude/shrink cũ = mới trùng bit; mask ε weight trùng full; log_share
  không đổi ε; sam+mask bị chặn. Với MW thật tập bias/LN chỉ thêm 5 tensor `graph.*` KHÔNG có gradient (fusion mw trả về trước nhánh đồ thị) ⇒
  hành vi MW không đổi. Bản trước: `_FinalPaperExperiment/state/{optim,train}.py.before_mask_0510`. **Chưa port vào refactor** (repo ngoài vùng ntat).
- **Bước 1:** 5 ô của chỉ ASAM trên JS + C/C++ (gốc: 4cwe_jscpp f1 SẬP 0,607 bình nguyên > 17; common_jscpp f3 SẬP 0,523 > 17; common_jscpp f1 bình nguyên 16;
  common_jscpp f4 15; full_jscpp f4 14) × 3 nhánh `asamAfull_*` (chạy lại, phải trùng bit ô gốc), `asamAmask_*`, `asamAexcl_*`, mọi nhánh `--sam_log_share 1`;
  chạy khô: chỉ khác ô gốc ở hai cờ đó, cùng checkpoint Pha 1. 15 ô trên **158** (ô gốc chạy trên 3 máy A4000 khác nhau ⇒ nhánh full kiêm kiểm trùng bit
  giữa máy), thứ tự theo ô: full → mask → exclude. ~20-25 phút/ô ⇒ ~00:15 → ~06:00.
  **00:1x CHIA LẠI (người dùng: "share việc cho vast và 161 đi")** - giữ ba nhánh của MỘT ô trên cùng một máy (so trong ô sạch; trùng bit giữa A4000 đã
  kiểm: ô full 4cwe f1 trên 158 trùng từng chữ số epoch 1 với ô gốc trên vast): 158 = 4cwe f1 (đang chạy) + common f1; paper_mw3 = common f3 + f4
  (checkpoint `p1_common_jscpp` tải lên theo mảnh, mã + runs.json trùng md5); 161 = full f4 (`queue_after` sau trộn JS full f5). 158 `skip.txt` thêm 9 ô
  dời đi (plan: 4cwe f1 / common f1 chạy, common f3/f4 + full f4 bỏ qua). Ước xong cả bước 1 ~02:10-02:40.
  **00:38 chia lại lần 2:** tải checkpoint lên paper_mw3 chậm (4/14 mảnh sau ~30 phút, ước xong ~02:00) ⇒ common f4 sang 161 (`queue_after` nối sau
  hàng đợi full f4, PID 2024256; preflight đạt); paper_mw3 chỉ còn common f3, phóng khi tải xong. Ước xong cả bước 1 ~03:30.
  **00:47 sửa lại:** tải xong sớm (00:46, md5 trùng 161) ⇒ huỷ hàng đợi common f4 ở cuối chuỗi 161 (PID 2055528, chưa exec) và phóng trên paper_mw3
  `run.sh paper_mw3` common f3 (3 nhánh) → common f4 (3 nhánh), 1 driver, lock giữ. 161 chỉ còn full f4. Ước xong cả bước 1 ~02:55.
  **VIỆC CHỜ khi 15/15 xong (người dùng 06/10 00:3x: "thêm nhận định thật ngắn gọn về hướng A khi chạy xong nha, kết quả không tốt có thể ẩn khỏi kết quả
  chính luôn"):** (1) nhận định NGẮN ở tab Nhận định: tiêu chí đặt trước (mask thoát trước ep5 ở ≥ 4/5), bảng full / mask / exclude, chấm A0-A4;
  (2) ẩn khỏi bảng chính các biến thể không tốt (nhánh full = chạy lại; exclude nếu không hơn mask; asamblnex/asamblnsh common chỉ JS), giữ ở tab So sánh,
  thêm dòng nhắc số biến thể đã ẩn.
  Kết quả sớm: 4cwe f1 full trùng bit ô gốc (0,6071987480438183, vast → 158), mask ROC 0,898 (bình nguyên 1 epoch; AdamW 0,905, cột chính 0,918).
- **06/10 02:1x TỰ CHẠY trên 158 (máy rảnh, quy tắc "máy rảnh thì chạy việc đã bàn"; đề xuất P1 ở dạng mask):** thêm f2, f5 cho `asamAmask_common_jscpp`
  và `asamAexcl_common_jscpp` (runs.json folds [1,3,4] → [1..5], md5 35be4984 ở 161/158/paper_mw3) ⇒ nguồn common JS + C/C++ đủ 5/5 fold KHÔNG chọn
  lọc (5 ô bước 1 chọn vì kẹt). Nhánh full của f2/f5 = ô gốc `asamonly_common_jscpp` (trùng bit đã kiểm 5/5 ô). 4 lượt × ~15-20 phút, fold-major
  (mask f2, excl f2, mask f5, excl f5). Dừng: `kill` driver `run.sh 158` trên 158 (lock `$OUT/state/fpe.lock`).
  **PHÓNG 02:06** (0 tiến trình của mình, lock tự do, preflight đạt, plan in đúng checkpoint `p1_common_jscpp`, chạy khô chỉ khác ô gốc ở 2 cờ,
  VRAM trống 16 GB). LỖI CỦA TÔI: truyền 4 mục rời cho `run.sh` ⇒ driver PID 4001830 chỉ chạy mask f2; 02:07 nối 3 mục còn lại bằng `queue_after.sh`
  (PID 4004078, nhận đủ danh sách). Đã vá cổng đối số cho `run.sh` / `queue_after.sh` (CLAUDE.md §13). Ước xong ~03:15.
  **02:14 dời f5 sang paper_mw3** (xong bước 1 lúc 02:12; nhãn không phải `ntat` nên KHÔNG tự huỷ, chờ người dùng như paper_mw/paper_mw2 - trong lúc
  chờ thì chạy thay vì nằm không): 158 `skip.txt` thêm `asamA{mask,excl}_common_jscpp:5` (plan: f5 bỏ qua, excl f2 vẫn chạy); paper_mw3 0 tiến trình,
  lock tự do, preflight đạt, runs.json/run.sh trùng md5 ⇒ `run.sh paper_mw3 'asamAmask_common_jscpp:5 asamAexcl_common_jscpp:5'`, dòng đầu log
  nhận đủ 2 mục. Ước: paper_mw3 xong ~02:45, 158 (mask f2 → excl f2) ~02:45.
  Ô gốc: f2 thoát ep10, test 0,926 (best 19); f5 thoát ep16, test 0,893 (best 27); AdamW 0,900 / 0,901; chính 0,911 / 0,908.
  **Dự đoán (ghi TRƯỚC):** C1 mask và exclude thoát (train loss < 0,65) ở ≤ ep5 ở cả f2 lẫn f5; C2 ở f2, f5 |ROC mask - gốc| ≤ 0,03 (cứu chỉ có
  nghĩa ở fold sập); C3 trên 5 fold, mask - AdamW ROC không 5/5 cùng dấu; C4 mask thấp hơn cột chính ở ≥ 3/5 fold.
- **Tiêu chí đặt trước (tài liệu):** mask thoát bình nguyên (train loss < 0,65) trước epoch 5 ở ≥ 4/5 ô ⇒ nhiễu bias/LN gây kẹt; mask ≈ full ⇒ không phải,
  chuyển ASAM muộn; so exclude với mask để đọc riêng tác dụng tăng ε trên weight. ROC chỉ để tham khảo.
- **Dự đoán (ghi TRƯỚC khi phóng):**
  - A0: nhánh full trùng bit ô gốc (test ROC đủ 16 chữ số) ở 5/5 ô.
  - A1: mask thoát bình nguyên trước ep5 ở ≥ 4/5 ô (dựa trên 22:55: bỏ nhiễu bias/LN làm khởi động như AdamW trên common chỉ JS).
  - A2: exclude cũng thoát trước ep5 ở ≥ 4/5 ô.
  - A3: nhánh full, epoch 1: phần bias + LN của ‖g·T‖² ≥ 80 % (probe batch 8 của tài liệu: ~92 %).
  - A4 (tham khảo): mask ở hai ô sập thật (4cwe f1, common f3) có test ROC ≥ 0,85.

### ➕ 05/10 21:1x BA VIỆC THÊM (người dùng 21:0x: "Chạy thêm mw adamw js thuần (không transfer). Chạy thêm trộn naive JS full + sven rồi val và test trên sven (naive merge)"; "Chạy lại 2 ô sập"; giữ paper_mw3) - 161 + 158 + paper_mw3, bậc 2 (n=5, seed 42)

- **`nop1_adamw_tojsfull`** = `nop1_adamw` (MW, AdamW, `--init none`) trên ĐÍCH JS full (`js_full_folds_622`, `--target_lang js`). Chạy khô: chỉ khác hai cờ đó.
- **`mix1p_adamw_full_jsonly`** = `mix1p_adamw_common_jsonly` nhưng pool JS full: `data/final_experiment_data/mixsrc_jsfull/` (1 254 JS + 456 SVEN = 1 710 hàm/fold,
  val/test = SVEN trùng byte; `tools/build_mixsrc_folds.py`, dựng hai lần trùng 15/15; bản sao 158 + paper_mw3 trùng md5). Chạy khô: chỉ khác `--data_root`.
- **Chạy lại 2 ô sập** với `--min_epochs 30` (`--epochs` giữ 30 nên lịch LR không đổi; chạy khô chỉ khác cờ đó): `rasam_sven_python_tojsfull_me30` f4
  (158, cùng máy ô gốc), `mix1p_adamw_common_jsonly_me30` f5 (161, cùng máy ô gốc).
- **Lịch (fold trọn vẹn):** paper_mw3: nop1 JS f1 → trộn full f1 → nop1 JS f2 → trộn full f2; 158: me30 SVEN → JS f4 → nop1 JS f3, f4;
  161: me30 trộn f5 → trộn full f3, f4, f5 → nop1 JS f5. Ước xong ~23:45-00:15.
  **PHÓNG 22:06** cả ba máy (mỗi máy 1 driver + 1 trainer, lock giữ; trước đó 0 tiến trình, lock tự do, preflight đạt, skip.txt không chặn fold nào,
  VRAM trống ≥ 15,6 GB). Monitor v8 ba máy; trang: 4 dòng mới (nhóm Biến thể phụ + nhóm Đích JS), 12 ô gieo sẵn.
  **23:48 dời `nop1_adamw_tojsfull` f5 sang paper_mw3** (xong việc lúc 23:47): 161 `state/skip.txt` thêm `:5` (plan: f5 bỏ qua, trộn JS full f5 vẫn chạy),
  paper_mw3 0 tiến trình / lock tự do / preflight đạt ⇒ phóng `run.sh paper_mw3 "nop1_adamw_tojsfull:5"`. Ước: paper_mw3 xong ~00:20, 161 ~00:55, 158 ~00:15.
- **Dự đoán (ghi TRƯỚC khi phóng):**
  - **KẾT QUẢ `nop1_adamw_tojsfull` 5/5 (00:06; paper_mw3 f1, f2, f5 + 158 f3, f4):** ROC 0,436 / 0,485 / 0,471 / 0,557 / 0,542, TB **0,498** (quanh ngẫu
    nhiên), best ep 1 / 4 / 17 / 6 / 10, bình nguyên 6 / > 17 / > 25 / 6 / 8. So với baseline JS: ROC -0,013 (+2/-3); SVEN → JS - MW JS: ROC **+0,054 (+4/-1)**,
    PR +0,043 (+4/-1), F1@0.5 +0,062 (+4/-1), F1@val +0,054 (+4/-1) ⇒ trên đích JS, Pha 1 SVEN hơn MW thuần ở 4/5 fold (fold thua là f4 sập của SVEN → JS).
    **Chấm:** A1 SAI biên (0,498 < 0,50); A2 ĐÚNG (2/5 kẹt > 13); A3 SAI (+0,054 vượt ±0,04, không 5/5 cùng dấu thì đúng).
  - A1: `nop1_adamw_tojsfull` ROC TB trong [0,50; 0,60] (baseline JS 0,511); A2: ≥ 1/5 fold kẹt ln2 (train loss chưa < 0,6 ở ep13);
    A3: SVEN → JS - nop1 JS, ROC TB trong ±0,04, không 5/5 cùng dấu.
  - M1: `mix1p_adamw_full_jsonly` ≤ 1/5 sập; M2: bỏ ô sập, so với `nop1_adamw` ROC dương ở ≥ 3 fold còn lại; M3: so với trộn common (cùng fold, bỏ ô sập)
    ROC TB trong ±0,015.
  - **KẾT QUẢ `mix1p_adamw_full_jsonly` 5/5 (06/10 01:18; paper_mw3 f1-f2, 161 f3-f5, rc 0):** ROC 0,911 / 0,934 / 0,901 / 0,927 / 0,929, TB **0,921**,
    best ep 8 / 9 / 6 / 10 / 13, 0/5 sập. Ghép cặp theo fold, thứ tự ROC / PR / F1@0,5 / F1@val:
    - so với MW không Pha 1 AdamW: +0,020 (+4/-0, f4 hoà) / +0,022 (+4/-0) / +0,021 (+4/-1) / +0,029 (+5/-0);
    - so với baseline: +0,024 (+4/-1) / +0,023 (+4/-1) / +0,031 (+4/-1) / +0,027 (+3/-1);
    - so với trộn common (bỏ f5 sập, n=4): -0,015 (+0/-4) / ±0,000 (+2/-2) / -0,024 (+1/-3) / -0,036 (+1/-3);
    - so với HAI PHA cùng nguồn JS full: AdamW (noRAS) +0,018 (+4/-1) / +0,027 (+5/-0) / -0,002 (+2/-3) / -0,014 (+2/-2);
      cột chính (RecAdam + ASAM) -0,012 (+1/-4) / -0,011 (+0/-5) / -0,027 (+1/-4) / -0,041 (+0/-4).
    **Chấm:** M1 ĐÚNG (0/5 sập); M2 ĐÚNG (ROC dương 4/5, f4 hoà); M3 SAI biên (-0,0152, ngoài ±0,015, 0/4 dương).
    Đọc (bậc 2, một seed, chưa lặp máy): trộn naive hơn MW không Pha 1 và baseline ở 4/5 fold; ngang hai pha AdamW về thứ hạng nhưng
    thua ở ngưỡng; thua cột chính ở cả bốn chỉ số (4-5/5 fold). Chạy lại trộn common f5 với min_epochs 30: test ROC 0,4856 trùng ô gốc (kẹt cả 30 epoch).
  - R1: SVEN → JS f4 me30 thoát bình nguyên trước ep30 (best ep > 10) và test ROC ≥ 0,55 (f3/f5 thoát ở ep18, test 0,60 / 0,58).
  - R2: trộn common f5 me30 thoát trước ep30 và test ROC ≥ 0,85 (`nop1_adamw` cùng fold 0,909). Độ chắc thấp: AdamW thuần đã kẹt 17 epoch.

### 🧩 05/10 19:0x - XONG 21:05 (paper_mw3 f1-f2, 158 f3, 161 f4-f5; 5/5 rc 0): MỘT PHA TRỘN NGUỒN, MW không đồ thị (người dùng: "trộn nguồn sau đó train 1 pha kiểu baseline nhưng trộn nguồn khác, cái này sẽ không dùng graph. Chạy thêm vào n=5") - 158 f1-f3 + 161 f4-f5, bậc 2 (n=5, seed 42)

- **Chọn (hỏi 18:5x):** MW không đồ thị (src_mwonly), nguồn common chỉ JS, AdamW.
- **Run** `mix1p_adamw_common_jsonly` = `nop1_adamw` (MW không Pha 1, AdamW, lr 5,66e-5, 30 epoch, không warmup, min 10, patience 8) nhưng train =
  990 hàm JS common (pool `mwonly5_sources/js_common`, train + val) + SVEN train của fold (1 446 hàm, js 990 / python 456); val/test = SVEN của fold
  (trùng byte). Bỏ `--target_lang` (train có cả hai ngôn ngữ). Chạy khô: lệnh train/test chỉ khác `nop1_adamw` ở `--data_root` và thiếu `--target_lang`.
- **Dữ liệu:** `data/final_experiment_data/mixsrc_jsonly_common/` (`tools/build_mixsrc_folds.py`, `MANIFEST.json` md5 từng file; không hàm nào của pool
  trùng val/test đích; dựng hai lần trùng 15/15); bản trên 158 trùng md5.
- **Lịch:** chia fold trọn vẹn: 158 f1-f3 sau `rasam_sven_python_tojsfull` f4-f5 (~20:10), 161 f4-f5 sau kiểm 2.3 (~20:30); ~3× SVEN mỗi epoch ⇒ ~25-30 phút/fold.
  **20:10 dời f1-f2 sang paper_mw3** (rảnh sau baseline JS): dữ liệu + runs.json trùng md5, 158 `state/skip.txt` thêm `:1` `:2` (plan: f1/f2 bỏ qua, f3 chạy),
  paper_mw3 preflight đạt, 0 tiến trình, lock tự do ⇒ phóng `run.sh paper_mw3 "mix1p_adamw_common_jsonly:1 mix1p_adamw_common_jsonly:2"` (1 driver + 1 trainer).
  161 chạy f4 (từ 19:41) → f5; 158 chỉ còn f3 sau SVEN → JS f5.
- **Số đối chứng (ROC f1-f5):** MW không Pha 1 AdamW 0,892 / 0,897 / 0,879 / 0,927 / 0,909 (TB 0,901); 2 pha AdamW common chỉ JS (`noras_common_jsonly`)
  0,902 / 0,914 / 0,912 / 0,928 / 0,928 (TB 0,917); cột chính 0,934; baseline 0,897.
- **KẾT QUẢ 5/5 fold (21:05; bậc 2, một seed, ba máy A4000 chia theo fold):** ROC 0,925 / 0,963 / 0,911 / 0,935 / **0,486 (f5 SẬP)**, TB 0,844;
  best ep 5 / 6 / 11 / 11 / 1. Ghép cặp ROC theo fold:
  - so với MW không Pha 1 AdamW: +0,032 / +0,066 / +0,032 / +0,008 / -0,423 ⇒ TB -0,057 (+4/-1); **bỏ f5: +0,035 (4/4)**;
  - so với 2 pha AdamW cùng nguồn (`noras_common_jsonly`): +0,023 / +0,049 / -0,001 / +0,007 / -0,442 ⇒ TB -0,073 (+3/-2); bỏ f5: +0,019;
  - so với cột chính (RecAdam + ASAM 2 pha): -0,018 / +0,028 / -0,023 / +0,009 / -0,445 ⇒ TB -0,090 (+2/-3); bỏ f5: -0,001;
  - so với baseline: TB -0,053 (+3/-2); bỏ f5: +0,033.
  PR / F1 cùng hướng (bỏ f5 thì dương so với không Pha 1). Đọc: khi KHÔNG sập, trộn 990 hàm JS vào train hơn hẳn chỉ SVEN (+0,035 ROC, 4/4) và
  ngang cột chính 2 pha (-0,001) - nhưng f5 SẬP với AdamW thuần (ln2 suốt 17 epoch, `nop1_adamw` cùng fold 0,909): trộn nguồn làm mất ổn định.
  Một seed, một khối; giả thuyết, chưa lặp.
  **Chấm:** X1 SAI (TB -0,057 vì f5; bỏ f5 +0,035 vẫn vượt [-0,005; +0,025]); X2 SAI (TB -0,073; bỏ f5 +0,019 thì trong ±0,015 cũng không);
  X3 SAI (kém cột chính 3/5, cần ≥ 4/5); X4 SAI (1/5 sập).
- **Dự đoán (ghi TRƯỚC khi phóng):**
  - X1: so với MW không Pha 1 AdamW, ROC TB trong [-0,005; +0,025] (thêm 990 hàm JS vào train; Pha 1 tuần tự cùng optimizer chỉ +0,016).
  - X2: so với 2 pha AdamW cùng nguồn, ROC TB trong ±0,015, không 5/5 cùng dấu (trộn ≈ tuần tự khi cùng optimizer).
  - X3: kém cột chính (RecAdam + ASAM 2 pha) về ROC ở ≥ 4/5 fold.
  - X4: 0/5 sập.

### 🔀 05/10 17:0x - XONG 20:19 (paper_mw3 f1-f3 + 158 f4-f5, 5/5 rc 0; baseline JS 5/5): NGUỒN SVEN → ĐÍCH JS full (người dùng: "lên lịch chạy task nguồn SVEN, đích JS nhé. JS chia lại 5 fold, tỷ lệ 6:2:2. lưu data thí nghiệm này cẩn thận"; 16:5x: "tôi đã thêm máy paper_mw3, chia việc để chạy task js") - vast paper_mw3, bậc 2 (n=5, seed 42)

- **Chọn (hỏi 16:4x):** đích JS **full**; **chỉ cột chính**; xoá checkpoint Pha 2 như khối (giữ Pha 1 + kết quả + xác suất từng mẫu + log).
- **Dữ liệu (lưu cẩn thận):**
  - Đích `data/final_experiment_data/js_full_folds_622/` (1 254 hàm 627/627, 5 fold xoay vòng như SVEN, 752/251/251; `tools/build_target_folds.py`;
    README mục "Đích JS full" + `MANIFEST.json` md5 từng file; dựng hai lần trùng md5).
  - Nguồn `data/mwonly5_sources/sven_python/` (760 hàm SVEN đã xoá comment, 646/114, test = val; `tools/build_hf_pools.py --only sven_python`, pool cũ giữ nguyên).
  - Bản sao trên 158 `/data/ntat/MultiVD/data/...` trùng md5 20/20 file.
- **Run:** `p1_sven_python` (Pha 1, cờ như mọi Pha 1 của khối) → `rasam_sven_python_tojsfull` f1-f5 (cờ = `rasam_common_jsonly` + `--data_root
  data/final_experiment_data/js_full_folds_622 --target_lang js`; chạy khô: lệnh train/test chỉ khác đúng hai cờ đó + checkpoint Pha 1).
- **Máy:** cả task trên **paper_mw3** (vast 54294215, A4000, 0,091 $/h; `runs.json` hosts.paper_mw3), vì 161 nhận phép kiểm 2.3. Không chép checkpoint
  Pha 1 giữa máy. Môi trường: `state/vast_papermw3_54294215/setup_papermw3.sh` (= setup paper_mw), phiên bản trùng vdenv; mã trùng md5 24/24.
  CodeBERT tải lên theo mảnh 40 MB (đường 161 → vast đứt mỗi ~1 phút), ghép lại trùng md5 6/6 file (17:11).
  **Thử khói ĐẠT** (`state/vast_papermw3_54294215/smoke_papermw3.sh`): `nop1_adamw` f1 epoch 1-2 trên paper_mw3 trùng từng chữ số với 161
  (train loss 0,7475 / 0,7552, val loss 0,8369 / 0,6789, val ROC 0,5450 / 0,6672, val F1 0,3274 / 0,5368); không còn tiến trình GPU sau khi dừng.
  **PHÓNG 17:13** (`run.sh paper_mw3 "p1_sven_python rasam_sven_python_tojsfull"`, 1 driver + 1 trainer, lock giữ thật); Pha 1 ~29 s/epoch.
  Pha 1 xong 17:21: chọn ep6, val ROC nguồn 0,955 (J1 ĐÚNG). f1 đọc lại từ log: khởi tạo từ `p1_sven_python`, `RecAdam | cof=500 | k=0.05 |
  t0=7/720 | neo 218 tensor`, `SAM | asam rho 0,5 | bias_ln=full`, `--data_root js_full_folds_622 --target_lang js`, train 752 / val 251.
  Epoch JS ~122 s (hàm dài hơn, nhiều cửa sổ) ⇒ 5 fold xong ~21:30.
  **18:43 DỜI f4-f5 SANG 158** (158 rảnh sau Java + JS; người dùng 16:5x "chia việc để chạy task js"): checkpoint Pha 1 SVEN chép paper_mw3 → 161 → 158
  (md5 best.pt 817d3407…, trùng 3 nơi; 161 giữ bản lưu `model/mwonly5/p1_sven_python/`), mã + runs.json 158 trùng md5 161; paper_mw3 `state/skip.txt`
  thêm `:4` `:5` (plan: f3 chạy, f4/f5 SKIP); 158 plan f4/f5 trỏ checkpoint thật, preflight đủ, 0 tiến trình, lock tự do, VRAM trống 15,7 GB.
  Phóng `run.sh 158 "rasam_sven_python_tojsfull:4 rasam_sven_python_tojsfull:5"` (1 driver + 1 trainer, lock giữ). Hai máy cùng A4000, chia theo
  fold trọn vẹn (CLAUDE.md §4). paper_mw3 còn f2 → f3 (~19:40); 158 f4 → f5 (~20:10).
  **18:5x THÊM BASELINE ĐÍCH JS** (người dùng: "lên lịch chạy thêm baseline js"): `baseline_tojsfull` = `baseline` của khối (CodeBERT 512 thuần, TF32,
  lr 4e-5, 30 epoch, patience 8) chỉ đổi `--data_root data/final_experiment_data/js_full_folds_622 --target_lang js` (chạy khô: lệnh train/test chỉ khác
  đúng hai cờ đó). `src_final` + `runs.json` lên paper_mw3 trùng md5 18/18, preflight đạt. Xếp trên paper_mw3 SAU f3 (`queue_after.sh`), f1-f5,
  ~10 phút/fold ⇒ ~19:40 → ~20:30. **XONG 20:07 (5/5 rc 0):** ROC 0,509 / 0,531 / 0,512 / 0,528 / 0,478 (TB 0,512 - quanh ngẫu nhiên; f5 train loss
  kẹt ln2 suốt 15 epoch dù AdamW thuần). Kết quả paper_mw3 đã kéo hết về 161 (Pha 1 + baseline 5/5 + SVEN → JS f1-f3).
  **Dự đoán (ghi TRƯỚC):** BJ1 ROC TB trong [0,52; 0,66]; BJ2 SVEN → JS - baseline: ROC TB trong ±0,03 và
  KHÔNG 5/5 cùng dấu (Pha 1 SVEN mang ít sang JS); BJ3 rc 0 ở 5/5.
  **Quan sát 18:2x:** bình nguyên ln2 ~10 epoch ở CẢ f1 lẫn f2 (f2: train loss 0,75 → 0,708 ep10 → 0,651 ep13; val ROC best 0,612 ep11) - trên đích
  JS đây là hành vi CHUNG của RecAdam + ASAM, không phải ô lẻ. Luật "train loss chưa < 0,6 sau 13 epoch" sẽ báo ở mọi fold JS: chỉ ghi ở đây,
  không mở cảnh báo mới trừ khi val ROC tụt hẳn hoặc train loss kẹt tới hết patience. Luật "best val < 0,7" đã bỏ cho `_tojs` (fpe.py 18:3x).
- **Monitor:** `mon_fpe_v8.sh` (= v7 + không gắn '⚠ SẬP' < 0,75 cho run `_tojs`: ngưỡng đó của đích SVEN; JS full trong miền chỉ đạt val ROC 0,59).
  Trang: nhóm "Đích JS full (nguồn SVEN Python)" ở cuối bảng chính; tab So sánh không tính Δ giữa hai đích khác nhau.
- **KẾT QUẢ 5/5 fold (20:19; bậc 2, một seed, hai máy A4000 chia theo fold):** SVEN → JS ROC 0,583 / 0,528 / 0,604 / 0,468 / 0,580 (TB 0,553),
  best ep 12 / 16 / 18 / **1** / 18; baseline JS ROC 0,509 / 0,531 / 0,512 / 0,528 / 0,478 (TB 0,511). Ghép cặp SVEN → JS - baseline:
  ROC **+0,041 (+3/-2**, -0,060..+0,103), PR +0,034 (+3/-2), F1@0.5 +0,036 (+4/-1), F1@val +0,095 (+4/-1).
  Đọc: đích JS rất khó (baseline quanh ngẫu nhiên, f5 baseline kẹt ln2 cả với AdamW thuần). Pha 1 SVEN hơn baseline ở fold nào thoát được bình
  nguyên ln2 (f1, f3, f5: +0,07..+0,10), thua ở fold sập (f4: -0,060) - phương sai lớn, ROC chỉ 3/5 dương nên CHƯA nói được Pha 1 SVEN giúp JS.
  Bình nguyên ln2 10-18 epoch ở cả 5 fold (RecAdam + ASAM) là trở ngại chính trên đích này.
  **Chấm:** J1 ĐÚNG (val 0,955); J2 ĐÚNG (TB 0,553 trong [0,50; 0,68]); J3 SAI (epoch chọn ≤ 3 chỉ 1/5 - f4 sập; các fold khác chọn muộn ep12-18);
  J4 ĐÚNG (rc 0 5/5). BJ1 SAI (baseline TB 0,511, dưới 0,52); BJ2 SAI (Δ ROC +0,041 vượt ±0,03; không 5/5 cùng dấu thì đúng); BJ3 ĐÚNG.
- **Dự đoán (ghi TRƯỚC khi phóng):**
  - J1: Pha 1 SVEN chọn checkpoint có val ROC (nguồn, 114 hàm) ≥ 0,80.
  - J2: test ROC trên JS full TB trong [0,50; 0,68] (tham chiếu duy nhất: Pha 1 JS full trong miền val ROC 0,59, khác cách chia nên chỉ để định cỡ).
  - J3: ở ≥ 2/5 fold epoch được chọn ≤ 3 (val chia theo hàm trên dữ liệu cặp làm val ROC giảm khi học - memory paired-val-split).
  - J4: rc = 0 ở 5/5 fold.

### 🧪 05/10 17:0x - XONG 19:41 (161, 10/10 ô rc 0): KIỂM 2.3: nhiễu ε của ASAM trên bias + LayerNorm (người dùng: "local test thêm cho tôi theo tham khảo mục 2.3 /drive1/cuongtm/metaProject/MAML/multiBABEL/audits/PHA2_SUA_THEO_BANG_CHUNG_0510.md") - 161, BẬC 1 → BẬC 2 từ 18:0x (n=5, seed 42)

- **Chọn (hỏi 16:5x):** cả hai biến thể; công thức chỉ ASAM (AdamW + ASAM ρ 0,5 η 0,01); chỉ nguồn common chỉ JS; f1-f3.
  - `asamblnex_common_jsonly`: `--sam_bias_ln exclude` - bias + LayerNorm KHÔNG bị nhiễu, bỏ khỏi cả ε lẫn chuẩn ‖T·g‖ (tập bị nhiễu thu hẹp, ρ giữ 0,5 ⇒ ε trên weight LỚN hơn trước).
  - `asamblnsh_common_jsonly`: `--sam_bias_ln shrink --sam_bias_ln_scale 0.1` - T của bias + LayerNorm nhân 0,1.
  - Ghép cặp với `asamonly_common_jsonly` f1-f3 (cùng checkpoint Pha 1 `p1_common_jsonly`; chạy khô: lệnh train/test chỉ khác đúng các cờ trên).
- **Mã:** `src_mwonly/mwg/optim.py` SAMStep thêm `bias_ln` (mặc định `full` = như cũ) + `train.py` hai cờ; bản trước ở
  `_FinalPaperExperiment/state/{optim,train}.py.before_biasln_0510`. Thay vào 161 lúc 16:56 SAU khi `rasamanchor` f5 đã nạp mã. Thử CPU hai chiều:
  `full` cho ε và w+ε trùng bit mã cũ; `exclude` ε = 0 trên bias/LN và ‖T⁻¹ε‖ = 0,500000; `shrink` khớp tính tay; `sam` + `exclude` bị chặn.
  Trên mô hình đồ chơi, tỉ phần ‖ε‖² trên bias/LN: full 0,920, shrink 0,001, exclude 0.
- **Số đối chứng (f1-f3):** chỉ ASAM ROC 0,9317 / 0,9484 / 0,9451, bình nguyên 1/1/1 epoch, best ep 7/5/7; AdamW ROC 0,9016 / 0,9139 / 0,9125
  (ASAM hơn AdamW +0,030 / +0,035 / +0,033). Nguồn này KHÔNG có ô kẹt nên phép kiểm chủ yếu trả lời: bỏ / co nhiễu bias-LN có GIỮ được lợi của ASAM không.
- **KẾT QUẢ n=5 (19:41; bậc 2, một seed, một máy 161):** ghép cặp theo fold với chỉ ASAM / với AdamW (`noras_common_jsonly`):
  - **bỏ nhiễu bias/LN** (ROC 0,914 / 0,932 / 0,913 / 0,931 / 0,946): so với chỉ ASAM ROC **-0,009 (+2/-3)**, PR -0,012 (+2/-3), F1@0.5 -0,009 (+1/-4),
    F1@val -0,016 (+2/-3); so với AdamW ROC +0,010 (+4/-0), PR +0,007 (+4/-1), F1@0.5 +0,010 (+2/-2), F1@val +0,001 (+3/-2).
  - **co T ×0,1** (ROC 0,915 / 0,932 / 0,915 / 0,925 / 0,951): so với chỉ ASAM ROC **-0,009 (+1/-4)**, PR -0,007 (+1/-4), F1@0.5 -0,018 (+1/-4),
    F1@val -0,032 (+0/-3); so với AdamW ROC +0,011 (+4/-1), PR +0,012 (+5/-0), F1@0.5 +0,001 (+2/-2), F1@val -0,016 (+2/-3).
  - Bình nguyên 0-1 epoch ở mọi ô (như chỉ ASAM). hyperparameters chỉ khác chỉ ASAM ở `sam_bias_ln(_scale)` (+ hai cờ RecAdam mới ở mặc định).
  - **Đọc:** ở n=3 cả hai biến thể kém chỉ ASAM 3/3 (ROC -0,022 / -0,021) - **CO LẠI ở n=5** còn -0,009 (dưới sàn nhiễu 0,010, f4-f5 đổi dấu).
    Câu "lợi của ASAM phần lớn đến từ nhiễu trên bias/LN" (báo 18:3x) RÚT LẠI. Còn đứng: cả hai biến thể vẫn hơn AdamW về ROC ở 4/5 fold
    (+0,010 / +0,011), tức bỏ hay co nhiễu bias/LN KHÔNG làm mất hẳn lợi của ASAM; trên nguồn này cũng không có lợi (không có bình nguyên để cứu).
    Mục 2.3 của tài liệu (giảm/loại nhiễu bias/LN) chưa được ủng hộ trên công thức chỉ ASAM, nguồn common chỉ JS; cần nguồn có ô kẹt mới trả lời phần "cứu bình nguyên".
  - **Chấm:** B1 ĐÚNG (dòng `SAM |` in đúng); B2 ĐÚNG (bình nguyên ≤ 1 epoch 10/10); B3 ĐÚNG (ROC TB -0,009 / -0,009 trong [-0,020; +0,010], dương ≤ 2/5);
    B4 ĐÚNG (hơn AdamW 4/5 cả hai).
- **📌 NHẬN ĐỊNH 22:55** (tab "Nhận định" + `_FinalPaperExperiment/meta/insights/asam_biasln/asam_biasln.md`, dựng bằng `build_asam_biasln.py`):
  không hơn chỉ ASAM về độ chính xác (ROC -0,009 cả hai, dưới sàn nhiễu), vẫn hơn AdamW (+0,010 / +0,011, 4/5) ⇒ giữ ~một nửa lợi ROC của ASAM
  (+0,020, toàn bộ ở f1-f3). Cơ chế: hai biến thể khởi động như AdamW (train loss ep3 0,25 so với ASAM 0,44, AdamW 0,21; val ROC ep1 0,82 so với 0,73)
  ⇒ phần lớn "phanh" của ASAM nằm ở ε trên bias/LN. Bỏ hẳn và co T không phân biệt được (|Δ ROC| ≤ 0,006 mọi fold).
  Không thay ASAM chuẩn; giá trị của mục 2.3 nằm ở nguồn/đích có kẹt - chưa kiểm.
- **Dự đoán cho bước tiếp (CHƯA CHẠY, cần duyệt):**
  - P1: bỏ nhiễu bias/LN, chỉ ASAM, nguồn common JS + C/C++ (ô gốc kẹt 4/5): bình nguyên ≤ 2 epoch ở ≥ 4/5 fold; 0/5 sập; ROC dương so với chỉ ASAM ở mọi fold gốc kẹt.
  - P2: bỏ nhiễu bias/LN với cột chính trên đích JS full (nguồn SVEN; gốc kẹt 10-18 epoch cả 5 fold, f4 sập): bình nguyên ≤ 3 epoch ở ≥ 4/5; 0/5 sập; ROC TB > 0,553.
  - P3: lặp seed 1234 trên common chỉ JS (chỉ ASAM, bỏ nhiễu, AdamW): |ROC bỏ nhiễu - chỉ ASAM| TB < 0,010, không 5/5 cùng dấu; bỏ nhiễu hơn AdamW ở ≥ 3/5.
- **Dự đoán (ghi TRƯỚC khi phóng):**
  - B1: dòng `SAM |` của mỗi ô in đúng `bias_ln=exclude` / `bias_ln=shrink scale=0.1`.
  - B2: bình nguyên ≤ 1 epoch ở 3/3 fold cho cả hai biến thể (không dài hơn chỉ ASAM).
  - B3: ROC so với chỉ ASAM trong [-0,020; +0,010] cho cả hai biến thể, không 3/3 cùng dấu dương.
  - B4: cả hai biến thể vẫn hơn AdamW về ROC ở ≥ 2/3 fold (giữ phần lớn lợi của ASAM).
- **Lịch:** 161, xếp sau `rasamanchor` (xong ~17:20), fold-major: ex f1, sh f1, ex f2, sh f2, ex f3, sh f3 ≈ 6 × 23 phút ⇒ xong ~19:40.
- **18:0x LÊN BẬC 2 (người dùng: "đẩy lên n=5"):** thêm f4-f5 cho cả hai biến thể (`runs.json` folds 1-5), `queue_after.sh` chờ driver f1-f3
  (PID 1508369) rồi chạy ex f4, sh f4, ex f5, sh f5 (`state/queue_161_biasln45.out`) ⇒ xong ~20:50. Plan f4/f5 trỏ đúng `p1_common_jsonly`.
  Dự đoán B2-B4 giữ nguyên, áp cho n=5. Kết quả f1 (bậc 1, mới 1/3 fold lúc lên bậc): bỏ nhiễu ROC -0,018 / F1@0.5 +0,034 so với chỉ ASAM; co T
  ROC -0,016 / F1@0.5 -0,018; cả hai hơn AdamW về ROC (+0,012 / +0,014); train loss giảm nhanh như AdamW.

### ⚓ 05/10 15:25 - XONG 17:15 (161, 5/5 ô rc 0): RecAdam NEO THEO BÀI GỐC + ASAM, nguồn common chỉ JS (người dùng: "chạy thêm js common với SOTA chính nhưng có recadam neo theo đúng các phân tích của các nghiên cứu khác") - 161, bậc 2 (n=5, seed 42)

- **Run** `rasamanchor_common_jsonly` (161, f1-f5, ~25 phút/fold ⇒ ~2 giờ): cột chính `rasam_common_jsonly` (cùng checkpoint Pha 1 `p1_common_jsonly`,
  cùng ASAM ρ 0,5, cùng mọi cờ) nhưng sửa ĐÚNG ba chỗ lệch so với Chen et al. 2020 (đã đối chiếu mã gốc, xem `asam_collapse.md` mục 6 và
  `archive_20261004/meta/review_pack/collapse_fix_D.md` C5): (1) γ (`--pretrain_cof`) 500 → **5 000** (mặc định mã gốc); (2) t0 4 bước (1 %, ngoài lưới
  của bài) → **100 bước** tuyệt đối (`--anneal_t0_steps 100`, cờ mới), k giữ **0,05** - cả hai là giá trị NHỎ NHẤT trong lưới §4.1 của bài
  (k ∈ {0,05; 0,1; 0,2; 0,5; 1}, t0 ∈ {100; 250; 500; 1 000}); (3) head mới **không neo** (`--recadam_free_head 1`, cờ mới ⇒ `anneal_w=0` cho `vul_head.*`
  như `run_glue_with_RecAdam.py`).
- **Vì sao không lấy mặc định mã gốc (k 0,5, t0 250):** Pha 2 chỉ có 450 bước (15 bước/epoch × 30). t0 250 khoá backbone tới epoch ~16, cắt 80 %
  ngân sách LR; còn k 0,5 thì chuyển 5 % → 95 % trong 12 bước nên neo gần như không bao giờ kéo (trước t0 θ = θ*). Với k 0,05 / t0 100: λ cuối
  ep1-12 = 0,01 0,03 0,06 0,12 0,22 0,38 0,56 0,73 0,85 0,92 0,96 0,98; ep1-2 gần như chỉ head học (giống LP-FT), ep3-11 cả hai số hạng cùng tác dụng
  (lực kéo tối đa lr·γ = 0,28/bước, dưới ngưỡng ổn định 0,5); backbone nhận 61 % ngân sách LR của cột chính.
- **Mã:** `src_mwonly/train.py` trên **161** thêm hai cờ, mặc định giữ hành vi cũ (bản trước ở `_FinalPaperExperiment/state/train.py.before_anchor_0510`;
  158 vẫn bản cũ, không cần). Thử khói CPU hai chiều: cờ mới ⇒ 3 nhóm, nhóm head `anneal_w=0` dịch 1,7e-4 khi backbone dịch 1,2e-6;
  cờ 0 ⇒ 2 nhóm, t0 = 4 như cũ. Chạy khô: lệnh train chỉ khác cột chính ở `--pretrain_cof 5000 --anneal_t0_steps 100 --recadam_free_head 1`
  (+ tên/checkpoint).
- **Tiến độ:** f1 xong 15:48 (rc 0): test ROC 0,9367 (cột chính 0,9430, Δ -0,0063), PR -0,0010, F1@0.5 -0,0062, F1@val +0,0009;
  best ep16 (cột chính ep8), dừng ep24. Val ROC phẳng 0,68-0,71 suốt ep1-7 (neo giữ backbone, chỉ head học), lên từ ep8. hyperparameters
  chỉ khác cột chính ở `pretrain_cof` / `anneal_t0_steps` / `recadam_free_head` (+ đường dẫn; ô cột chính chạy trên paper_mw, A4000).
  f2 xong 16:11: test ROC 0,9309 (Δ -0,0043), PR 0,9112 (Δ -0,0337), F1@0.5 ±0,0000, F1@val -0,0064; best ep17 (cột chính ep11),
  dừng ep25; val ROC phẳng 0,60-0,61 ep1-6, lên từ ep8. Sau 2/5 fold: ROC âm 2/2, PR âm 2/2.
  f3 xong 16:34: test ROC 0,9293 (Δ -0,0050), PR -0,0128, F1@0.5 -0,0068, F1@val -0,0133; best ep18 (cột chính ep7), dừng ep26.
  Sau 3/5 fold (ghép cặp, ngưỡng hoà 1e-3): ROC -0,0052 (0+/3-, min -0,0063 max -0,0043), PR -0,0158 (0+/2-), F1@0.5 -0,0043 (0+/2-),
  F1@val -0,0063 (0+/2-); epoch chọn +8,3 (3/3). ROC âm cả 3 fold nhưng mọi |Δ| ROC dưới sàn nhiễu 0,010.
- **KẾT QUẢ 5/5 fold (17:15; bậc 2, một seed, một máy):** so với cột chính `rasam_common_jsonly`, ghép cặp theo fold, ngưỡng hoà 1e-3:
  ROC **-0,0015** (+1/-4, min -0,0063 max +0,0133), PR -0,0065 (+1/-3, min -0,0337 max +0,0163), F1@0.5 -0,0013 (+1/-3), F1@val -0,0022 (+1/-2);
  epoch chọn +5,6 (muộn hơn ở 4/5). Từng fold ROC: 0,9367 / 0,9309 / 0,9293 / 0,9398 / 0,9255 (cột chính 0,9430 / 0,9352 / 0,9344 / 0,9265 / 0,9309).
  hyperparameters 5/5 ô chỉ khác cột chính đúng 3 cờ neo. Mọi |Δ| ROC trung bình dưới sàn nhiễu 0,010 ⇒ **neo đúng bài gốc KHÔNG đổi độ chính
  xác** so với cấu hình hiện tại (RecAdam như warmup ngầm); chỉ làm mô hình học muộn hơn ~6 epoch. Không ô sập.
  **Chấm dự đoán:** N1 ĐÚNG (muộn hơn ≥ 3 epoch ở 4/5: +8/+6/+11/+5, f4 -2); N2 ĐÚNG (val ROC ep1-2: 0,68 / 0,60 / 0,70 / 0,69 / 0,66);
  N3 ĐÚNG (TB -0,0015 trong [-0,020; +0,005], dương 1/5); N4 ĐÚNG (0/5 sập).
- **Dự đoán (ghi TRƯỚC khi phóng):**
  - N1: epoch được chọn muộn hơn cột chính ≥ 3 epoch ở ≥ 4/5 fold (backbone chỉ học thật từ ep5-8).
  - N2: val ROC ep1-2 (gần như chỉ head trên đặc trưng Pha 1) trong [0,55; 0,80].
  - N3: ROC test so với cột chính TB trong [-0,020; +0,005], dương ≤ 2/5 fold (K8: đóng băng lâu hơn là hại; mất 39 % ngân sách LR).
  - N4: 0/5 ô sập (ROC < 0,75). Rủi ro đã biết: nếu best rơi vào ep1-2 rồi bình nguyên ASAM khi backbone nhả thì patience 8 dừng ở ep10 lúc
    λ mới 0,92 - đọc log từng epoch trước khi đọc số.

### ➕ 05/10 15:0x - XONG 18:30 (158, Pha 1 + 5/5 ô rc 0): NGUỒN MỚI common Java + JS (người dùng: "chạy thêm nguồn java + js common")

Pool `data/mwonly5_sources/jsjava_common` = `merged_undersampled/common.jsonl` (HF 387d9dff) lọc lang {js, java}: Java 5 184 hàm (CleanVul
Java không có CWE nên phần Java của common TRÙNG full), JS 990 hàm common; 6 174 hàm, 50/50 nhãn, chia 15 % seed 42 (train 5 248 / val 926,
test = bản sao val; 762 cặp bị tách giữa train/val). Pool cũ trong MANIFEST không đổi. Chạy như nguồn C/C++: Pha 1 `p1_common_jsjava` rồi cột
chính `rasam_common_jsjava` f1-f5 (biến thể phụ chưa chạy, hỏi sau). Máy: 158 (Pha 1 ~2,5 h rồi Pha 2).
**KẾT QUẢ 5/5 fold (18:30; bậc 2, một seed, một máy 158):** Pha 1 chọn ep7, val ROC nguồn 0,569. Cột chính ROC từng fold 0,842 / 0,951 / 0,895 /
0,913 / 0,908 (TB 0,902), 0/5 sập. Ghép cặp theo fold (ngưỡng hoà 1e-3):
- so với **chỉ JS** (`rasam_common_jsonly`): ROC **-0,032 (+1/-4**, -0,101..+0,016), PR -0,059 (+1/-4), F1@0.5 -0,025 (+2/-3), F1@val -0,030 (+2/-3);
- so với **baseline**: ROC +0,005 (+3/-2, dưới sàn nhiễu), PR -0,023 (+2/-2), F1@0.5 **+0,043 (5/5)**, F1@val **+0,040 (5/5)**;
- so với MW không Pha 1 AdamW: ROC +0,001 (+2/-2), F1@0.5 +0,032 (+4/-1);
- so với JS + C/C++ common: ROC -0,015 (+1/-3), F1@0.5 +0,016 (+4/-1).
Đọc: thêm Java vào Pha 1 KÉO XUỐNG so với chỉ JS ở 4/5 fold (như thêm C/C++); so với baseline chỉ ăn ở NGƯỠNG (F1 5/5), thứ hạng (ROC) ngang.
hyperparameters Pha 2 trùng cột chính chỉ JS (chỉ khác checkpoint Pha 1). **Chấm:** J1 SAI phần ROC (cổng mở ep3 đúng, nhưng val 0,569 < 0,80);
J2 ĐÚNG (0/5 sập); J3 ĐÚNG (TB 0,902 < 0,934, âm 4/5); J4 SAI (ROC dương 3/5, cần ≥ 4/5).
**Dự đoán (ghi TRƯỚC):** (J1) Pha 1 mở cổng chọn theo val ≤ ep3, val ROC tốt nhất ≥ 0,80. (J2) cột chính 0/5 sập. (J3) TB ROC dưới
common chỉ JS (0,934) - common_jsjava - common_jsonly ROC < 0 ở ≥ 3/5 fold (thêm ngôn ngữ khác như C/C++ đã kéo xuống). (J4) so baseline
ROC > 0 ở ≥ 4/5 fold. Vùng không kết luận |Δ| < 0,010.

### 📌 NHẬN ĐỊNH 05/10 14:41 - Đóng góp của RecAdam trong Pha 2 (agent; tab "Nhận định" có 6 hình + bảng cho paper)

Toàn văn + hình (SVG/PDF, chữ tiếng Anh) + bảng (CSV/LaTeX/MD): `_FinalPaperExperiment/meta/insights/recadam/`; tạo lại bằng
`make_recadam_figures.py`, bản web bằng `refresh_web.sh`. Mức tin: trung bình (bậc 2, một seed, 6 nguồn chung fold đích).
- Độ chính xác: RecAdam KHÔNG tăng - bỏ 2 ô sập, cột chính - chỉ ASAM ROC +0,001 [-0,008; +0,011] (14/28 cặp), F1@val +0,009;
  chỉ RecAdam - noRAS ROC -0,005 (12/30); trên JS + C/C++ không ASAM RecAdam -0,012 (0/3 nguồn). Phần hơn của cột chính trên chỉ JS
  (+0,020 ROC 5/5 fold) là của ASAM.
- Ổn định: 0/30 sập (RecAdam + ASAM) vs 2/30 (chỉ ASAM) - hai ô đó tạo toàn bộ hiệu trung bình +0,025 và tương tác; ngoài chúng độ
  tản theo fold không giảm, kẹt > 10 epoch 6 vs 7. Ô chạy lại common_jscpp f3 kẹt cả 30 epoch (0,523) trong khi cột chính cùng fold
  0,916 ⇒ bằng chứng một ô rằng RecAdam có cứu thật (không do dừng sớm).
- Động học (chắc nhất): RecAdam = warmup ngầm, chậm ~1 epoch tới val ROC 0,85 (+1,07; 23/30 chậm hơn, 0 nhanh hơn).
- Chống quên: không đo được (checkpoint Pha 2 đã xoá); tín hiệu gián tiếp mơ hồ. Văn liệu: Zheng et al. ACL 2023 (RecAdam std cao hơn
  6/7 tập), Xu et al. EMNLP 2021 (+0,75 điểm).
- Đề xuất cho bài: RecAdam vào ablation 2×2 như quan sát ổn định một seed, không gọi đóng góp chính / "chống quên"; hoặc bỏ RecAdam
  và sửa bình nguyên ASAM bằng ASAM muộn / ρ nhỏ. Phép kiểm rẻ nhất (CHƯA chạy): chấm zero-shot 6 checkpoint Pha 1 trên test SVEN.

### 🔁 05/10 14:0x CHẠY LẠI 3 Ô SẬP với `--min_epochs 30` (người dùng: "thử luôn bằng cách kiểm --min_epochs 30 với 3 epoch sập")

Run mới (không ghi đè ô gốc): `nop1_asamonly_me30` f4 + `asamonly_4cwe_jscpp_me30` f1 trên 161, `asamonly_common_jscpp_me30` f3 trên 158.
Cùng mọi cờ của ô gốc, chỉ khác `--min_epochs 30` (dừng sớm không thể xảy ra trước epoch 30; lịch LR không đổi vì `--epochs` vẫn 30).
**Dự đoán (ghi TRƯỚC):** (M1) 17 epoch đầu TRÙNG BIT với ô gốc (tất định, cùng loại GPU) - lệch là lỗi tái lập, phải dừng đọc kết quả.
(M2) ≥ 2/3 ô rời bình nguyên (train loss < 0,6) trước epoch 30 - các ô chỉ ASAM khác thoát muộn tới ep 16-21. (M3) ô nào thoát thì
test ROC cách cột chính cùng nguồn/fold ≤ 0,03. Nếu M2 sai (kẹt vĩnh viễn) thì "sập" là thật chứ không phải do dừng sớm cắt.
**Kết quả (từng ô, 14:32):** 17 epoch đầu trùng ô gốc ở cả 2 ô đã xong (so 4 chữ số log; 158 còn trùng test ROC 16 chữ số) ⇒ M1 ĐÚNG.
- `nop1_asamonly_me30` f4 (161): **THOÁT** - val ROC ep18 0,567 (> best cũ 0,546: ô gốc dừng ở ep17, thiếu đúng 1 epoch), ep18-24
  0,57 → 0,85, train loss 0,70 → 0,59 ở ep24; chọn ep28; test ROC **0,929** (gốc 0,464; nop1_adamw cùng fold 0,927), F1@0.5 0,841,
  F1@val 0,841, PR 0,928 ⇒ bình nguyên TẠM THỜI, "sập" do dừng sớm cắt.
- `asamonly_common_jscpp_me30` f3 (158): **KẸT VĨNH VIỄN** - train loss 0,69-0,72 suốt 30 epoch, val ROC 0,39-0,46 từ ep10, best
  vẫn ep9; test ROC 0,5231 (trùng ô gốc tới 16 chữ số) ⇒ sập thật (lịch LR giảm tuyến tính về 0 càng khó thoát về cuối).
- `asamonly_4cwe_jscpp_me30` f1 (161, xong 14:57): **KẸT VĨNH VIỄN** - train loss ở ln2 suốt 30 epoch, best vẫn ep1, test ROC 0,607
  (trùng ô gốc tới 16 chữ số).
**Chấm:** (M1) ĐÚNG - 3/3 ô trùng ô gốc tới ep17 (và 2 ô kẹt trùng cả test ROC). (M2) **SAI** - chỉ 1/3 ô thoát (nop1 f4, thiếu đúng
1 epoch); 2/3 ô có Pha 1 JS + C/C++ kẹt suốt 30 epoch ⇒ "sập" của chỉ ASAM trên JS + C/C++ là THẬT, không do dừng sớm. (M3) ĐÚNG
cho ô thoát (0,929 vs AdamW 0,927; không có cột chính cho nop1).

### 📌 NHẬN ĐỊNH 05/10 12:57 - ASAM có gây sập Pha 2 không, RecAdam có đỡ không (agent phân tích, tab "Nhận định" trên artifact)

Toàn văn: `_FinalPaperExperiment/meta/insights/asam_collapse.{md,json}` (+ script và số trung gian cùng thư mục). Mức tin: trung bình
(bậc 2, một seed). ĐỨNG: ASAM ρ 0,5 bật từ bước đầu gây bình nguyên ln2 - ghép cặp thêm ASAM làm bình nguyên dài hơn 33/35 cặp
(không RecAdam) và 24/30 (có RecAdam), 0 cặp ngắn hơn; lặp ở khối cũ 49/49, 64/64; can thiệp ASAM muộn (K15) xoá bình nguyên 15/15.
"Sập" = bình nguyên dài bị dừng sớm cắt (0/165 ô học rồi sụp). KHÔNG ĐỨNG: "RecAdam giúp" - cột chính vs chỉ ASAM ngắn hơn 8 / dài
hơn 9 / bằng 13; kẹt > 10 epoch 14/45 vs 10/35; 0/45 vs 3/35 sập chỉ ở đuôi (p ≈ 0,08). Warmup ngầm λ(t) có thật (λ = 0,46 ở bước 1,
≥ 0,9 từ epoch 4) nhưng không rút ngắn bình nguyên ASAM. SE/AI4SE: 17 bài đã kiểm, chưa thấy bài dùng SAM/ASAM/RecAdam cho phát hiện
lỗ hổng (cần dò toàn văn ACM DL / IEEE Xplore trước khi gọi là điểm mới). 6 đề xuất phép kiểm - CHƯA chạy, cần duyệt.

### ❓ CHỜ NGƯỜI DÙNG QUYẾT (05/10 11:5x)

1. **[XONG 13:1x - ĐÃ DESTROY 54059221 (paper_mw) + 54113977 (paper_mw2), `vastai show instances` = 0]** Trước khi huỷ: 0 tiến trình,
   0 file results/logs còn trên máy; 6 `best_train_loss.pt` kéo về md5 khớp 6/6 (11 `best.pt` đã khớp từ trước); `state/` khớp tên + byte
   (19 + 17 file) ở `state/remote_logs/<máy>/state_full/`. `artifact_tick.sh` + `live_all.sh` bỏ hai máy này. Người dùng 12:4x: "vast xong
   kéo kết quả, checkpoint và hủy giúp tôi". Kéo 6 `best_train_loss.pt` còn thiếu (11 `best.pt`
   đã có trên 161, md5 khớp) + toàn bộ `state/` về `state/remote_logs/<máy>/state_full/`, đối chiếu md5, rồi destroy. Ghi chú gốc:
   **Hai máy vast `paper_mw` (54059221) / `paper_mw2` (54113977)** - nhãn KHÔNG phải `ntat` nên KHÔNG tự huỷ theo VAST_RULES (quy tắc
   "hết việc ⇒ huỷ" chỉ cho máy của mình). paper_mw2 rảnh từ 11:50 (~$0,09/h); paper_mw xong noRAS 4cwe_jscpp ~12:10. Đã sẵn sàng huỷ:
   kết quả + log fold đã chuyển hết (0 file còn trên paper_mw2), log driver/queue kéo về `state/remote_logs/paper_mw2/` khớp 7/7 tên +
   byte; 3 checkpoint Pha 1 chỉ có trên vast (`p1_common_ccpp` paper_mw2, `p1_4cwe_ccpp` + `p1_4cwe_jscpp` paper_mw) **đã kéo về
   `model/mwonly5/` trên 161 lúc 12:04, md5 khớp nguồn 3/3** ⇒ 161 có đủ checkpoint Pha 1 của cả 9 nguồn; huỷ vast không mất gì.
2. **Việc còn hoãn (chưa được nêu rõ):** noRAS 3 nguồn chỉ C/C++ (15 ô), chỉ ASAM / chỉ RecAdam 3 nguồn chỉ C/C++ (30 ô).
3. **3 ô sập thật, đều là biến thể chỉ ASAM:** `nop1_asamonly` f4 (0,464), `asamonly_4cwe_jscpp` f1 (0,607), `asamonly_common_jscpp` f3
   (0,523) - giữ nguyên / chạy lại patience lớn hơn / khác.

### ⚠ CẢNH BÁO (sập / nghi sập / lỗi - mới nhất ở trên; chạy tiếp, không dừng chờ trả lời)

_Quy ước 05/10: có diễn biến sau cảnh báo (thoát muộn / báo nhầm / sập thật) thì ghi thêm ngay dưới dòng cảnh báo, kèm kết cục; báo nhầm hiện màu vàng trên artifact._

- **07/10 11:00 - NGHI KẸT `p1_jspy41jsval_full_s1234` f5 (paper_night, tab Kết quả paper):** train loss ở ln2 (0,705-0,764) 8 epoch, val JS 0,46-0,54,
  checkpoint đang giữ ep6 (chưa học). Fold 5 full từng kẹt hẳn (p1_jspy31_full f5). Đã ghi artifact (`a202610071100_p1_jspy41jsval_full_s1234_f5`); chạy tiếp.
  → 11:14 KẾT CỤC: KẸT THẬT - hết 16 epoch train loss 0,70-0,72, checkpoint ep14 (val JS 0,543, chưa học); Pha 2 chạy tiếp từ checkpoint này.
  → 11:30 Pha 2 `rasam_jspy41jsval_full_s1234` f5: **SẬP** - train loss vẫn ở ln2, dừng sớm ep17 với checkpoint ep1, test ROC 0,570 (seed 42 cùng
  fold 0,933). Cả hai pha kẹt. Giữ ô trong n = 15, CHỜ người dùng quyết định (giữ / chạy lại / báo cáo kèm); artifact đã cập nhật (v4).

- **07/10 10:15 - SẬP `baseline_s7` f1 (158, tab Kết quả paper):** train loss ở ln2 (0,690-0,762) 9 epoch, dừng ep10, checkpoint ep1; test ROC 0,502. Baseline
  seed 42 / 1234 cùng fold 0,860 / 0,891. Δ (cấu hình - baseline) ở ô (seed 7, f1) bị thổi phồng ~0,4. Đã ghi artifact (`a202610071015_baseline_s7_f1`) +
  push; chạy tiếp, CHỜ người dùng quyết định (giữ / chạy lại / báo cáo kèm).

- **07/10 08:48 - NGHI KẸT `p1_jspy41jsval_common_s1234` f4 (161, tab Kết quả paper):** train loss ở ln2 (0,706-0,762) ep1-7, ep8 mới 0,6825 (lượt khác
  thoát ep4-6); checkpoint đang giữ ep5 (train loss 0,734, gần như chưa học, val JS 0,522). Đã ghi artifact (`a202610070848_p1_jspy41jsval_common_s1234_f4`);
  chạy tiếp.
  → 08:57 KẾT CỤC: BÁO NHẦM - thoát muộn từ ep9, chọn ep11 (train loss 0,478, val JS 0,609), đã học bình thường.

- **07/10 06:45 - NGHI SẬP `rasam_jspy41jsval_4cwe_s7` f2 (161, tab Kết quả paper):** Pha 2 ở ln2 suốt 12 epoch (train loss 0,70-0,74, val ROC 0,51-0,61),
  ep13 bắt đầu thoát (0,679 / 0,736). Nguyên nhân: Pha 1 `p1_jspy41jsval_4cwe_s7` f2 chọn ep2 (train loss 0,725, gần như chưa học) vì val CHỈ JS 88 hàm cho
  ROC cao nhất ở ep2 (0,572) - ô DUY NHẤT trong 28 Pha 1 val chỉ JS như vậy kèm Pha 2 chậm. Đã ghi artifact (`a202610070645_rasam_jspy41jsval_4cwe_s7_f2`)
  + push; chạy tiếp, không chạy lại khi chưa có quyết định.
  → 06:55 KẾT CỤC: thoát từ ep13, test ROC 0,916 - không sập nhưng thấp hơn seed 42 (0,951) / 1234 (0,938) cùng fold; ghi "báo nhầm" (vàng) kèm ghi chú.

- **06/10 19:49 - NGHI KẸT `p1_jspy31_full` f5 (158, Pha 1 trộn JS:Py 3:1 full):** train loss ở ln2 (0,703-0,732) suốt 8 epoch, val ROC 0,48-0,54
  (best 0,540 ep8), patience 0/8, trần 16 ⇒ còn tối đa 8 epoch. 15 lượt Pha 1 trộn khác (4:1 / 3:1 × common / full) đều thoát ở ep4-5. Cùng fold 5
  với ô kẹt `mix1p_adamw_common_jsonly` f5 (05/10, cũng trộn JS + SVEN train). Không thoát thì Pha 2 `rasam_jspy31_full` f5 vẫn chạy và ghi kèm val
  Pha 1. Đã ghi artifact (doc `a202610061949_p1_jspy31_full_f5`); chạy tiếp.
  → 20:06 KẾT CỤC: KẸT THẬT - hết trần 16 epoch, train loss 0,70-0,71 (min 0,7015), val ROC ep9-16 0,47-0,50, checkpoint ep8 (val 0,540).
  Pha 2 `rasam_jspy31_full` f5 chạy tiếp từ checkpoint này. Không chạy lại khi chưa có quyết định của người dùng.

- **05/10 20:42 - NGHI KẸT `mix1p_adamw_common_jsonly` f5 (161, đích SVEN):** train loss ở ln2 (0,70-0,73) suốt 12 epoch với AdamW THUẦN (không ASAM),
  val ROC 0,41-0,58, best ep1 (0,579), patience 3/8 ⇒ dừng ep17 với checkpoint ep1 nếu không vượt. Cùng fold, `nop1_adamw` (chỉ SVEN) học bình
  thường (ep4 train loss 0,35, val 0,90) - khác duy nhất là 990 hàm JS trong train. Đã ghi artifact; chạy tiếp.
  → 20:49 KẾT CỤC: SẬP THẬT - ln2 suốt 17 epoch, dừng ep17, checkpoint ep1. Test ROC 0,486, PR 0,543, F1@0.5 0,410 (`nop1_adamw` f5 0,909).
  Ô sập đầu tiên với AdamW thuần trên đích SVEN trong khối. Không chạy lại khi chưa có quyết định của người dùng.
  → 23:00 **NHẬN ĐỊNH** (tab "Nhận định" + `_FinalPaperExperiment/meta/insights/mix_f5/mix_f5_collapse.md`): bình nguyên ln2 do điều kiện khởi đầu
  (head ngẫu nhiên + CodeBERT gốc + cặp JS gần trùng ngược nhãn: trung vị khác 5 token, độ giống 0,955, 53 % cặp ≥ 0,95; JS = 68 % train), không do
  ASAM sinh ra. 41 ô AdamW đích SVEN: bình nguyên > 5 epoch CHỈ ở ô trộn (f3 6, f4 6, f5 > 17); AdamW trên JS thuần kẹt 6-7, baseline đích JS f5 > 15;
  Pha 1 xoá bình nguyên (2 pha AdamW: 0 ở mọi fold). ASAM vẫn KÉO DÀI bình nguyên (2×2: 28/0/2) nhưng không phải điều kiện cần. Chạy lại min_epochs 30
  đến ep22 vẫn kẹt (17 epoch đầu trùng bit). Đề xuất (cần duyệt): Q1 seed 1234/7 cho f5; Q2 warmup 0,10; Q3 chặn tỉ lệ JS 1:1.
  → 23:03 CHẠY LẠI min_epochs 30 XONG: kẹt ĐỦ 30 epoch (train loss thấp nhất 0,696 ở ep30, val ROC max 0,579 ở ep1), checkpoint ep1, test ROC 0,48559
  trùng bit ô gốc ⇒ bình nguyên VĨNH VIỄN trong ngân sách LR của lịch này. Dự đoán R2 SAI.
- **05/10 19:21 - SẬP THẬT `rasam_sven_python_tojsfull` f4 (158, đích JS):** train loss ở ln2 suốt 17 epoch (0,761 → 0,673), val ROC 0,48-0,51,
  best ep1 (0,514) ⇒ dừng sớm ep17 với checkpoint ep1. Test ROC 0,468, PR 0,462, F1@0.5 0,338 - dưới ngẫu nhiên. Cùng bình nguyên với f1-f3
  (f1, f2 thoát ep11-12; f3 đến ep13 chưa thoát). Đã ghi artifact; chạy tiếp f5; không chạy lại khi chưa có quyết định của người dùng.
  → 23:11 CHẠY LẠI min_epochs 30 (người dùng duyệt 21:0x, 158): bình nguyên 17 epoch rồi THOÁT ở ep18 (ô gốc dừng đúng ep17, thiếu một epoch),
  train loss 0,67 → 0,34, best ep21; test ROC 0,557, PR 0,576, F1@0.5 0,531 (gốc 0,468; baseline JS cùng fold 0,528) ⇒ sập do luật dừng sớm cắt
  bình nguyên TẠM THỜI (khác ô trộn f5: vĩnh viễn). Dự đoán R1 ĐÚNG (thoát trước ep30, best ep > 10, ROC ≥ 0,55).
- ~~**05/10 17:38 - NGHI KẸT `rasam_sven_python_tojsfull` f1 (paper_mw3, đích JS):**~~ (ĐÃ XEM 18:10 → `ALERT_HISTORY.md`, BÁO NHẦM) train loss ở ln2 (0,724 → 0,693) suốt 8 epoch, val ROC tụt
  0,500 → 0,458, best ep1, patience 7/8 ⇒ dừng ep10 với checkpoint ep1 nếu không vượt 0,500. Bình nguyên ASAM như các ô sập trên đích SVEN.
  Đã ghi artifact (alerts); chạy tiếp.
  → 17:45 SỬA LẠI: patience chỉ đếm từ ep10 (min_epochs; log ep8-9 "patience 0/8"), nên KHÔNG dừng ở ep10 - sớm nhất ep18. Train loss đã rời
  ln2 (ep9 0,671, ep10 0,680, ep11 0,650), val ROC nhích 0,451 → 0,489; có thể đang thoát bình nguyên. Theo dõi tiếp.
  → 18:02 KẾT CỤC: BÁO NHẦM - đã thoát bình nguyên, dừng sớm ep20, checkpoint ep12 (val 0,533). Test ROC 0,583, PR 0,617, F1@0.5 0,570,
  F1@val 0,539 - ngang Pha 1 JS full trong miền (val 0,59) ⇒ mức của đích JS, không phải sập (J2 [0,50; 0,68] đúng ở f1).
- ~~**05/10 10:43 - NGHI SẬP `asamonly_common_jscpp` f5 (161):**~~ (ĐÃ XEM 13:36 → `ALERT_HISTORY.md`, BÁO NHẦM) train loss ở ln2 suốt 12 epoch, nhưng val ROC nhích dần 0,53 → 0,58
  (best 0,583 ở ep11), patience 1/8 - khác hai ô sập thật (val tụt). Đã ghi artifact; chạy tiếp.
  → 10:51 KẾT CỤC: BÁO NHẦM - đã thoát, ep20 train loss 0,322, val ROC 0,850; chờ test.
- ~~**05/10 10:10 - NGHI SẬP `asamonly_common_jscpp` f4 (161):**~~ (ĐÃ XEM 13:36 → `ALERT_HISTORY.md`, BÁO NHẦM) train loss ở ln2 suốt 12 epoch, val ROC đi ngang 0,50-0,52, best 0,526 ở
  ep2, patience 3/8 ⇒ dừng ep17 nếu val không vượt 0,526. Giống hai ô chỉ ASAM vừa sập thật. Đã ghi artifact; chạy tiếp.
  → 10:18 KẾT CỤC: BÁO NHẦM - val nhích ep13-14 (0,533 / 0,539 > 0,526) đặt lại patience, rồi thoát ep16 (0,71 → 0,86 ở ep22).
  Sống sót nhờ hai lần nhích nhỏ đó. Giả thuyết (chưa kiểm): ở chỉ ASAM, "thoát muộn" hay "sập" phụ thuộc nhiều vào patience/thời điểm
  val nhích; phép kiểm rẻ là chạy lại một ô sập với patience lớn hơn - cần người dùng duyệt.
- ~~**05/10 09:57 - SẬP THẬT `asamonly_common_jscpp` f3 (158):**~~ (ĐÃ XEM 14:01 → `ALERT_HISTORY.md`, SẬP THẬT) train loss ở ln2 suốt quá trình, best val ROC 0,521 ở ep9 rồi tụt về
  0,40-0,43, dừng sớm ep17. **Test ROC 0,523**, F1@0.5 0,444, PR-AUC 0,509 (cột chính cùng fold 0,916). Ô sập thứ hai của chỉ ASAM trên
  JS + C/C++ ⇒ **dự đoán T1 (≤ 1/15 sập) SAI**. Đã ghi artifact ("sập thật"). Không chạy lại khi chưa có quyết định của người dùng.
- ~~**05/10 09:31 - NGHI SẬP `asamonly_full_jscpp` f5 (161):**~~ (ĐÃ XEM 13:36 → `ALERT_HISTORY.md`, BÁO NHẦM) train loss ở ln2 suốt 13 epoch (min 0,685), nhưng val ROC tăng liên tục
  ep10-13 (0,63 → 0,77), val loss giảm, patience 0/8. → KẾT CỤC: BÁO NHẦM (đang thoát lúc cảnh báo). Không ping. 09:46 xong: đủ 30
  epoch, chọn ep27, test ROC 0,852 (cột chính cùng fold 0,886) - không sập nhưng thấp.
- ~~**05/10 09:22 - SẬP THẬT `asamonly_4cwe_jscpp` f1 (paper_mw):**~~ (ĐÃ XEM 14:01 → `ALERT_HISTORY.md`, SẬP THẬT) train loss ở ln2 suốt quá trình, best val ROC 0,626 chỉ ở ep1 rồi tụt
  về 0,46-0,48, dừng sớm giữ checkpoint ep1. **Test ROC 0,607**, F1@0.5 0,438, PR-AUC 0,639 (cột chính cùng fold 0,918). Khác hai ô
  08:5x (val vẫn nhích rồi thoát). Đã ghi artifact ("sập thật"). Ô giữ trong bảng, không chạy lại khi chưa có quyết định của người dùng.
- ~~**05/10 08:53 - NGHI SẬP `asamonly_full_jscpp` f4 (161):**~~ (ĐÃ XEM 13:36 → `ALERT_HISTORY.md`, BÁO NHẦM) train loss ở ln2 suốt 12 epoch, val ROC 0,48-0,54; ep12 đạt 0,543 nên
  patience đặt lại 0/8. Kiểu dự đoán T1. Đã ghi artifact; chạy tiếp.
  → 08:59 KẾT CỤC: BÁO NHẦM - đã thoát, ep19 train loss 0,328, val ROC 0,889; chờ test.
- ~~**05/10 08:50 - NGHI SẬP `asamonly_common_jscpp` f1 (158):**~~ (ĐÃ XEM 13:36 → `ALERT_HISTORY.md`, BÁO NHẦM) train loss ở ln2 suốt 13 epoch, best val ROC 0,580 ở ep2, patience 4/8 ⇒
  dừng ep17 nếu val không vượt 0,580 (đang nhích 0,52 → 0,56). Đúng kiểu dự đoán T1. Đã ghi artifact; chạy tiếp.
  → 08:59 KẾT CỤC: BÁO NHẦM - đã thoát, ep23 train loss 0,157, val ROC 0,932; chờ test.
- ~~**05/10 06:59 - NGHI SẬP `rasam_common_ccpp` f4 (paper_mw2):**~~ (ĐÃ XEM 13:36 → `ALERT_HISTORY.md`, BÁO NHẦM) train loss ở ln2 suốt 12 epoch, val ROC 0,43-0,53; ep11-12 nhích 0,485 →
  0,527 nên patience đặt lại 0/8. Giống f1 cùng nguồn (thoát từ ep10, test 0,914). Đã ghi artifact; chạy tiếp.
  → 07:14 KẾT CỤC: BÁO NHẦM - val ROC nhích dần ep13-16 (0,54 → 0,69), thoát ep17-19 (0,74 → 0,90); chọn ep22, test ROC 0,946, F1@0.5 0,862.
- ~~**05/10 05:59 - NGHI SẬP `rasam_common_ccpp` f1 (paper_mw2):**~~ (ĐÃ XEM 13:36 → `ALERT_HISTORY.md`, BÁO NHẦM) train loss ở ln2 ep2-11, val ROC 0,45-0,50 ep1-9; ep10-12 val ROC
  0,55 → 0,61 → 0,65, val loss giảm, patience 0/8 - đang có dấu hiệu thoát. Đã ghi artifact; chạy tiếp.
  → 06:09 KẾT CỤC: BÁO NHẦM - chọn ep16 (val 0,938), test ROC 0,914, F1@0.5 0,838. (ROC trùng 16 chữ số với rasam_4cwe_ccpp f1 = 74/81:
  đã đối chiếu F1 / PR / epoch / ngưỡng đều khác ⇒ hai mô hình khác nhau, ROC trên tập test cố định là giá trị rời rạc.)
- ~~**05/10 04:58 - NGHI SẬP `rasam_full_jscpp` f5 (161) - CỘT CHÍNH:**~~ (ĐÃ XEM 13:36 → `ALERT_HISTORY.md`, BÁO NHẦM) train loss ở ln2 (0,69-0,78) suốt 12 epoch, val ROC 0,50-0,56,
  best 0,558 ở ep5, patience 3/8 ⇒ dừng ~ep17 nếu không thoát. f1-f4 cùng nguồn không sập (0,899 / 0,921 / 0,916 / 0,946). Đã ghi
  artifact; chạy tiếp. 04:59 ep13 luật train loss cũng báo (min 0,691).
  → 05:11 KẾT CỤC: BÁO NHẦM - thoát ở ep14-16 khi patience mới 4/8 (val ROC 0,61 → 0,74 → 0,82); best 0,908 ở ep24; chờ test.
- ~~**05/10 03:35 - NGHI SẬP `nop1_asamonly` f5 (paper_mw):**~~ (ĐÃ XEM 13:36 → `ALERT_HISTORY.md`, BÁO NHẦM) train loss ở ln2 (0,69-0,75) suốt 13 epoch (min 0,691); khác f4, val ROC vẫn
  nhích 0,50 → 0,73 (ep12) nên patience liên tục đặt lại. Đã ghi artifact; chạy tiếp.
  → 03:36 KẾT CỤC: BÁO NHẦM - thoát ở ep14 (train loss 0,637, val ROC 0,746). Thoát muộn giống f3. 03:50 xong: đủ 30 epoch, chọn ep26,
  test ROC 0,875 (nop1_adamw cùng fold 0,909).
- ~~**05/10 03:13 - SẬP THẬT `nop1_asamonly` f4 (paper_mw):**~~ (ĐÃ XEM 14:01 → `ALERT_HISTORY.md`, SẬP THẬT) train loss ở ln2 (0,70-0,74) suốt 17 epoch, val ROC 0,48-0,55, best 0,546 ở ep8;
  dừng sớm ep17. **Test ROC 0,464**, F1@0.5 0,374, PR-AUC 0,409. Khác f3 cùng cấu hình (thoát ep21, test 0,915): f4 hết patience trước
  khi kịp thoát. Đã ghi artifact (kết cục "sập thật"). Ô vẫn giữ trong bảng, không chạy lại khi chưa có quyết định của người dùng.
- ~~**05/10 02:45 - NGHI SẬP `nop1_asamonly` f3 (paper_mw):**~~ (ĐÃ XEM 03:00 → `ALERT_HISTORY.md`, BÁO NHẦM) train loss ở ln2 (0,70-0,73) suốt 13 epoch, val ROC 0,37-0,45, chọn ep2 (0,455);
  min 10 + patience 8 ⇒ dừng ~ep17-18 nếu không thoát. f1 cùng cấu hình thoát ep9-10. Đã ghi artifact; chạy tiếp. Kết cục: chờ kết quả.
  → 02:53 KẾT CỤC: BÁO NHẦM nhưng thoát RẤT MUỘN - đứng tới ep20, học ep21-25, chọn ep27, chạy đủ 30; test ROC 0,915 (vs baseline +0,025,
  vs nop1_adamw +0,037). Sống sót nhờ val ROC nhích nhẹ ở ep15, ep18 làm đặt lại patience - không thì đã dừng ep17 với checkpoint ep2.
- ~~**05/10 02:40 - NGHI SẬP `rasam_common_jscpp` f5 (158):**~~ (ĐÃ XEM 02:49 → `ALERT_HISTORY.md`, BÁO NHẦM) train loss ở ln2 (0,68-0,75) ep2-11, val ROC 0,52-0,55; ep12-13 loss 0,647 / 0,635
  nhưng val ROC vẫn 0,56. Giống f2 cùng nguồn (báo nhầm, thoát ep13, test ROC 0,911). Đã ghi artifact; chạy tiếp. Kết cục: chờ kết quả.
  → 02:48 KẾT CỤC: BÁO NHẦM (thoát muộn) - học tới ep24, test ROC 0,908 / F1@0.5 0,816, hơn baseline cùng fold cả 4 chỉ số (ROC +0,028).
- ~~**05/10 01:40 - NGHI SẬP `rasam_common_jscpp` f2 (158):**~~ (ĐÃ XEM 01:53 → `ALERT_HISTORY.md`, BÁO NHẦM) train loss ở bình nguyên ln2 (0,69-0,73) ep2-12, val ROC 0,46-0,54;
  ep13 mới nhích (0,640, val ROC 0,608). Cùng kiểu sập RA + ASAM tab cũ. Fold 1 cùng nguồn thoát ep6-7 (ROC 0,893). Đã push + ghi artifact; chạy tiếp.
  → 01:49 KẾT CỤC: BÁO NHẦM (thoát muộn) - học tới ep20, dừng ep28, test ROC 0,911 / F1@0.5 0,802, ngang baseline cùng fold (ROC +0,001).
- ~~**04/10 22:03 - KẸT Pha 1 `p1_4cwe_jsonly` (paper_mw2, batch 32):**~~ (ĐÃ XEM 22:17; xử lý: lr căn bậc hai + chọn checkpoint 2 giai đoạn, chạy lại 22:17. 05/10 01:5x người dùng: không còn liên quan config mới ⇒ ĐÃ XOÁ khỏi artifact và ALERT_HISTORY.md) ep8 train loss chưa từng < 0,68 (thấp nhất 0,6951 ≈ ln2), chọn ep1
  (val ROC 0,450); batch 4 cùng nguồn xuống 0,674 ở ep7. Kèm: `baseline` f1 batch 32 ROC 0,8588 / PR 0,8895 / F1@0.5 0,7415 / F1@val 0,7393
  so với batch 8 0,8964 / 0,9148 / 0,7958 / 0,8146 (n=1). Nghi lr 2e-5 giữ nguyên khi số bước giảm 4-8 lần. Đã push + ghi artifact; chạy tiếp,
  chờ người dùng quyết (tăng lr theo batch hay giảm batch).
  22:06 cập nhật: thoát bình nguyên MUỘN ở ep10-11 (0,688 → 0,637); checkpoint chọn vẫn ep1 (val ROC 0,450 > 0,444 ở ep11).

Người dùng 04/10 20:4x: dùng mã nhánh `refactor` của maytinhdibo/GraphTransferVD, archive toàn bộ kết quả `_FinalPaperExperiment`, chạy lại thí nghiệm
không đồ thị, artifact mới điền sẵn ô, n=5, seed 42, TF32, cố định seed tái lập được (lấy từ mã latent bottleneck cũ); Pha 1 warmup 0,25, lr 2e-5,
16 epoch, patience 8, chọn val ROC, min_epochs 10; 6 nguồn {full, common, 4CWE} × {chỉ JS, JS + C/C++}; chạy lại baseline; chính = RecAdam + ASAM
(bản thường nhất), phụ = noRAS / chỉ ASAM / chỉ RecAdam; dùng 4 máy, dừng mọi việc cũ. Trả lời 20:5x: Pha 2 30 epoch, patience 8, min 10; chính =
"cái bình thường nhất, không quan tâm muộn sớm"; baseline = CodeBERT 512 thuần, không warmup, không decay, chỉ khác TF32; archive vào thư mục con.
Dặn thêm: ô sập thì ping (push + dải cảnh báo ghim trên artifact + mục cảnh báo ở trên), không dừng chờ; xoá checkpoint cũ trên vast (đã xoá
20:5x, 8 P1 trùng md5 với 161 + 10 checkpoint Pha 2/kiểm); được dùng nhiều agent.

- **22:17 PHÓNG LẠI (lượt hiện hành) - batch 32 + lr theo căn bậc hai + chọn checkpoint Pha 1 hai giai đoạn** (người dùng 22:1x: "tăng lr theo batch,
  chạy lại toàn bộ"; "trong khi pha 1 chưa giảm train loss > 2% thì lấy check point theo train loss, sau đó mới lấy theo val ROC"; chọn quy tắc
  căn bậc hai và mốc = train loss epoch 1). MW lr 2e-5 × √(32/4) = 5,66e-5 (cả hai pha); baseline 2e-5 × √(32/8) = 4e-5. Pha 1 `--select_after_drop
  0.02` (src_mwonly/train.py, cờ mới; kiểm GPU 3 chiều: cờ 0 trùng bit mã cũ, không giảm đủ thì chọn theo train loss, giảm đủ thì chuyển val ROC).
  Lý do: lượt batch 32 lr 2e-5 (21:56-22:10) học chậm - baseline f1 ROC 0,8588 vs 0,8964 ở batch 8; p1_4cwe_jsonly kẹt ln2 tới ep9, thoát ep10-11.
  Lượt đó ở `archive_20261004/aborted_batch32_2210/`. Hàng đợi, skip.txt, chia máy giữ nguyên.
- **05/10 03:1x XẾP chỉ ASAM + chỉ RecAdam trên 3 nguồn CHỈ JS (người dùng: "nhánh chỉ recadam và asam ưu tiên thực hiện trên js trước"):** 30 ô
  `asamonly_*_jsonly`, `raonly_*_jsonly` trên 161, queue_after (PID 556970) chờ driver 221090 (Pha 1 + cột chính full JS + C/C++, xong ~05:00);
  `state/q161_side_js.txt` fold là vòng ngoài; checkpoint Pha 1 chỉ JS chép paper_mw2 -> 161 (md5 trùng); bỏ 6 dòng khỏi skip.txt của 161
  (noRAS vẫn hoãn). **Dự đoán:** (S1) chỉ ASAM có ≥ 1/15 ô đứng ở ln2 > 10 epoch (đã thấy ở mọi biến thể có ASAM); (S2) chỉ RecAdam 0/15 ô
  đứng; (S3) cột chính ≥ chỉ RecAdam cùng fold ở ≥ 3/5 fold ROC cho ≥ 2/3 nguồn JS; (S4) cột chính ≥ chỉ ASAM ở ≥ 3/5 fold cho ≥ 2/3 nguồn.
- **05/10 03:02 cột chính common chỉ JS XONG 5/5 (paper_mw2):** ROC 0,943 / 0,935 / 0,934 / 0,927 / 0,931, TB 0,934 (cao nhất 4 nguồn đã đủ);
  vs baseline F1@0.5 +0,068 (+5/5), F1@val +0,070 (+5/5), ROC +0,037 (+4/5, fold 4 -0,017), PR +0,036 (+4/5). Không ô nào sập / đứng ln2.
- **05/10 02:48 cột chính 4CWE chỉ JS XONG 5/5 (paper_mw2):** ROC TB 0,926; vs baseline F1@0.5 +0,070 (+5/5), F1@val +0,062 (+5/5), ROC +0,030 (+4/5),
  PR +0,031 (+4/5). **common JS + C/C++ XONG 5/5 (158):** ROC TB 0,916; F1@0.5 +0,026 (+4/5), F1@val +0,017 (+5/5), ROC +0,020 (+4/5, min +0,001),
  PR +0,023 (+5/5); f2 và f5 đứng ở ln2 12 epoch rồi thoát muộn (báo nhầm cả hai). 158 sang Pha 1 full chỉ C/C++ lúc 02:48.
- **05/10 01:23 cột chính 4CWE JS + C/C++ XONG 5/5 (paper_mw):** ROC 0,918 / 0,932 / 0,907 / 0,945 / 0,907, TB 0,922; ghép cặp với baseline
  (batch 32): F1@0.5 +0,057 (+5/5), F1@val +0,051 (+4/5), ROC +0,025 (+5/5, min +0,002), PR +0,009 (+3/5). Không ô nào sập (tab cũ 2/5 sập).
  paper_mw chuyển sang hàng nop1 lúc 01:23:58 (queue_after nối đúng).
- **05/10 02:3x THÊM nguồn CHỈ C/C++ (người dùng: "sau bộ các task chính, chạy thêm nguồn CCPP"):** `mwonly5_sources/ccpp_{full,common,4cwe}`
  (merged_undersampled lọc lang ccpp, val 15 % seed 42; toàn hàm lẻ nên val có nghĩa). full 11 778 (5 889/5 889), common 5 742 (2 871/2 871),
  ⚠ 4CWE 3 745 (105/3 640, 2,8 % nhãn 1 - bộ HF không undersample). Mỗi nguồn: Pha 1 + 5 ô cột chính; noRAS / chỉ ASAM / chỉ RecAdam hoãn (skip.txt).
  queue_after nối sau driver hiện tại: 158 full (`state/q158_ccpp.txt`), paper_mw2 common, paper_mw 4CWE. Artifact: nhóm "Chỉ C/C++", 204 ô.
  **Dự đoán:** (K1) Pha 1 chỉ C/C++ thoát bình nguyên trước ep6 (val có nghĩa, giống phần C/C++ của nguồn JS + C/C++); (K2) cột chính chỉ C/C++
  thấp hơn cột chính JS + C/C++ cùng mức ở ≥ 3/5 fold ROC (bỏ JS là bỏ ngôn ngữ gần Python hơn); (K3) 4CWE chỉ C/C++ thấp nhất trong 9 nguồn.
- **05/10 00:4x THÊM (người dùng: "lên lịch chạy thêm MW only (nghĩa là chỉ phase 2) dùng AdamW và ASAM only"):** `nop1_adamw`, `nop1_asamonly`
  (cờ Pha 2 của khối, `--init none`), 10 ô, xếp sau cột chính của paper_mw qua `queue_after.sh` (PID 67909 chờ driver 59791;
  `state/qpapermw_nop1.txt`, fold là vòng ngoài). Artifact: dòng "MW không Pha 1" + nhóm ghép cặp "So với MW không Pha 1".
  **Dự đoán (ghi TRƯỚC khi chạy):** (N1) cả hai 0/5 ô sập; (N2) nop1_adamw ROC ở mức baseline ± 0,02 TB (tab cũ batch 8: 0,910);
  (N3) RecAdam + ASAM nguồn bất kỳ hơn nop1_adamw cùng fold ở ≥ 4/5 fold ROC cho ≥ 4/6 nguồn; (N4) nop1_asamonly > nop1_adamw ROC ở ≥ 4/5 fold, TB +0,005 tới +0,03 (tab cũ batch 8: 0,926 vs 0,910, +0,016 5/5;
  sửa 00:5x trước khi chạy - bản đầu ghi "|Δ| ≤ 0,010" trái bằng chứng tab cũ).
  **Chấm 05/10 03:2x (f1-f4, f5 đang chạy):** (N1) **SAI** - nop1_asamonly f4 sập (test ROC 0,464, ln2 suốt 17 epoch), f3 suýt sập
  (thoát ep21). (N4) **SAI dù f5 ra sao** - ROC ghép cặp asamonly - adamw: f1 -0,025, f2 +0,027, f3 +0,037, f4 -0,463 ⇒ tối đa 3/5.
  Bỏ f4 sập thì 2/3 dương. Tab cũ batch 8 không sập ô nào (0,926 vs 0,910, 5/5); khối này khác ở batch 32 + lr căn bậc hai - chỉ là
  giả thuyết từ MỘT ô sập, chưa kiểm.
  **Chấm 05/10 03:5x - nop1_asamonly đủ 5/5** (ROC 0,8672 / 0,9241 / 0,9155 / 0,4641 / 0,8747, TB 0,8091). asamonly - adamw: ROC -0,092 (+2/5),
  F1@0.5 -0,085 (+2/5), F1@val -0,070 (+1/5), PR -0,095 (+2/5). Bỏ f4 sập: ROC +0,001 (2/4), F1@0.5 +0,006 (2/4), F1@val -0,011 (1/4),
  PR +0,013 (2/4) ⇒ không có Pha 1 thì ASAM không hơn AdamW, và 3/5 fold đứng ở ln2 > 10 epoch (1 sập thật). N4 SAI xác nhận.
  **Chấm 05/10 03:3x - nop1_adamw đủ 5/5** (ROC 0,8922 / 0,8967 / 0,8790 / 0,9274 / 0,9085, TB 0,9008; bậc 2, n=5, seed 42). Cả 4 máy
  đều RTX A4000 và đã kiểm trùng bit giữa máy, nên ghép cặp khác máy không lệch phần cứng.
  (N2) **ĐÚNG** - baseline - nop1_adamw: ROC -0,0042 (+3/5), F1@0.5 -0,0107 (+2/5), F1@val +0,0028 (+3/5), PR -0,0013 (+3/5):
  MW không Pha 1 ≈ CodeBERT 512 thuần, dưới sàn nhiễu.
  (N3) **ĐÚNG** (đã 4/6 nguồn đạt ≥ 4/5 ROC, không phụ thuộc ô còn lại) - cột chính - nop1_adamw, ROC / F1@0.5 / F1@val / PR:
  4cwe_js +0,025 5/5 / +0,059 5/5 / +0,065 5/5 / +0,029 4/5; common_js +0,033 4/5 / +0,057 5/5 / +0,073 5/5 / +0,035 4/5;
  full_js +0,035 4/4 / +0,047 4/4 / +0,071 4/4 / +0,036 4/4 (f5 đang chạy); 4cwe_jscpp +0,021 4/5 / +0,046 5/5 / +0,053 5/5 / +0,008 4/5;
  common_jscpp +0,016 3/5 / +0,016 4/5 / +0,020 3/5 / +0,022 4/5; full_jscpp chưa có.
- **22:42 baseline XONG 5/5 (batch 32, lr 4e-5, paper_mw):** ROC 0,8595 / 0,9097 / 0,8908 / 0,9431 / 0,8797, TB 0,8966. Ghép cặp với baseline
  batch 8 lr 2e-5 (= baseline_tf32 cũ, trùng bit đã kiểm f1-f2): F1@0.5 -0,0230 (+1/5), F1@val -0,0299 (+0/5), ROC -0,0107 (+1/5), PR -0,0108
  (+2/5) - batch 32 làm baseline kém hơn nhẹ nhưng nhất quán (bậc 2, n=5, một máy; ROC chỉ cỡ sàn nhiễu 0,010, F1@val 0/5).
- **21:56 phóng lần 4 (batch 32, lr 2e-5) rồi DỪNG 22:10** (người dùng 21:5x: "Dừng lại và chạy lại toàn bộ với batch gấp đôi ... ở mọi thí
  nghiệm"; "Nếu đo mà thấy đủ lên thẳng batch 16 ... đo theo cái nặng nhất"; "Nếu để 32 không tràn để luôn 32 cho nhanh"). Đo trên 161 ca nặng
  nhất: MW RA + ASAM 32 hàm × 8 cửa sổ (256 cửa sổ/bước, hàm C/C++ dài nhất của jsccpp_full) đỉnh 9 446 MiB; baseline CodeBERT 512 batch 32 đỉnh
  11 722 MiB (A4000 16 GB). Cổng VRAM trước mỗi ô nâng lên 12 500 MiB (VRAM_MIN). eval_batch_size giữ (MW 8, baseline 16). lr giữ 2e-5.
  Hàng đợi, skip.txt, chia máy giữ như 21:35. Lượt batch 4 (21:35-21:50) ở `archive_20261004/aborted_batch4_2150/` (baseline f1-f3 TRÙNG BIT
  baseline_tf32 cũ - xác nhận tái lập khác máy; p1_4cwe_jsonly xong, best ep 10 val ROC 0,553).
- **21:35 phóng lần 3 (batch 4) rồi DỪNG 21:50** - chỉ CỘT CHÍNH + baseline, chia đều 4 máy theo nguồn** (người dùng 21:3x: "ưu tiên xong cột chính mới chạy
  mấy config khác"; "Baseline dùng adamw làm mặc định, chưa cần chạy mấy cái khác vội"; "Đổi lại phase 2 không warmup"; "Share bớt việc cho 2 máy
  161 và 158 ... Chia đều đều cho 4 máy"). Đổi so với 21:22: Pha 2 `--warmup_ratio 0` (LR giảm tuyến tính từ bước 1); Pha 1 thêm
  `--also_select train_loss` (lưu thêm `best_train_loss.pt`; `best.pt` vẫn chọn theo val ROC) vì val ROC Pha 1 trên nguồn cặp chia ngẫu nhiên
  ~0,4-0,55 (nửa kia của cặp ở train, nhãn ngược) ⇒ có thể chọn epoch rất sớm - có sẵn checkpoint thứ hai nếu cần đổi, không phải chạy lại.
  noRAS / chỉ ASAM / chỉ RecAdam HOÃN qua `state/skip.txt` (90 ô). Hàng đợi: 161 `p1_full_jscpp` + 5 ô RA + ASAM (~13,4 h, đường dài nhất);
  158 `p1_common_jscpp` + 5 ô (~7,9 h); paper_mw baseline f1-f5 + `p1_4cwe_jscpp` + 5 ô (~6,2 h); paper_mw2 3 Pha 1 chỉ JS + 15 ô (~6,8 h).
  Luật cảnh báo Pha 1 đổi (fpe.py stall): ep ≥ 8 mà train loss chưa từng < 0,68 (luật ep5 < 2 % của refactor báo nhầm khi không pair loss).
  Kiểm tái lập miễn phí: lượt 21:35 của p1_4cwe_jsonly ra epoch 1 TRÙNG lượt 21:22 (train loss 0,7385, val ROC 0,4161).
- **21:22 phóng lần 2 rồi DỪNG 21:3x** (để thêm also_select + bỏ warmup Pha 2; log ở `archive_20261004/aborted_2133/`).
- **21:04 phóng lần 1 rồi DỪNG 21:1x** (người dùng: "không chạy pair loss, nguồn chia random như thường"; "dùng đúng nguồn trên huggingface";
  "dùng các bộ trong merged undersample và lọc lại ra thành các source theo lang"). Pha 1 lần 1 có pair loss + nguồn `final_experiment_data`
  (JS trùng HF cleanvul_js 100 %, nhưng C/C++ là bản tự dựng 24/09, KHÔNG trùng file nào trên HF). Log bỏ dở: `archive_20261004/aborted_pairloss_2104/`.
  Lỗi của tôi lúc dừng: lọc tiến trình 158 theo tên user, mà `ps` cắt `tranmanhcuong` còn 8 ký tự ⇒ báo "còn 0" trong khi driver vẫn sống;
  đã bắt lại theo UID và dừng đúng PID.
- **Nguồn mới (21:3x) `data/mwonly5_sources/`** (`tools/build_hf_pools.py`, README + MANIFEST.json md5): HF `LyanDumpling/MultiDataSource4VD`
  @387d9dff (sha256 khớp LFS), `merged_undersampled/<mức>` lọc theo lang (giữ thứ tự), xáo `random.Random(42)`, val 15 %, test = val - đúng
  quy tắc `pool_from_export.py` (đối chiếu: trùng thứ tự + nội dung). Chỉ JS: 1 254 / 990 / 588 hàm (= cleanvul_js); JS + C/C++: 13 032 / 6 732 /
  4 333. ⚠ C/C++ của `merged_undersampled/4cwe` trên HF KHÔNG undersample (105 lỗi / 3 640 sạch) ⇒ 4CWE JS + C/C++ chỉ 9 % nhãn 1.
  Đích SVEN trên HF trùng từng byte bản đang dùng. Pha 1 KHÔNG pair loss (`--pair_loss 0`).
- **Đã dừng 20:4x:** driver 161 (`mw_assemble_asamonly_4cwe_jsonly` f5 bỏ dở ep26, `recadamonly_4cwe_jsonly` f5 không chạy). K15 lên n=5 HUỶ.
- **Archive:** `_FinalPaperExperiment/archive_20261004/` (results, logs, state, meta, _repro_keep_vast*, scripts_snapshot; 1 929 file, không xoá gì);
  158 / vast: `<staging>/archive_20261004/`. Checkpoint cũ ở `model/final_paper` (161, 158) giữ nguyên. Artifact cũ XWwCLp7edw8dvjBTi1oRcj giữ nguyên.
- **Mã:** `src_mwonly/` = refactorMWG @ab094cf (04/10 20:28) + `--deterministic 1` (use_deterministic_algorithms không warn_only, CUBLAS :4096:8,
  PYTHONHASHSEED 42 tự chạy lại) + `--tf32 1` + DataLoader trộn bằng Generator riêng (`src_mwonly/SOURCE.md`). Thử GPU 161: Pha 1 và Pha 2 (RA + ASAM)
  chạy 2 lần trùng từng bit. md5 cây mã trùng 4 máy (2e461e22…). Vẫn phóng qua `det_launch.py` (guard own_trainers giữ nguyên). Baseline:
  `src_final/train_baseline.py` (LR hằng: warmup 0 ⇒ không scheduler).
- **Setting (runs.json mới, bản cũ ở archive):**
  Pha 1 `p1_<nguồn>`: `--fusion mw`, batch 4 (mặc định refactor; tab cũ 8), micro 16, lr 2e-5, wd 0,01, warmup 0,25, 16 epoch, min_epochs 10,
  patience 8, chọn val ROC, KHÔNG pair loss (từ 21:3x), không SAM, `--stuck_epoch 0` (không tự chạy lại - luật kẹt ep5 < 2 % chỉ để BÁO).
  Pha 2 (đích `sven_python_folds_nocomment`, `--init all`): 30 epoch, warmup 0 (từ 21:35; trước 0,10), min_epochs 10, patience 8, chọn val ROC, batch 4;
  `rasam_` RecAdam (cof 500, sigmoid k 0,05, t0 0,01) + ASAM ρ 0,5 η 0,01 từ bước 1 | `noras_` AdamW | `asamonly_` ASAM | `raonly_` RecAdam.
  `baseline`: CodeBERT 512 head_middle_tail, batch 8, lr 2e-5 hằng, wd 0, 30 epoch, min 3, patience 8 (đúng baseline_tf32 cũ).
  Nguồn (từ 21:3x): full_jsonly / common_jsonly / 4cwe_jsonly = `mwonly5_sources/js_{full,common,4cwe}`; full_jscpp / common_jscpp /
  4cwe_jscpp = `mwonly5_sources/jsccpp_{full,common,4cwe}`. 131 ô = 6 Pha 1 + 5 baseline + 6 × 4 × 5 Pha 2. Checkpoint mới ở `model/mwonly5/`.
- **Chia máy (fold là vòng ngoài trong từng máy; 4 biến thể của cùng nguồn - fold luôn cùng máy):** 161 `state/q161.txt` Pha 1 full JS + C/C++
  (~12-13 h) rồi 20 ô của nó; 158 `state/q158.txt` Pha 1 common JS + C/C++ (~6-7 h) rồi 20 ô; paper_mw `state/qpapermw.txt` Pha 1 full chỉ JS +
  common chỉ JS rồi 40 ô; paper_mw2 `state/qpapermw2.txt` Pha 1 4CWE chỉ JS + 4CWE JS + C/C++, baseline f1-f5 xen theo fold, 40 ô. Khi máy vast
  xong sớm (~12 h) sẽ chia bớt fold của full JS + C/C++ qua skip.txt. Ước xong cả khối ~14-15 h.
- **Dự đoán (ghi TRƯỚC khi phóng 21:1x; mốc = tab MW-only cũ, batch 8, min 3, mã multibabel):**
  (P1) ≥ 5/6 Pha 1 qua luật kẹt ep5 (giảm ≥ 2 %; luật hiệu chỉnh trên lượt CÓ pair loss nên chỉ là cảnh báo), mọi Pha 1 best val ROC ≥ 0,60;
  4CWE JS + C/C++ (9 % nhãn 1) có val ROC Pha 1 thấp nhất trong ba nguồn JS + C/C++.
  (C1) noRAS và chỉ RecAdam: 0/30 ô sập mỗi cột (sập = test ROC < 0,75; cũ 0/45 và 0/43).
  (C2) RA + ASAM sập ≥ 3/30 ô, tập trung ở full chỉ JS / 4CWE JS + C/C++ / full JS + C/C++ (cũ 5/5, 2/5, 2/5); chỉ ASAM sập ≤ RA + ASAM ở từng nguồn.
  (C3) common chỉ JS × RA + ASAM vẫn cao nhất hoặc cách cao nhất ≤ 0,010 ROC (cũ 0,941). (C4) ở các ô không sập, RA + ASAM hơn noRAS cùng fold về
  ROC ở ≥ 3/5 fold cho ≥ 3/6 nguồn (cũ: common chỉ JS 0,941 vs 0,920). (C5) noRAS hơn baseline ROC (cũ 0,907) ở ≥ 3/5 fold cho ≥ 4/6 nguồn.
  Vùng không kết luận: |Δ| < 0,010 (sàn chạy lại khác máy); một khối n=5 trên 4 máy chưa đủ để viết (CLAUDE.md §2).
- **05/10 12:09 noRAS 3 nguồn JS + C/C++ XONG 15/15, 0/15 sập** (bậc 2, n=5, seed 42). TB ROC 4cwe 0,927 · common 0,901 · full 0,916.
  noRAS - baseline (ROC / F1@0.5 / F1@val / PR): 4cwe +0,030 4/5 / +0,052 5/5 / +0,048 4/5 / +0,032 4/5; common +0,004 3/5 / +0,023 3/5 /
  +0,019 4/5 / +0,007 4/5; full +0,019 5/5 / +0,018 3/5 / +0,009 2/5 / +0,022 5/5.
  **Cột chính - noRAS: 4cwe -0,005 2/5 / +0,005 4/5 / +0,003 3/5 / -0,023 2/5; common +0,015 5/5 / +0,004 3/5 / -0,002 2/5 / +0,016 5/5;
  full -0,002 2/5 / -0,011 2/5 / +0,018 4/5 / -0,004 2/5** - trên JS + C/C++ cột chính ≈ noRAS (chỉ common hơn rõ ở ROC/PR).
  Chỉ RecAdam - noRAS: ROC -0,017 0/5 · -0,012 1/5 · -0,007 1/5 (PR 4cwe -0,031 0/5) - RecAdam KÉM fine-tune thường trên JS + C/C++.
  Chỉ ASAM - noRAS: ROC -0,059 · -0,070 · -0,014 (kéo bởi 2 ô sập).
  **Chấm (JS + C/C++):** (R1) **ĐÚNG** 0/15 sập. (R2) **ĐÚNG** 3/3 nguồn noRAS > baseline ≥ 3/5 fold ROC. (R3) **SAI** - chỉ 1/3 nguồn
  cột chính - noRAS > 0,010 (common). (R4) **SAI** - chỉ 1/3 nguồn |chỉ RecAdam - noRAS| ≤ 0,010; hai nguồn RecAdam thua rõ.
  **Hệ quả cho bài (giả thuyết, một seed):** lợi thế của cột chính so với chuyển giao ngây thơ (noRAS) lặp ở cả 3 nguồn chỉ JS (+0,013 /
  +0,017 / +0,030 ROC) nhưng KHÔNG lặp ở JS + C/C++ (-0,005 / +0,015 / -0,002). Đây là phép so then chốt cho đóng góp; cần seed khác và
  có thể nguồn chỉ C/C++ trước khi viết.
- **05/10 12:10 MỌI VIỆC ĐÃ DUYỆT XONG: 159/159 ô kỳ vọng có kết quả, 0 thiếu, 0 dòng artifact treo "đang chạy"** (baseline 5, MW không
  Pha 1 10, Pha 1 9, cột chính 45, chỉ ASAM 30, chỉ RecAdam 30, noRAS 30). Kết quả + log mọi máy đã về 161 (paper_mw, paper_mw2: 0 file
  còn lại; log driver/queue `state/remote_logs/` khớp tên + byte). Cả 4 máy rảnh; vast chờ người dùng quyết (mục ❓ ở trên).
- **05/10 11:36 noRAS 3 nguồn chỉ JS XONG 15/15, 0/15 sập** (bậc 2, n=5, seed 42). TB ROC 4cwe 0,913 · common 0,917 · full 0,902.
  noRAS - baseline (ROC / F1@0.5 / F1@val / PR): 4cwe +0,017 4/5 / +0,053 4/5 / +0,050 3/5 / +0,010 3/5; common +0,020 4/5 / +0,051
  5/5 / +0,052 5/5 / +0,017 4/5; full +0,006 3/5 / +0,033 4/5 / +0,041 4/5 / -0,003 2/5.
  Cột chính - noRAS: 4cwe +0,013 3/5 / +0,017 3/5 / +0,012 3/5 / +0,021 4/5; common +0,017 4/5 / +0,017 4/5 / +0,018 4/5 / +0,019
  4/5; full +0,030 5/5 / +0,026 3/5 / +0,027 5/5 / +0,037 5/5.
  Chỉ RecAdam - noRAS: 4cwe ROC +0,009 4/5; common -0,001 2/5; full +0,000 1/5 (F1@val full -0,017 0/5).
  Chỉ ASAM - noRAS: ROC +0,017 4/5 · +0,020 3/5 · +0,025 5/5.
  **Chấm (JS):** (R1) **ĐÚNG** 0/15 sập. (R2) **ĐÚNG** 3/3 nguồn noRAS > baseline ≥ 3/5 fold ROC. (R3) **ĐÚNG** 3/3 nguồn cột chính -
  noRAS > 0,010. (R4) **ĐÚNG** 3/3 nguồn |chỉ RecAdam - noRAS| ≤ 0,010. Bức tranh JS (một seed): Pha 1 + fine-tune thường đã hơn baseline;
  RecAdam thêm vào ≈ 0; ASAM thêm vào +0,017 tới +0,025 ROC; RecAdam + ASAM ≈ chỉ ASAM nhưng không sập.
- **05/10 11:24 NHÁNH PHỤ JS + C/C++ (chỉ ASAM / chỉ RecAdam × 3 nguồn × 5 fold) XONG 30/30** (bậc 2, n=5, seed 42).
  TB ROC chỉ ASAM: 4cwe 0,868 · common 0,831 · full 0,902; chỉ RecAdam: 0,910 · 0,889 · 0,908; cột chính: 0,922 · 0,916 · 0,914.
  Chỉ ASAM: 7/15 ô đứng ở ln2 > 10 epoch, **2/15 sập thật** (4cwe f1 0,607, common f3 0,523). Chỉ RecAdam: 0/15 đứng, 0/15 sập.
  Cột chính - chỉ RecAdam (ROC / F1@0.5 / F1@val / PR): 4cwe +0,012 4/5 / +0,019 3/5 / +0,005 3/5 / +0,009 3/5; common +0,027 5/5 /
  +0,033 5/5 / +0,022 5/5 / +0,019 5/5; full +0,005 3/5 / -0,013 2/5 / +0,009 3/5 / +0,001 3/5.
  Cột chính - chỉ ASAM: tính đủ ô 4cwe +0,054 1/5, common +0,085 3/5, full +0,012 3/5 (do 2 ô sập); BỎ ô sập: 4cwe -0,010 0/4,
  common +0,008 2/4, full +0,012 3/5.
  **Chấm:** (T1) **SAI** - đứng ≥ 1/15 đúng (7/15) nhưng sập 2/15 > 1/15. (T2) **ĐÚNG** - 0/15 sập, 0/15 đứng. (T3) **SAI** tính đủ ô
  (0/3 nguồn |Δ| ≤ 0,010); bỏ ô sập thì 2/3 nguồn trong ~0,010. (T4) **ĐÚNG** - 3/3 nguồn cột chính ≥ chỉ RecAdam.
  Mẫu hình chung JS + JS/C++ (giả thuyết, một seed): khi KHÔNG sập, chỉ ASAM ≈ cột chính; ASAM thêm vào RecAdam ăn ROC (5/6 nguồn
  ≥ +0,005, 3 nguồn 5/5); RecAdam thêm vào ASAM KHÔNG tăng trung bình nhưng cột chính 0/45 sập trong khi các biến thể chỉ ASAM sập
  3/35 (nop1 f4, 4cwe_jscpp f1, common_jscpp f3) ⇒ vai trò của RecAdam có thể là ỔN ĐỊNH chứ không phải độ chính xác. Cần lặp seed.
- **05/10 11:0x TỰ CHẠY: noRAS trên 3 nguồn JS + C/C++ (15 ô)** - bước kế theo thứ tự (noRAS JS đang xong dần; đã báo người dùng lúc 10:2x
  "nếu chưa trả lời sẽ cho vast chạy noRAS JS + C/C++"). Chia theo NGUỒN theo checkpoint sẵn có (kéo từ vast về chậm): paper_mw2
  full_jscpp f1-f5 (ngay); 161 common_jscpp f1-f5 (sau ô cuối); paper_mw 4cwe_jscpp f1-f5 (sau 2 ô cuối). **Dự đoán (ghi TRƯỚC):**
  R1-R4 áp nguyên cho JS + C/C++ (0/15 sập; noRAS > baseline ở ≥ 3/5 fold cho ≥ 2/3 nguồn; cột chính - noRAS > 0,010 ở ≥ 2/3
  nguồn; |chỉ RecAdam - noRAS| ≤ 0,010 ở ≥ 2/3 nguồn).
  **Đã phóng / xếp hàng 11:08** (mỗi máy gỡ đúng 1 dòng `noras_<nguồn>_jscpp` khỏi skip.txt RIÊNG - bản lưu `skip.txt.bak_1108`, plan hai
  chiều, preflight 5 mục, cờ `--init all --recadam 0 --sam_rho 0`): paper_mw2 `run.sh` 11:07 (`driver_paper_mw2_noras_jscpp.out`);
  161 `queue_after` PID 1120211 chờ driver 1038845; paper_mw `queue_after` PID 105844 chờ driver 96341.
  **11:37 dời noRAS common_jscpp f4-f5 từ 161 sang 158** (158 hết noRAS JS lúc 11:36): 161 skip.txt thêm 2 dòng `run:fold` (hai chiều:
  f4 bỏ qua, f3 chạy); 158 gỡ dòng hoãn noras_common_jscpp (bản lưu `skip.txt.bak_1137`), hai chiều + preflight; phóng 11:37
  (`state/driver_158_noras_jscpp.out`).
- **05/10 10:1x TỰ CHẠY (máy local rảnh, việc đã duyệt): noRAS trên 3 nguồn chỉ JS (15 ô) trên 158** - người dùng 04/10: "ưu tiên xong cột
  chính mới chạy mấy config khác" (cột chính xong 08:35) và xác nhận định nghĩa noRAS = Pha 1 rồi fine-tune thường (AdamW, không RecAdam,
  không ASAM). JS trước theo cùng thứ tự đã nêu cho nhánh phụ. Checkpoint 3 nguồn chỉ JS chép 161 → 158, md5 khớp 3/3. Fold là vòng ngoài.
  **Dự đoán (ghi TRƯỚC khi phóng):** (R1) noRAS 0/15 sập; (R2) noRAS - baseline: ROC > 0 ở ≥ 3/5 fold cho ≥ 2/3 nguồn (tab cũ common
  chỉ JS 0,920 vs 0,907); (R3) cột chính - noRAS: TB ROC > 0,010 ở ≥ 2/3 nguồn (như cột chính - chỉ RecAdam +0,018 / +0,030);
  (R4) chỉ RecAdam - noRAS: |TB ROC| ≤ 0,010 ở ≥ 2/3 nguồn (giả thuyết RecAdam ≈ không tác dụng). Vùng không kết luận |Δ| < 0,010.
  **Đã phóng 10:15** (158: đếm tiến trình 0, lock rảnh, gỡ 3 dòng `noras_*_jsonly` khỏi skip.txt - bản lưu `skip.txt.bak_1015`, plan hai
  chiều: noras_common_jsonly chạy, noras_common_jscpp vẫn hoãn; preflight 15 mục; cờ `--init all --recadam 0 --sam_rho 0`, batch 32);
  `state/driver_158_noras_js.out`. Dừng: `kill <PID run.sh 158>` hoặc thêm `run:fold` vào skip.txt của 158.
  **10:20 dời noRAS chỉ JS f4-f5 (6 ô) sang paper_mw2** (máy thuê hết việc 10:18; checkpoint 3 nguồn JS trên paper_mw2 khớp md5 với 161):
  158 skip.txt thêm 6 dòng `run:fold` (hai chiều: f4 bỏ qua, f3 chạy); paper_mw2 gỡ 3 dòng hoãn noras_*_jsonly (bản lưu
  `skip.txt.bak_1020`), hai chiều (noras_full_jscpp vẫn hoãn) + preflight 6 mục, cờ `--init all --recadam 0 --sam_rho 0`;
  phóng 10:19 (`state/driver_paper_mw2_noras_js.out`). Ước xong: paper_mw2 ~11:10, 158 ~11:30.
- **05/10 09:11 NHÁNH PHỤ JS (chỉ ASAM / chỉ RecAdam × 3 nguồn chỉ JS × 5 fold) XONG 30/30, 0/30 sập** (bậc 2, n=5, seed 42).
  TB ROC chỉ ASAM: 4cwe 0,930 · common 0,936 · full 0,928; chỉ RecAdam: 0,922 · 0,916 · 0,903 (cột chính 0,926 · 0,934 · 0,932).
  Cột chính - chỉ RecAdam (ROC / F1@0.5 / F1@val / PR): 4cwe +0,004 3/5 / +0,024 4/5 / -0,003 2/5 / +0,008 3/5; common +0,018 5/5 /
  +0,025 5/5 / +0,027 5/5 / +0,023 5/5; full +0,030 5/5 / +0,027 5/5 / +0,044 5/5 / +0,040 5/5.
  Cột chính - chỉ ASAM: 4cwe -0,004 2/5 / +0,011 3/5 / +0,003 2/5 / +0,002 2/5; common -0,002 2/5 / -0,003 2/5 / +0,001 2/5 / -0,000 2/5;
  full +0,005 3/5 / -0,004 1/5 / +0,017 2/5 / +0,011 3/5 - mọi TB |Δ| ≤ 0,017, ROC ≤ 0,005: KHÔNG phân biệt được.
  **Chấm:** (S1) **SAI** - chỉ ASAM 0/15 ô đứng ở ln2 > 10 epoch (min train loss ep1-11 < 0,68 ở cả 15 ô). (S2) **ĐÚNG** - 0/15. (S3)
  **ĐÚNG** - 3/3 nguồn (common, full 5/5 cả bốn chỉ số). (S4) đúng THEO CHỮ (2/3 nguồn ≥ 3/5, nhờ một ô hoà ±0,001) nhưng thực chất
  cột chính ≈ chỉ ASAM. Mẫu hình (giả thuyết, một khối, một seed): trên nguồn chỉ JS, phần ăn là ASAM; RecAdam thêm vào ASAM không
  đổi gì, còn ASAM thêm vào RecAdam +0,018 / +0,030 ROC ở common / full. Cần lặp (nguồn JS + C/C++ đang chạy; seed khác) mới viết.
- **05/10 08:4x TỰ CHẠY (máy rảnh, việc đã bàn - memory `fill-idle-rented-gpu`): chỉ ASAM / chỉ RecAdam trên JS + C/C++ (30 ô)** - bước kế
  theo thứ tự người dùng ("nhánh chỉ recadam và asam ưu tiên thực hiện trên js trước"); noRAS và nhánh phụ chỉ C/C++ vẫn hoãn. Chia theo
  NGUỒN vì kéo checkpoint từ vast về 161 chỉ ~100 KB/s (đẩy lên thì nhanh): 158 common_jscpp f1-f5; paper_mw2 full_jscpp f1-f3; 161 full_jscpp
  f4-f5 (sau nhánh phụ JS); paper_mw 4cwe_jscpp f1-f5 (sau nhánh phụ JS). Mỗi máy fold là vòng ngoài; 4 máy đều A4000, trùng bit đã kiểm.
  **Dự đoán (ghi TRƯỚC khi phóng):** (T1) chỉ ASAM: ≥ 1/15 ô đứng ở ln2 > 10 epoch, ≤ 1/15 sập (test ROC < 0,75); (T2) chỉ RecAdam: 0/15
  sập, không ô nào đứng ở ln2 > 10 epoch; (T3) cột chính - chỉ ASAM: |TB ROC| ≤ 0,010 ở ≥ 2/3 nguồn (như 4cwe chỉ JS -0,004 2/5);
  (T4) cột chính - chỉ RecAdam: TB ROC ≥ 0 ở ≥ 2/3 nguồn. Vùng không kết luận |Δ| < 0,010; n=5 một seed chưa đủ để viết.
  **Đã phóng (mỗi máy: đếm tiến trình = 0, lock rảnh, gỡ đúng 2 dòng hoãn của nguồn đó khỏi skip.txt RIÊNG của máy - bản lưu
  `skip.txt.bak_0845` - rồi plan hai chiều, preflight, cờ):** 158 08:39 `run.sh` PID 1473444 (`state/driver_158_side_jscpp.out`);
  paper_mw2 08:42 PID 64940 (`state/driver_paper_mw2_side_jscpp.out`; checkpoint full_jscpp đẩy từ 161, md5 1b4a7f4d…); 161 08:42
  `queue_after` → run.sh (driver nhánh phụ JS thoát đúng lúc, f4-f5 JS bỏ qua theo skip như dự tính); paper_mw `queue_after` PID 96341 chờ
  driver 88363 (`state/queue_paper_mw_side_jscpp.out`). Thử kéo checkpoint 4cwe_jscpp từ paper_mw về 161 bị ~100 KB/s ⇒ đã dừng, xoá file tạm.
  Dừng: `kill <PID run.sh>` trên máy đó (ô đang chạy dở sẽ mất), hoặc thêm `run:fold` vào skip.txt của máy để bỏ fold chưa tới.
  **10:00 dời common_jscpp f4-f5 từ 158 sang 161** (161 hết phần full_jscpp f4-f5 lúc 09:58): checkpoint p1_common_jscpp trên 161 khớp md5
  với 158 (5b31f16c…); 158 skip.txt thêm 4 dòng `run:fold` (hai chiều: f4 bỏ qua, f3 chạy); 161 gỡ 2 dòng hoãn common_jscpp (bản lưu
  `skip.txt.bak_1000`), hai chiều + preflight đạt, phóng 09:59 (`state/driver_161_side_jscpp2.out`).
- **05/10 08:35 CỘT CHÍNH ĐỦ 9 nguồn × 5 fold, 0/45 sập** (bậc 2). full_ccpp ROC 0,906 / 0,930 / 0,929 / 0,933 / 0,917, TB 0,923; so baseline
  ROC +0,027 4/5, F1@0.5 +0,021 3/5, F1@val +0,010 4/5, PR +0,030 4/5. Xếp TB ROC: common_js 0,934 · full_js 0,932 · 4cwe_js 0,926 ·
  full_ccpp 0,923 · 4cwe_jscpp 0,922 · 4cwe_ccpp 0,921 · common_ccpp 0,917 · common_jscpp 0,916 · full_jscpp 0,914. (K2) full **SAI**
  (ccpp < jscpp 1/5) ⇒ K2 đúng 1/3 mức. full_ccpp - full_jscpp: ROC +0,010 4/5 nhưng F1@val -0,017 1/5.
- **05/10 07:38 cột chính 4cwe_ccpp + common_ccpp ĐỦ 5/5** (bậc 2, n=5, seed 42). 4cwe_ccpp ROC 0,914 / 0,932 / 0,914 / 0,940 / 0,906,
  TB 0,921; common_ccpp 0,914 / 0,911 / 0,913 / 0,946 / 0,901, TB 0,917; 0/10 sập (2 nghi sập ở common_ccpp f1/f4 đều thoát muộn).
  So baseline (ROC / F1@0.5 / F1@val / PR): 4cwe_ccpp +0,025 4/5 / +0,012 3/5 / +0,009 3/5 / +0,027 5/5; common_ccpp +0,021 5/5 /
  +0,033 3/5 / +0,012 2/5 / +0,022 5/5 - ROC/PR rõ, F1 yếu. So chỉ JS cùng mức: 4cwe ROC -0,005 2/5, F1@0.5 -0,058 0/5; common ROC
  -0,017 1/5, F1@val -0,058 0/5. So JS + C/C++ cùng mức: ROC ±0,001 (1/5 dương mỗi mức).
  **Chấm:** (K1) **ĐÚNG** - cả 3 Pha 1 chỉ C/C++ mở cổng chọn theo val ở ep2, val ROC 0,831 / 0,845 / 0,850. (K2) 4cwe **SAI** (ccpp <
  jscpp chỉ 2/5), common **ĐÚNG** (3/5), full chờ. (K3) **SAI** - 4cwe_ccpp TB 0,921, cao hơn full_jscpp 0,914 / common_jscpp 0,916 /
  common_ccpp 0,917. Đã rà ROC trùng hệt giữa các ô cùng fold: 3/487 cặp, xác suất từng mẫu đều khác ⇒ trùng ngẫu nhiên, không chép nhầm.
- **05/10 07:33 DỜI fold 4-5 cột chính `rasam_full_ccpp` từ 158 sang paper_mw2** (paper_mw2 hết việc sau common_ccpp f5 ~07:35; 158 vừa
  xong Pha 1 full_ccpp lúc 07:29, val ROC 0,850). Checkpoint `p1_full_ccpp/.../best.pt` chép 158 → 161 → paper_mw2, md5 khớp cả ba
  (3df32c4b…). 158 `state/skip.txt` thêm `rasam_full_ccpp:4/:5` (kiểm hai chiều bằng plan TRÊN 158: f4 bỏ qua, f3 chạy). paper_mw2:
  skip.txt không chặn rasam_full_ccpp, preflight đạt, plan trỏ checkpoint thật, cờ đúng cột chính. `queue_after.sh paper_mw2 45971`
  (PID 61567, `state/queue_paper_mw2_fullccpp.out`). Ước xong: 158 f1-f3 ~08:45, paper_mw2 f4-f5 ~08:25.
- **05/10 06:41 DỜI fold 4-5 nhánh phụ JS (chỉ ASAM / chỉ RecAdam, 12 ô) từ 161 sang paper_mw** (paper_mw hết việc lúc 06:34 sau 4cwe_ccpp;
  chia theo fold trọn vẹn, CLAUDE.md §4). 161 `state/skip.txt` thêm 12 dòng `run:fold` (kiểm hai chiều bằng plan: f4 bỏ qua, f3 chạy);
  checkpoint Pha 1 3 nguồn JS chép 161 → paper_mw (md5 khớp); preflight đạt; chạy khô đúng cờ. Lần phóng 06:40 bỏ qua sạch 12/12 vì
  skip.txt RIÊNG của paper_mw còn dòng hoãn 04/10 - đã gỡ 6 dòng asamonly/raonly *_jsonly (bản lưu `skip.txt.bak_0640`), phóng lại 06:41
  (`state/driver_paper_mw_side_js.out`). 161 giữ f1-f3 (f1 xong, f2 đang chạy). Ước xong: 161 ~08:45, paper_mw ~09:00.
- **05/10 05:13 CỘT CHÍNH (RA + ASAM) XONG ĐỦ 6 nguồn × 5 fold** (bậc 2, n=5, seed 42; 4 máy đều RTX A4000, đã kiểm trùng bit giữa máy).
  TB ROC: 4cwe_js 0,926 · common_js **0,934** · full_js 0,932 · 4cwe_jscpp 0,922 · common_jscpp 0,916 · full_jscpp 0,914; **0/30 ô sập**
  (min 0,886); 3 ô có nghi sập rồi thoát muộn (common_jscpp f2/f5, full_jscpp f5).
  So với baseline (ROC / F1@0.5 / F1@val / PR, +k/5): 4cwe_js +0,030 4/5 / +0,070 5/5 / +0,062 5/5 / +0,031 4/5; common_js +0,037 4/5 /
  +0,068 5/5 / +0,070 5/5 / +0,036 4/5; full_js +0,036 5/5 / +0,059 5/5 / +0,068 5/5 / +0,034 5/5; 4cwe_jscpp +0,025 5/5 / +0,057 5/5 /
  +0,051 4/5 / +0,009 3/5; common_jscpp +0,020 4/5 / +0,026 4/5 / +0,017 5/5 / +0,023 5/5; full_jscpp +0,017 5/5 / +0,007 3/5 / +0,027 4/5 /
  +0,018 4/5. So với nop1_adamw: cùng chiều, ROC +0,013 tới +0,033, ≥ 3/5 ở mọi nguồn (full_jscpp F1@0.5 -0,004 3/5).
  Chỉ JS - JS + C/C++ cùng mức (ghép cặp fold): 4cwe ROC +0,004 3/5 (F1@val +0,011 3/5); common ROC +0,018 4/5 (F1@val +0,053 4/5);
  full ROC +0,019 4/5 (F1@val +0,041 5/5, min +0,020). Mẫu hình "thêm C/C++ không giúp" - giả thuyết, một khối, chưa lặp.
  **Chấm:** (C2) **SAI** - 0/30 sập (dự đoán ≥ 3/30; cũ full chỉ JS sập 5/5). (C3) **ĐÚNG** - common chỉ JS cao nhất. C1, C4, C5 cần
  noRAS / chỉ RecAdam (noRAS hoãn; chỉ RecAdam JS đang chạy trên 161).

## 04/10 03:34 — ĐANG CHẠY (161 + 158, vast paper_mw đang dựng): TAB MW-ONLY TF32 - 9 NGUỒN PHA 1 × PHA 2, bậc 2 (n=5, seed 42)

Người dùng 04/10 03:0x: "tạm dừng graph, tập trung vào MW only - tab riêng MW-only - nguồn full, common, 4CWE × JS only, JS + CCPP, cả 3 - SOTA, seed 42,
5 fold - thêm pha 2 only AdamW - nếu JS only common tốt nhất hoặc tương đương full (rẻ hơn) thì chạy tiếp ASAM only và RecAdam only - lên bảng trống trước -
dùng song song 2 máy". 03:2x: "tạm cho SOTA là RecAdam + ASAM thay vì chỉ ASAM. Java không lọc được chấp nhận nhét vào toàn bộ. Tôi đã thuê thêm vast,
cài môi trường. Bật TF32 nếu độ lệch không quá lớn" → 03:3x "không cần đo, dựa vào kinh nghiệm (công bố của NVIDIA trên A4000), xem tính toán gì nhanh mà
không giảm nhiều độ chính xác".

- **Setting:** MW (`--fusion mw`), Pha 1 `p1_mw_<src>` (8 epoch, warmup 0,25, pair loss, chọn val ROC). **Giai đoạn 1:** `mw_assemble<hậu tố>` (RecAdam + ASAM
  = SOTA mới) và `mw_assemble_noRAS<hậu tố>` (AdamW). **Giai đoạn 2 (có điều kiện):** `mw_assemble_asamonly<hậu tố>`, `mw_assemble_recadamonly<hậu tố>`.
  9 nguồn: 4cwe_jsonly, common_jsonly, full_jsonly, 4cwe (JS + C/C++), common_nojava, full_nojava, **4cwe_withjava** (mới: phase1_4cwe + toàn bộ 5 184 Java,
  6 077 hàm, `scripts/build_4cwe_withjava.py`, không chia lại), common, full. Đối chứng: `mw_nop1_rasam` (mới), `mw_nop1_adamw_tf32`, `baseline_tf32`
  (+ `mw_nop1_asamonly_tf32`, `mw_nop1_recadamonly` cho giai đoạn 2).
- **TF32 BẬT cho CẢ tab** (`FPE_TF32=1` trong `det_launch.py`, mặc định tắt; kiểm hai chiều). Chọn TF32 thay vì AMP FP16/BF16: TF32 không sửa trainer, A4000
  lý thuyết x2 (đo 28/09 x1,84; 04/10 Pha 2 AdamW 25-26 s/epoch so với 57 s FP32 = x2,2), NVIDIA báo độ chính xác ngang FP32; AMP cần autocast + GradScaler
  đụng ASAM hai bước và RecAdam. Vì luật TF32 cả khối (memory tf32-speed-vs-comparability), mọi run FP32 cũ dùng lại được chạy lại thành bản `_tf32`
  (`p1_mw_common_jsonly_tf32`, `p1_mw_common_tf32`, `mw_assemble_noRAS_jsonly_tf32`, `mw_assemble_tf32`, ...). Không ghép cặp với run FP32 của khối final.
- **Đã dừng (có chủ ý):** lượt FP32 03:11-03:23 (p1_mw_full_jsonly ep3, p1_mw_common_nojava) khi người dùng đổi SOTA; kiểm TF32 03:25-03:30 (4 run `chk_tf32_*`,
  `stopped` trong runs.json) khi người dùng nói không cần đo. Checkpoint dở đã xoá, log ở `state/stopped/`. Dữ liệu duy nhất thu được: ep1 Pha 2 AdamW
  TF32 vs FP32 train loss 0,7411 / 0,7411, val ROC 0,6344 / 0,6346.
- **03:4x người dùng: "không dùng lại pha 1 kết quả cũ. Goodnight"** ⇒ đã kiểm: 36 run Pha 2 + 4 đối chứng của tab đều trỏ tới 9 Pha 1 MỚI của
  tab (TF32), không run nào dùng checkpoint Pha 1 cũ. (Ngoại lệ duy nhất: thử khói vast `chk_vast_repro_noRAS_jsonly` dùng `p1_mw_common_jsonly` FP32 cũ
  CHỈ để so bit môi trường với 161, không vào bảng.)
- **13:3x thêm vast paper_mw2 (54113977, A4000, người dùng thuê):** dựng môi trường như paper_mw (venv311 trùng vdenv, md5 40/40 mã + dữ liệu đích +
  CodeBERT + ckpt thử khói). Hàng đợi `state/qpapermw2.txt`: thử khói `chk_vast2_repro_noRAS_jsonly` f1 (FP32, phải trùng 0.9090592940358199) →
  `baseline_tf32` + `mw_nop1_adamw_tf32` f1-f5 → `mw_nop1_rasam` f4-f5. Đã bỏ hai đối chứng khỏi 161 (`state/skip.txt`) và `mw_nop1_rasam` f4-f5 khỏi
  paper_mw (skip.txt trên vast); đã thử plan cả hai chiều. Ước xong giai đoạn 1 ~16:15 (thay vì 17:00).
- **GIAI ĐOẠN 1 XONG 04/10 16:08 - đủ 105/105 ô** (9 nguồn × 2 optimizer × 5 fold + 3 đối chứng × 5; ô cuối `mw_assemble_noRAS` f2 chạy bù trên paper_mw).
  Bảng đầy đủ: `_FinalPaperExperiment/meta/tmp/mwonly_stage1_matrix.txt`. Bậc 2, TF32, seed 42. ROC TB RA+ASAM / AdamW: 4CWE chỉ JS 0,938 / 0,912;
  common chỉ JS **0,941** / 0,909; full chỉ JS 0,570 (sụp 5) / 0,899; 4CWE JS + C/C++ 0,800 (sụp 2) / **0,923**; common JS + C/C++ 0,918 / 0,902; full JS + C/C++
  0,758 (sụp 2) / 0,903; 4CWE cả 3 0,924 / 0,886; common cả 3 0,929 / 0,872; full cả 3 0,831 (sụp 1) / 0,889. Không Pha 1: RA+ASAM 0,926, AdamW 0,910; baseline 0,907.
  **Lợi của transfer (Pha 1 - không Pha 1, cùng optimizer):** RA+ASAM common chỉ JS ROC +0,015 (4/5), PR +0,013 (4/5), F1@0.5 +0,073 (5/5), F1@val +0,061 (5/5);
  4CWE chỉ JS ROC +0,012 (3/5), F1@0.5 +0,047 (5/5). AdamW common chỉ JS ROC -0,001 (2/5) nhưng F1@0.5 +0,027 (5/5), F1@val +0,029 (5/5); 4CWE JS + C/C++
  ROC +0,013 (4/5), F1@0.5 +0,032 (5/5). Transfer ăn chủ yếu ở NGƯỠNG (F1), ROC nhỏ; bậc 2 một khối, chưa lặp phần cứng khác (CLAUDE.md §2).
  **Đối chiếu dự đoán 03:34:** P1 đúng trừ RA+ASAM full chỉ JS (sụp). P2 sai phần lớn (full JS + C/C++ RA+ASAM sụp 2/5 chứ không ≥ 3/5; AdamW common
  -0,006 ngoài khoảng; 4CWE JS + C/C++ không ≈ 0). P3 đúng một nửa (RA+ASAM common cả 3 -0,012 ✓, full cả 3 sụp ✗; AdamW common cả 3 -0,036 ✗, full -0,019 ✓).
  P4: RA+ASAM ✓ (common chỉ JS cao nhất); AdamW ✗ (4CWE JS + C/C++ 0,923 hơn common chỉ JS 0,014). P6 ✓ (TF32 - FP32 cùng cấu hình |TB ROC| ≤ 0,015 ở 4/4 cặp).
- **14:5x ĐIỀU KIỆN GIAI ĐOẠN 2 ĐẠT** (mọi ô full đã đủ n=5; Δ ghép cặp ROC full - common chỉ JS): RA+ASAM full chỉ JS -0,371 0/5 (sụp 5/5), full
  JS + C/C++ -0,184 0/5, full cả 3 -0,111 0/5; AdamW -0,010 1/5, -0,005 1/5, -0,019 0/5 ⇒ cả 6 ô TB ≤ +0,010, không ô nào 5/5 dương ⇒ tự chạy giai đoạn 2
  theo lời người dùng. **Hàng đợi giai đoạn 2 (100 ô = 18 run × 5 fold + 2 đối chứng × 5, fold là vòng ngoài trên mỗi máy, queue_after sau driver
  giai đoạn 1 của chính máy đó):** 161 `state/q161_s2.txt` (24: common chỉ JS + 4CWE chỉ JS f1-f5, 4CWE JS + C/C++ f4-f5); 158 `state/q158_s2.txt`
  (26: common JS + C/C++ + full JS + C/C++ f1-f5, 4CWE cả 3 f1-f3); paper_mw `state/qpapermw_s2.txt` (24: common cả 3 + full cả 3 f1-f5, 4CWE cả 3 f4-f5);
  paper_mw2 `state/qpapermw2_s2.txt` (26: full chỉ JS f1-f5, 4CWE JS + C/C++ f1-f3, `mw_nop1_asamonly_tf32` + `mw_nop1_recadamonly` f1-f5). Chia theo
  nơi có sẵn checkpoint Pha 1 (p1_mw_full_jsonly + p1_mw_4cwe chép 161 → paper_mw2, md5); checkpoint dùng chung (4cwe_withjava b6e44cc7…, common_tf32
  26a6207b…) trùng md5 ở 161 / 158 / paper_mw; train_mwg tất định bit-identical giữa 4 máy (FACTS §86, thử khói vast) nên chia nguồn giữa máy không
  thêm nhiễu phần cứng. Preflight đủ ở cả 4 máy; runs.json đồng bộ (chỉ khác metadata máy). Ước: chỉ ASAM ~14 phút/ô, chỉ RecAdam ~8 phút/ô ⇒ xong ~20:00-20:30.
- **Dự đoán giai đoạn 2 (ghi TRƯỚC 14:5x):** (S1) chỉ RecAdam: 0 ô sụp trên cả 45 ô có Pha 1 (trước đây 0/18) và trên `mw_nop1_recadamonly`;
  ROC gần AdamW cùng nguồn (|TB Δ| ≤ 0,010 ở ≥ 6/9 nguồn). (S2) chỉ ASAM: sụp tập trung ở đúng 4 nguồn RA+ASAM đã sụp (full chỉ JS, 4CWE JS + C/C++,
  full JS + C/C++, full cả 3), số ô sụp mỗi nguồn ≤ của RA+ASAM (vì RecAdam đóng băng đầu cộng thêm vào bình nguyên); không sụp ở 5 nguồn còn lại.
  (S3) `mw_nop1_asamonly_tf32`: 0/5 sụp (không Pha 1 0/20 trước đây). (S4) ở cả hai cột mới, common chỉ JS vẫn cao nhất hoặc cách cao nhất ≤ 0,010 ROC.
- **14:5x người dùng: "nên gọi agent bổ sung vào kiểm, hướng giải quyết vụ sụp mà không đơn thuần chỉ chạy lại"** ⇒ 3 agent Opus 5.5 chỉ đọc (C cơ chế
  trong mã → cách sửa, D tài liệu, E thiết kế phép chứng minh), báo cáo `_FinalPaperExperiment/meta/review_pack/collapse_fix_{C,D,E}.md`, mục K15
  trên tab kiểm. Khối kiểm nào họ đề xuất sẽ HỎI người dùng trước khi chạy (CLAUDE.md §9).
- **15:2x người dùng quyết:** (1) giai đoạn 2 chỉ ASAM CHỈ chạy nhóm chỉ JS (15 ô: `mw_assemble_asamonly_4cwe_jsonly`, `_jsonly_tf32`, `_full_jsonly`); 35 ô còn lại
  (6 nguồn + `mw_nop1_asamonly_tf32`) HOÃN qua `state/skip.txt` ở cả 4 máy (đã thử plan hai chiều). Chỉ RecAdam vẫn đủ 9 nguồn + đối chứng.
  (2) DUYỆT phép kiểm "ASAM muộn + luật dừng" (K15), bậc 1 (n=3, f1-f3, seed 42, TF32).
- **Phép kiểm K15 - setting:** mã `src_final_lateasam` = bản sao `src_final` + 2 cờ trong `train_mwg.py` (mặc định TẮT = hành vi cũ): `--sam_start_loss 0.65`
  (ASAM không nhiễu tới khi train loss trung bình một epoch < 0,65, từ epoch sau bật và giữ bật; ρ 0,5, η 0,01, tập tham số nhiễu KHÔNG đổi) và
  `--patience_start_loss 0.65` (patience chỉ đếm từ epoch đầu tiên có train loss < 0,65). Run: `fix_lateasam{_full_jsonly,_4cwe,_full_nojava,_full,_jsonly}`
  f1-f3 (= `mw_assemble<hậu tố>` cùng Pha 1, cùng cờ, chỉ khác mã + 2 cờ) + cổng `chk_lateasam_null_full_jsonly` f1 (mã mới, cờ TẮT). Thử khói CPU hai chiều trước khi phóng.
- **Dự đoán K15 (ghi TRƯỚC khi phóng, 15:3x):** (G0) `chk_lateasam_null_full_jsonly` f1 trùng `mw_assemble_full_jsonly` f1 tới chữ số cuối (ROC 0,625…, md5 xác suất) -
  lệch ⇒ dừng khối, không đọc số. (G0') ở mỗi ô fix, các epoch TRƯỚC khi ASAM bật trùng bit với `mw_assemble_recadamonly<hậu tố>` cùng fold.
  (L1) 0/12 ô sụp ở 4 nguồn rủi ro (ROC ≥ 0,80 mọi ô), train loss < 0,65 ở ep ≤ 6 trong ≥ 14/15 ô. (L2) sau khi bật ASAM không tái kẹt: train loss các epoch
  sau giữ < 0,65 ở ≥ 14/15 ô. (L3) common chỉ JS (ô vốn bình thường): |Δ ROC so với `mw_assemble_jsonly`| ≤ 0,010 ở ≥ 2/3 fold. (L4, kết cục phải viết
  lại nếu xảy ra) ô fix hơn AdamW cùng nguồn - fold ở ≥ 9/15 ô ROC; nếu ≤ 7/15 thì "ASAM giúp" chỉ là hiệu ứng của ô vốn bình thường. (L5) luật dừng
  hai giai đoạn hầu như không ràng buộc khi ASAM bật muộn (mô phỏng luật cũ trên log mới cho cùng epoch dừng ở ≥ 13/15 ô) ⇒ phần lớn hiệu ứng thuộc về ASAM muộn.
- **15:5x đã xếp hàng K15 (queue_after nối sau driver giai đoạn 2 của từng máy):** 158 `state/q158_fix.txt` (cổng null f1 CHẠY ĐẦU, rồi full JS + C/C++ và
  common chỉ JS f1-f3), paper_mw `state/qpapermw_fix.txt` (full cả 3 + 4CWE JS + C/C++ f1-f3), paper_mw2 `state/qpapermw2_fix.txt` (chỉ RecAdam 4CWE JS + C/C++
  f4-f5 chuyển từ 161 + full chỉ JS f1-f3). Checkpoint Pha 1 chép thêm (md5 trùng): full_jsonly + common_jsonly_tf32 → 158, 4cwe → paper_mw. Mã md5 trùng 4 máy
  (`find|LC_ALL=C sort|md5sum` = a614ece5…). Thử khói CPU ĐẠT cả ba chiều. Ước xong K15 ~19:30-19:45, giai đoạn 2 ~19:35.
- **K15 XONG 04/10 20:10 (15/15 ô, bậc 1 kiểm chứng n=3, TF32; bảng `_FinalPaperExperiment/meta/tmp/k15_final.txt`):** 0/15 ô sụp (RA+ASAM gốc cùng ô: 8 sụp).
  Dự đoán: G0 ĐÚNG (trùng bit 161 ↔ 158), G0' ĐÚNG 15/15, L1 ĐÚNG (0/12, ROC ≥ 0,80 12/12), L2 ĐÚNG 14/15 (full chỉ JS f2 tái kẹt 3 epoch rồi thoát), **L3 SAI**
  (common chỉ JS -0,004/-0,018/-0,017), L4 ĐÚNG 15/15, L5 ĐÚNG 15/15. K15 - RA+ASAM gốc: ô gốc sụp ROC +0,366 8/8 (cả bốn chỉ số 8/8); ô gốc không sụp ROC -0,007
  1/7, PR -0,001, F1@0.5 -0,007, F1@val +0,001. K15 - chỉ RecAdam ROC +0,027 15/15; K15 - AdamW ROC +0,025 15/15, PR +0,034 15/15. Số ô sụp cả tab: RA+ASAM
  gốc 10/45, chỉ ASAM 2/14 (nhóm chỉ JS), chỉ RecAdam 0/43, AdamW 0/45, ASAM muộn 0/15. paper_mw + paper_mw2 rảnh từ 19:53 / 20:10 (người dùng chọn GIỮ paper_mw,
  chờ chọn việc); `p1_mw_full` đã kéo về 161 (md5 44363f42…), 0 file chưa đồng bộ trên paper_mw. 161 còn 3 ô giai đoạn 2 (f5 nhóm chỉ JS, ~20:50).
- **15:48 SỰ CỐ (lỗi của tôi):** thử khói CPU chạy qua `det_launch.py` ⇒ guard `own_trainers` của run.sh 161 tự dừng driver giai đoạn 1 ⇒ bỏ ô
  `mw_assemble_noRAS` f2. Đã gỡ skip trên paper_mw và chèn ô này vào đầu hàng giai đoạn 2 của paper_mw (queue_after 39481, chạy ~16:05). Bài học ghi vào memory.
- **11:50 giữa chừng (bậc 2, TF32, Δ ghép cặp so với common chỉ JS cùng Pha 2):** RA+ASAM: common chỉ JS 0,941; 4CWE chỉ JS 0,938
  (Δ -0,003, F1@0.5 -0,025 0/5); common JS + C/C++ 0,918 (Δ -0,023 0/5, F1@0.5 -0,073 0/5); SỤP: full chỉ JS 5/5 (0,570), 4CWE JS + C/C++ 2/2,
  full JS + C/C++ 2/3. AdamW (không ô nào sụp): common chỉ JS 0,909; 4CWE chỉ JS 0,912 (+0,003 3/5); full chỉ JS 0,899 (-0,010 1/5); common
  JS + C/C++ 0,902 (-0,006 2/5, F1@0.5 -0,032 0/5); 4CWE JS + C/C++ n=2 0,931; full JS + C/C++ n=3 0,895. Nhóm cả 3 chưa có ô nào.
- **07:3x cân tải nhóm cả 3:** vast `state/skip.txt` bỏ Pha 2 4cwe_withjava + common (vast còn: Pha 1 full → Pha 2 full f1-f5 → queue_after
  `mw_nop1_rasam`). 161 (queue_after `state/q161_mwonly_c.txt`): f1-f2 × {4cwe_withjava, common} × 2 + `baseline_tf32` + `mw_nop1_adamw_tf32`.
  158 (queue_after `state/q158_mwonly_c.txt`): f3-f5 × {4cwe_withjava, common} × 2. Checkpoint Pha 1 chép vast → 161 → 158 (md5; p1_mw_4cwe_withjava
  b6e44cc7… đã xong, p1_mw_common_tf32 tự chép khi Pha 1 xong ~07:45). Sụp: `mw_assemble_full_jsonly` f1, f2 (RA+ASAM, chữ ký ln2) - `state/mwonly_events.txt`.
  Ước xong giai đoạn 1: ~17:00.
- **Kế hoạch qua đêm (người dùng ngủ):** xong giai đoạn 1 thì tính ma trận; **điều kiện giai đoạn 2** = ở cả hai cột (RA+ASAM, AdamW), với MỖI nguồn full
  (chỉ JS, JS + C/C++, cả 3): Δ ghép cặp ROC (full - common chỉ JS) có TB ≤ +0,010 và không 5/5 dương ⇒ ĐẠT ⇒ tự chạy giai đoạn 2 (chỉ ASAM + chỉ RecAdam,
  9 nguồn, + 2 đối chứng không Pha 1) theo lời người dùng "nếu ... thì chạy tiếp". KHÔNG đạt ⇒ dừng, báo. Hết việc đã duyệt ⇒ kéo kết quả + checkpoint Pha 1
  của vast về 161 (đối chiếu byte/md5) rồi `vastai destroy instance 54059221 -y` (memory destroy-rented-gpu-when-queue-empties).
- **Hàng đợi (04/10 03:34):** 161 (`state/q161_mwonly.txt`, 44 mục): Pha 1 4cwe_jsonly, common_jsonly_tf32, full_jsonly, 4cwe → f1-f5 × {4cwe_jsonly,
  common_jsonly, full_jsonly} × {RA+ASAM, AdamW} → f1-f5 4cwe × 2. 158: Pha 1 common_nojava → f1-f5 nojava × 2 → (queue_after) Pha 1 full_nojava →
  f1-f5 full_nojava × 2. vast paper_mw (54059221, A4000, đĩa 22 GB): đang dựng môi trường (agent), sau đó nhận Pha 1 full / 4cwe_withjava / common_tf32 +
  Pha 2 của chúng + đối chứng. Ước TF32: Pha 1 ≈ 0,55 × FP32; Pha 2 RA+ASAM ~22 phút, AdamW ~9 phút.
- **Mốc FP32 cũ (chỉ để tham khảo, KHÔNG ghép cặp):** common chỉ JS ROC ASAM 0,941 / AdamW 0,920; mw_assemble (common cả 3, RA+ASAM) 0,916; không Pha 1
  ASAM 0,926 / AdamW 0,896; baseline 0,906. Phía MWG RA+ASAM: full chỉ JS - common chỉ JS ROC -0,000; 4CWE chỉ JS -0,003; 4CWE JS + C/C++ +0,005;
  common JS + C/C++ -0,017 (F1@0.5 -0,065 0/5); full JS + C/C++ -0,294 (SỤP 0/5); common cả 3 -0,003; full cả 3 +0,007.
- **Dự đoán (ghi TRƯỚC, Δ = nguồn - common chỉ JS, cùng Pha 2, TF32):** (P1) chỉ JS: full và 4CWE |TB ROC| ≤ 0,010, không 5/5 cùng dấu; F1@val của full
  ≤ 3/5 dương. (P2) JS + C/C++: common ROC -0,01 đến -0,03, F1@0.5 âm ≥ 4/5; full JS + C/C++ với RA+ASAM SỤP ≥ 3/5 ô (như MWG), AdamW không sụp nhưng ROC
  thấp hơn ≥ 0,01; 4CWE JS + C/C++ |TB ROC| ≤ 0,010. (P3) cả 3: common và full |TB ROC| ≤ 0,02, không 5/5 dương; 4CWE cả 3 (Java 85 %) ≈ common cả 3.
  (P4) ở cả hai cột, common chỉ JS cao nhất hoặc cách cao nhất ≤ 0,010 ROC ⇒ điều kiện giai đoạn 2 nhiều khả năng đạt. (P5) Pha 1 val ROC 0,50-0,65;
  full và full JS + C/C++ chọn epoch 1-2. (P6) mw_assemble_tf32 - mw_assemble (FP32 cũ, cùng cấu hình) |TB ROC| ≤ 0,015 (TF32 như đổi quỹ đạo, không như đổi seed).

## 02/10 21:41 — XONG 03/10 03:18 (158 10 ô + 161 2 ô f5): PHẦN CÒN LẠI BẢNG 2×2 C/C++ f2-f5, bậc 2 (n=5, seed 42)

- **Bảng 2×2 đủ 20/20 ô (hp 48/48):** ROC TB RA+ASAM 0,916 · ASAM 0,918 · RA 0,911 · AdamW 0,916. **ASAM - AdamW: 022+079 -0,299 (0/5), CWE-macro
  -0,138 (0/5)**, F1@0.5 -0,028 (1/5). FACTS §87.5. Cả 161 và 158 RẢNH từ 03:18; không còn việc đã duyệt trong hàng đợi.

- 158 xong khối phép kiểm A C/C++ lúc 21:38 (driver in XONG). Theo "chủ động chạy task khi đang rảnh": lấy việc đã xếp kế tiếp = `mwg_assemble_asamonly_ccpponly`,
  `mwg_assemble_noRAS_ccpponly`, `mwg_assemble_recadamonly_ccpponly` f2-f5 (12 ô, fold-major), vốn nằm trong hàng đợi 161 sau khối ưu tiên (~03:00).
  161 `state/skip.txt` đã thêm 12 dòng (plan 161 trả SKIP, đã thử). Checkpoint Pha 1 `p1_mwg_common_ccpponly` md5 trùng 161/158 (9902801a…);
  train_mwg tất định giữa hai máy (FACTS §86) ⇒ f1 (161) và f2-f5 (158) ghép được. Preflight 12/12, dry-run đúng cờ, 1 driver, lock giữ. Xong ~04:00.
- **Kết quả phép kiểm A C/C++ phía MW (n=5, seed 42), Δ = có Pha 1 C/C++ - không Pha 1:** ASAM: ROC -0,014 (0/5 dương, 4 âm), F1@0.5 -0,024 (0/5),
  F1@val -0,001 (3/5), 022+079 +0,026 (3/5). AdamW: ROC -0,031 (0/5), PR -0,030 (0/5), F1@0.5 -0,020 (0/5), F1@val -0,024 (0/5), 022+079 +0,043 (4/5).
  ⇒ không đồ thị, Pha 1 nguồn C/C++ HẠI ở mức tổng (AdamW 5/5 cả bốn chỉ số). Phía MWG f2-f5 đang chạy (dòng trên) để tính tương tác.
- **03/10 02:25:** 158 xong 10/12 ô. Tương tác đồ thị × transfer C/C++ (n=5 ASAM, n=4 AdamW): ASAM ROC +0,018 (4/5), F1@0.5 +0,008 (3/5),
  F1@val -0,009 (2/5), 022+079 -0,093 (2/5) - LẪN LỘN; AdamW ROC +0,045, PR +0,046, F1@0.5 +0,065, F1@val +0,073, 022+079 +0,274 - 20/20 ô dương.
- **02:27:** GPU 158 bị job khác chiếm (2 × train.py của tài khoản tranmanhcuong + job EEG, VRAM trống 8,3 GB < 9 GB) ⇒ driver 158 kẹt ở
  "chờ VRAM" ⇒ dừng driver (PID 628635 + sleep con, chưa chạy ô nào; lock tự do) và gỡ skip `noRAS:5`, `recadamonly:5` trên 161 (plan đã thử:
  hai ô chạy, `asamonly:5` vẫn skip). Hàng đợi 161 (queue_after 345585) sẽ chạy chúng sau ô cuối khối ưu tiên (~02:40). Không đụng job người khác.
- **03:03 phép kiểm A C/C++ ĐỦ n=5:** tương tác AdamW 25/25 ô dương (ROC +0,039, F1@0.5 +0,054, F1@val +0,064, 022+079 +0,227, đều 5/5);
  ASAM lẫn lộn (ROC +0,018 4/5, F1@val -0,009 2/5, 022+079 -0,093 2/5). FACTS §87.4. Còn `mwg_assemble_recadamonly_ccpponly` f5 (161) cho bảng 2×2.

## 02/10 16:10 — XONG 23:22, ĐÃ HUỶ 23:4x (vast `paper`, 53831025, RTX 4070 SUPER 12 GB): LẶP 3 RUN NGUỒN CHỈ JS TRÊN LOẠI GPU KHÁC, bậc 2 (n=5, seed 42)

- **Kết quả (Δ = paper - A4000, ghép cặp theo fold, 15/15 ô, hparam khớp 48/48 mọi ô):** MWG common ROC -0,001 (2+/3-), F1@0.5 -0,009, F1@val -0,012;
  MWG full ROC -0,002, F1@0.5 +0,016 (5/5), 022+079 +0,108 (5/5); MW common ROC -0,006 (1+/4-), F1@val -0,018, 022+079 +0,078 (4/5).
  So cùng máy: MWG - MW ROC paper +0,002 / A4000 -0,003; 022+079 paper -0,006 / A4000 +0,081. common - full ROC paper +0,008 / A4000 +0,007;
  F1@0.5 paper -0,005 (2/5) / A4000 +0,019 (4/5); **022+079 paper -0,043 (1/5) / A4000 +0,074 (5/5) - ĐẢO DẤU**; CWE-macro -0,016 / +0,040 (5/5).
  ⇒ chỉ số tổng lặp được qua GPU; hiệu ứng theo CWE (một seed, một máy) KHÔNG lặp - CLAUDE.md §2 đúng ở đây.
- **Dự đoán:** P1 ĐÚNG (|TB ROC| ≤ 0,006, không 5/5 cùng dấu; F1@0.5 |TB| ≤ 0,016 - riêng full 5/5 dương). P2 ĐÚNG (MWG - MW ROC +0,002).
  P3 SAI nửa (common - full ROC +0,008 đúng; F1@0.5 -0,005 âm). P4 gần đúng (ep4 cả ba; val 0,548 / 0,524 / 0,556 - MWG common hụt 0,55 một chút).
- **Huỷ:** driver XONG 23:22 → `paper_finish.sh`: sync 15/15 + 3 Pha 1, 3 ckpt Pha 1 md5 trùng (`model/final_paper/p1_*_paper/`), state + script dựng môi
  trường ở `state/vast_paper53831025/`, kiểm kê thiếu 0 → `vastai destroy instance 53831025 -y` in "destroying instance 53831025.", show instances rỗng.

Người dùng 02/10: "tôi đã thuê 1 máy trên vast tên paper, tạo tôi 1 tab dùng để lưu thông số máy này kiểm. Test lại cho tôi với nguồn JS: common, full với
phương pháp SOTA và thêm bản 2 pha không đồ thị của common" + 15:4x "dựa trên các runname chuẩn tôi đã gửi mwg_assemble_asamonly_jsonly và
mw_assemble_asamonly_jsonly" ⇒ setting = **AdamW + ASAM ρ 0,5 / η 0,01, KHÔNG RecAdam** (đọc lại từ artifact; 3 run RecAdam + ASAM khai báo trước đó đã
bỏ, chưa chạy). Cả hai pha chạy lại trên `paper`; mọi cờ giống hệt run gốc trên A4000 (chỉ khác đường dẫn).
- **Run (tab "Máy vast paper"):** Pha 1 `p1_mwg_common_jsonly_paper`, `p1_mwg_full_jsonly_paper`, `p1_mw_common_jsonly_paper` → Pha 2 fold-major f1-f5:
  `mwg_assemble_asamonly_jsonly_paper` (so `mwg_assemble_asamonly_jsonly`), `mwg_assemble_asamonly_full_jsonly_paper` (so
  `mwg_assemble_asamonly_full_jsonly`), `mw_assemble_asamonly_jsonly_paper` (so `mw_assemble_asamonly_jsonly`).
- **Môi trường (xong 16:09):** `/workspace/venv311` = Python 3.11.14 + torch 2.9.1+cu128, cuDNN 91002, transformers 4.57.1, numpy 2.3.4, sklearn 1.7.2
  (trùng vdenv). uv của image đặt `UV_NO_CACHE=1` + `UV_LINK_MODE=copy` ⇒ đứt mạng là tải lại từ đầu, cài thì nhân đôi đĩa (hết 18 GB) ⇒ tải từng wheel
  bằng `curl -C -` + sha256, cài `UV_LINK_MODE=hardlink`. CodeBERT tải từ HF `refs/pr/8` (sha256 trùng 161). Mã + dữ liệu + model trùng md5 47/47 file.
- **Phóng 16:10** `run.sh paper` (18 mục: 3 Pha 1 rồi f1-f5 × 3 run), 1 driver, lock giữ thật; log đọc lại: tất định, TF32 tắt, pair loss 364 cặp + 116 lẻ.
- **21:5x người dùng: "xong task hãy huỷ vast giúp tôi"** ⇒ script canh `paper_finish.sh` (job tmp): chờ dòng `FPE paper XONG` của driver → `sync_remote.sh`
  → state về `state/vast_paper53831025/` → 3 checkpoint Pha 1 về `model/final_paper/p1_*_paper/` (md5) → kiểm kê tên + byte (đã thử chiều lệch: báo
  9 thiếu, CHƯA ĐỦ) → in READY; phiên Claude mới chạy `vastai destroy instance 53831025 -y`. KHÔNG đụng 51144271 (máy đồng nghiệp).
- **Mốc A4000 (n=5, seed 42):** ROC MWG common 0,938, MWG full 0,932, MW common 0,941; MWG - MW ROC -0,003 (1/4), F1@0.5 +0,005 (3/2);
  common - full (MWG) ROC +0,007 (4/1), F1@0.5 +0,019 (4/1).
- **Dự đoán (ghi TRƯỚC):** (P1) mỗi run, Δ ghép cặp paper - A4000: |TB ROC| ≤ 0,015 và nằm dưới sàn khác GPU 0,028, KHÔNG 5/5 cùng dấu; F1@0.5
  dao động lớn hơn (|TB| ≤ 0,03). (P2) MWG - MW trên paper: |TB ROC| ≤ 0,010 (không phân biệt được, như A4000). (P3) common - full (MWG) trên paper:
  F1@0.5 TB dương, ROC |TB| ≤ 0,010. (P4) Pha 1 common chỉ JS chọn epoch 3-5 với val ROC 0,55-0,60 như A4000.

## 02/10 15:3x — XONG 03/10 02:47 (161): MWG so với MW, chỉ ASAM, nguồn common chỉ JS - BẬC 3 (n=15 = 5 fold × seed 42, 1234, 7)

- **KẾT QUẢ (24/24 ô đủ json + npz, hp khớp 48/48):** seed 1234 sụp 9/10 ô Pha 2 ⇒ không cặp nào dùng được; 10 ô dùng được (seed 42 + 7 × 5 fold).
  Δ MWG - MW: ROC +0,002 (5+/5-, Wilcoxon p 0,63) · PR -0,006 (4+/4-) · F1@0.5 +0,006 (5+/3-) · F1@val +0,008 (6+/3-) · **022+079 +0,082 (7+/3-,
  p 0,027)** · **CWE-macro +0,038 (7+/3-, p 0,049)**. Dự đoán S1, S2, S3 ĐÚNG. Cùng phép so trên 4070S (seed 42): 022+079 -0,006 ⇒ chưa vững qua GPU.

Người dùng 02/10 15:2x: "ưu tiên chạy cho tôi cái này so sánh mwg_assemble_asamonly_jsonly và mw_assemble_asamonly_jsonly ... chạy ở 1 trong 2 máy 158 và
161 ... dừng task đó và ưu tiên chạy cái này trước n=15 42, 1234, 7" + "Nếu có n=5 rồi chạy n=10 tiếp vào là được". Seed 42 đã có (n=5) ⇒ chạy thêm seed
1234 và 7. **Chọn 161** (fold đang chạy mới ~8 phút; 158 đang giữa Pha 1 dài ~1,3 giờ). **15:25 đã dừng** driver C/C++ trên 161 (PID 242779 + trainer
331278, đúng PID; GPU về 1 MiB, lock trống); `mwg_assemble_asamonly_ccpponly` f2 dở dang - chạy lại SAU khối này cùng phần còn lại của bảng 2×2 C/C++.
- **Run (mỗi seed CẢ HAI pha, như tiền lệ `_s1234` của full C/C++):** `p1_mwg_common_jsonly_s{1234,7}`, `p1_mw_common_jsonly_s{1234,7}` (thêm `--seed`),
  `mwg_assemble_asamonly_jsonly_s{1234,7}`, `mw_assemble_asamonly_jsonly_s{1234,7}`. Tham số hiệu lực chỉ khác bản seed 42 ở `seed`.
- **Thứ tự:** 4 Pha 1 (~80 phút) → fold-major: f1 (MWG s1234, MW s1234, MWG s7, MW s7), f2, … f5 (~10 giờ). Tổng ~11,5 giờ.
- **Dự đoán (ghi TRƯỚC), Δ = MWG - MW ghép cặp theo (fold, seed), 15 ô:** (S1) ROC gộp |TB| ≤ 0,010, số ô dương 5-10/15; (S2) CWE-022 + 079 TB dương
  (seed 42: +0,056) nhưng ≤ 11/15 ô dương; (S3) ROC macro theo CWE TB dương, ≤ 11/15 ô dương.
- **Pha 1 xong 16:52**, cả 4 thoát bình nguyên: MWG s1234 ep4 0,582 · MW s1234 ep6 0,568 · MWG s7 ep5 0,559 · MW s7 ep4 0,565 (val ROC).
- **17:09 `mwg_assemble_asamonly_jsonly_s1234` f1 SỤP**: Pha 2 train loss kẹt ~0,69 (ln2) suốt 10 epoch, val ROC ep1 0,603 không bị vượt ⇒ early stop ep10,
  best ep1, test ROC 0,575. Rà 227 ô Pha 2 có dừng sớm: **11 ô** train loss không bao giờ rời bình nguyên (min > 0,66) ⇒ test 0,44-0,62; **216 ô** đã học ⇒
  test thấp nhất 0,854. Mọi ô "sụp" của dự án (full C/C++, full bỏ Java ×3 biến thể, và ô này) cùng một chữ ký: dừng ở ep10-11 khi còn trên bình nguyên
  (`min_epochs 3` + `patience 8`). Cùng giờ, `paper` f1 thoát ở ep9 (val 0,612 vượt 0,559 sát hạn) rồi lên 0,945. GIẢ THUYẾT (chưa kiểm): sụp = luật dừng
  sớm cắt run còn trên bình nguyên; nguồn/seed chỉ làm bình nguyên dài hơn. Đã hỏi người dùng: giữ nguyên / kiểm rẻ `--min_epochs 15` cho đúng ô này /
  đổi luật. Chưa đổi gì.
- **17:28 `mw_assemble_asamonly_jsonly_s1234` f1 CŨNG SỤP** cùng chữ ký (loss ~0,70 cả 10 epoch, best ep1, test ROC 0,501) ⇒ cặp MWG - MW seed 1234 f1
  là sụp - sụp, không so được. 12 ô sụp / 229 ô có dừng sớm. Đang chạy MWG seed 7 f1.
- **18:35 lát f1 xong** (Δ = MWG - MW): seed 42 ROC -0,009, 022+079 +0,080; seed 7 ROC +0,031, 022+079 +0,132; seed 1234 sụp cả hai (không tính).
- **18:51 `mwg_assemble_asamonly_jsonly_s1234` f2 SỤP** (ô seed 1234 thứ 3): loss ~0,71 cả 10 epoch dù mốc ep1 chỉ 0,504 ⇒ test ROC 0,603.
  Seed 7 (2/2 ô) và máy paper (4/4 ô) vẫn thoát bình nguyên. Vẫn chờ người dùng chọn cách xử lý.
- **19:38 `mw_assemble_asamonly_jsonly_s1234` f2 SỐNG nhờ val nhích dần (patience reset liên tục), thoát bình nguyên ở ep13-16 (muộn nhất
  từng thấy) ⇒ best ep26, test ROC 0,892 (seed 42 cùng fold 0,948). Bằng chứng tự nhiên: seed 1234 không làm hỏng mô hình, chỉ kéo dài bình nguyên;
  thoát muộn thì còn ít bước LR (lịch tuyến tính theo 30 epoch) nên vẫn thấp hơn ~0,05.
- **21:09 `mwg_assemble_asamonly_jsonly_s1234` f3 SỤP** (ô seed 1234 thứ 4; MWG s1234 sụp 3/3 fold): loss 0,70-0,72 cả 10 epoch, best ep1.
  Lát f2 (4 ô dùng được f1-f2, seed 42 + 7): Δ MWG - MW ROC +0,011 (3/4), F1@0.5 +0,020 (4/4), 022+079 +0,166 (4/4), CWE-macro +0,084 (4/4).
  Chưa tách được: sụp do checkpoint Pha 1 seed 1234 hay do RNG Pha 2 seed 1234 (cả hai cùng đổi). Phép kiểm tách cần người dùng duyệt.
- **21:24 `mw_assemble_asamonly_jsonly_s1234` f3 SỤP** (test ROC 0,526) ⇒ seed 1234 sụp 5/6 ô (MWG 3/3, MW 2/3); seed 7 4/4 ổn; paper 10/10 ổn.
- **22:36 `mwg_assemble_asamonly_jsonly_s1234` f4 SỤP** (test 0,538) ⇒ MWG s1234 sụp 4/4. Lát f3: 6 ô dùng được (s42 + s7, f1-f3): Δ MWG - MW ROC
  +0,002 (3/6), F1@0.5 +0,008 (4/6), 022+079 +0,144 (6/6), CWE-macro +0,065 (6/6).
- **Rà sụp theo nguồn (261 ô Pha 2, 22:4x):** 15 sụp, cùng chữ ký bình nguyên + dừng ep10-11. full_nojava 8/15, full_ccpponly s42 2/5 (s1234 0/5),
  common_jsonly s1234 5/6 (s42/s7/paper 0/23), mọi nguồn khác s42 0/~200, không Pha 1 0/25, không ASAM 0/77 (nhưng AdamW CHƯA chạy trên
  nguồn full C/C++ hay seed 1234). ASAM chỉ ở Pha 2 (38 run Pha 1 đều sam_rho 0). ⇒ "nguyên nhân là NGUỒN" (30/09) QUÁ SỚM: nguồn + seed
  làm bình nguyên Pha 2 dài, luật dừng sớm biến thành sụp.
- **00:45 `mwg_assemble_asamonly_jsonly_s1234` f5 SỤP** (0,559) ⇒ MWG seed 1234 sụp **5/5**, MW seed 1234 sụp 3/4 (f5 đang chạy). Lát f4 (8 ô dùng được,
  s42 + s7): Δ MWG - MW ROC +0,002 (4/8), F1@0.5 +0,010 (5/8), 022+079 +0,112 (7/8), CWE-macro +0,051 (7/8).

## 02/10 12:3x — ĐANG CHẠY (161 + 158): CÁC PHÉP KIỂM CỦA NGUỒN JS, LÀM LẠI VỚI NGUỒN C/C++, bậc 2 (n=5, seed 42)

Người dùng 02/10: "kiểm tương tự js cho bộ ccpp làm nguồn, các thí nghiệm chưa xong". Sáu run mới, mỗi run là bản C/C++ của một run JS (tham số hiệu lực
TRÙNG run JS làm khuôn, chỉ khác đường dẫn nguồn / checkpoint; Pha 1 MWG dùng lại `p1_mwg_common_ccpponly` seed 42):
- **161 - bảng 2×2 optimizer (K11) + đối chứng không Pha 1:** `mwg_assemble_asamonly_ccpponly` (chỉ ASAM), `mwg_assemble_noRAS_ccpponly` (AdamW, train 2 pha
  thuần), `mwg_assemble_recadamonly_ccpponly` (chỉ RecAdam); ô RecAdam + ASAM là `mwg_assemble_ccpponly` đã có. Ghép với `mwg_nop1_asamonly` /
  `mwg_nop1_adamw` (không phụ thuộc nguồn, đã có n=5). Fold-major, ~5,3 giờ.
- **158 - phép kiểm A (đồ thị × transfer) với nguồn C/C++:** `p1_mw_common_ccpponly` (~2,7 giờ) → `mw_assemble_asamonly_ccpponly`, `mw_assemble_noRAS_ccpponly`;
  ghép với `mw_nop1_asamonly` / `mw_nop1_adamw` đã có. ~6,5 giờ.
- **Không chạy, lý do:** bản 60 cặp (cỡ common C/C++ 2 868 nhãn 1 ≈ uncommon C/C++ 3 018 - đã cùng cỡ, phép kiểm B suy biến); các run `chk_*` (kiểm cơ chế
  RecAdam / checkpoint sớm, kết luận chung cho mọi nguồn). **Hỏi người dùng:** bảng 2×2 cho full C/C++ (Pha 1 seed 42 SẬP, seed 1234 không - dùng bản nào?);
  uncommon CHỈ C/C++ chặt (gần trùng uncommon C/C++ + JS chặt, chỉ khác 60 cặp JS / 6 138 hàng).
- **Dự đoán (ghi TRƯỚC):** (C1) chỉ ASAM so với RecAdam + ASAM (C/C++): ROC trong ±0,010, không 5/5 cùng dấu (RecAdam ≈ không tác dụng, K11);
  (C2) chỉ ASAM có - không Pha 1: ROC > 0 ở ≥ 4/5 fold (như JS +0,025), CWE-022 + 079 < 0 ở ≥ 3/5 fold (nguồn C/C++ không chuyển giao sang CWE web);
  (C3) AdamW có - không Pha 1: ROC trong ±0,015, F1@0.5 |TB| ≤ 0,02 (JS +0,044 nhờ CWE web; C/C++ không có phần đó);
  (C4) chỉ ASAM so với AdamW (cùng Pha 1 C/C++): ROC > 0 ở ≥ 4/5 fold (ASAM là thành phần có tác dụng, K11);
  (CA1) MW C/C++ có - không Pha 1 (ASAM): ROC > 0 ở ≥ 3/5 fold, CWE-022 + 079 ≤ 0 ở ≥ 3/5 fold;
  (CA2) tương tác đồ thị × transfer (C/C++, ASAM): ROC |TB| ≤ 0,015 và không 5/5 cùng dấu.

## 02/10 10:0x — XONG 02/10 12:20 (158 f1-f4, 161 f5): KIỂM TẤT ĐỊNH train_mwg.py + PHÉP KIỂM A VỚI AdamW, bậc 2 (n=5, seed 42)

**Kết quả A-AdamW (n=5):** MW AdamW có - không Pha 1: ROC +0,024 (+4/-1), PR +0,008 (+3/-2), F1@0.5 +0,061 (5/5), F1@val +0,056 (5/5), CWE-022 + 079 +0,446 (5/5), macro +0,216 (5/5). Tương tác Δ(MWG) - Δ(MW) với AdamW: ROC -0,016 (+1/-4), F1@0.5 -0,017 (0/5), CWE-022 + 079 +0,017 (+2/-3), macro +0,006 (+2/-3) - đồ thị làm lợi ích của transfer nhỏ đi một chút (ngược bản ASAM +0,010). **Dự đoán:** AW1 nửa đúng (CWE-022 + 079 dương 5/5 - đúng; ROC +0,024 ngoài ±0,015 - sai); AW2 nửa đúng (ROC -0,016 ngoài ±0,015 - sai, sát ngưỡng; CWE-022 + 079 không nhất quán - đúng). Hai máy rảnh từ 12:20.


Tự quyết theo luật "chủ động chạy task khi đang rảnh" + "chủ động kiểm theo hướng của tôi" (người dùng 02/10); 158 hết hàng 09:53.
- **(D) XONG 10:15 - dự đoán ĐÚNG: 152 xác suất test / val trùng từng bit với bản 161, mọi chỉ số và best epoch trùng (FACTS §86).** `chk_det_mwg_nop1_adamw` f1: cùng mọi cờ với `mwg_nop1_adamw` (tham số hiệu lực trùng hoàn toàn), chạy lại f1 (bản gốc ở 161) trên 158.
  **Dự đoán:** 152 xác suất test trùng từng bit với bản 161 (như FACTS đã đo cho train_baseline.py). Lệch ⇒ ghép cặp khác máy của train_mwg.py
  (fold 5 của MWG / MW không Pha 1, phép kiểm B f2/f4) phải coi là có sàn nhiễu 0,010.
- **(A-AdamW) `mw_assemble_noRAS_jsonly` (Pha 1 `p1_mw_common_jsonly`) + `mw_nop1_adamw`, f1-f5:** như phép kiểm A nhưng Pha 2 AdamW thường; ghép
  với cặp MWG AdamW đã có (`mwg_assemble_noRAS_jsonly` / `mwg_nop1_adamw`). Tham số hiệu lực chỉ khác cặp MWG ở `fusion` (chiều lệch bắt đủ 2 khoá).
  **Dự đoán (ghi TRƯỚC):** (AW1) MW AdamW có - không Pha 1: CWE-022 + 079 > 0 ở ≥ 4/5 fold, ROC gộp |TB| ≤ 0,015 (như MWG AdamW +0,008);
  (AW2) tương tác Δ(MWG) - Δ(MW) với AdamW: ROC gộp |TB| ≤ 0,015; CWE-022 + 079 không nhất quán (≤ 3/5 cùng dấu) - vì bản ASAM đã đảo dấu ở f5.
- Thứ tự fold-major: D f1 → (MW AdamW có, không Pha 1) f1 → … → f5. ~2,7 giờ.
- **11:44:** 161 hết hàng (phép kiểm B xong) ⇒ chuyển f5 của phép kiểm A với AdamW sang 161 (`state/skip.txt` của 158 thêm 2 dòng; train_mwg tất định giữa hai máy, FACTS §86). 158 làm f3, f4; 161 làm f5.

## 02/10 03:2x — XẾP HÀNG (161 + 158): HAI PHÉP KIỂM THEO HƯỚNG CỦA NGƯỜI DÙNG, bậc 2 (n=5, seed 42)

Người dùng 02/10: "chủ động kiểm theo hướng của tôi" + "n=5 if you need" (hướng = tab Kiểm K13: transfer có ích, đặc biệt với đồ thị đồng nhất;
cần nguồn giống nhãn; common gọn hơn full). Thay cho đề xuất P1/P2 ở mục dưới.

**(A) XONG 11:00 (n=5) - tương tác đồ thị × transfer.** Kết quả: MW có - không Pha 1: ROC +0,015 (+4/-1), PR +0,015 (+4/-1), F1@0.5 +0,049 (5/5), F1@val +0,053 (5/5), CWE-022 + 079 +0,338 (5/5), macro +0,160 (5/5) - transfer có ích cả khi KHÔNG có đồ thị. Tương tác Δ(MWG) - Δ(MW): ROC +0,010 (+4/-1), F1@0.5 +0,008 (+3/-2), CWE-022 + 079 +0,021 (+3/-2; theo fold +0,070 / +0,181 / +0,082 / -0,067 / -0,162), macro +0,025 (+3/-2) - cỡ sàn nhiễu, không nhất quán. Thăm dò cùng phép tách với nguồn common 3 ngôn ngữ (không chạy thêm): ROC +0,029 (+4/-1), CWE-022 + 079 -0,099 (+1/-4). **Dự đoán:** A1 ĐÚNG; A2 SAI (MW không Pha 1 cũng xếp ngược CWE-022 + 079, TB 0,323); A3 nửa đúng (ROC gộp +0,010 trong ±0,015, không 5/5 - đúng; CWE-022 + 079 dương chỉ 3/5 - sai). MW (không đồ thị) có / không Pha 1, nguồn common chỉ JS, Pha 2 chỉ ASAM - khớp cặp MWG đã có
(`mwg_assemble_asamonly_jsonly` / `mwg_nop1_asamonly`). Run: `p1_mw_common_jsonly` → `mw_assemble_asamonly_jsonly`; `mw_nop1_asamonly`. Tham số hiệu lực
chỉ khác cặp MWG ở `fusion` (mw / cat); chiều lệch của phép so bắt đúng 3 khoá. Máy: f1-f4 trên 161 (cùng máy với cả hai nhánh MWG của các fold đó),
f5 trên 158 (cùng máy với `mwg_nop1_asamonly` f5). Pha 1 chạy trên 158, `copy_p1_when_done.sh` chép về 161 (đối chiếu md5).
- **Dự đoán (ghi TRƯỚC):** (A1) MW có - không Pha 1: CWE-022 + 079 > 0 ở ≥ 4/5 fold, ROC gộp > 0 ở ≥ 3/5 fold; (A2) MW không Pha 1: CWE-022 + 079 TB
  trong [0,38; 0,58] (gần baseline CodeBERT 0,477) và cao hơn MWG không Pha 1 ở ≥ 4/5 fold ⇒ chính nhánh đồ thị chưa qua Pha 1 làm xếp ngược nhóm này;
  (A3) tương tác Δ(MWG) - Δ(MW): ở CWE-022 + 079 dương ở ≥ 4/5 fold (đồ thị CẦN transfer), ở ROC gộp |TB| ≤ 0,015 và không 5/5 cùng dấu.

**(B) XONG 11:41 (n=5) - cỡ nguồn hay giống nhãn.** Kết quả: 60 cặp - uncommon chặt: ROC -0,001 (+2/-3), PR -0,000 (+2/-3), F1@0.5 -0,004 (+0/-2), F1@val +0,006 (+3/-2), CWE-022 + 079 +0,008 (+2/-3; TB 0,329 so với 0,320), macro +0,008 (+2/-2); dự đoán hai bên tương quan 0,975-0,991 theo fold. common đầy đủ - 60 cặp: F1@0.5 +0,063 (5/5), F1@val +0,046 (5/5), CWE-022 + 079 +0,376 (5/5), macro +0,174 (5/5), ROC +0,009 (+3/-2). **Dự đoán:** B1 ĐÚNG, B2 SAI (⇒ theo luật ghi trước: khoảng cách trong JS chủ yếu do cỡ nguồn), B3 ĐÚNG. Hạn chế: Pha 1 ở 60 cặp chọn epoch 1. `phase1_common_jsonly_n60`: common chỉ JS rút ngẫu nhiên (seed 42) còn 60 cặp, cùng kiểu chia với uncommon chỉ JS
chặt ⇒ cùng 104 hàm train (54 / 50) và 16 hàm val (6 / 10); 58,3 % hàm nhãn 1 mang CWE của SVEN (đầy đủ: 57,8 %); kiểm hai chiều, md5 trùng ở 158.
Run: `p1_mwg_common_jsonly_n60` → `mwg_assemble_common_jsonly_n60` (MWG, RecAdam + ASAM; tham số hiệu lực TRÙNG uncommon chặt, chỉ khác đường dẫn dữ
liệu / checkpoint). Máy: f1, f3, f5 trên 161; f2, f4 trên 158 (cùng máy với uncommon chặt từng fold). Pha 1 trên 158, chép về 161.
- **Dự đoán (ghi TRƯỚC):** (B1) ROC gộp trong ±0,010 so với uncommon chặt, không 5/5 cùng dấu; (B2) CWE-022 + 079 TB ≥ 0,40 và hơn uncommon chặt
  (0,320) ở ≥ 4/5 fold ⇒ giống nhãn có tác dụng kể cả khi cùng cỡ; nếu chênh TB < 0,05 hoặc ≤ 3/5 fold ⇒ khoảng cách common - uncommon trong JS chủ
  yếu do cỡ nguồn; (B3) kém common chỉ JS đầy đủ ở CWE-022 + 079 ở ≥ 4/5 fold (cỡ nguồn cũng có tác dụng).
- **03:50 - lưu ý khi đọc B:** Pha 1 `p1_mwg_common_jsonly_n60` chọn checkpoint epoch 1 (val ROC 0,40 rồi giảm dần tới 0,08), giống hệt
  `p1_mwg_uncommon_jsonly_strict` (epoch 1). Lý do: cả 16 hàm val đều là nửa của một cặp có nửa kia (nhãn NGƯỢC) trong train, nên học càng
  khớp train thì val ROC càng ngược ⇒ chọn theo val luôn rơi vào epoch đầu (13 bước, còn trong warmup). B vì vậy so hai bản Pha 1 "gần như chưa
  học"; nếu B ra "không khác" thì kết luận đúng là ở cỡ 60 cặp quy trình này không để Pha 1 học được, CHƯA phải "nhãn không quan trọng".
  Muốn tách nhãn khỏi cỡ thật sự cần Pha 1 học đủ số bước cố định cho cả hai bộ (src_final chỉ chọn theo val: roc_auc / macro_f1 / pr_auc ⇒
  phải thêm cách chọn "epoch cuối" vào mã) - chưa làm, chờ người dùng.

**Thứ tự (fold-major).** 161, sau hàng đợi hiện tại (MWG không Pha 1 f4 → uncommon C/C++ + JS chặt f2, f4; ~05:30): A f1 (hai nhánh), B f1, A f2, A f3,
B f3, A f4, B f5 (~11:00). 158: Pha 1 uncommon C/C++ + JS chặt (~03:45) → Pha 1 của B (~5 phút) → Pha 1 của A (~25 phút) → Pha 2 uncommon C/C++ + JS
chặt f1, f3, f5 → MWG không Pha 1 f5 → B f2, f4 → A f5 (~09:00). Pha 2 chặt trên 158 lùi ~30 phút để hai Pha 1 mới xong sớm cho 161.
**Sửa 03:09:** hàng đợi 161 cũ sẽ tới uncommon chặt f2/f4 (~03:37) TRƯỚC khi checkpoint Pha 1 chép về (~03:48) ⇒ hai fold bị bỏ qua; đã dừng queue cũ
(PID 3591025, còn đang chờ) và xếp lại: MWG không Pha 1 f4 trước, uncommon chặt f2/f4 sau (PID 3699025).

## 02/10 01:37 — HÀNG ĐỢI KHI RẢNH (người dùng 02/10: "chủ động chạy task khi đang rảnh")

- **161**, sau driver MWG không Pha 1 f1-f3 (PID 3577192, ~04:10): `queue_after` → uncommon C/C++ + JS chặt **f2, f4** (checkpoint Pha 1 do
  `scripts/copy_p1_when_done.sh 158 p1_mwg_uncommon_nojava_strict` tự chép từ 158 khi Pha 1 in "xong rc=0", đối chiếu md5; md5 lệch ⇒ xoá bản chép, fold tự bỏ
  qua) → MWG không Pha 1 **f4** (ASAM, AdamW). Thử hai chiều helper: Pha 1 đã xong ⇒ chép 536 MB trong 51 s, md5 trùng; chưa xong ⇒ chờ (rc 124).
- **158**, sau Pha 2 uncommon C/C++ + JS chặt f1, f3, f5 (queue PID 1532597, ~05:55): MWG không Pha 1 **f5** (ASAM, AdamW).
- **Đề xuất 02/10 02:3x (tab Kiểm, K13), CHƯA xếp hàng, chờ người dùng duyệt:** (P1) tương tác đồ thị × transfer cho luận điểm "đặc biệt với đồ thị": `p1_mw_common_jsonly` (Pha 1 MW không đồ thị trên common chỉ JS) → `mw_assemble_asamonly_jsonly` + `mw_nop1_asamonly`, bậc 1 n=3 (~2 giờ GPU), so với cặp MWG đã có / đang chạy; (P2) common chỉ JS rút ngẫu nhiên còn 60 cặp (bằng cỡ uncommon chỉ JS chặt) để tách cỡ nguồn khỏi giống nhãn, bậc 1 n=3 (~1,5 giờ). Chưa có trong runs.json.
- Leo MWG không Pha 1 lên **bậc 2 (n=5)** vì đây là đối chứng quyết định cho phát biểu về chuyển giao trong bài; dự đoán N1-N3 giữ nguyên, đọc lại ở n=5.
  Dừng: `kill` PID queue_after tương ứng (161: `ps -u ntat -o pid,args | grep queue_after`; 158: tương tự với user tranmanhcuong).

## 02/10 01:3x — XONG 02/10 07:30 (161 f1-f4, 158 f5): MWG KHÔNG PHA 1 (ASAM; AdamW), leo lên bậc 2 (n=5, seed 42)

**Kết quả - có Pha 1 (common chỉ JS) trừ không Pha 1, ghép cặp theo fold, n=5:** ASAM: ROC +0,025 (5/5), PR +0,017 (4/1), F1@0.5 +0,056 (5/5), F1@val +0,056 (5/5), CWE-022 + 079 +0,359 (5/5), ROC macro theo CWE +0,185 (5/5). AdamW: ROC +0,008 (+2/-2), PR +0,008 (+3/-2), F1@0.5 +0,044 (5/5), F1@val +0,078 (5/5), CWE-022 + 079 +0,463 (5/5), macro +0,222 (5/5). MWG không Pha 1 - baseline: ROC +0,007 (ASAM, +4/-1) / +0,002 (AdamW). **Dự đoán:** N1 SAI (uncommon chỉ JS chặt - không Pha 1: ROC +0,015, +4/-1, ngoài ±0,010); N2 đúng phần số (TB CWE-022 + 079 không Pha 1 0,358 ≤ 0,45) nhưng sai phần suy luận "uncommon = không chuyển giao" ở bộ C/C++ + JS chặt (CWE-022 + 079 -0,078, +0/-4; macro -0,033, 0/5); N3 ĐÚNG (AdamW: ROC +0,008 trong ±0,010; CWE-022 + 079 0,321 so với 0,784, 5/5). Chưa lặp seed / loại GPU khác. (f5 chạy khác máy với nhánh có Pha 1, nhưng 10:15 đã kiểm: 161 và 158 cho kết quả train_mwg.py trùng từng bit.)


Người dùng 02/10 (sau "tại sao uncommon vẫn tốt được, trong khi rõ ràng phải có negative transfer"): duyệt "ASAM + AdamW, n=3". Phép kiểm quyết định:
uncommon là KHÔNG chuyển giao hay chuyển giao ÂM.
- **Run:** `mwg_nop1_asamonly` (MWG, `--init none`, Pha 2 AdamW + ASAM ρ 0,5 / η 0,01 - như mwg_assemble_asamonly_jsonly bỏ Pha 1) và `mwg_nop1_adamw`
  (MWG một pha, AdamW thường - như mwg_assemble_noRAS_jsonly bỏ Pha 1). Lệnh chỉ khác hai run đó ở `--init none` / không `--init_ckpt` (đã đối chiếu).
  161, fold-major (asam f1, adamw f1, …), ~3 giờ.
- **Dự đoán (ghi TRƯỚC):** (N1) MWG + ASAM không Pha 1: ROC gộp ≈ uncommon chỉ JS chặt (gần như không có Pha 1): trong ±0,010, không 3/3 cùng dấu;
  (N2) CWE-022 + 079 của MWG + ASAM không Pha 1 ≤ 0,45 (≈ mức ~0,3 của các nguồn không có dữ liệu web) ⇒ uncommon = KHÔNG chuyển giao, phần hơn của common
  chỉ JS ở CWE-022/079 là chuyển giao DƯƠNG; (N3) MWG + AdamW không Pha 1: ROC gộp trong ±0,010 so với train 2 pha thuần common chỉ JS, và CWE-022 + 079
  thấp hơn của train 2 pha thuần (0,784) ⇒ common chỉ JS chuyển giao dương ở CWE-022/079 kể cả khi không có ASAM.

## 02/10 00:3x — XONG 02/10 06:25 (158 f1, f3, f5; 161 f2, f4): UNCOMMON C/C++ + JS CHẶT, bậc 2 (n=5, seed 42)

**Kết quả (ghép cặp theo fold, n=5):** chặt - bản cũ (chỉ hàm có CWE): ROC -0,001 (+3/-2), PR -0,001 (+2/-2), F1@0.5 -0,014 (+1/-4), F1@val -0,006 (+3/-1); chặt - common C/C++ + JS: ROC +0,009 (+4/-1), PR +0,011 (+3/-2), F1@0.5 +0,001 (+3/-2), F1@val +0,034 (+4/-1); chặt - baseline: ROC +0,022 (5/5), PR +0,025 (5/5). CWE-022 + 079 theo fold 0,206 / 0,225 / 0,342 / 0,403 / 0,222 (TB 0,280; bản cũ 0,287; common 0,315). Val Pha 1 0,569 @ep2. **Cả bốn dự đoán V1-V4 ĐÚNG.** Ở nguồn nhiều C/C++, bỏ hẳn mẫu có nhãn common không đổi gì ⇒ giống nhãn không tạo khác biệt (tab Kiểm K13).


Phóng theo dặn của người dùng ("chỉ bù khi máy rảnh trong vòng 30 phút không thấy tôi rep"; "kiểm bộ js trước"): 158 rảnh từ 23:59, tới 00:30
không có trả lời; bộ JS chặt đã có 4/5 fold. Dữ liệu `phase1_uncommon_nojava_strict` (6 138 hàng: C/C++ 3 018 / 3 000, JS 60 cặp). Pha 1
`p1_mwg_uncommon_nojava_strict` trên 158 (~3 giờ), Pha 2 `mwg_assemble_uncommon_nojava_strict` xếp hàng sau trên 158, fold 2, 4 sẽ chuyển sang 161.
- **Dự đoán (ghi TRƯỚC):** (V1) val Pha 1 tốt nhất 0,50-0,60 (bản cũ 0,554); (V2) Pha 2 ≈ bản cũ uncommon_nojava_knowncwe: ROC trong ±0,010, không
  5/5 cùng dấu (phần C/C++ gần như giữ nguyên, JS chỉ bớt 43 cặp); (V3) ROC trên CWE-022 + 079 dưới 0,5 (xếp ngược) ở ≥ 4/5 fold (bản cũ 0,287);
  (V4) không kém common C/C++ + JS về ROC (TB ≥ -0,005; bản cũ +0,010).

## 01/10 22:0x — XONG 02/10 00:33 (bộ JS; bộ C/C++ + JS đang chạy ở mục trên): UNCOMMON CHẶT, chạy BỘ JS TRƯỚC, bậc 2 (n=5, seed 42)

**Kết quả bộ JS chặt (tab Kiểm, K12):** 5/5 ô rc=0 (161 f1/f3/f5, 158 f2/f4). ROC 0,928 (bản cũ 0,918: +0,010, 4/5), PR 0,936, F1@0.5 0,814. So với common chỉ JS: ROC -0,007 (1/5 dương), F1@0.5 -0,059 (0/5); so với baseline: ROC +0,022 (5/5). CWE-022 + 079: 0,320 (xếp ngược 4/5 fold; baseline 0,477, common chỉ JS 0,705; 0/5). ROC macro theo CWE 0,632 < baseline 0,688 (0/5). Dự đoán S1, S2, S3 ĐÚNG. Ghi chú: train 2 pha thuần common chỉ JS có ROC macro cao nhất (0,844) - ASAM nâng 078 + 089 nhưng kéo 022 + 079 xuống.

Người dùng 01/10: "uncommon của JS gồm những CWE nào. Nếu 1 mẫu có nhiều hơn 1 CWE và 1 trong số đó nằm trong common thì phải bỏ nó đi"; nhãn 0
C/C++: "bỏ, chỉ bù khi máy rảnh trong vòng 30 phút không thấy tôi rep"; chạy lại: "kiểm bộ js trước".
- **Dữ liệu** (`tools/v4/build_final_experiment_data.py --uncommon_strict_only`): giữ hàng mà MỌI nhãn CWE (`cwe_labels` đầy đủ trong meta sources_v4)
  là mã CWE thật và KHÔNG áp dụng cho Python theo đúng luật dựng common (`tools/cwe_rules.py` R1-R6). `phase1_uncommon_jsonly_strict` = 60 cặp
  (CWE-1321 56, CWE-843 4; bỏ 42 cặp có nhãn common như CWE-79/78/74/305/200 và 30 cặp nhãn NVD/rỗng); train 104, val 16.
  `phase1_uncommon_nojava_strict` = C/C++ 3 018 nhãn 1 (bỏ 2: CWE-326, 703) / 3 000 nhãn 0 + JS 60 cặp = 6 138 hàng (3 078 / 3 060); train 5 198,
  val 940. **Sửa 22:0x theo người dùng** ("uncommon và common chúng ta chỉ quan tâm label 1 thôi. Với JS thì 1 đi đôi với 0, nhưng với C thì chúng ta
  sampling data 0 cho bằng 1"): bản dựng đầu (4 651 hàng) đã LỌC nhầm nhãn 0 C/C++ theo CWE commit (bỏ 1 372); nay nhãn 0 C/C++ = phần bù 3 000 trong
  6 000 hàm nhãn 0 đã undersample của full, không lọc CWE (bản đầu chưa phóng run nào). Kiểm độc lập theo luật mới: mọi phép kiểm = 0; chiều lệch bắt
  đúng bộ cũ (45 hàm nhãn 1 có nhãn common/unknown, thiếu 115 nhãn 0, thừa 88). Đã chép sang 158 (md5 trùng). Bộ JS chặt không đổi (md5 như cũ).
- **Run:** `p1_mwg_uncommon_jsonly_strict` → `mwg_assemble_uncommon_jsonly_strict` (chỉ khác bản cũ ở --data_root / --init_ckpt);
  `p1_mwg_uncommon_nojava_strict` → `mwg_assemble_uncommon_nojava_strict` (CHƯA phóng - bộ JS trước). Bản "bỏ rồi bù" nhãn 0 C/C++ chỉ dựng + chạy khi
  một máy rảnh ≥ 30 phút mà chưa có trả lời của người dùng.
- **Dự đoán bộ JS chặt (ghi TRƯỚC):** (S1) ROC so với uncommon chỉ JS cũ trong ±0,010, không 5/5 cùng dấu (Pha 1 trên ~100 hàm gần như không dịch
  trọng số ở cả hai bản); (S2) kém common chỉ JS về ROC và F1@0.5 ở ≥ 4/5 fold; (S3) ROC trên CWE-022 + 079 TB ≤ 0,45 (dưới baseline 0,477), vì nguồn
  không còn hàm nào có nhãn liên quan (bản cũ 0,310 còn có CWE-79/78/74).

## 01/10 — XONG 22:40 (161 + 158): TRAIN 2 PHA THUẦN (AdamW) TRÊN NGUỒN CHỈ JS, bậc 2 (n=5, seed 42)

**Kết quả (tab Kiểm, K11):** 10/10 ô rc=0. So với train 2 pha thuần cùng nguồn (common chỉ JS / full chỉ JS): chỉ ASAM ROC +0,023 / +0,026 (5/5),
RecAdam + ASAM +0,020 / +0,030 (5/5), chỉ RecAdam +0,005 / +0,002 (2/5, 3/5); F1@0.5: +0,035 / +0,031, +0,037 / +0,045, +0,011 / +0,013. Train 2 pha
thuần so với baseline: ROC +0,010 (3/5) / -0,000 (2/5), F1@0.5 +0,028 / +0,012. Dự đoán T1, T2, T3 ĐÚNG ở cả hai nguồn. Đọc: phần hơn train 2 pha
thuần là của ASAM; RecAdam không thêm gì; train 2 pha thuần gần ngang baseline về ROC. Câu hỏi mở: MWG + ASAM không Pha 1.

(ghi lúc phóng:)

Người dùng 01/10: "kiểm việc 1 trước" (việc 1 = train 2 pha thuần trên nguồn JS - ô còn thiếu của bảng 2×2 AdamW / chỉ ASAM / chỉ RecAdam /
RecAdam + ASAM; "thế thì khác gì train 2 pha?").
- **Run:** `mwg_assemble_noRAS_jsonly` (Pha 1 p1_mwg_common_jsonly) và `mwg_assemble_noRAS_full_jsonly` (p1_mwg_full_jsonly): Pha 2 AdamW thường
  (`--recadam 0 --sam_rho 0`, flag set plain_adamw như mwg_assemble_noRAS*). Lệnh sinh ra chỉ khác chỉ ASAM ở cờ SAM, chỉ khác chỉ RecAdam ở cờ
  RecAdam, chỉ khác mwg_assemble_noRAS_noccpp ở --init_ckpt (đã đối chiếu bằng `fpe.py plan`).
- **Máy:** 161: common f1, full f1, common f3, full f3, common f5; 158: common f2, full f2, common f4, full f4, full f5 (A4000 trùng bit). ~1,4 giờ.
- **Dự đoán (ghi TRƯỚC):** (T1) train 2 pha thuần ≈ chỉ RecAdam (RecAdam gần như AdamW vì neo nhả trong ~0,3 epoch đầu): ROC trong ±0,010, không
  5/5 cùng dấu, ở cả hai nguồn; (T2) RecAdam + ASAM hơn train 2 pha thuần về ROC ≥ +0,010 ở ≥ 4/5 fold, cả hai nguồn (≈ phần của ASAM: +0,016 /
  +0,028; nguồn nhiều ngôn ngữ: +0,020-0,030); (T3) chỉ ASAM hơn train 2 pha thuần ở ≥ 4/5 fold về ROC (cả hai nguồn).

## 01/10 — XONG 18:39 (161): KIỂM NEO RecAdam VỪA (K9), bậc 1 (n=3: f1-f3, seed 42)

**Kết quả (tab Kiểm, K9):** 6/6 ô rc=0; V1 (t0=85/1710, không ASAM, hp chỉ khác đúng cờ neo, anneal_k 0,02 tra trong hyperparameters), V2 (6/6
trùng), V3 đạt. Neo vừa NGANG neo ngắn: ROC so với neo ngắn t0 5 % +0,0027, nhả chậm +0,0023 (cùng 2/3 fold dương); F1@0.5 +0,011 / -0,002.
Vẫn kém RecAdam + ASAM: F1@0.5 -0,013 / -0,026 (0/3), ROC -0,012 / -0,013. t0 5 % hơn t0 10 % ROC +0,015 (3/3) ⇒ đóng băng quá ~1 epoch là hại.
Đường loss / val của hai cách gần trùng nhau: phần neo rơi vào 3 epoch warmup LR nên lực kéo về neo nhỏ. Dự đoán: t0 5 % (a)(b)(c) ĐÚNG; nhả
chậm (a) SAI (epoch chọn chỉ lùi 1/0/1), (b)(c)(d) ĐÚNG. Gộp K8 + K9: không lịch neo nào (t0 1-30 %, k 0,02-0,05) đưa chỉ RecAdam lên ngang
RecAdam + ASAM. Phân tích `meta/checks/anchor_long.py --set k9` → `anchor_mid.json` / `.txt`; checkpoint 6 ô đã xoá sau khi đối chiếu md5.

(ghi lúc phóng:)

Người dùng 01/10 (sau K8): "kiểm recadam neo vừa thì sao, không quá dài". K8 cho thấy t0 chỉ là độ dài ĐÓNG BĂNG (bước học nhân λ), nên chạy
HAI cách hiểu "neo vừa", cùng common chỉ JS, cùng Pha 1 p1_mwg_common_jsonly, src_final_eptest (lưu test mỗi epoch), stage check:
- `chk_ra_t0p05_jsonly`: --anneal_t0_ratio 0,05 (k 0,05) ⇒ t0 85/1710, λ cuối ep1/2 = 0,20 / 0,81: đóng băng ~1 epoch rồi nhả (96 % LR tới ep16).
- `chk_ra_k0p02_jsonly`: --anneal_k 0,02 + --anneal_t0_ratio 0,05 ⇒ λ = 0,15 lúc đầu, 0,36 / 0,64 / 0,85 / 0,95 cuối ep1-4: học ngay từ đầu, lực kéo về
  neo giảm dần trong ~4 epoch (neo TRONG LÚC học; 94 % LR). Lệnh sinh ra chỉ khác mwg_assemble_recadamonly_jsonly ở các cờ này (+ src/env/stage).
- Máy: 161, thứ tự fold-major (A f1, B f1, A f2, …), ~1,7 giờ ⇒ ~18:50 (trước khi 161 cần cho Pha 2 uncommon).
- **Dự đoán (ghi TRƯỚC):** (A) epoch chọn ≥ neo ngắn ở 3/3 fold; ROC so với neo ngắn TB trong [-0,010; +0,005], không 3/3 cùng dấu (nằm giữa t0 1 %
  và t0 10 %: K8 -0,013); kém RecAdam + ASAM về ROC ≥ 2/3 fold. (B) epoch chọn muộn hơn neo ngắn ≥ 2 epoch ở ≥ 2/3 fold; ROC so với neo ngắn trong
  ±0,010, không 3/3 cùng dấu; kém RecAdam + ASAM về ROC ≥ 2/3 fold; khoảng cách CWE-078 tới RecAdam + ASAM KHÔNG thu hẹp (Δ ≤ -0,02 ở ≥ 2/3 fold).
  Lý do chung: checkpoint Pha 1 gần ngẫu nhiên trên đích (val 0,53-0,65 lúc đóng băng ở K8), nên kéo về nó trong lúc học khó mang thêm gì.

## 01/10 — XONG 21:16 (158 + 161): NGUỒN UNCOMMON C/C++ + JS CHỈ HÀM CÓ CWE, bậc 2 (n=5, seed 42)

**Kết quả:** Pha 1 val tốt nhất 0,554 (epoch 2); Pha 2 5/5 ô rc=0, hparam khớp 48 tham số; f1, f3 trên 158, f2, f4, f5 trên 161 (skip.txt chia
đúng, 158 bỏ qua f2/f4/f5). ROC 0,9207 / 0,9286 / 0,9238 / 0,9558 / 0,9142. So với **common C/C++ + JS** (mwg_assemble_nojava): ROC +0,0101
(3/5 dương, 1 hoà), PR +0,0121 (3/5), F1@0.5 +0,0154 (4/5), F1@val +0,0403 (5/5) ⇒ uncommon KHÔNG kém common ở cùng ngôn ngữ, cỡ gần bằng.
So với baseline: ROC +0,0230 (5/5), PR +0,0260 (5/5), F1@0.5 +0,0150 (3/5). So với common chỉ JS: ROC -0,0068 (1/5), F1@0.5 -0,0492 (0/5).
Theo CWE so với common C/C++ + JS: mọi CWE gần 0 (022 -0,037, 078 +0,014, 079 -0,029, 089 +0,010 5/5); so với common chỉ JS: CWE-022 -0,461,
CWE-079 -0,375 (cả hai 0/5) ⇒ phần hơn của common chỉ JS nằm ở CWE-022/079 (JS giàu XSS / path traversal), không ở 078/089. Dự đoán: U1 ĐÚNG;
U2 SAI (ngược chiều); U3 SAI (hơn baseline 5/5 về ROC); rủi ro Pha 1 sập không xảy ra. Đọc: với nguồn C/C++ + JS, lọc CWE "áp dụng cho Python"
ở phần C/C++ KHÔNG mang thêm gì; lợi ích "đúng target" chỉ thấy ở common chỉ JS và chỉ ở CWE-022/079 (nhóm nhỏ: 8-20 hàm mỗi fold).
**Phân tích K10 (tab Kiểm, người dùng: "tại sao uncommon gần đạt sota, trong khi đây là negative transfer"):** ROC gộp bị CWE-089 + 078 (81 % test) chi phối, nơi uncommon tốt nhất (0,979); trên CWE-022 + 079 uncommon xếp NGƯỢC (0,287; baseline 0,477, SOTA 0,66-0,72; 0/5); toàn bộ khoảng cách ROC gộp tới SOTA (-0,0137) đến từ các khối cặp có CWE-022/079. ROC macro theo CWE: uncommon 0,612 < baseline 0,688 (0/5) < SOTA 0,80-0,81. Thêm C/C++ vào common chỉ JS: 022 + 079 -0,390 (0/5); thêm Java vào common C/C++ + JS: +0,403 (5/5). `_FinalPaperExperiment/meta/checks/uncommon_why.py` → `uncommon_why.json` / `.txt`.

(ghi lúc phóng:)

Người dùng 01/10: "chạy thêm bản uncommon của CCPP kèm với uncommon của js (nhưng không bao gồm các mẫu unknown cwe)".
- **Dữ liệu:** `data/final_experiment_data/phase1_uncommon_nojava_knowncwe/` (dựng bằng `tools/v4/build_final_experiment_data.py
  --uncommon_nojava_knowncwe_only`) = hàng của `phase1_full_nojava` KHÔNG có trong `phase1_common_nojava` (so nguyên văn code), bỏ hàng
  `cwe` rỗng (115 hàm nhãn 0 C/C++, 29 cặp JS), giữ train/val của full, gỡ pair_id lẻ. 6 111 hàm (3 123 / 2 988): C/C++ 3 020 / 2 885,
  JS 103 / 103; train 5 179, val 932; 1 298 cặp đủ. C/C++ nhãn 0 không lọc theo CWE (hàm sạch) nên là nửa ngẫu nhiên còn lại của full.
  Kiểm độc lập: 9 phép kiểm = 0 (đủ/thừa so với kỳ vọng, cách chia, test = val, pair_id); chiều lệch bắt đúng (uncommon chỉ JS: 58 hàng
  không CWE; common_nojava: 6 858 hàng trùng common). Đã chép sang 158, md5 trùng 5/5 file.
- **Run:** `p1_mwg_uncommon_nojava_knowncwe` (Pha 1, như p1_mwg_common_nojava, chỉ khác --data_root) → `mwg_assemble_uncommon_nojava_knowncwe`
  (Pha 2 RecAdam + ASAM, như mwg_assemble_nojava, chỉ khác --init_ckpt; đã đối chiếu lệnh sinh ra bằng `fpe.py plan`).
- **Máy:** Pha 1 trên 158 (~3 giờ, như common không Java 12 000 s); Pha 2 fold 1, 3, 5 xếp hàng sau trên 158; fold 2, 4 trên 161 sau khi chép
  checkpoint Pha 1 (A4000 trùng bit giữa hai máy).
- **Dự đoán (ghi TRƯỚC):** (U1) val ROC Pha 1 tốt nhất 0,50-0,60 như common / full không Java (0,577 / 0,531); (U2) so với
  mwg_assemble_nojava (common C/C++ + JS, 6 858 hàm - cùng ngôn ngữ, cỡ gần bằng): ROC thấp hơn ở ≥ 4/5 fold, TB ≤ -0,010, và F1@0.5 thấp
  hơn ở ≥ 4/5 fold (uncommon chỉ JS đã cho ROC -0,017 4/5, F1@0.5 -0,067 5/5 nhưng lẫn cỡ dữ liệu 264 hàm; ở đây cỡ đã khớp); (U3) so với
  baseline: không hơn, ROC TB trong [-0,03; +0,01]. Rủi ro: Pha 1 sập như full không Java (Pha 2 Δ tới -0,45 ở 3 fold) - khi đó phép so đo
  "Pha 1 sập" chứ không đo "CWE đúng target"; vẫn chạy đủ, ghi val Pha 1 (CLAUDE.md §3).

## 01/10 — XONG 16:30 (161 + 158): KIỂM NEO RecAdam DÀI HƠN, bậc 1 (n=3: f1-f3, seed 42)

**Kết quả (tab Kiểm, K8):** 6/6 ô rc=0; hợp lệ V1 (dòng "RecAdam bat | cof=500 | t0=171/1710" / "t0=513/1710", không ASAM, hp chỉ khác
anneal_t0_ratio), V2 (xác suất test ở epoch chọn trùng kết quả cuối 6/6), V3 (nhãn khớp test.jsonl). **Neo của RecAdam là công tắc:** bước Adam
của nhiệm vụ đích nhân λ(t), λ đi 5 % → 95 % trong 118 bước (~2 epoch) bất kể t0 ⇒ trước t0 mô hình ĐỨNG YÊN (t0 30 %: val loss / val ROC y hệt
nhau 6 epoch đầu), t0 30 % chỉ còn 40 % ngân sách LR tới ep16 cho nhiệm vụ đích. Dự đoán: (a) t0 30 % ĐÚNG (epoch chọn 14/14/14 so với 8/5/5),
t0 10 % SAI ở f1 (8/8/7); (b) SAI cả hai, lệch về phía KÉM: ROC so với neo ngắn -0,0127 / -0,0204 (1/3 fold dương), PR -0,015 / -0,036;
(c) ĐÚNG cả hai: kém RecAdam + ASAM ở cả bốn chỉ số 3/3 fold (ROC -0,028 / -0,035). Theo CWE so với RecAdam + ASAM: CWE-022 +0,14, CWE-078
-0,06 / -0,05 (0/3), CWE-089 -0,02 / -0,04 (0/3) ⇒ neo dài không thu hẹp khoảng cách ở 078/089. Muốn thử neo TRONG LÚC HỌC thì đổi độ dốc k
(vd 0,005) hoặc phạt tách rời kiểu L2-SP - chưa chạy, cần hỏi. Phân tích `meta/checks/anchor_long.py` → `anchor_long.json` / `.txt`;
số theo epoch `results/chk_ra_t0p{1,3}_jsonly/ep_test/fold<k>/`; checkpoint của 6 ô đã xoá sau khi đối chiếu md5 bản chép.

(ghi lúc phóng:)

Người dùng 01/10: "kiểm với lực neo dài hơn" (sau khi tóm: RecAdam gần như không thêm gì vì neo tắt sau ~2 epoch: t0 = 17/1710, k = 0,05).
- **Run (stage check, tab Kiểm K8):** `chk_ra_t0p1_jsonly` (--anneal_t0_ratio 0,1 ⇒ t0 = 171/1710, neo giữ tới ~epoch 3-4) và `chk_ra_t0p3_jsonly`
  (0,3 ⇒ t0 = 513/1710, tới ~epoch 9-10; phủ quãng ep5-8 chỉ RecAdam đạt đỉnh). Đổi ĐÚNG một biến so với mwg_assemble_recadamonly_jsonly
  (γ 500, sigmoid k 0,05, không ASAM; chạy khô chỉ khác anneal_t0_ratio + thư mục mã / tên / FPE_EPOCH_TEST). src_final_eptest (lưu test mỗi epoch).
  161: f1, f3 (cả hai mức mỗi fold); 158: f2.
- **Dự đoán (ghi TRƯỚC):** (a) neo dài hơn làm epoch được chọn MUỘN hơn (t0 0,3: ≥ 3 epoch so với 8/5/5); (b) ROC so với chỉ RecAdam neo ngắn trong
  [-0,010; +0,010], không 3/3 cùng dấu (Pha 1 yếu - val nguồn ~0,55 - nên giữ gần nó lâu hơn không mang thêm gì); (c) vẫn kém RecAdam + ASAM về
  ROC ở ≥ 2/3 fold. Rủi ro đã biết: neo dài + patience 8 có thể dừng sớm đúng lúc neo vừa nhả (đọc log từng epoch trước khi đọc số).

## 01/10 11:14 — XONG (161 + 158): KIỂM CHECKPOINT SỚM, bậc 1 (n=3: f1-f3, seed 42)

**Kết quả (tab Kiểm, K7):** 6/6 ô rc=0, TRÙNG run gốc ở val mọi epoch, chỉ số test và từng xác suất test tại epoch chọn (V1, V2 đạt). Giả thuyết checkpoint sớm BỊ BÁC: (P1) RecAdam + ASAM tại epoch chỉ RecAdam chọn so với epoch tự chọn: CWE-078 -0,042 (0/3 dương), CWE-089 -0,129 (0/3), CWE-022 -0,052 (1/3) ⇒ phần 022 SAI; (P2) cùng epoch, chênh CWE-078 nhỏ hơn 2/3 nhưng ở CWE-089 và ROC chung cùng epoch chênh LỚN hơn (RecAdam + ASAM chưa học kịp: ROC chung -0,040 / -0,182 / -0,162); (P3) chỉ RecAdam học tiếp thì CWE-078 GIẢM 0,036-0,068 (0/3) ⇒ SAI. Đọc: không ASAM thì đạt đỉnh sớm rồi quá khớp; ASAM học chậm hơn nhưng tới đỉnh cao hơn ở CWE-089 (3/3) và 078 (2/3). CWE-022: RecAdam + ASAM kém ở mọi điểm so được (3/3 fold, cả cùng epoch lẫn epoch tự chọn) - khớp giả thuyết ASAM bỏ rơi nhóm ít dữ liệu (kiểu ImbSAM), nhưng n=3 và 8-19 hàm mỗi fold. Số theo epoch: `results/chk_ep_*/ep_test/fold<k>/epNN.npz`, phân tích `meta/checks/ckpt_early.py` → `ckpt_early.json`.

(ghi lúc phóng:)

Người dùng 01/10: "kiểm check point sớm, các lần kiểm tóm tắt bằng 1 2 câu. Bổ sung thêm 1 tab kiểm." Câu hỏi: chỉ RecAdam (và AdamW) tốt hơn ở
CWE-022 nhưng kém ở CWE-078/089 so với RecAdam + ASAM - do checkpoint được chọn SỚM (ep5-8 so với 9-28) hay do ASAM đổi cái được học?
- **Mã:** `src_final_eptest` = `src_final` + train_mwg.py: FPE_EPOCH_TEST=1 ⇒ cuối mỗi epoch chấm thêm test, lưu `<checkpoint>/ep_test/epNN.npz`
  (xác suất test + val của epoch đó); lưu rồi khôi phục MỌI RNG quanh phần thêm. Thử khói hai chiều (24 hàm, 3 epoch): bật biến + khôi phục
  trùng từng số với không bật; bản cố ý bỏ khôi phục LỆCH từ epoch 2 (train loss 0,6922 so với 0,6516) ⇒ khôi phục là cần và đủ.
- **Run (stage check, không hiện ở bảng run):** `chk_ep_rasam_jsonly` (= mwg_assemble_jsonly), `chk_ep_recadamonly_jsonly`
  (= mwg_assemble_recadamonly_jsonly); chạy khô: chỉ khác thư mục mã, run_name, đường dẫn và FPE_EPOCH_TEST=1. Pha 1 p1_mwg_common_jsonly
  md5 659e809f… trùng hai máy; mã md5 trùng hai máy. 161: f1, f3 (cả hai nhánh mỗi fold); 158: f2.
- **Kiểm hợp lệ (ghi TRƯỚC):** (V1) val ROC mọi epoch trùng log run gốc; (V2) test ROC ở epoch chọn trùng kết quả gốc, và xác suất test ở
  epoch chọn trùng probs.npz gốc. Lệch ⇒ phần thêm làm đổi quỹ đạo ⇒ dừng, không đọc số.
- **Dự đoán (ghi TRƯỚC; giả thuyết H = checkpoint sớm):** (P1) ở nhánh RecAdam + ASAM, tại epoch mà chỉ RecAdam đã chọn (f1 8, f2 5, f3 5) so
  với epoch nó tự chọn (13, 13, 9): ROC trong CWE-078 và CWE-089 THẤP hơn, CWE-022 CAO hơn, mỗi điều ở ≥ 2/3 fold; (P2) cùng một epoch, chênh
  CWE-078 giữa hai nhánh NHỎ hơn chênh giữa hai nhánh ở epoch mỗi bên tự chọn, ≥ 2/3 fold; (P3) nhánh chỉ RecAdam: ROC CWE-078 ở epoch cuối
  (trước dừng sớm) ≥ ở epoch nó chọn, ≥ 2/3 fold. Giả thuyết đối H' (ASAM đổi cái được học): cùng epoch mà RecAdam + ASAM đã hơn ở CWE-078 và
  kém ở CWE-022. CWE-022 chỉ 8-19 hàm mỗi fold ⇒ đọc CWE-078/089 là chính.

## 30/09 18:19 — XONG (161 + 158): full setting với nguồn UNCOMMON CHỈ JS (= full chỉ JS bỏ common), bậc 2 (n=5, seed 42)

**Đối chiếu:** 6/6 ô kỳ vọng (Pha 1 + Pha 2 f1-f5) `xong rc=0`; hp khớp 48 tham số cả 5 fold; driver 161 in `FPE 161 XONG 18:19:52`. 161 TRỐNG từ 18:20 (không tự thêm việc - chờ người dùng quyết các đối chứng đề xuất).

| uncommon chỉ JS − | F1@0,5 | F1@val | ROC | PR |
|---|---|---|---|---|
| common chỉ JS | -0,0671 (0/5) | -0,0489 (0/5) | -0,0171 (1/5) | -0,0135 (1/5) |
| full chỉ JS | -0,0595 (0/5) | -0,0487 (1/5) | -0,0169 (1/5) | -0,0143 (1/5) |
| common | -0,0553 (0/5) | -0,0328 (1/5) | -0,0139 (1/5) | -0,0074 (2/5) |
| baseline | -0,0029 (1/5) | -0,0064 (2/5) | +0,0128 (5/5) | +0,0179 (5/5) |

ROC trong CWE-079: uncommon TB 0,358 (0,388/0,278/0,444/0,389/0,291); − common chỉ JS -0,355 (0/5); − baseline -0,133 (1/5) ⇒ uncommon mất hẳn phần hơn ở XSS, còn kém cả baseline. Dự đoán: (a) Pha 1 val ~0,5 - ĐÚNG (0,458, 35 hàm val); (b) − common chỉ JS |Δ ROC| < 0,010, không 5/5 cùng dấu - SAI về độ lớn (-0,017; F1@0,5 và F1@val 0/5); (c) F1@0,5 > baseline ≥ 4/5 - SAI (1/5, gần như bằng baseline). Đọc: phần JS KHÔNG đúng target (264 hàm) không cho lợi F1 nào so với baseline, trong khi common chỉ JS (990 hàm) cho +0,064 (5/5) - nhưng lẫn yếu tố cỡ; cần đối chứng 264 hàm bốc ngẫu nhiên từ common chỉ JS (chưa chạy, chờ người dùng).

**Lưu ý 18:5x (phát hiện khi kiểm nhận định):** Pha 1 uncommon CHƯA cất cánh - train loss 1,92 → 1,50 sau 8 epoch, trong khi các nguồn JS khác cất cánh ở ep4-5 (common chỉ JS 1,87 → 0,69; 4cwe chỉ JS 1,82 → 0,82; full chỉ JS 1,87 → 0,51). Pha 1 chạy cố định 8 epoch nên 229 hàm train nhận ít bước hơn common chỉ JS (844) khoảng 3,7 lần ⇒ kết quả uncommon lẫn BA yếu tố: độ liên quan CWE, cỡ, và số bước Pha 1. Đối chứng 264 hàm bốc ngẫu nhiên từ common chỉ JS (cùng cỡ, cùng số bước) tách được hai yếu tố sau khỏi yếu tố đầu.

(ghi lúc chạy:)

**16:07 đổi theo người dùng:** "Đặt tên cái không common là bộ uncommon và chạy ưu tiên cái này, xếp nó vào 1 máy trống và có thể nhanh chóng kết thúc run hiện tại để chạy ưu tiên cái này ngay sau." ⇒ (1) ĐỔI TÊN (chưa ô nào chạy bằng tên cũ): dữ liệu `phase1_full_jsonly_nocommon` → `phase1_uncommon_jsonly` (mv ở 161 và 158, README + bảng md5 dựng lại, builder cờ `--uncommon_jsonly_only`), nguồn `uncommon_jsonly` ("uncommon chỉ JS"), run `p1_mwg_uncommon_jsonly` → `mwg_assemble_uncommon_jsonly`; effective vẫn khác full chỉ JS ĐÚNG data_root / init_ckpt; 158 md5 6/6. (2) DỪNG trên 161 driver 1201199 + trainer 1265240 (chỉ RecAdam f5 common mới chạy 1,5 phút, bỏ); gỡ queue_after 1007512 (tên cũ) trên 158. (3) 161 driver 1268768 (`state/driver_161_uncommon.out`) 16:09: Pha 1 → f1, f3, f5. 158: khi Pha 1 xong ⇒ chép checkpoint 161 → 158 + md5 ⇒ queue_after driver 888381 (chỉ RecAdam f4 full): uncommon f2, f4 ⇒ RỒI MỚI chạy bù chỉ RecAdam f5 (common + full) đã hoãn.
- **16:15 Pha 1 uncommon XONG (161, 6 phút):** chọn ep7, val ROC 0,458 (ep1–7: 0,363 · 0,307 · 0,346 · 0,373 · 0,405 · 0,448 · 0,458; val 35 hàm). Checkpoint c6311827… chép 161 → 158, md5 khớp. 161 chạy tiếp uncommon f1 → f3 → f5. 158 queue_after PID 1048238 (`state/queue_158_uncommon.out`, preflight đủ 4 mục) chờ driver 888381 ⇒ uncommon f2, f4 ⇒ chỉ RecAdam f5 common + full (chạy bù).
- 16:1x phiên Claude Code khởi động lại: monitor nền dừng, scratchpad cũ không dùng nữa ⇒ script monitor chuyển vào `scripts/mon_fpe_v4.sh` (trạng thái `state/monitor/`), bản nhận định vào `meta/review/`. Driver trên máy KHÔNG bị ảnh hưởng.
- **17:05 f1 XONG (161, rc=0, 50 phút):** chạy đủ 30 epoch, chọn ep22 (val ROC 0,949), ngưỡng val 0,41; hp khớp 48 tham số. Test: F1@0,5 0,8082 · F1@val 0,8078 · ROC 0,9171 · PR 0,9353. Δ (`scripts/pair_delta.py`, n=1): − common chỉ JS −0,0579 / −0,0513 / −0,0082 / −0,0091; − full chỉ JS −0,0724 / −0,0379 / −0,0066 / −0,0025; − baseline −0,0075 / −0,0142 / +0,0103 / +0,0079 (F1@0,5 ÂM ⇒ ngược dự đoán (c) ở fold này). Theo CWE (ROC/F1): CWE-079 0,388/0,497 (common chỉ JS 0,673/0,785, baseline 0,306/0,429), CWE-022 0,222/0,208 (common 0,500/0,269); CWE-078/089 ngang. Mới 1 fold, tập CWE-079 chỉ 14 hàm ⇒ chưa kết luận; đợi n=5.
- **17:15 f2 XONG (158, rc=0, 45 phút):** chọn ep18 (val ROC 0,919), hp khớp 48 tham số. Test: F1@0,5 0,7763 · F1@val 0,7858 · ROC 0,9127 · PR 0,9298. Δ n=2 (f1, f2): - common chỉ JS F1@0,5 -0,0881 (0/2) · F1@val -0,0703 (0/2) · ROC -0,0172 (0/2) · PR -0,0172 (0/2); - baseline F1@0,5 -0,0167 (0/2) · ROC +0,0154 (2/2). CWE-079 ROC 0,278 (common chỉ JS 0,678, full chỉ JS 0,778, baseline 0,633; 19 hàm); CWE-022 0,000 (8 hàm). Dự đoán (b) |Δ ROC| < 0,010 đang lệch (-0,017), (c) F1@0,5 dương đang 0/2.
- **17:36 f3 XONG (161, rc=0, 31 phút):** dừng sớm ep20, chọn ep12 (val ROC 0,946), hp khớp 48 tham số. Test: F1@0,5 0,8085 · F1@val 0,8132 · ROC 0,9102 · PR 0,9062. Δ n=3: - common chỉ JS F1@0,5 -0,0809 (0/3) · F1@val -0,0585 (0/3) · ROC -0,0213 (0/3) · PR -0,0212 (0/3); - baseline F1@0,5 -0,0026 (1/3) · ROC +0,0124 (3/3) · PR +0,0191 (3/3). CWE-079 ROC 0,444 (common chỉ JS 0,778; 20 hàm) ⇒ 3/3 fold CWE-079 của uncommon thấp hơn common chỉ JS (0,388/0,278/0,444 so với 0,673/0,678/0,778). 161 chạy tiếp f5.
- **17:55 f4 XONG (158, rc=0, 40 phút):** chọn ep16 (val ROC 0,909), hp khớp 48 tham số. Test: F1@0,5 0,8417 · F1@val 0,8282 · ROC 0,9562 · PR 0,9540. Δ n=4: - common chỉ JS F1@0,5 -0,0641 (0/4) · F1@val -0,0523 (0/4) · ROC -0,0126 (1/4; f4 +0,0134) · PR -0,0114 (1/4); - baseline F1@0,5 -0,0036 (1/4) · ROC +0,0122 (4/4) · PR +0,0177 (4/4). CWE-079 ROC 0,389 (common chỉ JS 0,889; 13 hàm) ⇒ 4/4 fold thấp hơn. ROC co lại từ -0,021 (0/3) xuống -0,013 (1/4) khi thêm fold; F1 vẫn 0/4. 158 chạy tiếp chỉ RecAdam f5.

(ghi cũ lúc xếp hàng, tên cũ:)

Người dùng 30/09: "Thêm giúp tôi 1 thí nghiệm n=5 chạy thử full setting nhưng dùng bộ full bỏ common để đối chứng, tức full loại đi các sample có trong common" rồi "Chỉ thử với ngôn ngữ JS thôi".
- **Dữ liệu** (`build_final_experiment_data.py --jsonly_nocommon_only`, hàm `build_minus`): `phase1_full_jsonly_nocommon` = hàng của `phase1_full_jsonly` KHÔNG có trong `phase1_common_jsonly` (so nguyên văn `code`; bỏ khoảng trắng cho cùng 264), không chia lại: **264 hàm (132 / 132), train 229 / val 35, 132 cặp đủ, 33 cặp bị tách**. common JS ⊂ full JS (990/990) nên phép trừ không cắt cặp nào. CWE: 1321 (prototype pollution) 112, không CWE 58, 14 CWE nhỏ; 19 cặp mang CWE 79/78/74/20 vẫn nằm ngoài common vì luật common đòi MỌI nhãn CWE áp dụng được cho Python (README clean_sources_v4). 91 file cũ trùng md5 trước/sau; README chỉ thêm 17 dòng; 158 md5 6/6 (4 file + README + runs.json).
- **Run:** `p1_mwg_full_jsonly_nocommon` (effective khác `p1_mwg_full_jsonly` ĐÚNG data_root) → `mwg_assemble_full_jsonly_nocommon` (khác `mwg_assemble_full_jsonly` ĐÚNG init_ckpt); plan khô 6/6 ô khác đúng một cờ. runs.json 60 run.
- **Máy:** 158 queue_after PID 1007512 (`state/queue_158_nocommon.out`) chờ driver 888381 (chỉ RecAdam f4) ⇒ Pha 1 (~10 phút) ⇒ Pha 2 f2, f4. 161: SAU khi Pha 1 xong — chép checkpoint 158 → 161, đối chiếu md5, rồi mới xếp f1, f3, f5 (tránh run.sh bỏ fold vì thiếu checkpoint).
- **Dự đoán ghi TRƯỚC:** (a) Pha 1 val ROC ~0,5 (cặp lỗi / vá gần như không tách được, như các nguồn JS khác); (b) − common chỉ JS (`mwg_assemble_jsonly`): |Δ ROC| TB < 0,010, không 5/5 cùng dấu (4cwe JS 588 hàm ≈ common JS 990 ≈ full JS 1 254 ⇒ mức liên quan CWE với Python không quyết định); (c) − baseline: F1@0,5 dương ≥ 4/5 fold.

---

## 30/09 18:25 — XONG (161 + 158): CHỈ RecAdam (tắt ASAM) cho hai nguồn chỉ JS, bậc 2 (n=5, seed 42)

**Đối chiếu:** 10/10 ô kỳ vọng (2 nguồn × 5 fold) `xong rc=0`, hp khớp 48 tham số mọi ô; 158 in `FPE 158 XONG 18:25:17`. Hai máy TRỐNG từ 18:25.

- **18:25 full chỉ JS ĐỦ n=5** (f5 158, 14 phút, ep7). Test f5: ROC 0,8861.

  | chỉ RecAdam (full chỉ JS) − | F1@0,5 | F1@val | ROC | PR |
  |---|---|---|---|---|
  | RecAdam + ASAM | -0,0318 (0/5) | -0,0274 (0/5) | -0,0284 (0/5) | -0,0365 (0/5) |
  | chỉ ASAM | -0,0186 (0/5) | -0,0214 (0/5) | -0,0247 (0/5) | -0,0322 (0/5) |
  | baseline | +0,0247 (4/5) | +0,0149 (3/5) | +0,0012 (2/5) | -0,0043 (1/5) |

  Dự đoán ở full chỉ JS: (a) SAI (ROC -0,028, 0/5), (b) SAI (cả bốn chỉ số 0/5 so với chỉ ASAM), (c) ĐÚNG (F1@0,5 4/5). ⇒ LẶP ở nguồn thứ hai: bỏ ASAM thì mất ở cả bốn chỉ số; trên full chỉ JS là 5/5 fold ở cả bốn. Phần lợi của Pha 2 đi cùng ASAM, RecAdam một mình không thêm gì (còn gần bằng baseline về ROC).

- **01/10 theo CWE (người dùng hỏi "chỉ RecAdam cứu CWE-022 nhưng tệ ở CWE khác"):** Δ chỉ RecAdam − RecAdam + ASAM, ROC trong CWE: CWE-022 +0,079 (3/5) / +0,020 (3/5) (common chỉ JS / full chỉ JS), CWE-078 -0,032 (1/5) / -0,079 (0/5), CWE-089 -0,012 (0/5) / -0,011 (0/5), CWE-079 +0,101 (4/5) / -0,011 (1/5). AdamW (noRAS, nguồn common) − RecAdam + ASAM cho CÙNG mẫu hình: CWE-022 +0,051 (3/5), CWE-078 -0,052 (0/5), CWE-089 -0,010 (2/5) ⇒ không phải tác dụng riêng của RecAdam mà của việc THIẾU ASAM (neo RecAdam tắt sau ~2 epoch: t0 = 17/1710 bước, k = 0,05). Epoch được chọn: không ASAM 5-8 (chỉ RecAdam JS [8,5,5,7,6], AdamW [9,8,6,9,8]) so với có ASAM 9-28. SVEN train mỗi fold: CWE-089 245, CWE-078 123, CWE-079 49, CWE-022 39. Giả thuyết: không ASAM thì val đạt đỉnh sớm ⇒ checkpoint sớm, ít khớp các CWE đông (078/089); phần huấn luyện thêm nhờ ASAM chủ yếu vào nhóm đông, nhóm ít (022) không hưởng (tương tự ImbSAM, ICCV 2023: SAM không giúp lớp đuôi). CWE-022 chỉ 8-19 hàm test mỗi fold ⇒ phần "cứu" nằm trong nhiễu; phần mất ở CWE-078 chắc hơn.

- **18:10 common chỉ JS ĐỦ n=5** (f5 trên 158, rc=0, 14 phút, ep6, hp khớp 48 tham số; log: `RecAdam bat | cof=500`, không có `SAM bat`). Test f5: F1@0,5 0,8421 · ROC 0,9207 · PR 0,9392.

  | chỉ RecAdam (common chỉ JS) − | F1@0,5 | F1@val | ROC | PR |
  |---|---|---|---|---|
  | RecAdam + ASAM | -0,0260 (0/5) | -0,0128 (1/5) | -0,0156 (1/5) | -0,0227 (1/5) |
  | chỉ ASAM | -0,0245 (0/5) | -0,0206 (2/5) | -0,0183 (0/5, 1 hoà) | -0,0158 (1/5) |
  | baseline | +0,0382 (4/5) | +0,0297 (4/5) | +0,0142 (3/5) | +0,0087 (2/5) |

  Dự đoán: (a) ROC TB trong [-0,010; +0,005] - **SAI** (-0,0156; F1@0,5 còn 0/5); (b) |Δ| < 0,010 với chỉ ASAM - **SAI** (cả bốn chỉ số -0,016 đến -0,025, F1@0,5 0/5); (c) F1@0,5 > baseline ≥ 4/5 - **ĐÚNG** (4/5). ⇒ ở nguồn common chỉ JS, phần lợi của Pha 2 đi cùng ASAM; chỉ RecAdam kém cả RecAdam + ASAM lẫn chỉ ASAM. full chỉ JS f5 đang chạy trên 158.

Driver: 161 PID 1201199 (`state/driver_161_recadamonly.out`), 158 PID 888381 (`state/driver_158_recadamonly.out`); trước khi phóng mỗi máy 0 driver / 0 trainer, lock trống; sau khi phóng lock GIỮ thật. Cấu hình hiệu lực đọc từ log tiến trình cả hai máy: `RecAdam bat | cof=500 | t0=17/1710`, KHÔNG có dòng `SAM bat` (run RecAdam + ASAM cũ có `SAM bat | variant=asam rho=0.5`), nạp đúng p1_mwg_common_jsonly; epoch 1 49 s (có ASAM ~90 s).

Người dùng 30/09: "với các thí nghiệm ASAM only, chạy thêm recadam only cho cùng cấu hình".
- **Run:** `mwg_assemble_recadamonly_jsonly` (Pha 1 `p1_mwg_common_jsonly`) và `mwg_assemble_recadamonly_full_jsonly` (Pha 1 `p1_mwg_full_jsonly`); flag set `recadam_only` = `recadam_asam` với `--sam_rho 0` (sam_variant / sam_eta giữ, trơ khi rho = 0). effective_args khác RecAdam + ASAM ĐÚNG `sam_rho 0,5 → 0` (+ tên / đường dẫn); plan khô 10/10 ô chỉ khác `--sam_rho`, init trùng; chiều lệch (so với chỉ ASAM) bắt được. Checkpoint Pha 1 trùng md5 trên 161 và 158 (659e809f… / 7cd4d868…). runs.json 58 run, rsync 158 md5 khớp; preflight hai máy đủ.
- **Máy (chia theo FOLD TRỌN VẸN, fold-major):** 161 f1, f3, f5 (cả hai nguồn mỗi fold); 158 f2, f4. Các A4000 trùng bit nên ghép cặp với run chỉ ASAM / RecAdam + ASAM (chạy ở 161) hợp lệ. Không ASAM ⇒ mỗi epoch ~một nửa thời gian.
- **Dự đoán ghi TRƯỚC:** (a) chỉ RecAdam − RecAdam + ASAM: ROC TB trong [−0,010; +0,005], không 5/5 cùng dấu (ASAM đóng góp ít, nếu có thì ở thứ hạng, CLAUDE.md §2b); (b) chỉ RecAdam − chỉ ASAM: |Δ| TB < 0,010 cả bốn chỉ số, lẫn dấu; (c) chỉ RecAdam − baseline: F1@0,5 dương ≥ 4/5 fold ở cả hai nguồn (lợi ích đến từ chuyển giao, không từ bộ tối ưu).

---

## 30/09 09:45 — XONG (158): full chỉ C/C++ LẦN 2, seed 1234 cả hai pha, ĐỦ n=5 · 0/5 SỤP ⇒ dự đoán (b) SAI

Bậc 2 (n=5), seed 1234, một máy (158). Đối chiếu: 6/6 ô kỳ vọng (Pha 1 + Pha 2 f1–f5) `xong rc=0`, không còn driver/hàng đợi; hp cả 5 fold seed 1234, 48 tham số khớp runs.json; kết quả + log đã `sync_158.sh` về 161.

| ROC | f1 | f2 | f3 | f4 | f5 |
|---|---|---|---|---|---|
| seed 1234 | 0,900 (ep16) | 0,922 (ep19) | 0,902 (ep16) | 0,944 (ep19) | 0,910 (ep27) |
| seed 42 | 0,922 (ep23) | **0,475 (ep2, sụp)** | **0,500 (ep2, sụp)** | 0,949 (ep27) | 0,894 (ep30) |
| baseline | 0,907 | 0,892 | 0,904 | 0,945 | 0,881 |

| s1234 − | F1@0,5 | F1@val | ROC | PR |
|---|---|---|---|---|
| baseline | −0,016 (2+/3−) | −0,005 (1+/3−/1=) | +0,010 (2+/2−/1=) | +0,014 (4+/1−) |
| full chỉ Java (seed 42) | −0,074 (0/5) | −0,038 (0/5) | −0,021 (0/5) | −0,019 (0/5) |
| nguồn full đầy đủ `mwg_assemble_full` (seed 42) | −0,070 (0/5) | −0,052 (0/5) | −0,027 (0/5) | −0,025 (0/5) |
| nguồn common đầy đủ `mwg_assemble` (seed 42) | −0,068 (0/5) | −0,032 (1+/4−) | −0,017 (0+/4−/1=) | −0,012 (0+/4−/1=) |
| common chỉ C/C++ (seed 42) | −0,007 (2+/2−/1=) | +0,004 (3+/2−) | −0,000 (2+/2−/1=) | −0,001 (2+/3−) |
| seed 42, chỉ f1/f4/f5 (không sụp) | −0,018 (0+/2−/1=) | +0,019 (2+/1−) | −0,004 (1+/2−) | −0,009 (1+/2−) |

- **Dự đoán ghi trước:** (a) Pha 1 chọn ≤ ep3 — ĐÚNG (ep1); (b) ≥1/5 fold sụp — **SAI (0/5)** ⇒ theo nhánh đã ghi: hai fold sụp của seed 42 là do seed; (c) fold cất cánh ≈ baseline ±0,02 — 3/5 trong khoảng (−0,007 / −0,002 / −0,000), 2/5 cao hơn (+0,030 / +0,029); TB +0,010.
- **Cơ chế đọc từ log:** cất cánh (train loss < 0,6) seed 1234 ở ep10 · 11 · 12 · 12 · 11; seed 42 ở ep17 · — · — · 16 · 15 (f2, f3 dừng sớm ep10 với best ep2 khi CHƯA cất cánh). Tính chất vững của nguồn C/C++ áp đảo là **cất cánh muộn** (nguồn có Java ep6–9); sụp = cất cánh chậm hơn cửa sổ patience 8, phụ thuộc seed. Đổi cả seed Pha 1 lẫn Pha 2 nên không tách được pha nào quyết định.
- So với run seed 42 khác là so KHÁC seed: FACTS §65 đo sàn nhiễu seed ~0,04 ROC (cấu hình khác) ⇒ chỉ đọc dấu nhất quán; vững nhất: thua full chỉ Java và thua nguồn full đầy đủ 0/5 cả bốn chỉ số (ROC −0,013 … −0,046 và TB −0,027). (Sửa 09:5x: bản đầu dòng này ghi `mwg_assemble` là "nguồn full" — thực ra là nguồn COMMON; báo người dùng lúc 07:25 cũng dán nhầm nhãn đó cho fold 2.)
- 158 RẢNH từ 09:46 (GPU 158 lúc đó 42 %, 542 MiB — không phải tiến trình của mình, trainer = 0).

---

## 30/09 04:1x — GỠ "common chỉ Java" khỏi artifact (người dùng: "common chỉ java không tồn tại nên bạn xoá luôn khỏi artifact")

Bỏ nguồn `common_javaonly` + run `p1_mwg_common_javaonly` / `mwg_assemble_javaonly` khỏi `scripts/runs.json` (58 → 56 run; `runs_effective`, `runs_resolved` lọc theo) — đã rsync sang 158, md5 khớp; `plan` khô 15 ô đang xếp hàng (161: 4cwe js/ccpp f1–f5; 158: s1234 f1–f5) cho lệnh TRÙNG HỆT bản cũ, còn run đã gỡ thì `plan` báo lỗi (không driver nào xếp nó). Xoá 3 doc `folds`, sửa `lines` của mục kết luận Java; trang v39. Kết quả f1–f2, log, checkpoint Pha 1 và dữ liệu `phase1_common_javaonly` trên đĩa GIỮ NGUYÊN (không được yêu cầu xoá); dòng skip `mwg_assemble_javaonly:3/4/5` trong skip.txt 161 để yên.

---

## 30/09 09:50 — XONG (161, bắt đầu 03:42): 4cwe CHỈ JS và 4cwe CHỈ C/C++, mã final, bậc 2 (n=5, seed 42)

Đối chiếu: 12/12 ô kỳ vọng (2 Pha 1 + 10 Pha 2, fold-major) `xong rc=0`; hp 10/10 ô seed 42, khớp runs.json; không còn driver/hàng đợi, 161 RẢNH từ 09:50. Không fold nào sụp: cất cánh (train loss < 0,6) chỉ JS ep6 · 5 · 6 · 6 · 5; chỉ C/C++ ep8 · 8 · 9 · 9 · 8.

| ROC | f1 | f2 | f3 | f4 | f5 |
|---|---|---|---|---|---|
| 4cwe chỉ JS | 0,865 | 0,947 | 0,949 | 0,968 | 0,932 |
| 4cwe chỉ C/C++ | 0,924 | 0,927 | 0,933 | 0,949 | 0,893 |
| 4cwe gốc (JS + C/C++) | 0,932 | 0,943 | 0,934 | 0,969 | 0,924 |
| baseline | 0,907 | 0,892 | 0,904 | 0,945 | 0,881 |

| Δ ghép cặp n=5 | F1@0,5 | F1@val | ROC | PR |
|---|---|---|---|---|
| chỉ JS − baseline | **+0,058 (5/5)** | **+0,045 (5/5)** | +0,026 (4+/1−) | +0,013 (4+/1−) |
| chỉ JS − 4cwe gốc | +0,010 (3+/1−/1=) | +0,015 (3+/2−) | −0,008 (3+/1−/1=) | −0,020 (3+/1−/1=) |
| chỉ JS − common chỉ JS | −0,007 (2+/3−) | +0,002 (1+/2−/2=) | −0,003 (3+/1−/1=) | −0,018 (3+/1−/1=) |
| chỉ C/C++ − baseline | +0,021 (4+/0−/1=) | +0,011 (4+/1−) | **+0,020 (5/5)** | **+0,025 (5/5)** |
| chỉ C/C++ − 4cwe gốc | −0,027 (1+/4−) | −0,019 (2+/3−) | **−0,015 (0/5)** | −0,009 (1+/4−) |
| chỉ C/C++ − chỉ JS | **−0,037 (0/5)** | **−0,034 (0/5)** | −0,007 (1+/4−) | +0,011 (1+/4−) |
| chỉ C/C++ − common chỉ C/C++ | **+0,030 (5/5)** | +0,020 (4+/1−) | +0,010 (3+/1−/1=) | +0,010 (4+/1−) |

- Chỉ JS: F1 hơn baseline 5/5; ROC/PR chỉ âm ở f1 (ROC −0,042, PR −0,114). So với 4cwe gốc và các nguồn JS khác: lẫn dấu, trong nhiễu (TB PR âm do riêng f1).
- **f1 KHÔNG sụp khi train** (đo 30/09 15:3x, người dùng hỏi "dừng ở đâu mà bị sập"): cất cánh ep5–7, best ep10 val ROC 0,949, dừng sớm ep18. Test tụt vì **CWE-022**: 10 hàm ĐÃ VÁ của CWE-022 bị chấm rất cao (TB 0,89, trung vị 0,94; hàm lỗi CWE-022 chỉ 0,66) ⇒ xếp trên nhiều hàm lỗi thật của CWE khác (7/30 điểm cao nhất là nhãn 0, 5 trong đó CWE-022). ROC tách cặp: trong cùng CWE 0,927 (baseline 0,931), khác CWE 0,829 (baseline 0,893). Bỏ 10 hàm đó: ROC 0,865 → 0,940, PR 0,813 → 0,926 (bỏ cả CWE-022: 0,945 / 0,926). Val fold 1 chỉ có 8 hàm CWE-022 (chấm 0,47 / 0,56) và 83/152 hàm CWE-089 (ROC 1,000) nên chọn checkpoint không thấy lỗi này. CWE-022 bất ổn với MỌI run (ROC trong CWE-022 theo fold 0,00–0,96, baseline f2 0,00) — 8–19 hàm mỗi fold; điểm riêng của f1 này là mức điểm của nhãn 0 quá cao nên phá thứ hạng TOÀN tập.
- Chỉ C/C++ (273 hàm train, Pha 1 val ROC 0,97 nhờ ĐƯỜNG TẮT CHỦ ĐỀ): vẫn hơn baseline 5/5 ở ROC/PR, cất cánh ep8–9 (sớm hơn các nguồn C/C++ lớn ep10–17), hơn common chỉ C/C++ ở F1@0,5 5/5; thua 4cwe gốc ROC 0/5 và thua chỉ JS ở F1 0/5 ⇒ gộp JS + C/C++ (4cwe gốc) là tốt nhất trong ba bản 4cwe về ROC. Giả thuyết (CHƯA đo): Pha 1 nhỏ (273 hàm × 8 epoch) ít xê dịch backbone hơn nguồn C/C++ lớn.


Người dùng: "chạy thêm 4cwe chỉ js" rồi "lên lịch 4 cwe js + ccpp" ⇒ hiểu là HAI nguồn tách từ 4cwe (bản gộp JS + C/C++ chính là 4cwe gốc, đã
đủ n=5 — đã nói với người dùng).
- **Dữ liệu** (`build_final_experiment_data.py --cwe4_langonly_only`: thêm khối ở CUỐI main + nhãn "4cwe" vào `langonly_readme_section`; dựng
  thử ở thư mục tạm rồi dựng thật, 8/8 file trùng bản thử; đường `--javaonly_only` dựng lại TRÙNG BYTE; README chỉ thêm 34 dòng; 158 md5 9/9):
  `phase1_4cwe_jsonly` 588 hàm (294 / 294; train 486 / val 102; 294 cặp, 80 tách), `phase1_4cwe_ccpponly` 305 hàm (105 / 200 — lệch 1:2;
  train 273 / val 32; CHỈ 3 cặp đủ ⇒ pair loss gần như không có tác dụng; val 32 hàm ⇒ chọn checkpoint Pha 1 theo val ROC rất nhiễu).
- **Run:** `p1_mwg_4cwe_{js,ccpp}only` → `mwg_assemble_4cwe_{js,ccpp}only` (chép khuôn 4cwe gốc; lệnh chỉ khác tên / đường dẫn; effective 56
  tham số, khác đúng checkpoint_path, data_root|init_ckpt, run_name). argv Pha 1: `--data_root …/phase1_4cwe_jsonly --epochs 8 --warmup_ratio 0.25`.
- **Máy:** 161 `run.sh 161 "p1 js, p1 ccpp, Pha 2 js/ccpp f1 → f5 (fold-major)"` từ 03:42 (log `state/driver_161_cwe4lang.out`), dự kiến ~11:00.
  158 vẫn chạy full chỉ C/C++ seed 1234 (Pha 1 ~05:50 rồi Pha 2 5 fold ~09:50); khi 158 xong có thể chia bớt fold 4cwe sang (skip.txt 161).
- **Pha 1 chỉ JS xong 03:54:** chọn ep5, val ROC 0,512 (ep1–8: 0,475 → 0,489, không epoch nào quá 0,52) — trên cặp CleanVul trước/sau vá
  (294 cặp) Pha 1 gần như không tách được hai nhãn, giống 4cwe gốc.
- **Pha 1 chỉ C/C++ ep2 val ROC 0,931 (03:57) — ĐƯỜNG TẮT CHỦ ĐỀ, không phải học lỗ hổng.** Nhãn 1 chỉ thuộc bốn CWE (22/78/79/89: train 37/23/
  27/5); nhãn 0 là hàm sạch bốc từ common, mang CWE của ~40 lớp khác (416, 119, 125, 476, 787, …) ⇒ nhãn trùng với "mã xử lý chuỗi/lệnh/
  đường dẫn" ↔ "mã bộ nhớ". Val chỉ 32 hàm; F1@0,5 0,3725 = đoán một lớp. Ghi TRƯỚC khi có Pha 2: số Pha 1 này không nói gì về chất lượng
  biểu diễn lỗ hổng; phần C/C++ của 4cwe gốc cũng mang đúng đường tắt này.

## 30/09 02:40 — PHƯƠNG ÁN 2 (full bỏ Java, Pha 1 ep8) ĐỦ n=5 · 3/5 SỤP ⇒ theo dự đoán ghi trước: nguyên nhân là NGUỒN

- `mwg_assemble_full_nojava_tl8` ROC 0,576 / 0,534 / 0,935 / 0,920 / 0,620 (f1, f2, f5 dừng ep10, chọn ep1; f3, f4 cất cánh ep16 / ep19, chọn
  ep26 / ep30). Dự đoán (c) ghi 29/09 04:2x: cất cánh = train loss < 0,6 TRƯỚC ep12 ⇒ **0/5**; sụp 3/5 ⇒ nhánh "≥3/5 vẫn sụp ⇒ do nguồn
  (C/C++ 90,5 %)". (d) không áp dụng. (a) (b) đã đúng (8/8 epoch trùng log gốc, chọn ep8).
- So ba checkpoint Pha 1 trên CÙNG nguồn, cùng Pha 2: ep1 (final) sụp f1, f3, f4 · ep3 (refactor seed 7) sụp f1, f3 · ep8 (phương án 2) sụp f1,
  f2, f5 — f1 sụp cả ba; fold nào cất cánh cũng muộn (train < 0,6 ở ep12–21; nguồn có Java ep7–9). tl8 − có Java: ROC −0,225 (0/5), F1@val
  −0,202 (0/5); tl8 − baseline ROC −0,189 (1+/4−). ⇒ Đổi epoch Pha 1 không chữa được: sụp là tính chất của nguồn C/C++ áp đảo (cùng mẫu với
  full chỉ C/C++ Pha 1 ep1: 2/5 sụp). 161 rảnh từ 02:40 (chờ checkpoint Pha 1 seed 1234 ~05:50).

## 30/09 01:20 — CHỈ ASAM (tắt RecAdam) ĐỦ n=5 cả hai nguồn chỉ JS (161)

| chỉ ASAM − | F1@0,5 | F1@val | ROC | PR |
|---|---|---|---|---|
| common: RecAdam + ASAM (`mwg_assemble_jsonly`) | −0,002 (2+/3−) | +0,008 (3+/2−) | +0,003 (3+/2−) | −0,007 (2+/3−) |
| common: baseline | +0,063 (5/5) | +0,050 (5/5) | +0,033 (5/5) | +0,025 (5/5) |
| full: RecAdam + ASAM (`mwg_assemble_full_jsonly`) | −0,013 (1+/4−) | −0,006 (1+/2−/2=) | −0,004 (1+/4−) | −0,004 (1+/4−) |
| full: baseline | +0,043 (5/5) | +0,036 (5/5) | +0,026 (5/5) | +0,028 (5/5) |

ROC chỉ ASAM: common 0,918 / 0,960 / 0,944 / 0,948 / 0,920; full 0,914 / 0,944 / 0,937 / 0,956 / 0,907. hp khác RecAdam + ASAM đúng
`recadam 1→0` (+ 2 mặc định RecAdam trơ). ⇒ Trên nguồn chỉ JS, RecAdam đóng góp rất ít (common lẫn dấu; full nghiêng âm 4/5 nhưng trong nhiễu
0,010); chỉ ASAM vẫn hơn baseline 5/5 cả bốn chỉ số ở cả hai nguồn. Bậc 2, một khối. 161 chuyển sang phương án 2 f1, f3, f5 (queue_after 209477).

## 29/09 23:1x — XẾP HÀNG: full chỉ C/C++ LẦN 2, seed 1234 cả hai pha (người dùng)

Người dùng: "nguồn full c sập rất nặng, chạy lại thêm lần 2 để kiểm" → hỏi lại (lặp y nguyên seed 42 sẽ TRÙNG BIT, không kiểm được gì) → chọn
"Seed 1234 cả hai pha".
- **Run:** `p1_mwg_full_ccpponly_s1234` (= p1_mwg_full_ccpponly + `--seed 1234`) → `mwg_assemble_full_ccpponly_s1234` (init từ lượt trên,
  `--seed 1234`, compare_to lần 1). resolve: chỉ khác tên / đường dẫn / seed 42→1234; effective_args khác đúng checkpoint_path, run_name, seed
  (+ init_ckpt ở Pha 2). Thư mục checkpoint vẫn tên `seed_42` (quy ước khối), seed thật 1234.
- **Máy:** 158 queue_after 2521913 (log `state/queue_158_ccpp1234.out`) chờ driver 1312036 (phương án 2 f4) ⇒ Pha 1 s1234 (~5,8 h) ⇒ Pha 2
  s1234 f1 → f5 (sẽ chuyển một phần sang 161 khi có checkpoint). Phương án 2 f1, f3, f5 (việc tự xếp) nhường chỗ: gỡ queue_after 1796709 trên
  158; bỏ chặn tl8 f1/f3/f5 trong skip.txt 161 (driver 161 cũ có ba ô này đã thoát; driver hiện tại chỉ có ô chỉ ASAM) ⇒ 161 queue_after 209477
  (log `state/queue_161_tl8c.out`) chờ driver 4069638 (chỉ ASAM) ⇒ tl8 f1, f3, f5. Kiểm hai chiều bằng `fpe.py plan`.
- **Dự đoán ghi TRƯỚC khi đo:** (a) Pha 1 s1234 lại chọn một epoch trên bình nguyên (≤ ep3), như seed 42 chọn ep1 — vì ở nguồn này val ROC rơi
  khi train loss bắt đầu giảm; (b) Pha 2 s1234: ≥1/5 fold sụp (ROC < 0,6) ⇒ sụp là tính chất của checkpoint bình nguyên trên nguồn này; 0/5 ⇒
  hai fold sụp của seed 42 là do may rủi seed; (c) fold nào cất cánh thì ROC ≈ baseline (±0,02), như seed 42 (+0,004 … +0,015).
- **30/09 05:57 Pha 1 s1234 XONG (158, 20 899 s):** chọn **ep1**, val ROC 0,5388 (ep1–8: 0,539 · 0,539 · 0,524 · 0,501 · 0,476 · 0,423 · 0,399 · 0,413; train loss 1,585 → 1,308) ⇒ **dự đoán (a) ĐÚNG** - cùng mẫu với seed 42 (ep1, 0,543): val ROC rơi đều khi train loss bắt đầu giảm. hp: seed 1234 (thư mục checkpoint `seed_42` theo quy ước). Kết quả + log đã `sync_158.sh` về 161. Pha 2 s1234 f1 bắt đầu 05:57 trên 158; (b), (c) chờ 5 fold.

## 29/09 22:47 — FULL CHỈ C/C++ ĐỦ n=5 · nhóm chỉ Java / chỉ C/C++ XONG · Pha 2 phương án 2 bắt đầu (158)

- **Full chỉ C/C++ (11 888 hàm), n=5:** ROC 0,922 / 0,475 / 0,500 / 0,949 / 0,894, chọn ep 23 / 2 / 2 / 27 / 30 (Pha 1 ep1 ⇒ f2, f3 SỤP;
  f1, f4, f5 cất cánh muộn ep14–16, f5 chọn ep30 = trần). − có Java: ROC −0,194 (0+/4−/1=), F1@0,5 −0,193 (0/5), PR −0,189 (0/5); − chỉ Java
  ROC −0,189 (1+/4−); − baseline ROC −0,158 (3+/2−), ba fold cất cánh chỉ +0,015 / +0,004 / +0,013 ⇒ kể cả khi cất cánh, C/C++ ≈ baseline.
- **Nhóm chỉ Java / chỉ C/C++ XONG** (20/20 ô dự kiến, trừ `mwg_assemble_javaonly` f3–f5 bị bỏ theo người dùng — 2/5 ô, có chủ ý).
- **22:47 Pha 2 phương án 2 bắt đầu trên 158** (`mwg_assemble_full_nojava_tl8` f2 → f4 → f1 → f3 → f5): argv `src_refactor train.py --recadam 1
  --sam_variant asam --sam_rho 0.5 --epochs 30 --init_ckpt …/p1_mwg_full_nojava_tl8/…/best.pt`; log "RecAdam | cof=500 | t0=17/1710 | neo 218".

## 29/09 21:55 — COMMON CHỈ C/C++ ĐỦ n=5 (158) · full chỉ C/C++ f5 đang chạy · chỉ ASAM f3 (161)

- **Common chỉ C/C++ (5 868 hàm C/C++), n=5, bậc 2:** ROC 0,900 / 0,920 / 0,903 / 0,949 / 0,906 (hp chỉ khác đường dẫn / tên).

  | common chỉ C/C++ − | F1@0,5 | F1@val | ROC | PR |
  |---|---|---|---|---|
  | có Java (`mwg_assemble`) | −0,062 (0/5) | −0,035 (0/5) | −0,017 (1+/4−) | −0,010 (1+/4−) |
  | Java + JS (`mwg_assemble_noccpp`) | −0,070 (0/5) | −0,045 (0/5) | −0,020 (0+/4−/1=) | −0,017 (0/5) |
  | chỉ JS (`mwg_assemble_jsonly`) | −0,073 (0/5) | −0,051 (0/5) | −0,020 (1+/4−) | −0,017 (1+/4−) |
  | C/C++ + JS (`mwg_assemble_nojava`) | −0,009 (2+/3−) | +0,016 (4+/1−) | −0,003 (2+/3−) | +0,001 (2+/3−) |
  | baseline | −0,009 (1+/3−/1=) | −0,009 (2+/3−) | +0,010 (3+/1−/1=) | +0,015 (4+/1−) |

  ⇒ C/C++ một mình ≈ C/C++ + JS ≈ baseline; thua mọi nguồn có JS hoặc Java mà không bị C/C++ áp đảo, 0/5 ở F1. Cùng với chỉ JS ≈ chỉ Java ≈
  Java + JS ≈ nguồn đầy đủ: chuyển giao đến từ phần CleanVul (JS/Java), PrimeVul C/C++ không đóng góp. Bậc 2, một khối — giả thuyết mạnh, chưa lặp.

## 29/09 19:48 — FULL CHỈ JAVA ĐỦ n=5 · chỉ ASAM bắt đầu trên 161 · chỉ C/C++ trên 158 (f4)

- **Full chỉ Java (5 184 hàm Java), n=5, bậc 2:** ROC 0,913 / 0,935 / 0,948 / 0,964 / 0,923 (hp hai ô ghép cặp chỉ khác đường dẫn / tên).

  | full chỉ Java − | F1@0,5 | F1@val | ROC | PR |
  |---|---|---|---|---|
  | có Java (`mwg_assemble_full`) | +0,004 (3+/2−) | −0,014 (2+/3−) | −0,006 (2+/3−) | −0,006 (2+/3−) |
  | Java + JS (`mwg_assemble_full_noccpp`) | +0,008 (4+/1−) | −0,010 (2+/3−) | +0,001 (2+/3−) | +0,004 (2+/2−/1=) |
  | chỉ JS (`mwg_assemble_full_jsonly`) | +0,001 (3+/2−) | −0,010 (2+/2−/1=) | +0,001 (3+/2−) | +0,001 (3+/2−) |
  | baseline | +0,058 (5/5) | +0,032 (5/5) | +0,031 (5/5) | +0,033 (4+/1−) |
  | chỉ Java common (n=2, nhiễu cách chia) | +0,007 (1+/1=) | +0,008 (1+/1−) | +0,001 (1+/1−) | +0,006 (1+/1−) |

  ⇒ Java một mình ≈ JS một mình ≈ Java + JS ≈ nguồn đầy đủ (trong nhiễu), cùng hơn baseline 5/5. Nguồn KÉM đều là nguồn C/C++ áp đảo (bỏ
  Java; full chỉ C/C++ với Pha 1 ep1). Bậc 2, một khối — giả thuyết.
- 161 xong driver chỉ Java 19:48 (bỏ đúng tl8 f1/f3/f5 theo skip.txt) ⇒ queue_after chỉ ASAM phóng `mwg_assemble_asamonly_jsonly` f1 19:48:33;
  argv + log: `--recadam 0 --sam_variant asam --sam_rho 0.5 --sam_eta 0.01`, init p1_mwg_common_jsonly, KHÔNG có dòng "RecAdam bat".
- Full chỉ C/C++: f1 0,922 (cất cánh muộn ep16) · f2 0,475 · f3 0,500 (sụp, dừng ep10; xác suất test KHÔNG hằng — 152 giá trị khác nhau).

## 29/09 18:0x — XẾP HÀNG: CHỈ ASAM (không RecAdam) cho hai nguồn chỉ JS, bậc 2 (n=5, seed 42)

Người dùng: "2 nhóm nguồn chỉ js (full và common), chạy asam only nhé".
- **Cấu hình:** flag set mới `asam_only` = `--recadam 0 --sam_variant asam --sam_rho 0.5 --sam_eta 0.01` (ρ, η GIỐNG cấu hình đầy đủ). Run
  `mwg_assemble_asamonly_jsonly` (init p1_mwg_common_jsonly), `mwg_assemble_asamonly_full_jsonly` (init p1_mwg_full_jsonly) — dùng lại Pha 1
  (CLAUDE.md §5: đổi optimizer Pha 2 thì dùng lại được). Lệnh chỉ khác run RecAdam + ASAM ở `--recadam 1→0` và bỏ 4 cờ RecAdam; effective_args:
  khác đúng recadam + 2 mặc định RecAdam (pretrain_cof 5000, anneal_t0 0,05) — TRƠ vì `train_mwg.py:459` chỉ dùng chúng trong `if args.recadam`.
  Không RecAdam ⇒ AdamW cùng nhóm tham số / lr, cộng ASAM. So với RecAdam + ASAM: tách phần của RecAdam; so với noRAS (chưa chạy cho chỉ JS):
  tách phần của ASAM.
- **Máy (người dùng trước, việc tự xếp sau):** 161 queue_after 4069638 (log `state/queue_161_asamonly.out`) chờ driver 3846352 (full chỉ Java
  f3–f5, ~20:10) ⇒ chỉ ASAM f1 → f5 fold-major (10 ô, ~04:00). Pha 2 phương án 2 dồn HẾT sang 158: skip.txt 161 chặn tl8 f1, f3, f5 (driver
  161 có sẵn ba ô này); gỡ queue_after 4061000 (tl8 f2, f4 trên 161); skip.txt 158 bỏ chặn tl8 f2, f4 (driver 158 có sẵn, chạy sau chỉ C/C++);
  158 queue_after (log `state/queue_158_tl8.out`) chờ driver 1312036 ⇒ tl8 f1, f3, f5 (~04:30). Kiểm hai chiều bằng `fpe.py plan` trên cả
  hai máy. ⚠ skip.txt 161 GIỮ chặn tl8 f1, f3, f5 — đừng xếp tl8 lên 161 mà không gỡ dòng đó.

## 29/09 09:30 — ĐANG CHẠY: nguồn CHỈ Java và CHỈ C/C++ (common + full), mã final, bậc 2 (n=5, seed 42)

Người dùng 29/09 ~09:2x: "tôi nhầm vì tưởng nó chưa chạy, thêm bản riêng cho java và ccpp nhé (full/common)" (trước đó: "chạy thêm js only
và js common" — hai run chỉ JS đã xong, người dùng tưởng chưa chạy). Cùng cách với nguồn chỉ JS: chỉ bản đầy đủ (RecAdam + ASAM), n=5.

- **Dữ liệu** (`tools/v4/build_final_experiment_data.py --javaonly_only --ccpponly_only`; hàm lọc = `build_lang_subset` của chỉ JS, thêm
  `langonly_readme_section` + hai cờ ở CUỐI main; đường cũ `--jsonly_only` dựng lại TRÙNG BYTE 8/8 file; dựng thử ở thư mục tạm rồi dựng
  thật, 16/16 file trùng bản thử; README chỉ THÊM 68 dòng; chép 158, md5 17/17 trùng):

  | nguồn | số hàm | train / val | cặp đủ (tách train/val) |
  |---|---|---|---|
  | `phase1_common_javaonly` | 5 184 | 4 415 / 769 | 2 592 (615) |
  | `phase1_full_javaonly` | 5 184 | 4 438 / 746 | 2 592 (630) |
  | `phase1_common_ccpponly` | 5 868 | 4 977 / 891 | 1 175 (312) |
  | `phase1_full_ccpponly` | 11 888 | 10 053 / 1 835 | 4 602 (1 214) |

  ⚠ **Java common và Java full chứa CÙNG 5 184 hàm** (CleanVul Java không có CWE nên vào nguyên vẹn cả hai) — chỉ khác cách chia train/val
  (3 859 / 5 184 = 74 % cùng phía). Hiệu giữa hai run chỉ Java = nhiễu do cách chia Pha 1, không phải hiệu ứng nguồn.
- **Run** (runs.json, chép khuôn run chỉ JS): `p1_mwg_{common,full}_{java,ccpp}only` → `mwg_assemble_{,full_}{java,ccpp}only`. resolve: lệnh chỉ
  khác run chỉ JS ở tên / đường dẫn (16/16 lệnh; phép so chiều lệch có bắt khác biệt); effective_args 56 tham số khớp runs.json, khác run chỉ
  JS đúng 3 khoá (checkpoint_path, data_root|init_ckpt, run_name). argv tiến trình đọc lại: `--data_root …_{javaonly,ccpponly} --epochs 8
  --warmup_ratio 0.25 --pair_loss 1.0 --seed 42` — trùng run chỉ JS.
- **Máy** (preflight đạt, 0 tiến trình, lock tự do trước; lock bị giữ thật sau khi phóng):
  - 158 `run.sh 158 p1_mwg_full_ccpponly:1` từ 09:30 (~5,8 h), log `state/driver_158_langonly.out`.
  - 161 `run.sh 161 "p1_mwg_common_javaonly:1 p1_mwg_common_ccpponly:1 p1_mwg_full_javaonly:1"` từ 09:30 (~2,5 + 2,8 + 2,5 h),
    log `state/driver_161_langonly.out`.
  - Pha 2 (20 fold) xếp khi có checkpoint Pha 1, thứ tự fold-major giữa các nguồn đã sẵn sàng. Ước tính xong toàn bộ ~00:00 ngày 30/09
    (hai máy); Pha 2 phương án 2 (full bỏ Java tl8, 5 fold) xếp SAU việc này (người dùng chưa trả lời thứ tự).
- **vast 53121439:** người dùng "vast xong hãy destroy giúp tôi, nhớ kéo về" ⇒ khi driver in XONG (Pha 1 tl8, ~11:25): kéo results / logs /
  state + checkpoint tl8 (Pha 2 phương án 2 cần) + checkpoint lượt ep2 (chỉ có trên vast) về 161, đối chiếu byte / md5, rồi huỷ.
  **11:27 driver vast in "FPE vast XONG".** Pha 1 tl8: **dự đoán (a) ĐÚNG — 8/8 dòng epoch TRÙNG NGUYÊN VĂN log gốc** `p1_mwg_full_nojava`
  (train 1,5871 … 1,1052; val ROC 0,5307 … 0,3738); **(b) ĐÚNG — chọn ep8** (train loss 1,1052; val ROC nguồn 0,374). results + log đã
  chuyển về local. Đang kéo checkpoint (~0,75 MB/s — ~45 phút cho 2,1 GB) rồi mới huỷ.
  **12:1x ĐÃ HUỶ vast `final_paper` (53121439).** Trước khi huỷ: 0 tiến trình, GPU 1 MiB / 0 %, lock tự do; results/ logs/ trên máy TRỐNG sau
  sync (đã chuyển về); 15/15 file `state/` kéo về `_FinalPaperExperiment/state/vast_final/from_instance_53121439/state/` (tên + byte, `comm -3`
  rỗng); **12/12 checkpoint** trên máy có bản ở 161 TRÙNG md5: 9 bản đã có sẵn, kéo thêm `p1_mwg_full_nojava_tl8/{best.pt ada3a85c…,
  best_roc_auc.pt cdd1584b…}`, `p1_mwg_full_nojava_ep2/{best.pt 9112ee7f…, ep2.pt 87bef07d…}`; `repro_baseline/best.pt` của vast (7d45893a…)
  KHÁC md5 bản cùng tên trên 161 (c44c3855…, cùng 498 660 288 byte — hai lượt khác nhau) ⇒ kéo về đường riêng
  `model/final_paper/repro_baseline_vast53121439/`. `vastai destroy instance 53121439 -y` in "destroying instance 53121439." (không
  Aborted), `show instances` chỉ còn 51144271 (`cuongtm`, của đồng nghiệp — không đụng). **Không còn máy vast nào của mình.**
- Pha 1 common chỉ Java xong 11:07 (chọn ep3, val ROC 0,505; 11,6 phút/epoch). Pha 1 common chỉ C/C++ (161) từ 11:07, 20,5 phút/epoch
  (~13:59); full chỉ C/C++ (158) 44 phút/epoch (~15:31).
- **13:58 Pha 1 common chỉ C/C++ xong** (chọn ep2, val ROC 0,592 — cao nhất các nguồn Pha 1; 835 cặp đủ / 3 307 lẻ). Pha 1 full chỉ Java (161)
  từ 13:58 (~15:40), rời bình nguyên ep3 như common chỉ Java. Pha 1 full chỉ C/C++ (158): val tốt nhất vẫn ep1 (0,543) tới ep6 — nhiều khả năng
  chọn ep1 như full bỏ Java (đã từng làm Pha 2 cất cánh muộn → sụp).
- **14:40 XẾP HÀNG PHA 2** (queue_after sau driver Pha 1 mỗi máy; checkpoint Pha 1 chung đã chép 158, md5 trùng: common chỉ Java 971e2a2c…,
  common chỉ C/C++ 9902801a…, tl8 ada3a85c…):
  - **161** (PID 3846352, log `state/queue_161_langonly.out`): `mwg_assemble_javaonly` + `mwg_assemble_full_javaonly` f1 → f5 (fold-major), rồi
    `mwg_assemble_full_nojava_tl8` f1, f3, f5.
  - **158** (PID 1312036, log `state/queue_158_langonly.out`): `mwg_assemble_ccpponly` + `mwg_assemble_full_ccpponly` f1 → f5, rồi
    `mwg_assemble_full_nojava_tl8` f2, f4.
  - Chia THEO NGÔN NGỮ (không theo fold) để mỗi máy chỉ dùng checkpoint Pha 1 của chính nó (full chỉ Java train trên 161, full chỉ C/C++ trên
    158) và cặp common/full cùng ngôn ngữ nằm cùng máy; ghép cặp với run cũ vẫn hợp lệ vì các A4000 trùng bit (đã kiểm 158 / 161 / vast / vast2).
    init_ckpt Pha 2 = đường ra Pha 1 (5/5 run, đã kiểm); preflight các mục có checkpoint sẵn đạt.
  - Ước tính: mỗi máy 10 ô ngôn ngữ (~45 phút/ô) ⇒ ~23:00; phương án 2 thêm ~2–2,5 h ⇒ ~01:30 ngày 30/09.
- **15:43 ĐỦ 4 PHA 1** (chọn theo val ROC nguồn, mã final):

  | Pha 1 | epoch chọn | val ROC | thời gian | md5 best.pt (có trên CẢ 158 và 161) |
  |---|---|---|---|---|
  | common chỉ Java | ep3 | 0,505 | 11,6 phút/epoch | 971e2a2c… |
  | full chỉ Java | ep3 | 0,481 | 12 phút/epoch | ff4b4836… |
  | common chỉ C/C++ | ep2 | 0,592 | 20,5 phút/epoch | 9902801a… |
  | full chỉ C/C++ | **ep1** | 0,543 | 44 phút/epoch | 7f84b9ff… |

  full chỉ C/C++ chọn ep1 — checkpoint còn trên bình nguyên (train loss rời bình nguyên ep4–5), ĐÚNG tình huống đã làm full bỏ Java sụp.
- **17:5x NGƯỜI DÙNG: "bạn có nhầm không, java làm gì có common mà chạy nhỉ?"** — ĐÚNG: Java không có CWE nên common = full (cùng 5 184 hàm,
  chỉ khác cách chia train/val). Lỗi của tôi: đã phát hiện điều này lúc dựng dữ liệu nhưng vẫn chạy cả hai ("theo yêu cầu") thay vì hỏi trước
  ⇒ tốn ~3 h GPU (Pha 1 common chỉ Java 1,5 h + 2 fold Pha 2). Xử lý: `mwg_assemble_javaonly` **DỪNG sau f2** (skip.txt 161: f3–f5, kiểm hai
  chiều bằng `fpe.py plan`), giữ f1–f2 làm số đo nhiễu cách chia; run chỉ Java chính = `mwg_assemble_full_javaonly`. runs.json chỉ sửa title /
  desc — KHÔNG đổi `folds` khi driver đang chạy (plan thoát lỗi ⇒ `eval` chuỗi rỗng ⇒ run.sh dùng lại lệnh ô trước). Cân tải: phương án 2
  f2, f4 chuyển 158 → 161 (skip.txt 158 + queue_after 161 chờ PID 3846352, log `state/queue_161_tl8b.out`). Dự kiến: 158 ~23:35, 161 ~00:15.
  Pha 2 bắt đầu: 161 `mwg_assemble_javaonly` f1 15:37, 158 `mwg_assemble_ccpponly` f1 15:43 — argv + log đọc lại: init đúng checkpoint Pha 1,
  nạp 216 tensor, RecAdam cof 500 t0=17/1710 (trần 30), neo 218 tensor, ASAM ρ 0,5, patience 8.

## 29/09 04:1x — CHỈ JS common ĐỦ n=5 · refactor full bỏ Java f3 SỤP · f5 chuyển sang vast2 (bậc 2, n=5, seed 42)

- **07:4x ĐÃ HUỶ vast2 `final_paper2` (53148947)** — người dùng: "Hủy final paper 2 giúp tôi nhé" + "nếu không còn chạy gì". Trước khi huỷ:
  0 tiến trình, GPU 1 MiB / 0 %, lock tự do, dòng cuối driver = f5 xong 04:55. results/ và logs/ trên máy đã TRỐNG (sync_remote chuyển hẳn
  file về local) trừ `logs/repro_baseline/fold1.log` (8 381 byte, lượt repro dừng tay — KHÁC file cùng tên ở local) ⇒ kéo về
  `_FinalPaperExperiment/state/vast_final2/from_instance_53148947/` cùng 10/10 file `state/` (đối chiếu tên + byte, `LC_ALL=C comm -3` rỗng).
  Checkpoint Pha 1 trên máy: `p1_mwg_full_nojava` (md5 6c8e94d2…) và `p1_mwg_full_nojava_tl_s7` (3dec6911…) TRÙNG bản ở 161;
  `p1_mwg_full_nojava_tl_s1234` (best.pt 2f2b0059…, best_roc_auc.pt 16295d8c…; lượt kẹt, không run nào dùng) KHÔNG kéo — VAST_RULES: "chỉ tải
  model nếu tôi yêu cầu ở runname đó". `vastai destroy instance 53148947 -y` in "destroying instance 53148947." (không Aborted), `show instances`
  không còn 53148947. Còn lại: 53121439 `final_paper` (đang chạy Pha 1 tl8, KHÔNG huỷ) và 51144271 `cuongtm` (của đồng nghiệp, không đụng).

- **Common chỉ JS (990 hàm Pha 1) đủ 5 fold** — ROC 0,925 / 0,939 / 0,940 / 0,943 / 0,931. hp hai ô ghép cặp chỉ khác đường dẫn / run_name.

  | Δ ghép cặp (n=5) | F1@0,5 | F1@val | ROC | PR |
  |---|---|---|---|---|
  | chỉ JS − có Java (`mwg_assemble`, 12 042 hàm) | +0,012 (4+/1−) | +0,016 (4+/1−) | +0,003 (3+/1−/1=) | +0,006 (3+/2−) |
  | chỉ JS − Java + JS (`mwg_assemble_noccpp`) | +0,004 (3+/2−) | +0,006 (2+/3−) | −0,001 (2+/3−) | −0,0005 (3+/2−) |
  | chỉ JS − C/C++ + JS (`mwg_assemble_nojava`) | +0,065 (5/5) | +0,067 (5/5) | +0,017 (4+/1−) | +0,018 (5/5) |
  | chỉ JS − baseline | +0,064 (5/5) | +0,043 (4+/1=) | +0,030 (4+/1−) | +0,032 (4+/1−) |

  JS một mình ngang nguồn đầy đủ; nguồn KÉM là C/C++ + JS. Tỉ lệ C/C++ trong train+val Pha 1 (đếm 29/09): common 48,7 %, full 64,9 %,
  common bỏ Java 85,6 %, full bỏ Java 90,5 %. ⇒ Đọc lại giả thuyết 22:18 "chuyển giao chủ yếu từ Java": đúng hơn là **C/C++ áp đảo nguồn
  thì hại**. Bậc 2, một khối — GIẢ THUYẾT.
- **Full chỉ JS tạm n=4** (f5 trên vast từ 04:05, ~05:00): − có Java ROC −0,007 (1+/3−), F1@0,5 +0,013 (2+/1−/1=) — ngang; − baseline
  F1@0,5 +0,066 (4/4), ROC +0,028 (3+/1−).
- **Refactor full bỏ Java (STUCK_ALL → seed 7 ckpt ep3):** f1 0,444 (158) · f2 **0,935** (vast2, cất cánh, chọn ep17) · f3 **0,517** (158,
  chọn ep1, dừng ep10) · f4 (161) đi ngang 19 epoch rồi **CẤT CÁNH ep20–22**, chọn **ep30 = trần epoch** (vẫn đang lên) ⇒ ROC **0,956**
  (có Java 0,961; baseline 0,945) · f5 vast2. Fold cất cánh thì gần bản có Java (f2 −0,025, f4 −0,005 ROC). Checkpoint ep3 KHÔNG chữa được sụp: cất cánh muộn (ep15–20) × dừng sớm patience 8 — fold nào val
  không nhích trong 8 epoch thì chết trước khi cất cánh (f1, f3). Phương án cho full bỏ Java (ghi nhận / chạy lại Pha 1 đủ 8 epoch không cổng kẹt) — **CHỜ người dùng**.
- **04:02 f5 refactor chuyển vast → vast2:** gỡ queue_after 34262 trên vast (log `state/queue_vast_tl2.out`), preflight vast2 đạt, phóng
  `run.sh vast2 mwg_assemble_full_nojava_trainloss:5` (log `state/driver_vast2_tlp2b.out`). Không chuyển full chỉ JS f5 sang 158: vast bắt
  đầu nó lúc 04:05, chuyển cũng xong cùng giờ.
- **Máy:** 158 RẢNH từ 04:02 (chưa xếp việc) · 161 refactor f4 · vast chỉ JS full f5 · vast2 refactor f5. KHÔNG huỷ vast nào (người dùng).
- **04:55 REFACTOR FULL BỎ JAVA ĐỦ n=5** — ROC 0,444 / 0,935 / 0,517 / 0,956 / 0,911, chọn ep 1 / 17 / 1 / 30 / 28. Val ROC ≥ 0,7 lần đầu:
  f2 ep11, f5 ep14, f4 ep19 (có Java: ep7–9; common: ep6–7; full chỉ JS: ep4–6); f1, f3 không bao giờ (dừng ep10). Δ ghép cặp n=5:

  | refactor full bỏ Java − | F1@0,5 | F1@val | ROC | PR |
  |---|---|---|---|---|
  | quy trình final (ckpt Pha 1 ep1) | +0,133 (5/5) | +0,114 (4+/1−) | +0,111 (4+/1−) | +0,121 (5/5) |
  | có Java (`mwg_assemble_full`) | −0,177 (1+/4−) | −0,168 (0+/5−) | −0,190 (0+/5−) | −0,182 (0+/4−/1=) |
  | baseline | −0,123 (3+/2−) | −0,122 (3+/2−) | −0,153 (3+/2−) | −0,144 (3+/2−) |

  Trung bình do 2 fold sụp gánh (§2b: đọc số fold, không đọc trung bình): 3 fold cất cánh đều TRÊN baseline (+0,011 … +0,042 ROC) và DƯỚI
  bản có Java (−0,005 … −0,025). f4, f5 chọn ep30 / ep28 ⇒ trần 30 epoch cũng đang cắt. Sụp = cất cánh muộn × patience 8 × trần 30.
- **04:45 FULL CHỈ JS ĐỦ n=5** (f5 0,919, vast): − có Java ROC −0,007 (1+/4−), PR −0,006 (0+/4−/1=), F1@0,5 +0,003 (2+/2−/1=), F1@val −0,004
  (2+/3−) — trong nhiễu, nghiêng âm ở thứ hạng; − Java + JS ≈ 0 cả bốn; − baseline F1@0,5 +0,057 (5/5), F1@val +0,042 (5/5), ROC +0,030 (4+/1−),
  PR +0,032 (4+/1=); − common chỉ JS ≈ 0. JS common (990) nằm TRỌN trong JS full (1 254) theo md5 mã (745/990 cùng phía train/val).
  Rà artifact ↔ local: 118 fold, lệch 0.
- **04:47 Pha 1 tl8 CHUYỂN vast2 → vast 53121439** (vast rảnh 04:45, vast2 còn f5): gỡ queue_after 29916 trên vast2 (khớp argv), preflight
  vast đạt, 0 tiến trình, lock tự do, đĩa trống 6,4 G; phóng `run.sh vast p1_mwg_full_nojava_tl8:1` (log `state/driver_vast_tl8.out`). argv tiến
  trình: `--seed 42 --selection_metric train_loss --also_select roc_auc --epochs 8 --stuck_epoch 0 --data_root …/phase1_full_nojava`. Dự kiến
  ~11:15. vast2 rảnh sau f5. Kiểm dự đoán (a): dòng pair loss / độ phủ TRÙNG log gốc (3 747 / 3 645, 11 139, 65,2751…); **05:39 ep1 TRÙNG
  NGUYÊN VĂN** (train 1,5871 · val loss 0,8477 · val ROC 0,5307 · F1 0,5025).
- **04:2x TỰ QUYẾT — phương án 2 cho full bỏ Java, xếp hàng trên vast2 sau f5 (04:47 đã chuyển sang vast, xem dòng trên)** (máy thuê bị ghim "đừng huỷ", sắp hết việc, người dùng chưa
  trả lời; memory "máy thuê rảnh thì lấp bằng hướng đã bàn" — phương án 2 đã đề xuất, chưa bị bác; người dùng chọn phương án 1 thì dừng ngay):
  `p1_mwg_full_nojava_tl8` = `p1_mwg_full_nojava_tl_s42` + `--stuck_epoch 0` (bỏ `--stuck_min_drop`) — mã refactor, chọn theo train loss, đủ
  8 epoch (~6,4 h) → `mwg_assemble_full_nojava_tl8` (như `_trainloss`, init_from tl8) chia 5 fold cho các máy rảnh. Lệnh sinh ra chỉ khác run
  tham chiếu ở tên / đường dẫn / `stuck_epoch 3→0` (resolve + effective_args 59 tham số, đã kiểm).
  **Lý do có cơ sở:** hai nguồn có Java chọn theo train loss đều ra **ep8** (val ROC nguồn 0,444 / 0,370 — dưới ngẫu nhiên) mà Pha 2 vẫn ngang
  bản chọn ep2–3 (ROC −0,003 / −0,004) ⇒ checkpoint muộn không hại. Mọi nguồn Pha 1: khi train loss rời bình nguyên thì val ROC nguồn rơi dưới
  0,5 (0,35–0,46) và val loss tăng — val ROC nguồn không phải tiêu chí chọn tốt (cơ chế chưa đo; nghi học thuộc cặp bị tách train/val).
  **Dự đoán ghi TRƯỚC khi đo:** (a) dòng ep1–ep8 TRÙNG NGUYÊN VĂN log gốc `p1_mwg_full_nojava` (train loss 1,5871 … 1,1052; val ROC 0,5307 …
  0,3738) — lệch ⇒ dừng, báo; (b) chọn **ep8**; (c) Pha 2: **≥3/5 fold cất cánh** (train loss < 0,6 trước ep12) ⇒ sụp là do checkpoint Pha 1
  còn trên bình nguyên, không phải do nguồn; **≥3/5 vẫn sụp** ⇒ do nguồn (C/C++ 90,5 %); 2/5 ⇒ không kết luận; (d) nếu cất cánh: ROC − baseline
  trong [−0,02; +0,03] (common bỏ Java: +0,013) — tức full bỏ Java ≈ baseline, dưới bản có Java.

## 29/09 00:2x — (ĐÃ THAY bằng mục trên) full bỏ Java CHẠY LẠI bằng mã refactor + nguồn CHỈ JS (người dùng: "full no java bị sập nên chạy lại
## (dùng script của refactor) + chạy thêm nguồn js only full và common")

- **Full bỏ Java theo quy trình final (checkpoint Pha 1 ep1) — ghi nhận như kết quả, KHÔNG xoá:** f1 0,464 · f2 0,894 (cất cánh muộn ep15–16)
  · f3 0,460 · f4 0,551 · f5 0,840 (cất cánh ep19–20). Có Java: 0,922 / 0,960 / 0,942 / 0,961 / 0,927.
- **Chạy lại bằng mã refactor** (src_refactor/train.py: chọn Pha 1 theo train loss, kiểm kẹt ep3 < 2 %): ba lượt SONG SONG thay vì lần lượt
  (kết quả y hệt vì tất định; luật p1_retry áp sau: seed ĐẦU TIÊN không kẹt theo 42 → 1234 → 7; kẹt cả ba ⇒ lượt 7):
  `p1_mwg_full_nojava_tl_s1234` vast2 00:20, `p1_mwg_full_nojava_tl_s7` 161 00:20, `p1_mwg_full_nojava_tl_s42` vast (queue_after sau lượt ep2,
  ~01:25). Dự đoán: seed 42 KẸT (ep2→ep3 của log gốc giảm 0,71 % < 2 %). Pha 2 `mwg_assemble_full_nojava_trainloss` (RecAdam+ASAM, n=5)
  thêm vào runs.json KHI đã chọn seed, init_from = lượt được chọn; chia 5 fold cho các máy rảnh.
- **Nguồn chỉ JS (mã final, chỉ bản đầy đủ, n=5):** `phase1_common_jsonly` 990 (train 844 / val 146, 495 cặp), `phase1_full_jsonly` 1 254
  (train 1 086 / val 168, 627 cặp) — đúng phần JS, không chia lại (`--jsonly_only`). 158 từ 00:21: `p1_mwg_common_jsonly` → `p1_mwg_full_jsonly`
  → `mwg_assemble_jsonly` / `mwg_assemble_full_jsonly` f1 → f5 (fold sau chia cho máy rảnh qua skip.txt 158).
- **02:4x CẢ BA SEED KẸT ⇒ STUCK_ALL:** seed 7 (161) ep2→ep3 1,5828 → 1,5721 = −0,68 %; seed 1234 (vast2) 1,5783 → 1,5595 = −1,19 %; seed 42 −0,71 %
  (log gốc). Theo luật p1_retry: dùng LƯỢT CUỐI = seed 7, checkpoint theo train loss (ep3, loss 1,5721, val ROC 0,515; md5 3dec6911…, chép
  161/158/vast/vast2; stuck.pt giữ nguyên). Nhận xét: luật 2 % @ep3 quá sớm cho nguồn này (lượt gốc rời bình nguyên ep4–5).
  Pha 2 `mwg_assemble_full_nojava_trainloss` (src_refactor, init_from p1_mwg_full_nojava_tl_s7): vast2 f1 → f2 từ 02:59; 158 f3, 161 f4,
  vast f5 qua queue_after sau việc chỉ JS. Dự kiến xong ~05:05. Chỉ JS f3 chuyển 158 → 161 (skip.txt 158) lúc 02:45.
- **01:1x lượt seed 42 KHÔNG chạy lại** (gỡ queue 27037 trên vast): mã refactor cho quỹ đạo trùng từng số mã final (đã kiểm 8/8 epoch ở
  common và full); log gốc seed 42 của chính nguồn này (p1_mwg_full_nojava): ep2 1,5837 → ep3 1,5725 = giảm 0,71 % < 2 % ⇒ luật refactor
  CHẮC CHẮN đánh "kẹt". Quyết định dựa trên số đo tất định, không phải phỏng đoán. Luật chọn còn lại: 1234 nếu không kẹt, không thì 7.
- **Chỉ JS:** Pha 1 common xong 00:42 (chọn ep4, val 0,576; kẹt ở ep3 rồi rời bình nguyên ep4), full xong 01:09 (chọn ep3, val 0,511).
  Checkpoint md5 trùng 158/161/vast. Pha 2: 158 f1→f3 (cả hai nguồn), vast f4, f5 (queue sau lượt ep2, skip.txt 158).
- vast 53121439 vẫn chạy lượt tái tạo ep2 (tự quyết 23:40) tới ~01:25 — dòng ep1/ep2 so với log gốc vast2 = phép kiểm trùng bit cho vast2.
  **00:3x KIỂM TRÙNG BIT vast2 ĐẠT:** ep1 của bản tái tạo (vast 53121439 — máy đã trùng bit với 158) TRÙNG NGUYÊN VĂN log gốc vast2 (pair loss,
  độ phủ, best ckpt 0,530736, dòng epoch), và checkpoint ep1 hai máy trùng **218/218 tensor** (torch.equal). ⇒ vast2 (final_paper2) tất định
  trùng bit với các A4000 khác trên khối này; ghép cặp các ô chạy trên vast2 hợp lệ.
- KHÔNG huỷ vast nào (người dùng).

## 28/09 16:2x — BỎ JAVA: ĐANG CHẠY, bố trí lại (người dùng: "Sử dụng tag final_paper2 để chạy cái no java … không cần chạy noRAS …
## chạy để kiểm chứng phần Java trước", "ép các phiên bản bao gồm cả python xuống y hệt 2 A4000 ở local", "cùng máy là được … rất vội")

- **23:1x–23:4x FULL BỎ JAVA: PHA 2 SỤP CÓ HỆ THỐNG** — f1 0,464 (dừng ep11, best ep3), f3 0,460 (dừng ep10, best ep2), f4 0,551
  (dừng ep16, best ep8), f5 đứng yên (train loss ≥0,686 sau 17 epoch), f2 đang chạy. Mọi Pha 2 RecAdam+ASAM đều đi ngang 6–8 epoch rồi
  "cất cánh" ep7–9 (vd. full có Java f3: val 0,62 → 0,87 @ep8); ở đây KHÔNG cất cánh. Nghi phạm: Pha 1 full bỏ Java chọn ep1 (val 0,531 vs
  ep2 0,528 — tung đồng xu), rời CodeBERT ít nhất (‖θ−θ₀‖/‖θ₀‖ 0,0025; full có Java ep2 0,0085; common bỏ Java ep2 0,0045). Đã báo người
  dùng 3 phương án (23:2x), CHƯA có trả lời.
- **23:46 f5 (161) XÁC NHẬN CHẨN ĐOÁN "CẤT CÁNH MUỘN":** đi ngang tới ep18 (loss ~0,69, val 0,41→0,53 nhích dần nên patience không cạn),
  cất cánh ep19–20 (val 0,58 → 0,69 → 0,82, loss 0,69 → 0,28), test ROC **0,840** (full có Java f5: 0,927). ⇒ Từ checkpoint Pha 1 ep1,
  Pha 2 cất cánh ~ep20 thay vì ep7–9; f1, f3, f4 bị dừng sớm (patience 8) TRƯỚC khi cất cánh. Sụp là tương tác checkpoint × dừng sớm, không
  phải "không học được".
- **23:4x TỰ QUYẾT (vast 53121439 trống, người dùng dặn không huỷ; memory "máy thuê không được chạy không khi chờ hỏi"): tái tạo Pha 1
  full bỏ Java tới ep2** = run `p1_mwg_full_nojava_ep2` (mã `src_final_epsave`: y hệt src_final + `FPE_SAVE_EPOCH=2` lưu `ep2.pt` rồi dừng,
  KHÔNG đổi --epochs/lịch LR/RNG). Pha 2 từ ep2 (`mwg_assemble_full_nojava_ep2`) CHƯA phóng — chờ người dùng.
  **Dự đoán ghi TRƯỚC khi đo:** (a) dòng ep1, ep2 TRÙNG NGUYÊN VĂN log gốc trên vast2 (train 1,5871 / 1,5837; val ROC 0,5307 / 0,5282) —
  đồng thời là phép kiểm trùng bit còn thiếu của vast2; lệch ⇒ dừng, báo. (b) Nếu Pha 2 từ ep2 được chạy: giả thuyết "sụp do checkpoint
  ep1" đúng nếu ≥3/5 fold cất cánh (train loss < 0,6 trước ep12); ≥3/5 vẫn sụp ⇒ nguyên nhân là nguồn bỏ Java chứ không phải epoch; 2/5 ⇒ không kết luận.
- **23:11 `mwg_mixsrc` (một pha, trộn nguồn common + SVEN) ĐỦ 5 FOLD** (158; f5 dừng sớm ep24, chọn ep16, test ROC 0,854):

  | Δ ghép cặp (n=5) | F1@0,5 | F1@val | ROC | PR |
  |---|---|---|---|---|
  | mixsrc − `mwg_assemble` (hai pha) | −0,043 (1+/4−) | −0,017 (2+/3−) | −0,035 (1+/4−) | −0,047 (0+/5−) |
  | mixsrc − baseline | +0,010 (2+/3−) | +0,010 (3+/2−) | −0,008 (3+/2−) | −0,022 (0+/5−) |
  | mixsrc − `mwg_assemble_noRAS` | −0,007 (2+/3−) | +0,013 (3+/2−) | −0,007 (1+/4−) | −0,018 (1+/4−) |

  Trộn nguồn một pha KHÔNG hơn baseline (PR thấp hơn 5/5) và thua hai pha ở ROC/PR ⇒ lợi ích đến từ cách chuyển giao hai pha, không
  phải từ việc có thêm dữ liệu nguồn. 158 giờ chạy `mwg_assemble_full_nojava` f4 (chuyển từ 161).
- **22:18 COMMON BỎ JAVA ĐỦ 5 FOLD** (f1, f2 vast 53121439; f3–f5 161; hp trùng `mwg_assemble` trừ đường dẫn; Pha 1 chọn ep2, val 0,577):

  | Δ ghép cặp (n=5) | F1@0,5 | F1@val | ROC | PR |
  |---|---|---|---|---|
  | bỏ Java − có Java (`mwg_assemble`) | −0,053 (0+/5−) | −0,051 (0+/5−) | −0,014 (1+/4−) | −0,012 (1+/3−/1=) |
  | bỏ Java − baseline | −0,000 (3+/2−) | −0,024 (2+/3−) | +0,013 (4+/1−) | +0,014 (4+/1−) |
  | bỏ Java − bỏ C/C++ (`mwg_assemble_noccpp`) | −0,061 (0+/5−) | −0,061 (0+/5−) | −0,018 (1+/4−) | −0,018 (1+/4−) |
  | (tham chiếu) có Java − baseline | +0,052 (4+/1=) | +0,026 (4+/1−) | +0,027 (4+/1=) | +0,025 (5/5) |

  Bỏ Java mất gần hết phần hơn baseline ở F1 và khoảng một nửa ở ROC/PR; Java + JS (bỏ C/C++) hơn C/C++ + JS ở cả 4 chỉ số. Pha 1 rời
  CodeBERT ít nhất (‖θ−θ₀‖/‖θ₀‖ 0,0045 @ep2; common 0,0112, full 0,0085, noccpp 0,0058, 4cwe 0,0041 — 4cwe dời ít nhất mà Pha 2 vẫn +0,035 ⇒
  khoảng dời một mình không giải thích). Bậc 2, một khối, 5/5 = sàn Wilcoxon — GIẢ THUYẾT "đóng góp chuyển giao chủ yếu từ Java", chưa lặp.
- **17:3x TF32 (người dùng hỏi, "xem nếu hợp lý mới làm") — ĐO rồi KHÔNG bật:** CodeBERT fwd+bwd 8×512 ckpt FP32 trên A4000 161: TF32 tắt
  0,408 s/bước, bật 0,222 ⇒ x1,84. Không bật vì mọi run tham chiếu (có Java, baseline, Pha 1) chạy TF32 tắt ⇒ Δ ghép cặp lẫn hiệu ứng TF32
  (rẽ quỹ đạo như đổi hạt giống, sàn ~0,04 ROC — FACTS §65). Để dành cho khối mới bật TF32 cho MỌI nhánh từ đầu (người dùng ĐỒNG Ý 28/09 17:4x — ghi vào `block.determinism` của khối đó).
- **17:5x artifact v23: khung "Máy đang chạy gì" ở đầu trang** (người dùng) — job lấy tự động từ fold `status=running` của db theo `host`;
  hàng đợi + ghi chú từng máy ở doc `meta/machines` ({updated_at, hosts: {<host>: {next: [...], note, state}}}) — CẬP NHẬT mỗi lần xếp/đổi hàng đợi.
- **17:35 158:** mixsrc f5 best mới ep16 (val 0,847) ⇒ patience về 0, sớm nhất dừng ep24 (~23:00) ⇒ 158 KHÔNG nhận fold common bỏ Java; common
  chia cho vast 53121439 + 161 (~20:15 → ~22:15), full cho vast2 + 161 + vast (~22:55 → ~00:20).
- **16:3x người dùng: "Vast thì đừng huỷ vội và chạy no java n=5 luôn để đối chứng"** ⇒ n=5 (fold 1–5), KHÔNG huỷ vast nào khi chưa được
  bảo. Hàng đợi: vast 53121439 = queue_after 11675 chờ 5445 ⇒ `p1_mwg_common_nojava` → `mwg_assemble_nojava` f1–f5; vast2 = driver 1628
  (`p1_mwg_full_nojava` → f1) + queue_after 2236 ⇒ `mwg_assemble_full_nojava` f2–f5. Khi có checkpoint Pha 1: fold sau (f5, f4, f3) chuyển sang
  161/158/máy vast rảnh bằng skip.txt của máy vast + phóng driver ở máy nhận (chép ckpt, md5 trước).
- **Phạm vi (16:2x, đã đổi n ở dòng trên):** CHỈ bản đầy đủ (`mwg_assemble_nojava`, `mwg_assemble_full_nojava`, RecAdam + ASAM), ~~n=3~~;
  hai run noRAS bỏ Java đã gỡ khỏi runs.json (định nghĩa: như `mwg_assemble_noRAS_noccpp` với init_from p1_*_nojava — thêm lại khi cần).
- **vast `final_paper2` = host `vast2`** (instance **53148947**, RTX A4000, driver 595.84, 18 GB, 0,087 $/h; SSH `-p 63971 root@162.200.81.25`):
  `state/vast_final2/setup_env.sh` ⇒ `/workspace/venv311` = Python **3.11.14** (uv; bản dựng Clang, local là conda GCC — cùng số phiên bản)
  + ĐỦ **117/117 gói** của `pip list` vdenv 158 cùng phiên bản (kể cả tensorflow 2.20.0 / keras, pip/setuptools/wheel) — so bằng diff: rỗng.
  libtorch_cuda / libtorch_cpu / libc10_cuda / cuDNN 9 / cuBLAS / cuBLASLt: **6/6 trùng md5** với vdenv 161. Mã, script, dữ liệu, CodeBERT:
  58/58 file trùng md5. Kiểm trùng bit `repro_baseline` đã phóng rồi DỪNG TAY sau ~1 phút theo người dùng ("chạy luôn cái no java trước",
  "cùng máy là được") — **KHÔNG có phép kiểm trùng bit trên máy này.**
- **Đang chạy:** vast2 `p1_mwg_full_nojava` từ **16:24** (~6 h ⇒ ~22:25; pair loss 3 747 cặp / 3 645 lẻ, train 11 139; argv tiến trình khớp
  runs.json: epochs 8, warmup 0,25, pair 1,0/1,0, lr 2e-5) → `mwg_assemble_full_nojava:1`. vast 53121439: queue_after 11110 chờ 5445 (refactor
  full f5) ⇒ `p1_mwg_common_nojava` (~3 h ⇒ ~19:45) → `mwg_assemble_nojava:1`. Hàng đợi bỏ Java trên 161 đã gỡ.
- **Chia fold khi có checkpoint Pha 1:** f2, f3 của mỗi nguồn sang máy rảnh (161 rảnh từ ~16:55; 158 sau mixsrc f5) — chép ckpt qua 161,
  md5, rồi phóng. Dự kiến: common n=3 ~20:30, full n=3 ~23:10.
- **Huỷ:** 53121439 sau `mwg_assemble_nojava:1` + sync; 53148947 (final_paper2) sau `mwg_assemble_full_nojava:1` + sync (chỉ máy của mình,
  đối chiếu trước).

## 28/09 15:4x — (ĐÃ THAY bằng mục trên) XẾP HÀNG: nguồn common / full BỎ JAVA (161 + vast 53121439)

Người dùng 28/09: "java đang chạy ở full lẫn common mà không suy xét đến cwe nên hãy thử chạy 1 src không có cwe cho common và full của các
thí nghiệm cơ bản". Bậc 2 (n=5, seed 42), tab final.

- **Cách hiểu:** CleanVul Java không có nhãn CWE nên vào NGUYÊN VẸN (5 184 hàm) cả common lẫn full ⇒ dựng hai nguồn bỏ Java, không chia lại
  (hàng C/C++/JS giữ đúng tập train/val như nguồn gốc):

  | nguồn | số hàm (nhãn 1 / 0) | C/C++ | JS | train / val | cặp đủ (tách train/val) |
  |---|---|---|---|---|---|
  | `phase1_common_nojava` | 6 858 (3 363 / 3 495) | 5 868 | 990 | 5 821 / 1 037 | 1 670 (428) |
  | `phase1_full_nojava` | 13 142 (6 515 / 6 627) | 11 888 | 1 254 | 11 139 / 2 003 | 5 229 (1 370) |

  Bỏ Java thì mọi hàng nhãn 1 của common đều mang CWE dùng chung với Python; full vẫn lấy mọi CWE (kể cả 29 hàm JS nhãn 1 không ghi CWE).
- **Dựng:** `tools/v4/build_final_experiment_data.py --nojava_only` (`build_noccpp` thêm tham số `drop` ở CUỐI chữ ký; đường cũ drop=ccpp tái tạo
  TRÙNG BYTE 8/8 file noccpp đang dùng). README + bảng md5 cập nhật (chỉ thêm dòng). Đã chép sang vast, md5 8/8 khớp.
- **Run** (runs.json, như bộ full_noccpp): `p1_mwg_common_nojava`, `p1_mwg_full_nojava` (Pha 1) → `mwg_assemble_nojava`, `mwg_assemble_noRAS_nojava`,
  `mwg_assemble_full_nojava`, `mwg_assemble_noRAS_full_nojava` (Pha 2, f1–f5). Cờ chỉ khác run noccpp tương ứng ở data_root / run_name /
  checkpoint; effective 56 tham số khớp runs.json; 270 lệnh `plan` của run cũ TRÙNG NGUYÊN VĂN bản trước khi sửa.
- **Hàng đợi** — `queue_after.sh` nay nối chuỗi được (chờ cả một queue_after khác, vì nó `exec` thành run.sh cùng PID + lstart) và nhận driver
  phóng bằng đường dẫn tuyệt đối (driver 161); thử hai chiều bằng driver giả (nối 3 tầng đúng thứ tự; PID `sleep` bị từ chối, mã 2):
  - **15:57 ĐỔI THỨ TỰ (người dùng: "ưu tiên chạy 2 cái full setting trước thay vì chạy cái noRAS")**: mỗi nguồn chạy `mwg_assemble_*_nojava`
    (RecAdam + ASAM) đủ f1→f5 TRƯỚC, rồi mới `mwg_assemble_noRAS_*_nojava` f1→f5. Hai hàng đợi cũ (fold-major, còn đang chờ) đã dừng theo
    PID + lstart và phóng lại; log thứ tự cũ giữ ở `*.old_order`.
  - **vast 53121439**: PID 9653 chờ 5445 (refactor full f5) ⇒ `p1_mwg_full_nojava` (~6 h) → `mwg_assemble_full_nojava` f1–f5 → noRAS_full_nojava f1–f5.
    Log `state/queue_vast2.out`.
  - **161**: PID 2351686 chờ driver 1548916 (refactor full f3–f4) ⇒ `p1_mwg_common_nojava` (~3 h) → `mwg_assemble_nojava` f1–f5 → noRAS_nojava f1–f5.
    Log `state/queue_161_nojava.out`.
  - **Chia máy (người dùng: "chia việc chạy no java cho máy còn trống")**: 158 rảnh (sau mixsrc f5) + Pha 1 common xong ⇒ chép ckpt sang 158,
    chuyển fold 4–5 của common (bản đầy đủ trước, noRAS sau) qua skip.txt 161; Pha 1 full xong (~22:45) ⇒ chia fold full cho máy rảnh.
    Giữ hai nhánh của cùng fold trên cùng máy khi có thể (trùng bit giữa các A4000 đã kiểm nên ghép cặp khác máy vẫn hợp lệ).
  - Ước tính: 161 xong ~03:00, vast ~05:30 ngày 29/09. 158 rảnh (sau mixsrc f5) ⇒ chuyển bớt fold Pha 2 bằng skip.txt + chép ckpt Pha 1.
- **Monitor v4** (`scratchpad/mon_fpe_v4.sh <host>`): đọc "bắt đầu / xong rc=" từ `state/driver_<host>.log` (mọi driver, kể cả driver do queue_after
  exec — v3 để lọt) và "XONG / !! / phóng" từ mọi `state/{driver,queue}_<host>*.out`. Thử hai chiều (lần đầu im; lùi bộ đếm thì in lại).
- **Huỷ vast 53121439 dời tới khi xong hàng đợi này** (sync + đối chiếu trước).

## 28/09 13:08 — THUÊ LẠI VAST `final_paper` (instance **53121439**, RTX A4000, 0,087 $/h, đĩa 19 GB) — người dùng: "đã thuê thêm vast final_paper, cài giúp tôi các gói đồng bộ và dùng nó để chạy thêm"

- SSH `ssh -p 64226 root@162.200.81.25` (khoá gắn `vastai attach ssh 53121439`), hostname `c91e8843e860`, driver 595.84. runs.json `hosts.vast` đã cập nhật.
- Môi trường: `/workspace/setup_env.sh` (bản lưu ở `_FinalPaperExperiment/state/vast_final/`) ⇒ `/workspace/venv311`: Python 3.11.14, torch 2.9.1+cu128,
  cuDNN 91002, transformers 4.57.1, tokenizers 0.22.1, numpy 2.3.4, sklearn 1.7.2, scipy 1.16.3 — TRÙNG vdenv.
- Đã chép + md5 70/70 khớp (LC_ALL=C): src_final, src_refactor, SVEN nocomment, CodeBERT, ckpt Pha 1 `p1_mwg_common_trainloss` + P1_CKPT.
- Driver vast (13:08, PID 1460, lock giữ): `repro_baseline:1` (kiểm trùng bit với baseline f1 của 158) → `mwg_assemble_common_trainloss` f2–f5.
  161 bỏ 4 mục đó qua `state/skip.txt` (thử hai chiều: common f2/f5 bỏ qua; full f2 và common f1 không). Monitor `scratchpad/mon_fpe_vast_v3.sh`.
- **13:17 kiểm trùng bit ĐẠT:** `repro_baseline` f1 trên vast 53121439 — cả 23 dòng epoch trùng NGUYÊN VĂN baseline f1 của 158; test ROC 0,9067988175969397,
  F1@0,5 0,8157, F1@val 0,8220, PR 0,9274, best ep15 — trùng từng chữ số (bản lưu `_FinalPaperExperiment/_repro_keep_vast2/`). Ghép cặp fold chạy ở vast hợp lệ.
- HUỶ **53121439** (máy CỦA MÌNH, nhãn final_paper — KHÔNG phải 51144271) khi hàng đợi vast xong + đã `sync_remote.sh vast` và đối chiếu.
  (15:4x: hàng đợi vast nay gồm cả khối bỏ Java — xem mục trên; dòng kết thúc cần chờ là "FPE vast XONG" của driver do queue 8651 exec.)

## 28/09 01:5x — ĐANG CHẠY trên 161: TAB "REFACTOR" — mwg_assemble common + full, Pha 1 chọn theo TRAIN LOSS + thử lại khi kẹt

Người dùng 28/09: "tạo thêm 1 thư mục con để lưu mã nguồn nhánh refactor, chạy thêm 2 cấu hình mwg_assemble full và common với cách lấy
loss best và retry khi bị sụp. Thêm 1 tab riêng, tránh gây ảnh hưởng tới kết quả cũ. Điều phối các máy khi rảnh."

- **Mã**: `MultiVD/src_refactor/` = `maytinhdibo/GraphTransferVD` nhánh `refactor` @ `c1f1641` (thư mục refactorMWG/, bỏ data/), md5 trong
  `src_refactor/SOURCE.md`. Bộ kiểm của nhánh: test_symbols 32/32, test_stuck 20/20. KHÔNG sửa file nào trong đó.
- **Run** (runs.json, `"tab": "refactor"`, ID mới ⇒ không đụng kết quả cũ): `p1_mwg_common_trainloss`, `p1_mwg_full_trainloss` (Pha 1),
  `mwg_assemble_common_trainloss`, `mwg_assemble_full_trainloss` (Pha 2 RecAdam+ASAM, f1–f5). So với run gốc (`compare_to`):
  tham số hiệu lực trùng 56/56 khoá, CHỈ khác Pha 1 `--selection_metric train_loss --also_select roc_auc --stuck_epoch 3 --stuck_min_drop 0.02`.
  `best_roc_auc.pt` = cách chọn cũ, giữ để đối chiếu từng tensor với checkpoint Pha 1 cũ (kiểm tương đương mã).
- **Thử lại khi kẹt**: `scripts/p1_retry.py` (cơ chế run_p1.sh của nhánh refactor): seed 42 → 1234 → 7; kẹt ⇒ stuck.pt + `fold1_seed<S>_stuck.log`;
  ghi `<ckpt>/<run>/P1_CKPT` = `seed_<S>` (tương đối); kẹt cả 3 ⇒ vẫn chạy Pha 2 bằng lượt cuối, đánh dấu STUCK_ALL (CLAUDE.md §3).
  Thiếu P1_CKPT ⇒ Pha 2 bỏ qua fold (không nạp best.pt dở). Mô phỏng 4 ca (không kẹt / kẹt rồi học / kẹt cả 3 / lỗi thật) đạt.
- **fpe.py** mở rộng: `src_dir` theo run, P1_CKPT, dòng log kiểu mới (`Pair loss`, `Độ phủ`, `Kiểm kẹt`, `### P1`), chọn theo metric khác
  val ROC. Kiểm hai chiều: 156 + 78 `plan` của run cũ TRÙNG NGUYÊN VĂN bản trước; log giả có thử lại chỉ lấy lượt cuối; log cũ đọc như trước.
- **Artifact**: thanh tab cấp trang ("Khối final" không đổi hiển thị / "Refactor: train loss"); tab mới kèm hàng tham chiếu (baseline + run gốc)
  và dòng "Δ gốc". Monitor 161 v3 bắt thêm dòng log mới.
- **Thứ tự** (fold-major, có kết quả sớm): p1 common → f1 common → p1 full → f1 full → f2..f5 × [common, full]. Bậc 2 (n=5, seed 42) như
  lời người dùng "n=5 luôn dễ so". Ước tính: Pha 1 common ~5 h, full ~7,5 h, mỗi fold Pha 2 ~45 phút ⇒ xong ~21:30 28/09.
- **Kiểm tương đương mã (01:52):** Pha 1 common bằng src_refactor, seed 42 ⇒ pair loss 3 099 cặp / 4 038 lẻ, độ phủ train+val trùng khít,
  dòng Epoch 1 TRÙNG TỪNG SỐ với lượt cũ trên 158 (train 1,5090 · val loss 0,7014 · ROC 0,5332 · F1 0,4772) — mã refactor + det_launch tái tạo
  đúng quỹ đạo cũ; khác biệt Pha 2 sẽ chỉ do checkpoint được chọn. ~34 phút/epoch ⇒ Pha 1 common xong ~06:05.
- **05:56 Pha 1 common (refactor) xong train:** cả 8 dòng epoch trùng từng số với lượt cũ; kiểm kẹt ep3 giảm 7,45 % ⇒ không thử lại (seed 42);
  checkpoint theo train loss = **ep8** (train 0,5587, val ROC 0,444, val loss 2,53); `best_roc_auc.pt` (ep3) TRÙNG KHÍT 218/218 tensor với
  checkpoint Pha 1 cũ của p1_mwg_common ⇒ tương đương mã đã chứng minh ở Pha 1.
- **14:29 Pha 1 full (refactor) xong:** 8 epoch trùng từng số lượt cũ; kiểm kẹt ep3 giảm 5,82 % ⇒ không thử lại; chọn **ep8** theo train loss
  (train 0,8122, val ROC 0,370); `best_roc_auc.pt` (ep2) trùng khít 218/218 tensor với p1_mwg_full cũ. Chép ckpt + P1_CKPT sang vast (md5 khớp).
- **Chia việc 14:3x:** vast 53121439 = common f2–f5 rồi **full f5** (queue_after.sh chờ driver 1460); 161 = full f1–f4 (skip common f2–f5, full f5). Dự kiến
  hai máy xong ~17:30.
- **16:06 refactor COMMON đủ 5 fold** (f1 161, f2–f5 vast; hp hai ô ghép cặp chỉ khác đường model_name + cờ stuck của trainer mới):

  | Δ ghép cặp (n=5, hoà \|Δ\| ≤ 1e-3) | F1@0,5 | F1@val | ROC | PR |
  |---|---|---|---|---|
  | refactor − gốc `mwg_assemble` | −0,018 (1+/4−) | −0,004 (1+/3−/1=) | −0,003 (3+/2−) | −0,005 (3+/2−) |
  | refactor − baseline | +0,034 (5/5) | +0,023 (5/5) | +0,024 (5/5) | +0,020 (4/5) |

  Từng fold ROC refactor − gốc: +0,004 −0,031 +0,006 +0,011 −0,005 ⇒ chọn Pha 1 theo train loss KHÔNG khác cách chọn cũ ở ROC/PR (dấu lẫn lộn,
  trung bình dưới sàn nhiễu 0,010); F1@0,5 thấp hơn 4/5 fold. Bậc 2, một máy/khối — chưa phải kết luận.
- **17:00 TAB REFACTOR XONG — full đủ 5 fold** (f1–f4 161, f5 vast; hp chỉ khác đường dẫn + cờ stuck):

  | Δ ghép cặp (n=5) | F1@0,5 | F1@val | ROC | PR |
  |---|---|---|---|---|
  | refactor full − gốc `mwg_assemble_full` | −0,016 (1+/3−/1=) | −0,020 (1+/4−) | −0,004 (2+/3−) | −0,009 (1+/4−) |
  | refactor full − baseline | +0,038 (5/5) | +0,027 (4/5) | +0,033 (5/5) | +0,029 (5/5) |

  ROC từng fold refactor − gốc: −0,005 −0,022 −0,001 +0,007 +0,002. Cả hai nguồn: chọn Pha 1 theo train loss KHÔNG hơn cách chọn theo val ROC
  (ROC trong nhiễu; F1 và PR nghiêng âm). Kẹt Pha 1: không lượt nào kẹt (seed 42 cả hai) ⇒ cơ chế thử lại chưa được dùng tới.
- **Điều phối**: 161 rảnh ⇒ chạy tại đây. 158 bận mixsrc f4→f5 tới ~29/09; nếu 158 rảnh sớm thì chuyển fold Pha 2 còn lại sang (skip.txt,
  chép src_refactor + checkpoint Pha 1 + P1_CKPT), chia theo fold trọn vẹn.

## 28/09 00:1x — full > common: có phải do Pha 1 (val loss / checkpoint được chọn)? (người dùng hỏi)

| Pha 1 | ep chọn (theo val ROC) | train loss @chọn (bình nguyên) | val loss @chọn (min, ep) | ‖θ−θ₀‖/‖θ₀‖ backbone | Pha 2 mwg_assemble Δ ROC / Δ PR vs baseline (n=5) |
|---|---|---|---|---|---|
| common | **3** | 1,375 (1,48) — đã rời bình nguyên 0,105 | 0,776 (0,700 @2) | **0,0112** | +0,027 / +0,025 |
| full | 2 | 1,641 (1,64) — **còn ở bình nguyên** | 0,739 (0,735 @1) | 0,0085 | **+0,037 / +0,039** |
| common_noccpp | 2 | 1,780 (1,82) | 0,693 (min) | 0,0058 | +0,031 / +0,032 |
| full_noccpp | 2 | 1,772 (1,82) | 0,762 (0,732 @1) | 0,0062 | +0,030 / +0,029 |
| 4cwe | 4 | 1,082 (1,40) | 0,839 (0,651 @3) | 0,0041 | +0,035 / +0,034 |

- Val loss Pha 1 tự nó KHÔNG xếp được thứ tự Pha 2 (4cwe val loss cao nhất nhưng Δ thứ hai; common_noccpp thấp nhất nhưng Δ giữa).
- Nhưng common là ô DUY NHẤT được chọn ở ep3 (val ROC 0,540 vs 0,526 @ep2 — chênh trong nhiễu, val ROC Pha 1 ≈ ngẫu nhiên) ⇒ Pha 1 của
  common chạy xa hơn (rời CodeBERT nhiều nhất, val loss đã tăng) còn full dừng ngay ở bình nguyên. Hai giả thuyết cho full > common — (a) số
  cặp vá lỗi ccpp (835 vs ~3 280), (b) epoch chọn Pha 1 (ep3 vs ep2) — cùng khớp dữ liệu hiện có (bản noccpp: cùng ep2 VÀ không có ccpp ⇒ bằng nhau).
- Tách được bằng: Pha 2 từ checkpoint ep2 của common (tất định trùng bit ⇒ chạy lại p1_mwg_common tái tạo đúng quỹ đạo 158, kiểm bằng dòng log
  ep1/ep2). CHƯA chạy — chờ người dùng. Ckpt p1_mwg_common đã chép 158 → 161 (md5 5f727a80…).

## 27/09 22:2x — ~~LỆNH HUỶ MỘT LẦN vast 51144271~~ → ĐÃ HUỶ KẾ HOẠCH 28/09 00:2x (Claude hiểu SAI máy)

> **Đính chính của người dùng 28/09:** "vast tên tôi tưởng của người khác, của chúng ta là con a4000". Lệnh "khi vast xong hãy destroy"
> là cho A4000 `final_paper` (52805338) — máy đó đã huỷ từ 27/09 10:28. **51144271 (nhãn cuongtm) là máy của đồng nghiệp: KHÔNG huỷ,
> KHÔNG đụng.** Script chờ đã dừng khi còn ở vòng chờ: chưa rsync, chưa tạo `vast_51144271_final/`, chưa destroy; chỉ từng đọc ps/log.
> Phần dưới giữ lại làm dấu vết, KHÔNG thi hành.

(Kế hoạch cũ:) người dùng: "khi vast xong hãy destroy cho tôi (chỉ lần này)"

- Ghi đè lệnh ghim "KHÔNG HUỶ" (VAST_RULES) **đúng một lần**. 51144271 (nhãn cuongtm, RTX 5060 Ti) đang chạy 2 chuỗi của phiên khác:
  `chuoi_mc65.sh` (/workspace/dgmoe_dung, mc64 v17; kết thúc = `CHUOI MC65 XONG` trong chuoi_mc65.log) và `log/bl8_full.sh`
  (/workspace/cuongtm_refactor/refactorMWG, Pha 2 f2–5; kết thúc = `BL8_DONE` trong log/bl8_full.out). ETA ~00:00–00:40 ICT 28/09.
- `scratchpad/finalize_51144271.sh` (nền): chờ CẢ HAI dòng kết thúc (chuỗi chết mà thiếu dòng ⇒ KHÔNG huỷ) → không còn job mới + GPU < 500 MiB →
  rsync mọi file TRỪ trọng số (~3,4 GB, 2 188 file) về `/drive1/cuongtm/ntat/MultiVD/vast_51144271_final/` → md5 LC_ALL=C → in "SẴN SÀNG HUỶ";
  Claude mới `vastai destroy instance 51144271 -y` và kiểm output. Trọng số (~18 GB) KHÔNG kéo (VAST_RULES) — mất khi huỷ.
  Cổng đã thử hai chiều (mẫu kết thúc thật: 0 0 ⇒ chờ; mẫu có sẵn trong log: 1 1). Đã báo phiên MultiVD-AQ (không giữ gì, không xếp job).

## 27/09 18:35 — 158: mwg_mixsrc f3 XONG (test ROC 0,906, best ep16, dừng sớm ep24, 14,9 giờ) → đang f4 → f5

mixsrc n=3 (f1–f3), ghép cặp: − baseline ROC −0,006 2/3, PR −0,018 0/3, F1@0,5 +0,010 1/3; − mwg_assemble ROC −0,037 0/3,
PR −0,050 0/3, F1@0,5 −0,060 0/3. Vẫn như f1–f2: trộn nguồn một pha không hơn baseline, kém hẳn hai pha (mwg_assemble).

## 27/09 10:41 → XONG 17:19 trên 161: NGUỒN FULL KHÔNG C/C++ (người dùng: "thử full no ccpp lên máy đang trống", "n=5 luôn dễ so")

**Xong 17:19:40** — driver in `FPE 161 XONG`; 11/11 mục `xong rc=0`, đủ json + probs.npz; hparam khớp 48/48 mọi ô. GPU 161 rảnh.
n=5, seed 42, một máy (161, cùng máy với full + baseline). Δ ghép cặp theo fold (hoà = |Δ| ≤ 1e-3):

| so sánh | F1@0,5 | F1@val | ROC | PR |
|---|---|---|---|---|
| full_noccpp − baseline | +0,050 5/5 | +0,043 4/5 (+1 hoà) | +0,030 5/5 [+0,002..+0,062] | +0,029 5/5 |
| noRAS_full_noccpp − baseline | +0,033 4/5 | +0,015 3/5 | +0,010 3/5 | +0,007 3/5 |
| full_noccpp − full (có ccpp) | −0,004 (1+/3−/1 hoà) | −0,004 (2+/2−/1) | **−0,007 (0+/4−/1 hoà)** [−0,015..+0,000] | **−0,010 0/5** [−0,019..−0,003] |
| noRAS_full_noccpp − noRAS_full | +0,009 2/5 | −0,005 2/5 | +0,003 3/5 | −0,001 3/5 |
| RAS − noRAS (full_noccpp) | +0,017 4/5 | +0,028 4/5 (+1 hoà) | +0,020 4/5 | +0,022 4/5 (+1 hoà) |

ROC từng fold — baseline 0,907 0,892 0,904 0,945 0,881 · full 0,922 0,960 0,942 0,961 0,927 · full_noccpp 0,919 0,954 0,930 0,947 0,927.
Bỏ ccpp làm mwg_assemble thấp hơn full có ccpp một chút về thứ hạng (ROC −0,007, PR −0,010 cùng dấu mọi fold) — dưới sàn nhiễu chạy lại 0,010
(CLAUDE.md §2), một seed, một máy ⇒ quan sát, chưa phải kết luận. Không có tác dụng như vậy ở nhánh noRAS.

- Bậc **2 (xác nhận, n=5, seed 42)** theo lời người dùng — để ghép cặp thẳng với 5 fold của `full` (cùng máy 161).
- Nguồn `data/final_experiment_data/phase1_full_noccpp/` = đúng các hàng Java + JS của `phase1_full`, **không chia lại** (như `common_noccpp`):
  6 438 hàm (3 219/3 219) — Java 5 184, JS 1 254 (hơn common 264 hàm JS ngoài CWE chung); train 5 524 / val 914; 0 hàng ccpp; đúng id + thứ tự
  của phase1_full bỏ ccpp. Dựng bằng `tools/v4/build_final_experiment_data.py --full_noccpp_only` (hàm lọc tổng quát hoá; đường common dựng lại
  vào thư mục tạm khớp md5 từng byte với bản đang dùng).
- Run (runs.json): `p1_mwg_full_noccpp` (Pha 1 như p1_mwg_full, chỉ khác `--data_root`) → `mwg_assemble_full_noccpp` (RecAdam+ASAM) và
  `mwg_assemble_noRAS_full_noccpp` (AdamW), f1–f5. Chạy khô: lệnh trùng từng ký tự với bản full sau khi đổi tên; `fpe.py effective` chỉ khác
  run_name/checkpoint_path/init_ckpt/data_root. Log tiến trình phút đầu: train n=5524, val n=914, PAIR LOSS 2 369 cặp + 786 lẻ.
- Driver 161 PID 816419 (lock giữ thật), hàng đợi fold-major 11 mục; log khối trước lưu ở `state/driver_161_block_full.out`. Pha 1 ~15 phút/epoch (như common_noccpp) ⇒ xong ~12:47; cả khối ~17:00.

## 27/09 04:5x — KẾT QUẢ VAL PHA 1 (mọi nguồn, seed 42, warmup 0,25, 8 epoch; val 15 % — >50 % hàng val có nửa cặp nhãn ngược trong train nên val thấp là bình thường)

| Pha 1 | nguồn | train | val ROC @ep chọn | F1@0,5 | PR | thoát bình nguyên (train loss) | máy |
|---|---|---|---|---|---|---|---|
| p1_mwg_common | common | 10 236 | **0,540 @3** | 0,497 | 0,546 | ep3 (1,375 < 1,48) | 158 |
| p1_mw_common (không đồ thị) | common | 10 236 | **0,575 @4** | 0,547 | 0,556 | ep4 (kẹt tới ep3) | 158 |
| p1_mwg_origbabel_common | common | 10 236 | **0,566 @3** | 0,508 | 0,550 | ep3 | 158 |
| p1_mwg_4cwe | 4cwe | 759 | **0,645 @4** | 0,578 | 0,582 | ep3 (1,315 < 1,40) | 161 |
| p1_mwg_common_noccpp | common không C/C++ | 5 259 | **0,534 @2** | 0,501 | 0,518 | ep3 (1,547 < 1,82) | 161 |
| p1_mwg_full | full | 15 577 | **0,520 @2** | 0,511 | 0,506 | ep3 (1,546 < 1,64) | 161 |
| p1_mwg_full_noccpp | full không C/C++ | 5 524 | **0,496 @2** | 0,462 | 0,508 | ep3 (1,527) | 161 |

- Lượt warmup 0,10 của p1_mwg_common (dừng 25/09 00:58) KẸT: ep2 1,471 / 0,542.
- **Cảnh báo train loss tự động** (người dùng 27/09: "lưu ý cẩn thận các trường hợp loss train không giảm"): `fpe.py stall <run> <log>`, gắn vào
  monitor v5 (158) / v2 (161, vast) và nhãn đỏ trên artifact. Thử hai chiều: log cắt p1_mw_common@ep3 → báo KẸT; p1_mwg_common@ep3 → không;
  Pha 2 giả loss đứng → báo; bình thường → không; mixsrc giả val < 0,6 @ep6 → báo.
- **05:20 — luật Pha 2 báo NHẦM, đã hiệu chỉnh.** `mwg_assemble_full` f1 bị báo ở ep5 ("không giảm 3 epoch" 0,680 → 0,704) rồi ep6/ep7
  ("best val < 0,7"), trong khi nó đang lên (ep7 loss 0,660, val 0,678). Phát lại luật cũ trên **mọi tiền tố** của 35 log Pha 2/baseline đã xong:
  báo nhầm **8/35** (ở ep5) và **13/35** (ở ep6) — test ROC các log đó 0,89–0,96. RecAdam+ASAM đi ngang 4–8 epoch sau warmup là bình thường;
  chậm nhất: loss lần đầu < 0,6 ở **ep11** (mw_assemble f3), best val ≥ 0,7 ở **ep9**. Luật mới: ep ≥ 13 mà loss chưa từng < 0,6; ep ≥ 12 mà
  best val < 0,7. Thử hai chiều: 36 log thật (kể cả full f1 đang chạy) → **0** báo; giả kẹt (loss 0,69, val 0,55) → báo ở ep12/13;
  giả loss giảm nhưng val đứng 0,6 → báo ở ep12. Luật Pha 1 và mixsrc giữ nguyên.
- **09:29 — 161 XONG toàn bộ hàng đợi**: đối chiếu 33 mục kỳ vọng = 17 `xong rc=0` + 16 bỏ qua (skip.txt, chạy ở vast) + 0 thiếu.
  Driver KHÔNG in dòng `XONG` vì `run.sh` bị sửa TẠI CHỖ lúc 02:40 khi driver đang chạy (từ 26/09 16:53) ⇒ sau vòng lặp bash đọc
  lệch offset, `line 55: syntax error`. Vòng lặp đã parse trọn nên mọi fold chạy bằng mã cũ — kết quả không ảnh hưởng. 158 an toàn
  (fd 255 trỏ `(deleted)` — bản đó thay bằng rsync); vast khởi động SAU lần sửa. GPU 161 rảnh từ 09:30, chờ người dùng chọn việc.
- **Nguồn full n=5 (bậc xác nhận, 1 seed, 1 máy):** `mwg_assemble_full − baseline` F1@0,5 +0,054 5/5, F1@val +0,046 5/5, ROC +0,037 5/5,
  PR +0,039 5/5. `noRAS_full − baseline` F1@0,5 +0,024 4/5, ROC +0,007 3/5, PR +0,008 4/5. `full − noRAS_full` ROC +0,030 5/5, PR +0,031 5/5,
  F1@0,5 +0,030 4/5, F1@val +0,027 2/5.
- **10:24 — vast XONG, 10:28 ĐÃ HUỶ 52805338.** Driver in `FPE vast XONG`; `finalize_vast.sh` (scratchpad): sync kết quả, kéo
  `state/vast_final/{driver_vast.log,driver_vast.out,setup_env.sh}` + `_FinalPaperExperiment/_repro_keep_vast/` (md5 khớp LC_ALL=C), 0 file
  logs/results còn trên vast, 17/17 mục `xong rc=0`, 20/20 ô 4cwe+noccpp trên 161. `vastai destroy instance 52805338 -y` → biến khỏi danh sách;
  51144271 (nhãn cuongtm, ghim KHÔNG huỷ) nguyên. Ckpt Pha 1 `p1_mwg_4cwe`, `p1_mwg_common_noccpp` vẫn ở `/drive1/.../model/final_paper/`.
- **Nguồn noccpp n=5:** `mwg_assemble_noccpp − baseline` F1@0,5 +0,060 5/5, F1@val +0,037 4/5, ROC +0,031 5/5, PR +0,032 5/5.
  `noRAS_noccpp − baseline` F1@0,5 +0,038 4/5, F1@val +0,044 4/5, ROC +0,008 3/5, PR −0,008 2/5. `RAS − noRAS` PR +0,040 5/5, ROC +0,023 3/5.
- **Nguồn 4cwe n=5:** `mwg_assemble_4cwe − baseline` F1@0,5 +0,048 5/5, F1@val +0,030 5/5, ROC +0,035 5/5, PR +0,034 5/5.
  `noRAS_4cwe − baseline` F1@0,5 +0,045 4/5, ROC +0,022 4/5. `RAS − noRAS` ROC +0,013 4/5, F1@0,5 +0,003 2/5, PR +0,008 2/5.

## 27/09 02:41 — THÊM MÁY VAST `final_paper` (instance 52805338, RTX A4000, 0,087 $/h) — người dùng: "chuyển bớt task cho máy này… dùng vast cho task nào cũng được"

- SSH: `ssh -p 64350 root@162.200.81.25` (khoá gắn bằng `vastai attach ssh 52805338`). Đĩa 19 GB. Container giờ UTC ⇒ `run.sh` giờ `export TZ=ICT-7`.
- Môi trường `/workspace/venv311`: Python 3.11.14 + gói ghim TRÙNG vdenv (torch 2.9.1+cu128, CUDA 12.8, cuDNN 91002, transformers 4.57.1,
  tokenizers 0.22.1, numpy 2.3.4, sklearn 1.7.2, scipy 1.16.3; dựng bằng `uv`, script `/workspace/setup_env.sh`). venv sẵn của image (py3.10, numpy 2.2.6) KHÔNG dùng.
- Đã chép + đối chiếu md5 (LC_ALL=C hai phía): src_final, SVEN nocomment, CodeBERT, ckpt Pha 1 `p1_mwg_4cwe`, `p1_mwg_common_noccpp` — 41/41 khớp.
- runs.json `hosts.vast` (có `port`, `hostname`); script mới `remote_env.sh`, `sync_remote.sh <host>`, `live_remote.sh <host>` (sync_158/live_158 thành vỏ gọi chúng);
  `launch.sh` hỗ trợ cổng ssh.
- Driver vast PID 2003: `repro_baseline:1` (kiểm trùng bit với 158 — epoch 1 đã trùng) → f2..f5 × [mwg_assemble_4cwe, noRAS_4cwe, mwg_assemble_noccpp, noRAS_noccpp].
  161 bỏ 16 mục đó qua `_FinalPaperExperiment/state/skip.txt` (plan trên 161 đã kiểm). Monitor `scratchpad/mon_fpe_vast.sh`.
- HUỶ vast: khi hàng đợi vast xong + đã `sync_remote.sh vast` và đối chiếu — theo VAST_RULES (DESTROY, kiểm output). Ckpt Pha 2 tự xoá sau test; không cần kéo ckpt về.

## 26/09 16:5x — PHÂN LẠI VIỆC: 158 = mwg_mixsrc, 161 = các nguồn khác (người dùng + comment artifact 11:30)

- Người dùng: "Bạn chỉ dùng 1 máy chạy mwg_mixsrc thôi vì task này lâu (như comment)… Chạy thêm các nhánh của nguồn khác"; chọn: 158 giữ
  mixsrc, DỪNG f5 trên 161; nguồn khác = common không C/C++ + full + 4cwe, nhánh chính mỗi nguồn (p1_mwg + mwg_assemble + noRAS).
- 161: dừng driver 3617481 rồi trainer mixsrc f5 (ep9, best ep8 val 0,7287) lúc 16:4x; log dở → `clean/logs/_dung_giua_chung/final_mwg_mixsrc_f5_161_dung_ep9.log`,
  xoá ckpt dở. 158 `state/skip.txt` chỉ còn `p1_mwg_common_noccpp:1` ⇒ 158 chạy mixsrc f2 (đang) → f3 → f4 → f5 (plan trên 158 đã kiểm).
- 161 driver **3951578** (16:53), 33 mục: p1_mwg_4cwe → p1_mwg_common_noccpp → f1 [mwg_assemble_4cwe, noRAS_4cwe, mwg_assemble_noccpp,
  noRAS_noccpp] → p1_mwg_full → f1 [mwg_assemble_full, noRAS_full] → f2..f5 cả 6 run. Danh sách: `scratchpad/queue_161_sources.txt`.
- runs.json thêm 8 run (p1_mwg_full, p1_mwg_4cwe, mwg_assemble[_noRAS]_{noccpp,full,4cwe}); cùng công thức common (Pha 1 warmup 0,25;
  Pha 2 warmup 0,1; RecAdam+ASAM / AdamW). Nguồn full, 4cwe hết là "tham khảo".
- Artifact: thêm F1 dòng 2 trong ô fold (comment 10:35), nhóm nguồn mới tự hiện.

## 26/09 10:21 — CHIA HAI MÁY (người dùng: "Có thể chia bớt ra chạy trên cả 2 máy cho nhanh nhé")

- **158** (driver 2053162, đang `mwg_mixsrc` f1): làm tiếp mixsrc f2, f3; bỏ f4, f5, `p1_mwg_common_noccpp` qua
  `/data/ntat/MultiVD/_FinalPaperExperiment_staging/state/skip.txt` (driver đang chạy đọc `fpe.py plan` mỗi fold; không phải dừng nó).
- **161** (driver 3617481, `launch.sh 161`, lock `_FinalPaperExperiment/state/fpe.lock`): `repro_baseline:1` (kiểm lặp lại bit-với-bit
  baseline f1 của 158 — nếu trùng thì ghép cặp fold chạy hai máy là hợp lệ) → mwg_mixsrc f5 → f4 → p1_mwg_common_noccpp.
  Monitor cục bộ `scratchpad/mon_fpe_161.sh`. Kết quả 161 ghi thẳng `_FinalPaperExperiment/{logs,results}` (không cần sync).
- **Lỗi đã chặn trước khi gây hại**: `run.sh` làm `eval "$(fpe.py plan …)" || continue` — `eval` chuỗi rỗng trả 0 nên nhánh `||`
  KHÔNG BAO GIỜ chạy; nếu `plan` thoát lỗi, driver sẽ dùng lại lệnh của fold TRƯỚC (xoá kết quả rồi chạy lại). Vì thế skip KHÔNG thoát
  lỗi mà trỏ `INIT_CKPT` tới file không tồn tại ⇒ nhánh "thiếu checkpoint — bỏ qua" (chạy trước mọi lệnh xoá). Thử trong sandbox với
  CHÍNH `run.sh` đang chạy trên 158 (md5 8a24a081) và bản mới: fold bỏ qua giữ nguyên file mồi, fold khác chạy. `run.sh` mới: bắt lỗi
  plan trước khi eval (`PLAN=$(…) || continue; eval "$PLAN"`), thử chiều lỗi thật (fold 9) — không chạy lại fold trước.
- **10:31 repro_baseline f1 (161) TRÙNG TỪNG BIT baseline f1 (158)**: mọi chỉ số, 152 xác suất test + val, ngưỡng (md5 `b2b54bf9`).
  ⇒ ghép cặp fold của hai máy hợp lệ (FACTS §86). 161 chuyển sang mwg_mixsrc f5 lúc 10:31.
- **13:14 mwg_mixsrc f5 (161) ep4**: train loss 0,7014 (vẫn quanh ln2) nhưng val ROC 0,487 → **0,612**, val loss 0,700 → 0,680 ⇒ bắt đầu học
  (chậm hơn f1 một epoch) ⇒ KHÔNG coi là kẹt, chạy tiếp. Ghi chú: train loss của mixsrc tính trên tập trộn 96 % là nguồn nên đứng quanh
  ln2 cả khi mô hình đã học đích (f1 ep4: train 0,6885 mà val ROC 0,873) — phán kẹt ở mixsrc phải nhìn val SVEN, không nhìn train loss.
- **17:15 mwg_mixsrc f2 (158) ep4: val ROC 0,4997, train 0,7008 ⇒ theo luật mixsrc (val SVEN ≈ 0,5 ở ep4) là "kẹt". QUYẾT ĐỊNH: KHÔNG
  chạy lại riêng fold này với warmup 0,25** — lệch luật đã ghi, lý do: (1) một run phải cùng cấu hình ở mọi fold (f1 đã xong với 0,10);
  đổi riêng f2 là trộn hai cấu hình trong một run; (2) cứu riêng fold kém = chọn lọc theo chính kết quả. f5 cũng ì tới ep7 (0,62) rồi mới lên
  0,73 ở ep8 — patience 8 cho f2 tới ep11. Nếu hết patience vẫn ~0,5 thì ghi "mixsrc f2 không học được với warmup 0,10" và báo người dùng.
- Chỉ số theo CWE (comment của người xem artifact 26/09): `fpe.py dbrows` giờ tính ROC/F1@0,5 theo CWE từ `.probs.npz` — PHẢI chạy bằng
  python của vdenv (`/home/ntat/miniconda3/envs/vdenv/bin/python scripts/fpe.py dbrows 158`), python3 hệ thống không có numpy ⇒ bỏ cwe.

## 26/09 05:14 — KHỐI COMMON XONG 5 run × 5 fold trên **158** (bậc 2: n=5, seed 42, MỘT máy). Đang chạy tiếp: `mwg_mixsrc` rồi `p1_mwg_common_noccpp`

Kiểm tham số: mọi ô khớp `runs.json` (baseline 23, còn lại 48 tham số). Tất định strict chạy suốt, 0 lỗi kernel.

| run | ROC | F1@0,5 | F1@val | PR |
|---|---|---|---|---|
| baseline | 0,9056 ± 0,024 | 0,8090 ± 0,025 | 0,8160 ± 0,029 | 0,9118 ± 0,021 |
| **mwg_assemble** | **0,9323** ± 0,017 | 0,8614 ± 0,023 | 0,8424 ± 0,029 | 0,9371 ± 0,018 |
| mwg_assemble_noRAS | 0,9048 ± 0,012 | 0,8256 ± 0,026 | 0,8123 ± 0,027 | 0,9086 ± 0,016 |
| mwg_assemble_nofixbracket | 0,9317 ± 0,024 | **0,8731** ± 0,038 | **0,8508** ± 0,047 | **0,9379** ± 0,023 |
| mw_assemble | 0,9160 ± 0,021 | 0,8484 ± 0,013 | 0,8317 ± 0,018 | 0,9253 ± 0,012 |

Δ ghép cặp theo fold (TB, số fold dương /5, [min…max]):

| phép so | ROC | F1@0,5 | F1@val | PR |
|---|---|---|---|---|
| mwg_assemble − baseline (toàn gói) | +0,027 4/5 [−0,001…+0,064] | +0,052 5/5 | +0,026 4/5 | +0,025 5/5 |
| mwg_assemble − noRAS (RecAdam+ASAM) | **+0,028 5/5** [+0,004…+0,045] | +0,036 5/5 | +0,030 4/5 | **+0,029 5/5** |
| mwg_assemble − nofixbracket (bản sửa BABEL) | +0,001 2/5 | −0,012 2/5 | −0,008 3/5 | −0,001 3/5 |
| mwg_assemble − mw_assemble (đồ thị) | +0,016 4/5 [−0,008…+0,041] | +0,013 3/5 | +0,011 4/5 | +0,012 3/5 |
| noRAS − baseline | −0,001 3/5 | +0,017 3/5 | −0,004 3/5 | −0,003 2/5 |
| mw_assemble − baseline | +0,010 3/5 | +0,039 5/5 | +0,016 3/5 | +0,014 5/5 |

Đọc (KHÔNG viết vào bài khi chưa lặp phần cứng khác, CLAUDE.md §2): lợi ích toàn gói ~+0,027 ROC nằm gần hết ở RecAdam+ASAM
(Pha 1 + AdamW thường ≈ baseline); bản sửa BABEL ≈ 0; đồ thị +0,016 ROC 4/5. 5/5 = sàn Wilcoxon p = 0,0625.

## 25/09 00:59 — ĐÃ XONG KHỐI COMMON 26/09 05:14 (xem trên) trên **158**: khối common, Pha 1 **warmup 0,25** (lần 2). Người dùng TỰ QUYẾT từ 01:00

- Người dùng 25/09 ~01:00: "Đổi warm nhé"; rồi "Sau đây tự quyết nhé tôi sẽ nghỉ và không theo dõi được, chạy lượt giờ để đạt setting tốt nhất nhé".
- Lần 1 (warmup 0,10) KẸT: ep1 1,5082/0,5286, ep2 1,4711/0,5417 (train loss/val ROC); bình nguyên dự báo 0,69 + 1,31 × 0,606 = 1,48
  (6 198/10 236 hàng trong cặp) — y hệt ô A FACTS §84. Dừng chủ động 00:58 (rc 143); log giữ ở `clean/logs/_dung_giua_chung/final_p1_mwg_common_warm010_seed42.log`.
- Đổi: `runs.json` flag_set `phase1` `--warmup_ratio 0.25` (mọi run Pha 1 kể cả noccpp) + `mwg_mixsrc` 0,25 (nhánh đồ thị khởi tạo ngẫu nhiên,
  cùng cơ chế kẹt); Pha 2 giữ 0,10; baseline 0. Dry-run hai chiều đạt; log tiến trình xác nhận `--warmup_ratio 0.25`.
- Driver PID 2053162 (`launch.sh 158`, 34 mục = p1_mwg_common + hàng đợi fold-major như dưới). Monitor `scratchpad/mon_fpe_158_v3.sh`.
- **Luật tự quyết trong đêm** (ghi TRƯỚC khi có số):
  1. Pha 1 kẹt = ep3 train loss còn trong ±0,03 của bình nguyên (common 1,48) **và** val ROC < 0,60 ⇒ dừng, chạy lại với nút tiếp theo:
     ô D §84 = warmup **0,10** + `pair_margin 0,5` (graph_lr 2e-5) — thoát ngay ep1–2 trên pool jsCjv, 3 epoch. [Sửa 25/09 15:4x: bản đầu ghi nhầm
     "warmup 0,25 + margin 0,5"; file dựng sẵn `scratchpad/runs_D.json` vẫn đúng 0,10.] KHÔNG áp dụng — Pha 1 thoát ở ep3 với warmup 0,25.
     Không đổi seed (săn seed = thiên lệch chọn lọc, §84).
     Val ROC thấp mà train loss GIẢM rõ thì KHÔNG phải kẹt (common nặng C/C++ có thể khó hơn pool jsCjv) — để chạy tiếp.
  2. Ô lỗi vì kernel không tất định ⇒ tìm op, sửa trong `det_launch.py`/trainer theo cách vẫn tất định, xếp chạy lại ô đó. Không tắt strict.
  3. Mọi ô khác chạy đủ và ghi lại (CLAUDE.md §3), không bỏ ô vì Pha 1 yếu.
- **Kết quả (bậc 2, n=5 đang chạy — mới fold 1, CHƯA kết luận)**:
  | ô | xong | test ROC | F1@0,5 | F1@val | PR | best ep |
  |---|---|---|---|---|---|---|
  | p1_mwg_common f1 (warmup 0,25) | 06:10 | 0,5398 (= val) | 0,4975 | 0,5254 | 0,5464 | 3/8 |
  | baseline f1 | 06:20 | 0,9068 | 0,8157 | 0,8220 | 0,9274 | 15 (dừng 23) |
  | mwg_assemble f1 | 06:57 | 0,9169 | 0,8468 | 0,8468 | 0,9294 | 12 |
  | mwg_assemble_noRAS f1 | 07:16 | 0,8854 | 0,7953 | 0,7744 | 0,9013 | 9 (dừng 17) |
  | p1_mw_common f1 (không đồ thị) | 12:25 | 0,5750 (= val) | 0,5467 | — | 0,5562 | 4/8 (kẹt tới ep3, thoát ep4) |
  | p1_mwg_origbabel_common f1 (BABEL gốc) | 17:38 | 0,5664 (= val) | 0,5077 | — | 0,5501 | 3/8 (thoát ep3) |
  | mwg_assemble_nofixbracket f1 | 18:16 | 0,8910 | 0,8068 | 0,7904 | 0,9126 | 13 |
  | mw_assemble f1 | 19:02 | 0,9030 | 0,8553 | 0,8146 | 0,9289 | 18 (dừng 26) |
  **19:02 đủ lát fold 1** (5 run đích). Fold 2 bắt đầu 19:02 (baseline). Mỗi fold ~2,5 giờ ⇒ fold 5 xong ~05:00 26/09, rồi mwg_mixsrc.
  **21:41 đủ lát fold 2** — test ROC f2: baseline 0,8922 · mwg_assemble 0,9565 · noRAS 0,9112 · nofixbracket 0,9468 · mw_assemble 0,9156.
  Δ ghép cặp so với baseline (f1, f2): mwg_assemble ROC +0,010 / +0,064, F1@0,5 +0,031 / +0,099 — **2/2 trên cả 4 chỉ số**;
  noRAS, nofixbracket 1/2 (âm ở f1, dương ở f2); mw_assemble F1@0,5 và PR 2/2, ROC 1/2. n=2 — chỉ để nhìn hướng.
  **23:44 đủ lát fold 3** — test ROC f3: baseline 0,9036 · mwg_assemble 0,9243 · noRAS 0,9073 · nofixbracket 0,9345 · mw_assemble 0,9326.
  Δ ghép cặp n=3 (bậc 1 kiểm chứng — KHÔNG viết vào bài): mwg_assemble ROC +0,032 3/3 [+0,010…+0,064], F1@0,5 +0,070 3/3,
  F1@val +0,046 3/3, PR +0,031 3/3 · noRAS ROC +0,000 2/3, PR −0,002 1/3 · nofixbracket ROC +0,023 2/3 · mw_assemble ROC +0,016 2/3,
  F1@0,5 +0,048 3/3, PR +0,016 3/3. Fold 4 bắt đầu 23:44.
  **02:32 đủ lát fold 4** — test ROC f4: baseline 0,9446 (cao nhất) · mwg_assemble 0,9436 · noRAS 0,9040 · nofixbracket 0,9497 · mw_assemble 0,9396.
  Δ n=4: mwg_assemble ROC +0,024 **3/4** [−0,001…+0,064], F1@0,5 +0,052 4/4, F1@val +0,023 3/4, PR +0,024 4/4 (f4 ≈ 0 mọi chỉ số) ·
  noRAS ROC −0,010 2/4, PR −0,009 1/4 · nofixbracket ROC +0,019 3/4 · mw_assemble ROC +0,011 2/4, F1@0,5 4/4, PR 4/4.
  Chuỗi 3/3 của mwg_assemble đã gãy ở f4 (đúng mẫu hình "n=3 co lại ở n=5", CLAUDE.md §1). Fold 5 bắt đầu 02:32.
  **03:31 mwg_assemble + baseline ĐỦ 5/5** (bậc 2 xác nhận, MỘT máy 158, seed 42 — chưa lặp phần cứng khác, CLAUDE.md §2):
  | chỉ số | baseline TB | mwg_assemble TB | Δ ghép cặp | +/5 | min … max |
  |---|---|---|---|---|---|
  | ROC | 0,9056 | 0,9323 | +0,0266 | 4/5 | −0,001 … +0,064 |
  | F1@0,5 | 0,8090 | 0,8614 | +0,0524 | 5/5 | +0,000 … +0,099 |
  | F1@val | 0,8160 | 0,8424 | +0,0264 | 4/5 | −0,046 … +0,079 |
  | PR | 0,9118 | 0,9371 | +0,0253 | 5/5 | +0,001 … +0,072 |
  f5: baseline 0,8810 · mwg_assemble 0,9201. 5/5 ⇒ p=0,0625 (sàn Wilcoxon); f4 ≈ 0 trên mọi chỉ số.
  **03:49 noRAS ĐỦ 5/5**: TB ROC 0,9048 (Δ vs baseline −0,001 3/5), F1@0,5 +0,017 3/5, F1@val −0,004 3/5, PR −0,003 2/5 ⇒ ≈ baseline.
  **mwg_assemble − noRAS** (phần riêng của RecAdam+ASAM ρ 0,5 ở Pha 2, cùng checkpoint Pha 1): ROC **+0,028 5/5** [+0,004…+0,045],
  F1@0,5 +0,036 5/5, F1@val +0,030 4/5, PR +0,029 5/5. Tức Pha 1 + AdamW thường KHÔNG hơn baseline; lợi ích nằm ở RecAdam+ASAM.
  **04:31 nofixbracket ĐỦ 5/5** (TB ROC 0,9317): vs baseline ROC +0,026 4/5, F1@0,5 +0,064 4/5, PR +0,026 4/5.
  **mwg_assemble − nofixbracket** (phần riêng của bản sửa BABEL: typed 4 quan hệ, bỏ dòng ngoặc, co_use chuỗi, clean_gadget sửa literal):
  ROC +0,001 2/5 [−0,016…+0,026], F1@0,5 −0,012 2/5, F1@val −0,008 3/5, PR −0,001 3/5 ⇒ **≈ 0, không tách khỏi sàn nhiễu 0,010**.
- **26/09 04:5x — tự quyết: `mwg_mixsrc` warmup 0,25 → 0,10** (trước khi nó bắt đầu; chỉ đổi `runs.json`, rsync lên 158, md5 khớp,
  `fpe.py plan` trên 158: mixsrc 0,1, p1_mwg_common_noccpp vẫn 0,25). Lý do: 0,25 × 30 epoch = 7,5 epoch ≈ 8 giờ/fold chỉ warmup
  (12 498 hàm, ~65 phút/epoch); mixsrc `pair_loss 0` nên không có bình nguyên điểm-hằng của pair loss; 0,10 × 30 = 3 epoch đã dài hơn
  2 epoch warmup giúp mọi Pha 1 thoát. Nếu mixsrc kẹt (train loss đứng ở ln2 ≈ 0,69 qua ep4) ⇒ dừng, chạy lại 0,25.
  Strict tất định chạy được cho train_baseline.py, train_mwg.py Pha 1 và Pha 2 RecAdam+ASAM, Pha 2 AdamW.
- **09:14 — `p1_mw_common` (KHÔNG đồ thị) ep3 train loss 1,4982 > 1,43 ⇒ theo luật là KẸT** (ep1 1,5394/0,519, ep2 1,5160/0,530,
  ep3 1,4982/0,528). **Quyết định: KHÔNG dừng, KHÔNG đổi công thức cho riêng nhánh này** — lệch luật đã ghi, lý do: luật viết cho Pha 1 CHÍNH;
  áp margin 0,5 riêng cho mw thì mw_assemble vs mwg_assemble đổi HAI biến (đồ thị + công thức Pha 1). Giữ công thức Pha 1 ĐỒNG NHẤT
  (warmup 0,25, margin 1,0) cho mọi nhánh; ghi "Pha 1 kẹt" kèm val (CLAUDE.md §3). Nó vẫn có thể thoát ở ep4+ (chạy đủ 8 epoch).
  **Phát hiện phụ**: kẹt cả khi KHÔNG có nhánh đồ thị ⇒ cơ chế "9,45 M tham số đồ thị khởi tạo ngẫu nhiên kéo head về hằng số" (§84/§16.8b)
  không giải thích hết; pair loss/margin là nghi phạm. Chờ đủ 8 epoch rồi ghi FACTS.
  **09:51 ep4: train loss 1,4146 (−0,07 dưới bình nguyên), val ROC 0,5750 (best) ⇒ THOÁT MUỘN một epoch** — giữ chạy là đúng;
  luật "phán ở ep3" quá sớm cho nhánh không đồ thị (lần sau: phán ở ep4 khi warmup 0,25 × 8 epoch = 2 epoch).
  - **Bổ sung 01:50 (trước khi có ep2/ep3)**: val ROC của Pha 1 KHÔNG đo được "đang học" — **1 043/1 806 hàng val (57,8 %) có nửa kia
    của cặp (nhãn ngược, mã gần giống) nằm trong train**, nên học thuộc train làm val ROC **tụt dưới 0,5**. Tiền lệ: `clean/logs/p1_v2common/f1.log`
    (warmup 0,25, 8 ep): train loss 1,76 → 1,69 → 1,58 → 1,37 → … 0,88 (thoát bình nguyên 1,73) nhưng val ROC 0,47 → 0,52 → 0,50 → … 0,40, best ep2;
    checkpoint ep2 đó vẫn cho transfer +0,049 F1@0,5 5/5 ở Pha 2 (FACTS §85). ⇒ Phán kẹt CHỈ theo train loss: ep3 > bình nguyên − 0,05 (common: > 1,43).
    Epoch 1 lần 2: 1,5090 / 0,5332 (còn trong warmup 2 epoch). Ep2 1,4861 / 0,5263. **Ep3 1,3754 / 0,5398 (best) ⇒ THOÁT** bình nguyên
    (−0,11 < ngưỡng −0,05) ⇒ chạy tiếp, KHÔNG áp phương án D (đã chuẩn bị sẵn `scratchpad/runs_D.json`: warmup 0,10 + pair_margin 0,5).

## 24/09 23:19 — ĐÃ DỪNG 00:58 (kẹt, xem trên) trên **158**: khối final, `p1_mwg_common` fold 1 (Pha 1 MW + đồ thị BABEL đã sửa, batch 8)

- Người dùng: "Chạy giúp tôi trên 158 nhé batch 8 chạy luôn và nhớ cài monitor". Phóng bằng `_FinalPaperExperiment/scripts/launch.sh 158 "p1_mwg_common" "1"`.
- Driver `bash scripts/run.sh` PID **1828575** (session riêng), trainer `det_launch.py` PID 1828632; lock `/data/ntat/MultiVD/_FinalPaperExperiment_staging/state/fpe.lock` đang GIỮ.
- Log `/data/ntat/MultiVD/_FinalPaperExperiment_staging/logs/p1_mwg_common/fold1.log`; ckpt `/data/ntat/MultiVD/model/final_paper/p1_mwg_common/seed_42/fold1/best.pt`.
- Tham số hiệu lực (đọc từ log): det_launch deterministic=True, CUBLAS :4096:8, PYTHONHASHSEED 42, TF32 tắt; MWGraph typed R=4 Lmax 150,
  drop_bracket 1, co_mode chain, fusion cat; PAIR LOSS 1,00 / margin 1,00 (3 099 cặp + 4 038 lẻ); W 510 / S 384 / K 8; batch 8,
  lr = graph_lr = 2e-5, warmup 0,10, 8 epoch. Tiền xử lý 281 s.
- Monitor (local, `scratchpad/mon_fpe_158.sh`): báo từng epoch, best, lỗi, driver chết, GPU nhàn khi trainer sống, ssh hỏng.
- Xong: `bash scripts/sync_158.sh` -> `python3 scripts/fpe.py dbrows` -> ghi db artifact. Pha 2 `mwg_assemble*` phải chạy CÙNG máy (158).
- **Hàng đợi 158 — KHỐI COMMON ĐẦY ĐỦ** (người dùng 24/09 23:5x: "Chạy toàn bộ thí nghiệm với common trước nhé. Chạy cho đêm nay";
  "Dùng 158 tạm thôi cho đêm nay"). `scripts/queue_after.sh` PID **1911771** chờ driver 1828575 (PID + giờ khởi động) rồi
  `exec bash scripts/run.sh 158 "<33 mục run:fold>"` — fold-major (CLAUDE.md §1), Pha 1 đứng trước Pha 2 dùng nó:
  1. baseline:1 → mwg_assemble:1 → mwg_assemble_noRAS:1 (lát so sánh chính fold 1, ngay sau p1_mwg_common)
  2. p1_mw_common → p1_mwg_origbabel_common (mỗi cái ~7 giờ)
  3. mwg_assemble_nofixbracket:1 → mw_assemble:1
  4. fold 2..5: baseline → mwg_assemble → noRAS → nofixbracket → mw_assemble
  5. mwg_mixsrc fold 1..5 (ước 25–70 giờ) → 6. p1_mwg_common_noccpp (nguồn mới, xếp cuối vì người dùng: common trước)
  Queue cũ (1878877, chỉ noccpp) đã huỷ 23:56. Danh sách nguyên văn: `scratchpad/queue_night.txt`; log `state/queue_158.out`.
- Ước lượng: Pha 1 common ~55 phút/epoch (FACTS: ~0,3 s/hàng) ⇒ p1_mwg_common xong ~07:00 25/09. Khối common KHÔNG xong trong đêm.
- Artifact: mỗi sự kiện monitor -> `bash scripts/live_158.sh` (tiến độ epoch) -> ghi db `folds/<run>__f<F>`; fold xong ->
  `sync_158.sh` -> `fpe.py dbrows 158` -> ghi db. Monitor hết 30 phút thì bật lại ngay (người dùng: "không tắt monitor sớm").
- Nguồn mới `data/final_experiment_data/phase1_common_noccpp/` = đúng các hàng Java + JS của `phase1_common` (không chia lại):
  6 174 hàm (Java 5 184, JS 990); fold1 train 5 259 / val 915. 158 khớp md5. Cấu hình y hệt p1_mwg_common, chỉ khác data_root.
  4cwe giữ nguyên không Java (người dùng 24/09: "cứ để nguyên 4cwe thế đi" — CleanVul Java không có CWE/CVE).

## 24/09 22:55 — KHỐI FINAL PAPER ĐÃ CHUẨN BỊ (Pha 1 p1_mwg_common đã phóng 23:19, xem trên) (người dùng: "Chưa chạy bây giờ tạo và ghim sẵn trước"; chờ GPU 16 GB)

- Thư mục DUY NHẤT: `/drive1/cuongtm/ntat/MultiVD/_FinalPaperExperiment` — `scripts/` (mã + `runs.json`, thư mục để commit),
  `meta/`, `state/`, `logs/<run>/`, `results/<run>/` (một bản mỗi run, chỉ trên 161; 158 ghi tạm rồi `sync_158.sh` chuyển về).
- 9 run (n=5, seed 42, batch 8, lr = graph_lr = 2e-5): p1_mwg_common, p1_mwg_origbabel_common, p1_mw_common, baseline,
  mwg_assemble, mwg_assemble_noRAS, mwg_assemble_nofixbracket, mw_assemble, mwg_mixsrc — chi tiết `scripts/README.md`.
- Tất định strict qua `scripts/det_launch.py`. CHƯA thử khói trên GPU (cần trước lần chạy thật: kernel không tất định sẽ làm dừng).
- Artifact (ghim): https://claude.ai/artifact/XWwCLp7edw8dvjBTi1oRcj — kết quả ghi vào db collection `folds` bằng `fpe.py dbrows`.
- Pha 1 thử batch 4 trên 158 đã DỪNG 22:28 (người dùng: mọi batch 8); chưa có phép thử sập ở batch 8.

## 24/09 22:17 — ĐÃ DỪNG 22:28 (dừng theo yêu cầu, chưa xong epoch nào) trên **158**: Pha 1 MW+BABEL `p1_common_w010_glr2e-5` (thử xem có sập không). Chỉ Pha 1.

- Người dùng: "Pull lại mới nhất từ multibabel và chạy phase 1 mw graph trước… Log param hyper đúng"; "Dùng graph lr 2e-5";
  warmup "thử 0.1 trước"; val Pha 1 đổi 15 %; "Chạy trên 158"; chỉ `common` trước.
- Mã: `/data/ntat/MultiVD/src_final` = GraphTransferVD `multibabel` @ `8dc0321` (`code_snapshot/src`, md5 train_mwg `89ce73c5`,
  clean_gadget `8491b5f8` = bản xoá trắng mọi literal) + `train.py` của `src_mwg_theirs` (`e90630f5`, chỉ import).
- Dữ liệu: `/data/ntat/MultiVD/data/final_experiment_data` (29/29 file khớp md5 bản local), `phase1_common/fold1`: train 10 236 / val 1 806.
- Driver: `run/final_p1.sh` (lock `final_runs/final_p1.lock`, PID driver 1694300), log `final_runs/log/p1_common_w010_glr2e-5/f1.log`,
  ckpt `final_runs/model/p1_common_w010_glr2e-5/seed_42/best.pt`, kết quả `final_runs/results/p1_common_w010_glr2e-5/phase1/seed_42/fold1.json`.
- Tham số: seed 42, lr 2e-5, **graph_lr 2e-5**, **warmup 0,10**, wd 0,01, batch 4 (eval 8), micro 16, window 510 / stride 384 / K 8,
  agg mean, graph typed, drop_bracket 1, co_mode chain, epochs 8 (min 3, patience 8, chọn theo roc_auc), pair_loss 1,0 / margin 1,0, SAM 0.
- Dấu hiệu sập cần đọc: train loss đứng yên quanh ~1,7–1,9 và val ROC ~0,5 (kẹt §84) — hay train loss giảm mà val tụt (overfit).

## 24/09 — DỮ LIỆU FINAL: `data/final_experiment_data/` (người dùng: "chạy toàn bộ bằng folder này"). CHƯA CHẠY GÌ.

- Pha 1: `phase1_{full,common,4cwe}/fold1/{train,val,test}.jsonl` (+ `phase1_<loại>.jsonl` mọi hàng); đích: `sven_python_folds_nocomment/`.
- C/C++ từ PrimeVul paired+unpaired gộp (lọc T0–T4 toàn bộ): full = 5 888 nhãn 1 + 6 000 nhãn 0 (4 602 bản vá thuộc cặp + 1 398 lẻ,
  seed 42); common = 2 868 nhãn 1 + 3 000 nhãn 0 bốc từ 6 000; 4cwe = 105 nhãn 1 + 200 nhãn 0 bốc từ 3 000 (nhãn 0 KHÔNG xét CWE,
  người dùng chọn 24/09; 4cwe: nhãn 0 CWE ngoài 4 loại -> cwe_class -100). Java/JS như nguồn chuẩn.
- Tổng: full 18 326 (9 107/9 219), common 12 042 (5 955/6 087), 4cwe 893 (399/494). Dựng: `tools/v4/build_final_experiment_data.py`.
- ⚠️ Script chạy (`clean/scripts/qf_v3.sh`) vẫn trỏ pool v3 + `sven_python_folds_norm` — phải đổi sang thư mục này khi được yêu cầu chạy.

## 23/09 18:10 — DỮ LIỆU NGUỒN v4 XONG, CHỜ NGƯỜI DÙNG + ĐỒNG TÁC GIẢ SOÁT. KHÔNG CHẠY THÍ NGHIỆM.

Người dùng 23/09: *"Không chạy experience bây giờ… Tôi cần data để check trước… Dựng xong data đã chạy sau."*
- Hàng final 161 đã DỪNG 17:55 sau Pha 1 `v3common` (ROC 0,5376; số đo thời gian Pha 1 bẩn vì việc CPU song song).
  Không còn tiến trình train nào. Chạy lại SẠCH TỪ ĐẦU vào thư mục mới khi người dùng cho phép.
- Dữ liệu v4 (từ thô `/drive1/cuongtm/ntat/Archive/PrimeVulRaw` + CleanVul 3–4):
  `data/sources_v4/` (paired 4 622 cặp · unpaired 197 066 hàng · Java 2 592 cặp · JS 627 cặp),
  `data/sources_v4/_dropped/` + `_report.json` (mọi đơn vị bị loại), pool `data/mwsrc_v4{full,common,cwe4,common_nocc}`.
  **NGUỒN CHUẨN (chỉ dữ liệu giữ lại + README): `data/clean_sources_v4/{by_source,merged,merged_undersampled}/`.**
  Unpaired CHƯA vào pool (lệch nhãn 5 889 / 191 138) — chờ quyết định.
- 24/09: **PrimeVul gộp paired + unpaired** `data/sources_v4/ccpp_primevul_merged_*` (lọc T0–T4 trên toàn bộ, cặp đứng trước):
  197 063 hàm (5 888 / 191 175) = 4 602 cặp + 187 859 hàm lẻ (20 cặp ít hơn bộ paired riêng: một nửa trùng hàm NGƯỢC nhãn chỉ có
  ở unpaired → T1). **Undersample `full` trước** (giữ mọi nhãn 1 + mọi cặp, bốc nhãn 0 lẻ, seed 42) rồi mới lọc:
  `clean_sources_v4/by_source/primevul_merged_undersampled/` full 11 776 (5 888/5 888), common 5 658 (2 868/2 790), 4cwe 230 (105/125);
  `merged_undersampled/` (+ Java + JS) THAY bản cũ: full 18 214 (9 107/9 107), common 11 832 (5 955/5 877), 4cwe 818 (399/419).
  Luật phụ (common > 10 000, 4cwe lệch > 3:1) không kích hoạt. Chưa có pool `mwsrc` cho bộ này. Code + manifest ở worktree `clean_v3`, CHƯA push.
- Luật T0–T4 + xoá comment duy nhất: `tools/v4/` (local) = GraphTransferVD nhánh `clean_v3` @ `1d7153f` (dữ liệu; BABEL: `15f95a9`)
  (`scripts/clean_v3/build_sources_v4.py`, `export_clean_sources.py`, `code_snapshot/src/babel/strip_comments.py`; `train_mwg.py` = bản multibabel).
- Artifact soát: https://claude.ai/artifact/ERdvAAULmZptKRg6FyVrYq (bản 14).
- 24/09: `clean_v3` @ `15f95a9` = merge PR #3 (`fix_gadget`, xoá trắng chuỗi nhiều dòng) + bổ sung xoá trắng MỌI literal trong `babel/clean_gadget.py` (f-string một dòng, tiền tố f/L, regex JS, nháy lồng, chuỗi cắt ở Lmax). Chỉ đổi ĐỒ THỊ lúc chạy, không phải dựng lại `data/`; mọi run MW/BABEL trước đó dùng bản cũ (cạnh đổi ở 34/760 hàm SVEN, 155/1 254 hàm JS).
- ⚠️ `src_mwg/` trên 161 vẫn là BABEL cũ (trước 588ced1, không xoá comment lúc chạy). Chạy lại phải đồng bộ theo `clean_v3`. Nay gồm cả `babel/clean_gadget.py` @ `15f95a9`.
- ⚠️ Tập đích `sven_python_folds_norm` còn comment/docstring ở 392/760 hàm (251 có `#`, 235 có docstring). KHÔNG nhánh nào
  xoá comment trên văn bản đưa vào mô hình (baseline, MW, BABEL); BABEL chỉ bỏ qua comment khi tìm định danh dựng cạnh.
  24/09: đã dựng bản sạch **`data/sven_python_folds_nocomment`** (cùng cấu trúc: `data.jsonl` + `fold1..5/{train,val,test}.jsonl`,
  cùng thứ tự/nhãn/fold, chỉ thay `code`; `tools/v4/make_sven_nocomment.py`). Kiểm: 0 comment `#`, 0 docstring (Pygments + AST);
  AST trùng bản gốc bỏ docstring 748/748 hàm parse được, 12 hàm Python 2 trùng chuỗi token; 422 hàm đổi (392 comment/docstring +
  30 chỉ khoảng trắng cuối dòng); dòng không đổi trùng byte với gốc. Người dùng: TẠM giữ `norm` cho các run, chưa chuyển.

## 23/09 15:30 — 161: HÀNG FINAL SẼ DỪNG SAU PHA 1 `v3common`, CHỜ CHECK DATA RỒI CHẠY SẠCH TỪ ĐẦU

Người dùng 23/09: *"chạy xong fold đang dở rồi dừng; check dữ liệu xong chạy sạch từ đầu, tránh fill lỗ chỗ"*.
- Watcher `clean/scripts/qf_pause_after_p1.sh` (setsid) chờ dòng `PHA 1 v3common xong|LOI` rồi hãm + tắt
  `qf_v3.sh`/`qf_chain.sh`/`flock`; log ở `clean_final/logs/pause_after_p1.log`.
- Lý do chạy lại: (1) luật lọc mới **L1 một dòng (JS)** bỏ thêm 1 cặp (`sending_profiles.min.js`) có mặt
  trong `mwsrc_v3common` và `mwsrc_v3cwe4`; (2) epoch 3 Pha 1 final chậm ~22% vì tôi chạy việc CPU cùng lúc
  ⇒ số đo thời gian của ô này không sạch. Kết quả ROC/F1 không bị ảnh hưởng.
- `clean_final/` hiện có: baseline 5/5 fold (sạch), Pha 1 `v3common` (sắp xong). Lần chạy lại dùng
  **thư mục mới**, không tái dùng ô cũ.
- Luật mới + dữ liệu mới **chưa ghi vào `data/`, chưa commit** — chờ người dùng xác nhận qua artifact
  https://claude.ai/artifact/ERdvAAULmZptKRg6FyVrYq (bản 12).

## 23/09 11:57 — 161 (A4000): HÀNG ĐỢI FINAL, CHỈ NGUỒN CÓ C

Người dùng 23/09: *"Cứ chạy các thí nghiệm với src có cả C nhé còn không C chỉ để xem thôi"*.
⇒ `v3common_nocc` **đã bỏ khỏi** `clean/scripts/qf_chain.sh`; bản không C chỉ còn ở
`clean_local/` (thăm dò, Pha 1 val 0,5705 vs 0,5496 có C; Pha 2 đang chạy 5 fold).

**12:55** `q161_chain2.sh` xong `v3common_nocc` (5/5 fold) rồi tôi **dừng nó** — hai bước
còn lại của nó (3 đối chứng + `v3cwe4` vào `clean_local/`) trùng y hàng final (~5 h).
Không C, Pha 2, ghép cặp 5 fold (không C − có C): F1@0,5 +0,0090 3/5 · F1@val +0,0094 3/5 ·
ROC +0,0014 2/5 · PR −0,0069 2/5 ⇒ **không phân biệt được**.

```
12:56 BẮT ĐẦU : qf_after.sh -> qf_chain.sh  (flock /tmp/mvd_qf_chain.lock, log clean_final/logs/driver_final.log)
hàng final (clean_final/, src_mwg/, seed 42, n=5):
  1) baseline thường batch 16   2) v3common Pha 1+2   3) abl_noinit/nograph/norecsam
  4) v3cwe4 Pha 1+2 (ccpp 144 + js 475)
```

---

## 22/09 — DỮ LIỆU BỊ DỰNG LẠI. MỌI Ô CÓ PHA 1 TRÊN `v2*` ĐỀU PHẢI CHẠY LẠI.

Người dùng hỏi data đã dedup và bỏ bản build chưa. Kiểm ra **chưa**, và pool JS bẩn
tới mức đủ làm hỏng Pha 1:

| | js | java | ccpp |
|---|---:|---:|---:|
| đường dẫn `dist/ build/ .min.js vendor/` | **30,5 %** | 0,33 % | — |
| mã đã mangle (đo theo nội dung) | **20,1 %** | 0,02 % | 0,6 % |

Pool `jsCjv` của đồng tác giả — pool ĐÃ BIẾT là học được (Pha 1 ~0,68) — chỉ **0,93 %**
mangle. Gấp 24 lần. `v2cwe4` là pool js-nặng nhất (84 % js ⇒ 20,95 % hàng mangle) và
cũng là pool Pha 1 tệ nhất (0,4771, **dưới ngẫu nhiên**). Ba mảnh khớp một hướng.

**Bộ mới `data/sources_v3` + `data/mwsrc_v3*`** dựng bằng `tools/clean_v3.py` (lọc
`full` trước, `common`/`4cwe` thừa hưởng quyết định của `full`) và `tools/build_mwsrc_v3.py`.

### Phát biểu PHẢI RÚT LẠI

Bản trước của file này viết *"đường chuyển giao không mang thông tin ở cấu hình này"*.
**Không viết được nữa.** Nó dựa trên "cả hai checkpoint nguồn đều quanh mức ngẫu nhiên",
mà mức ngẫu nhiên đó phần lớn là **lỗi dựng pool**, không phải tính chất của transfer.

### Giữ nguyên có chủ ý — KHÔNG phải sót

- **Cặp bị chia đôi train/val** (~20 % hàng val có nửa kia trong train, độ giống trung
  vị 0,955, nhãn luôn ngược). Người dùng yêu cầu **giữ chia ngẫu nhiên** — cùng lý do
  reviewer nêu cho `sven_python_folds_norm` (CLAUDE.md §6): ca gần-trùng-ngược-nhãn là
  ca khó chứng minh mô hình học đặc trưng lỗ hổng chứ không học mẫu văn bản.
  ⚠️ Hệ quả: val NGUỒN của ta **khó hơn** val của `jsCjv` (4,5 % mẫu lẻ). So trực tiếp
  0,52 với 0,68 là **so hai thước đo khác nhau**.
- `test.jsonl` = bản sao `val.jsonl` (quy ước pool nguồn cũ của dự án).
- 6 hàng lẻ ở `ccpp common` (nửa kia bị luật CWE loại).

---

## ĐANG CHẠY — 161 (A4000), data v3, bắt đầu 23/09 00:43

```
driver : bash clean/scripts/q161_v3.sh        (flock /tmp/mvd_q161_v3.lock)
nối tiếp: q161_after.sh -> q161_chain.sh      (flock /tmp/mvd_q161_chain.lock, chờ theo PID)
log    : clean_local/logs/driver_161.log
kết quả: clean_local/            <- TÁCH KHỎI clean/ (đó là của vast, khác máy)
```

Chuỗi đêm: `v3common` (Pha 1+2 ×5) → 3 nhánh đối chứng → `v3cwe4` → `v3common_nocc`.
**Không có `v3full`** (7 h) — để sang khi 158 rảnh.

### Máy nào chạy gì

| máy | GPU | trạng thái |
|---|---|---|
| **161** | A4000 16 GB | **của ta, đang chạy chuỗi đêm** |
| **158** | A4000 16 GB | ⛔ **đang chạy job của đồng tác giả** (`mwgp1U_pvcomU_158`, `/data/tranmanhcuong/MultiVD`). KHÔNG phóng gì lên đó cho tới khi họ xong |
| vast 51144271 | — | ⛔ **đang cho người khác mượn**. Cron đã xoá, `qtune` vẫn là stub |

### Cấu hình hiệu lực (đã đọc lại từ chính tiến trình)

`MWGraph | graph=typed R=4 Lmax=150 layers=2 lstm=1 fusion=cat drop_bracket=1 co_mode=chain`

| | Pha 1 | Pha 2 |
|---|---|---|
| warmup / graph_lr | **0,25 / 2e-5** ← đổi, sửa lỗi kẹt §84 | 0,10 / 1e-4 giữ nguyên |
| lr · K · batch | 2e-5 · K=8 (W510 S384) · 4/8, micro 16 | như Pha 1 |
| SAM / RecAdam | 0 / tắt | 0,02 / `cof 500 t0 0.01` |
| epoch | 8 | trần 30, patience 8 |
| khởi tạo | — | `--init all` từ checkpoint Pha 1 |

**Baseline KHÔNG chạy lại** — đã có đủ 5 fold ở `clean/results/clean_baseline_codebert_py`
(ROC TB 0,9065 · F1@0.5 0,7970).

⚠️ Kết quả **161 và vast KHÔNG ghép cặp với nhau được** (sàn nhiễu cross-GPU 0,028).
Chia hai máy thì chia theo **FOLD TRỌN VẸN**, không bao giờ theo nhánh.

## HÀNG ĐỢI — MỌI Ô DÙNG `mwsrc_v3*`

| # | ô | Pha 1 dùng lại được? | giờ |
|---|---|---|---|
| 1 | `v3common` Pha 1 + Pha 2 × 5 fold | mới | ~3,7 h |
| 2 | 3 nhánh `abl_*` (noinit / nograph / norecsam) | dùng chung ck #1 | ~3,8 h |
| 3 | `qtune` 6 nhánh (glr2e5, K16, coall, nobabel, warm25, warm03) | K16 và coall cần Pha 1 riêng | ~21 h |
| 4 | `v3full` Pha 1 + Pha 2 | mới | ~7 h |
| 5 | `v3cwe4` Pha 1 + Pha 2 | mới | ~1,5 h |
| 6 | `v3common_nocc` Pha 1 + Pha 2 | mới | ~2,5 h |
| 7 | giữ ngoặc `--drop_bracket 0` | cần Pha 1 riêng | ~5 h |
| 8 | 4cwe **có java** | chưa dựng pool | ~6 h |
| — | SOTA (MVD, MulnVul…) | chờ người dùng chỉ định bài | — |

> Thứ tự do người dùng chốt. Mỗi ô xong **kéo ngay về 161**.

---

## KHÔNG phải chạy lại

`clean_baseline_codebert_py` — không có Pha 1, không đụng pool nguồn.
F1@0.5 **0,7970** · F1@val 0,7958 · ROC **0,9065** · PR 0,9183.

## Cấu hình chốt (FACTS §84)

| | Pha 1 | Pha 2 |
|---|---|---|
| `warmup_ratio` | **0,25** | 0,10 |
| `graph_lr` | **2e-5** | 1e-4 |
| `learning_rate` | 2e-5 | 2e-5 |
| `sam_rho` / RecAdam | 0 / tắt | 0,02 / `cof 500, t0 0.01` |
| đồ thị | `typed` R=4 · Lmax 150 · `drop_bracket 1` · `co_mode chain` · fusion `cat` | như Pha 1 |

seed 42 · K=8 (W=510 S=384) · batch 4 / eval 8 · micro 16 · patience 8 · epochs 8 / 30.

## Kiểm tay

```bash
ssh -p 56599 root@115.73.216.179 \
  'ps -eo comm= | grep -cx python; tail -3 /workspace/clean/logs/driver_chain.log'
rsync -a -e "ssh -p 56599" root@115.73.216.179:/workspace/clean/results/ \
  /drive1/cuongtm/ntat/MultiVD/clean/results/
```
