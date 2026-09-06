# Quy tắc chạy thực nghiệm — MultiVD

File này được nạp tự động mỗi phiên. Đọc trước khi xếp bất kỳ lịch chạy nào.
Bổ sung cho [VAST_RULES.md](VAST_RULES.md) (thuê/huỷ máy) và [METHOD.md](METHOD.md) (phương pháp).

> **Chỗ làm việc là `/drive1/cuongtm/ntat/MultiVD`** — đã chuyển xong 06/09/2026,
> đối chiếu 6 694 file lệch 0 byte. `/` trên 161 đầy 98% và từng làm `torch.save` ghi
> cụt checkpoint, nên **mọi thứ nặng, nhất là `model/`, đặt ở `/drive1`**. Bản ở
> `/home/ntat/workspace/MultiVD` giữ lại để tra cứu, không chạy thí nghiệm mới ở đó.
> Đổi thư mục làm việc là đổi khoá memory của Claude Code; memory đã chép sang khoá
> mới. Chi tiết ở [SERVER.md](SERVER.md).

---

## 1. Thứ tự chạy — FOLD LÀ VÒNG NGOÀI CÙNG

> **Khi so sánh các phương pháp, mọi phương pháp phải chạy trên CÙNG fold trước
> khi sang fold tiếp theo.**

```
for fold in 1 2 3 4 5:          ← vòng NGOÀI
    for source in ...:          ← vòng trong
        for method in ...:
            for optimizer in ...:
                chạy
```

**Vì sao**: xong fold 1 là đã có **một lát cắt so sánh được ngay** — đủ mọi phương
pháp, mọi nguồn, trên cùng một fold. Nó chưa nói lên tất cả, nhưng đủ để thấy
hướng và để quyết định có chạy tiếp không. Đến fold 3 thường đã đủ để **dừng**.

Thứ tự ngược lại (source-major hoặc method-major) khiến phải chạy gần hết mới có
ô nào so được với ô nào — mất khả năng dừng sớm, và nếu hỏng giữa chừng thì
không còn gì dùng được.

**Thứ tự trong một fold** (người dùng nêu 27/08, giữ nguyên trừ khi có yêu cầu mới):

```
baseline → none → cwe → latent_bottleneck → latent_proto
```

với mỗi phương pháp chạy AdamW và RecAdam **cạnh nhau**, rồi mới sang fold tiếp.
Đặt hai optimizer cạnh nhau để hiệu giữa chúng ghép cặp được theo fold.

### Bẫy đã mắc

`FOLDS="${FOLDS:-1 2 3 4 5}"` — dấu **hai chấm** biến `FOLDS=""` thành mặc định,
nên giai đoạn "chỉ Phase 1" chạy luôn cả 5 fold và thứ tự thành source-major.
Dùng `${FOLDS-...}` (không hai chấm) khi cần giữ giá trị rỗng.

---

## 2. Phép so sánh — luôn ghép cặp trong cùng fold

Δ **luôn** ghép cặp theo `(backbone, nguồn, optimizer, fold)`. **Không bao giờ**
lấy hiệu của hai trung bình.

Một fold ngoại lệ từng gánh cả một kết luận rồi bị rút lại. Và trong dự án này đã
có **bảy lần** một mẫu hình co lại hoặc biến mất khi thêm fold — nên mọi phát biểu
ở n nhỏ phải kèm n, số fold cùng dấu, và biên độ min–max.

**Ngưỡng thống kê**: sàn Wilcoxon ở n=5 là p=0.0625 — nghĩa là "cùng dấu ở cả 5
fold", không phải "gần có ý nghĩa".

**Hai sàn nhiễu đã đo trên chính dự án này:**

| | Giá trị |
|---|---|
| Chạy lại cùng seed, cùng cấu hình, cùng loại GPU, khác máy | **0.010** |
| Giữa các loại GPU khác nhau | **0.028** |

Hiệu ứng dưới ~0.01 không phân biệt được với việc chạy lại đúng một thứ.

---

## 3. Chạy đủ — sập KHÔNG phải lý do để dừng

> **Một ô bị chặn là một ô TRỐNG, và ô trống không viết được gì vào bài.**

Khi một nhánh hỏng giữa khối, mặc định là **chạy tiếp và ghi lại**, không phải bỏ
nhánh đó rồi dừng. Chỉ dừng khi người dùng yêu cầu.

- **Không bao giờ tự ý bỏ một ô** vì Phase 1 của nó yếu. Chạy Phase 2 và ghi kèm
  `val` Phase 1 của checkpoint. Người đọc nhìn `val=0.34` là biết ngay "Phase 1
  sập", còn ô trống thì không nói được gì.
- Cổng chất lượng đặt ở `run/matrix.sh:phase1_usable`, ngưỡng chỉnh bằng
  **`PHASE1_MIN_VAL`** (mặc định 0.40 — bắt checkpoint đoán một lớp). Đặt
  `PHASE1_MIN_VAL=0` để chạy hết, kể cả checkpoint suy biến.
- Cổng phải phân biệt **file hỏng** (RuntimeError lúc đọc, kích thước lệch mốc)
  với **chất lượng kém** (đọc được, val thấp). Đừng dán nhãn `.rejected` vĩnh
  viễn cho cái đầu.
- Khi khối chạy xong, **đối chiếu số ô thực tế với số ô kỳ vọng** và nêu rõ ô nào
  thiếu, vì sao. Driver in "xong" không có nghĩa là đã đủ.

**Vì sao có mục này:** ngày 30–31/08, ngưỡng cũ `val >= 0.55` từ chối
`codebert/latent_bottleneck/com` ở **0.549872** — hụt 0.000128, tức nhỏ hơn sàn
nhiễu 0.010 khoảng 78 lần. Nó chỉ loại `latent_bottleneck` ở đúng những nguồn
nhánh đó yếu, nên bảng kết quả chỉ còn chỗ nó mạnh: **thiên lệch chọn lọc**. Cùng
đợt, 46 ô khác biến mất vì đĩa đầy làm `torch.save` ghi cụt rồi bị cổng dán nhãn
"phương pháp kém".

Và ô "hỏng" lại thành bằng chứng tốt nhất: `codebert × full × none` với Phase 1
val 0.3403 cho **Δ −0.4395, 0/5 fold** — con số đó chứng minh vì sao phải gắn val
Phase 1 vào mọi ô, và nó chỉ có được vì đã chạy thay vì bỏ.

---

## 4. Chia việc giữa các máy

- **Một backbone nằm trọn trên một máy.** Baseline và mọi nhánh của nó cùng phần
  cứng thì Δ nội bộ sạch.
- Nếu buộc phải chia, **chia theo FOLD TRỌN VẸN**, không bao giờ theo nhánh. Cả
  nhánh chính lẫn nhánh đối chứng của một fold phải cùng máy, khi đó độ lệch phần
  cứng triệt tiêu trong Δ ghép cặp.
- Nhánh đối chứng phải chạy **cùng máy cùng phiên** với nhánh nó đối chứng —
  không so với kết quả cũ trên máy khác.

---

## 5. Phase 1 — khi nào dùng lại được

| Đổi cái gì ở Phase 2 | Dùng lại checkpoint Phase 1? |
|---|---|
| optimizer (AdamW ↔ RecAdam) | ✅ |
| SAM/ASAM, ρ, η | ✅ |
| cách chia fold đích | ✅ |
| **λ** | ❌ — λ nhân vào loss Phase 1 (`train.py:53`) |

Ngoại lệ: nhánh **`none`** dùng lại được ở **mọi λ**, vì `model.py:320` trả
`aux_loss=None` nên λ không vào hàm loss. Symlink thay vì huấn luyện lại.

Khi một khối chỉ đổi Phase 2, phải tách `PHASE1_TAG` khỏi `ARM_TAG`; nếu không
nhánh mới sẽ tự huấn luyện một Phase 1 khác và phép so đổi hai biến.

---

## 6. Dữ liệu

### Nguồn Phase 1 — ccpp (PrimeVul) + js (CleanVul), đã cân bằng nhãn 50/50

| nguồn | file | n | ccpp / js | số CWE | ghi chú |
|---|---|---|---|---|---|
| `4cwe` | `data/phase1_4cwe.jsonl` | 930 | 118 / 812 | **4** | CWE-79 692, CWE-78 100, CWE-22 92, CWE-89 46 |
| `com` | `data/phase1_common.jsonl` | 3 744 | 2 360 / 1 384 | 94 | CWE chung giữa hai ngôn ngữ |
| `full` | `data/phase1_full.jsonl` | 7 598 | 6 042 / 1 556 | 123 | toàn bộ |

Chỉ **js mới có nhãn CWE** trong CleanVul, nên nhánh `cwe` (head 4 lớp có nhãn)
**chỉ chạy được trên nguồn `4cwe`**. Hai nguồn kia chỉ có `none`,
`latent_bottleneck`, `latent_proto` — vì thế 4cwe có 4 nhánh còn com/full có 3.

`full` là nguồn **khó nhất** (123 CWE, lệch mạnh về ccpp) và là chỗ Phase 1 của
backbone yếu hay sập. Dựng lại từ bộ gốc bằng `src/build_sources.py`.

### Tập đích — Python

| | n | chia | dùng khi nào |
|---|---|---|---|
| **`data/sven_python_folds_norm`** | 760 (380/380) | ngẫu nhiên 60/20/20 → 456/152/152 mỗi fold | **mặc định** |
| `data/sven_python_twin` | ~760 | theo **cụm gần trùng**, không rò rỉ | phụ, chỉ khi được yêu cầu |

Reviewer yêu cầu phân phối ngẫu nhiên nên `norm` là tập chính. Nhưng phải biết
điểm yếu của nó: `src/build_folds.py` chia **theo từng dòng**, nên ~40% hàng test
có bản sao gần giống nằm trong train. `twin` gom cụm gần trùng rồi mới chia — đó
là câu trả lời cho phản biện rò rỉ, để dành chạy sau.

Trường mỗi dòng: `code, label, cwe, cwe_id, cwe_class, lang`.

---

## 7. Phương pháp đã chốt (31/08)

Sau 5 khối cấu hình so được với nhau (784 ô tất cả), cấu hình chốt:

```
latent_bottleneck   Linear(H→8) → Linear(8→C), num_latent=8
λ = 0.05            SAM/ASAM TẮT ở CẢ HAI PHA   (--sam_rho 0)
đối chứng bắt buộc: baseline (không Phase 1) + none (Phase 1 không head)
```

Đã loại, kèm bằng chứng — **đừng chạy lại nếu không có lý do mới**:

| bỏ gì | số đo |
|---|---|
| ASAM ρ=0.1 (Phase 2) | khối C/D cùng phiên cùng máy: +0.0020, 70/130, p=0.43 |
| λ=0.02 | ghép cặp với λ=0.05: −0.0053, 60/133, p=0.30 |
| SAM ρ=0.05 ở Phase 1 | ghim codebert ở ln2 suốt 13 epoch, F1 0.3333 |
| `latent_proto` | head riêng ≈ 0 mọi khối; tự sập ở codebert×full (0.3432) |
| `cwe` | độ tản giữa backbone 0.0443, **âm** trên codet5p; cần nhãn nên chỉ chạy được trên `4cwe` |

Hai phát biểu chính, **phải nêu cả hai vì chúng mâu thuẫn**:

- **AdamW**: head ăn về điểm — Δ vs `none` +0.0111, 33/43 fold, p=0.0006. Ô duy
  nhất trong toàn lưới vừa qua p<0.05 vừa vượt sàn nhiễu.
- **RecAdam**: head hết ăn về điểm (+0.0036, 23/43, p=0.76) nhưng **ổn định nhất
  giữa backbone** — +0.0186 / +0.0211 / +0.0228, độ tản 0.0042.

Giá trị chắc nhất của head **không phải độ chính xác mà là chống sập Phase 1**:
ở `codebert × full`, hai lần độc lập tại hai λ, `none` sập về ~0.34 còn
`latent_bottleneck` giữ 0.545–0.564. `latent_proto` không có tính chất này.

Seed 42 đã xong đủ cấu hình này ở **cả hai optimizer** (`none` 45 ô,
`latent_bottleneck` 48 ô, mỗi optimizer, cộng 15 baseline) — chạy đa seed chỉ
là thêm seed, không làm lại. Chi phí ~17 h/seed nếu codebert ở A4000 local, ~9 h
nếu ở 5070Ti, ba máy song song.

---

## 8. Báo cáo và canh giờ

- **Người dùng nói "báo mỗi N phút" nghĩa là báo CHO NGƯỜI DÙNG mỗi N phút**,
  không phải đặt chu kỳ nội bộ của watchdog thành N. Đã hiểu nhầm một lần.
- Chu kỳ watchdog và chu kỳ báo cáo là **hai thứ khác nhau**: watchdog nên dày
  hơn (~10 phút) vì driver chết ở độ mịn 30 phút là 30 phút tính tiền.
- Có job chạy trên vast thì **luôn cài monitor nền**: theo dõi driver/GPU/đĩa,
  báo khi xong fold hoặc khi hỏng, kéo kết quả về và đối chiếu từng byte trước
  khi huỷ máy.
- **Điều kiện tự huỷ máy phải là dòng kết thúc do chính driver in ra**, không bao
  giờ là "đếm đủ N ô". Ngưỡng đếm có thể không bao giờ đạt — đã suýt để máy chạy
  33 giờ vì `NEED=50` trong khi 30 ô đã chết vì lý do khác. Mất log ⇒ **không
  huỷ**. Driver chết mà còn việc ⇒ **phóng lại**, không huỷ.
- Thử mọi cổng xác minh **cả hai chiều** trước khi tin: cho nó một trường hợp
  khớp và một trường hợp lệch. Cổng báo nhầm "chưa an toàn" cũng là lỗi — nó làm
  máy nằm không mà vẫn tính tiền.
- Đếm tiến trình **không được tự khớp chính nó**. `pgrep -f`/`ps|grep` bắt luôn
  dòng lệnh của mình; dùng `flock -n <lock> -c true` hoặc lọc theo `comm`.

---

## 9. Hỏi, đừng tự quyết

- **Không tự thêm thí nghiệm vào hàng đợi.** Quét siêu tham số, nhánh đối chứng
  thêm, đổi tập đích — tất cả phải hỏi trước, kể cả khi có lý do khoa học tốt.
- **Không tự đảo thứ tự ưu tiên** người dùng đã nêu.
- Nếu **quên** một thứ tự hoặc yêu cầu đã được nhắc trước đó thì **hỏi lại** —
  người dùng đã nói rõ là cứ hỏi thoải mái. Đoán rồi chạy sai tốn nhiều hơn hỏi.
- Báo cáo phải nêu **cả hai optimizer**, không chỉ nhánh thắng.

Ngày 30/08 tôi tự thêm quét ρ, tự đổi ưu tiên sang `twin` dù đã được dặn đó là
phụ, và chèn nhánh đối chứng làm gấp đôi khối lượng — việc người dùng cần bị đẩy
lùi mất gần một buổi.

---

## 10. Trước khi xoá hay huỷ máy

- Đối chiếu **từng file kể cả kích thước byte**, không chỉ đếm số file. Một lần
  rsync đứt giữa chừng để lại file ngắn hơn mà `wc -l` vẫn thấy "đủ".
- `LC_ALL=C` cho **cả `sort` lẫn `comm`** — `comm` dưới locale khác trả rác và
  vẫn thoát 0.
- `vastai destroy instance` cần `-y` **và** phải kiểm output: nó in `Aborted.`
  rồi thoát 0.
- **Không bao giờ xoá thứ chưa xác minh là đã có ở local.** Đã mất 6 checkpoint
  Phase 1 vì xoá trước khi xác minh.

---

## 11. Đọc thêm

**Phiên mới thì đọc [HANDOFF.md](HANDOFF.md) trước tiên** — nó nói hiện trạng: dữ liệu
nằm ở thư mục nào, khối nào đã xong, cái gì chưa chạy, và ba cái bẫy trong dữ liệu.

| File | Nội dung |
|---|---|
| [HANDOFF.md](HANDOFF.md) | **hiện trạng bàn giao — đọc đầu tiên ở phiên mới** |
| [ARTIFACTS.md](ARTIFACTS.md) | **mọi trang kết quả đã xuất bản, kèm link và mô tả** |
| [CURRENT_RUN.md](CURRENT_RUN.md) | khối đang chạy (hoặc khối vừa xong), cấu hình chi tiết |
| [SERVER.md](SERVER.md) | **máy nào, thư mục nào được dùng — `/drive1` là chỗ chính** |
| [VAST_RULES.md](VAST_RULES.md) | thuê/huỷ máy, nhãn, giá, không đụng user khác |
| [METHOD.md](METHOD.md) | kiến trúc transfer, các nhánh phụ |
| [FACTS.md](FACTS.md) | mọi số đã đo, theo mục đánh số |
| [DEAD_ENDS.md](DEAD_ENDS.md) | hướng đã thử và bỏ |
| [VAST_TEMPLATE_ERROR.MD](VAST_TEMPLATE_ERROR.MD) | lỗi image vast để báo nhóm |



## 12. Lưu trữ run hiện tại:
Khi người dùng yêu cầu đặc biệt phải lưu ra 1 file CURRENT_RUN.md để chạy cái hiện tại và nêu ra nội dung hiện tại đang làm gì và đang chạy cái gì, setting ra sao để đảm bảo không nhầm. Khi xong sẽ phải update vào ở đầu file md này là đã xong + ngày giờ để biết hiện tại không còn chạy cái này nếu run sau không có gì đặc biệt hoặc chỉ là chạy lại phần nhỏ.

File này có thể lưu các queue, các yêu cầu có thể ngay cả khi không đặc biệt và chỉ cần lưu yêu cầu có thể khác ở các máy khác nhau để hiểu rõ ràng có thể note thêm tên máy để phân biệt, file này không cần thiết phải backup. Ở local có thể lưu thêm cả các việc ở trên máy trên vast để biết trên đó đang run gì hoặc update trực tiếp ở vast.
