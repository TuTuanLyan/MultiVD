# Sổ artifact — MultiVD

Danh sách các trang kết quả đã xuất bản, mới nhất trước. Artifact mặc định **riêng
tư**; muốn gửi cho người khác thì mở trang rồi chia sẻ từ menu của nó.

Cập nhật danh sách: trong terminal gõ `/artifacts` (phím `o` mở, `c` chép link),
hoặc vào <https://claude.ai/code/artifacts>.

---

## Đang dùng

### 📉 Trục ρ và chọn nguồn — đêm 08→09/09/2026

<https://claude.ai/code/artifact/b6ba617d-eef2-4bee-ad8b-e8db92f7eadf>

Hai khối chạy qua đêm trên bốn máy. **Trục ρ của ASAM** (7 mức, 10–15 ô/mức, đối chứng
ρ=0 cùng máy): đường cong đơn đỉnh, đỉnh ở ρ∈[1,2], ρ=8.0 cho 0/10 fold trên cả bốn chỉ
số — và ρ=0.1 mà dự án dùng từ 31/08 nằm ở đáy. **Lưới nguồn** tách trùng CWE khỏi tỉ lệ
ngôn ngữ. Trên 73% hàng test **không có bản gần trùng**, pha loãng nguồn xuống ≤25% dòng
đúng CWE cho 0/7 fold dương trên cả F1 lẫn ROC-AUC — liều–đáp ứng đơn điệu. Và điểm val
Phase 1 không dự báo được transfer (Spearman −0.191).

Kèm **ba phát biểu của chính tôi đã phải sửa hoặc rút lại** trong đêm — cả ba đều vì cùng
một phép kiểm: tách điểm test theo nhóm rò rỉ. Và bốn lỗi công cụ, cả bốn đều im lặng
(`bash -n` báo OK, mã thoát 0, "xong" trong log).

Bậc 1 và bậc 2, seed 42 — chưa số nào được đưa vào bài.

### Neo vào tri thức nguồn — tổng hợp nhánh `optimize-v1` (08/09/2026)

<https://claude.ai/code/artifact/406023e5-2d41-401f-8a83-f9a32ff42eae>

Bốn khối thực nghiệm tối ưu RecAdam gộp vào một trang: OPT1 (quét γ và lịch neo, 211 ô,
bậc 2 n=5), RET1 (giữ lại tri thức nguồn, 21 ô), SPD1 (đổi cơ chế neo sang điều kiện
gradient, 21 ô), INT1 (nội suy θ_Pha1 ⊕ θ_Pha2, đang chạy). Kèm hai phát biểu đã phải
**rút lại** và ba lỗi kỹ thuật đã ghi lại.

Số trong trang là **bậc 1 và bậc 2, seed 42** — chưa có số nào được đưa vào bài.

### 🎚️ ASAM ρ=0.1 có mua được gì không
<https://claude.ai/code/artifact/1a9ac269-85fd-4ad9-9745-bb60350d71fa> · 06/09

Trang gọn cho **một khối duy nhất**: NIGHT48, n=15, có nhánh đối chứng ρ=0 nên tách
được phần do riêng ASAM. Dựng để chia sẻ ra ngoài nhóm.

- `latent_bottleneck` · codet5p-220m-bimodal · RecAdam · λ=0.05 · ρ ∈ {0, 0.1} ·
  **cả ba nguồn** `4cwe`/`com`/`full` · seed 42/7/1234 · fold 1–5 · **90/90 ô + 15/15
  baseline**, kiểm toán không thiếu không chồng lấn
- **macro-F1 và ROC-AUC luôn nằm cạnh nhau** ở mọi bảng — hai chỉ số không luôn cùng
  nói một điều, để một cái là giấu mất điều đó
- Dải Δ: mỗi hàng vẽ 15 fold ghép cặp, trung bình, và **băng sàn nhiễu ±0.010** tô mờ
  ngay trên trục, nên hiệu ứng dưới sàn nhìn là biết
- Bảng 105 ô gốc, lọc theo nguồn / ρ / seed / fold, sắp xếp được, kèm cột máy đã chạy
- Phần đầu giải thích thuật ngữ cho người đọc ngoài nhóm

Kết luận: **ASAM ρ=0.1 không mua được gì vượt sàn nhiễu** (gộp +0.0034, 29/45, p=0.073),
cùng câu trả lời đã có ở ρ=0.2 và ρ=0.5. Điều đáng giá hơn nằm ở chỗ khác: toàn bộ hiệu
ứng dồn vào **fold 3** (+0.0233, 8/9), dương ở cả ba seed trên ba máy khác nhau — đặc
tính của một lát cắt, không phải của phương pháp. Số liệu gốc ở `FACTS.md` §18.

### 🎯 Xác nhận đa seed cho ASAM
<https://claude.ai/code/artifact/897a6a3e-5931-484c-88ff-62baacae4a21> · 05/09

Trang gọn, một câu hỏi duy nhất: **hiệu ứng của ρ có sống sót khi n đi từ 3 lên 15
không.** Hai cấu hình ASAM dẫn đầu ở trang quét được chạy lại với 3 seed × 5 fold,
**mỗi seed huấn luyện Pha 1 riêng**.

- `latent_bottleneck` · codet5p-220m-bimodal · RecAdam · nguồn `common` · λ=0.05 ·
  ρ ∈ {0, 0.2, 0.5} · seed 42/7/1234 · fold 1–5 · **torch 2.11.0**
- Baseline từng seed (chênh nhau tới 0.0175 nên bắt buộc ghép cặp theo seed)
- Bảng chính: macro-F1 và ROC-AUC, mỗi cái kèm Δ vs baseline, Δ vs ρ=0, dấu và p
- Bảng tách theo seed, cho thấy ρ **đảo dấu** giữa các seed

Kết luận: **hiệu ứng co lại và không qua ngưỡng** trên macro-F1 (+0.0089, 11/15,
p=0.12, dưới sàn nhiễu), dù ROC-AUC có qua (+0.0105, 12/15, p=0.035). Số liệu gốc
ở `results_confirm47_com/`.

### 🔬 Quét λ × ρ trên CodeT5+
<https://claude.ai/code/artifact/ff03d6cc-3f1c-4ae8-9c10-561e9c9d4405> · 04/09

Trang chuyên cho một câu hỏi: **trọng số loss phụ λ và bán kính nhiễu loạn ASAM ρ
có tương tác không, và ρ có mua được gì không.**

- Hai lưới nhiệt λ×ρ (nguồn `common` và `4cwe`), đổi được giữa macro-F1 và ROC-AUC,
  rê chuột ra cả hai chỉ số kèm từng fold
- 126 ô: 3 λ × 7 ρ × 3 fold × 2 nguồn, `latent_bottleneck` · RecAdam · seed 42
- Bảng gộp hai nguồn, bảng theo λ, hai bảng đầy đủ

Toàn bộ lưới là seed 42, fold 1–3, chỉ backbone `codet5p-220m-bimodal`. Phần xác
nhận đa seed nằm ở trang riêng phía trên.

### ⚖️ Bảng so sánh MultiVD
<https://claude.ai/code/artifact/84b176a7-faf0-450a-a3b5-3747de09f255> · 02/09

Bảng gọn để **so hai cấu hình bất kỳ cạnh nhau**. Dựng cho người đọc ngoài nhóm.

- Phần đầu giải thích thuật ngữ và cách áp dụng: hai pha, baseline, bốn đầu phụ,
  AdamW/RecAdam, ASAM ρ, λ, Δ ghép cặp, sàn nhiễu, "Pha 1 sập"
- Bảy nhóm lọc đa chọn, mỗi nhóm một dòng: backbone · đầu phụ · nguồn · bộ tối ưu ·
  sharpness Pha 2 · λ · đợt chạy. Mặc định chỉ bật đợt mới nhất cho gọn
- **Hai bảng, hai bộ lọc độc lập** — đặt cạnh nhau để so, kèm nút chép bộ lọc A→B
- Hai chế độ: trung bình (kèm ±sd và số fold cùng dấu) và từng fold
- Baseline luôn ghim ở đầu bảng, không bị bộ lọc làm mất

### 📊 MultiVD Seed-42 Ledger
<https://claude.ai/code/artifact/836edc35-69bf-4137-94a0-549a672af892> · 31/08

**Sổ tra cứu đầy đủ 784 ô** của seed 42 — nơi tra khi cần một con số cụ thể.

- Lọc và sắp xếp theo mọi trục: khối · backbone · nguồn · phương pháp · optimizer ·
  λ · biến thể SAM · ρ · fold
- Ba chế độ: từng ô, gộp theo nhánh, và **bảng trung bình xoay trục** (chọn hàng,
  cột và số liệu: macro-F1, ROC-AUC, PR-AUC, Δ vs baseline, Δ vs `none`, val Pha 1)
- Cột `val Phase 1` tô vàng ô ≤ 0.40 để lộ ngay ô có Pha 1 sập
- Chín khối cấu hình từ 20/08 đến 31/08

---

## Lưu trữ

Bốn trang dưới đây thuộc giai đoạn trước khi chốt lại phương pháp; giữ để tra lịch
sử lập luận, **không dùng làm nguồn số liệu hiện hành**.

### 📐 Ma trận reset MultiVD
<https://claude.ai/code/artifact/0c258966-c8a0-4c7e-b4e5-362257c60d80> · 26/08 —
kế hoạch chạy lại toàn bộ ma trận sau khi dựng lại bộ dữ liệu nguồn.

### 🔬 Latent Auxiliary Transfer
<https://claude.ai/code/artifact/83856b7a-4b6b-49f9-8048-654d6d802f68> · 20/08 —
mô tả ý tưởng đầu phụ tiềm ẩn và các nhánh `latent_bottleneck` / `latent_proto`.

### 📉 The Stronger-Backbone Penalty
<https://claude.ai/code/artifact/4d4f49ba-d471-46d2-9af8-acb271827faf> · 20/08 —
giả thuyết "backbone mạnh hơn lại chuyển giao kém hơn". **Đã bị rút lại** khi thêm
dữ liệu; xem `DEAD_ENDS.md`.

### 🧬 Distillation Between Equals
<https://claude.ai/code/artifact/61cdf5a3-f79b-45e7-846d-a055e30fe3ca> · 17/08 —
hướng chưng cất giữa hai mô hình ngang tài. Đã dừng theo đuổi.

---

## Quy ước

- **Δ luôn ghép cặp theo fold** trong cùng (backbone, nguồn, optimizer, seed). Không
  trang nào lấy hiệu của hai giá trị trung bình.
- **Sàn nhiễu 0.010** — đo trên chính dự án, chạy lại cùng seed cùng cấu hình trên
  máy khác. Hiệu ứng nhỏ hơn ngưỡng này được làm mờ hoặc ghi chú rõ.
- Sàn kiểm dấu phụ thuộc n: p=0.25 ở n=3, p=0.0625 ở n=5. Ô n nhỏ **không thể** đạt
  p<0.05 dù mọi fold cùng dấu.
- Số liệu gốc ở `FACTS.md` (theo mục đánh số) và trong `results/`.

## Dựng lại

Các trang được sinh từ script trong thư mục tạm của phiên, dữ liệu đọc từ
`results/`. Muốn dựng lại sau khi có kết quả mới thì chạy lại script sinh trang rồi
xuất bản **vào đúng URL cũ** — xuất bản mà không kèm URL sẽ tạo ra một artifact mới
thay vì cập nhật cái đang có.

## Trộn đều hai mô hình (09/09/2026)
https://claude.ai/code/artifact/d18d51d3-ac51-491d-b8b6-90685506a8a6

FACTS §25. Trộn xác suất baseline ⊕ chuyển giao ở α=0.5 vượt **cả hai** đầu mút trên hai chỉ số
xếp hạng (ROC +0.0165 so với chuyển giao thuần, 59/83, p=0.0002). Lặp trên hai backbone với
cùng biên độ. Cơ chế theo CWE: giữ 43%/64% phần được ở CWE-022/079, xoá sạch phần hại ở
CWE-078/089. Nêu rõ ba giới hạn, trong đó đối chứng "trộn hai mô hình nào cũng lợi" **chưa
chạy xong** — chưa trích được khi chưa có nó.

## Sổ kết quả thực nghiệm (09/09/2026) — TRANG CHO NGƯỜI HƯỚNG DẪN
https://claude.ai/code/artifact/35c5f832-c306-42e4-959d-9d51adfdcef0

Toàn bộ **1 158 phép so sánh** rút từ 3 063 ô (332 ô trùng đã khử), 5 backbone, 4 máy.
Cấu trúc theo yêu cầu 09/09: (1) ma trận "chứng minh đã qua những gì" · (2) chú giải ký hiệu
đầy đủ · (3) **bảng tương tác** có lọc + sắp xếp, hai chế độ (1 bảng tổng quan / 2 bảng đối
chiếu lọc độc lập) · (4) nhận xét · kết luận · đề xuất.

Nguồn sinh: `tools/build_table.py`; bản HTML rời ở `page/`. Chạy lại để cập nhật:
```
python3 tools/build_table.py results results_* > page/table.json
```
