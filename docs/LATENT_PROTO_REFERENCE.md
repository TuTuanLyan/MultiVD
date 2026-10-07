# Latent proto: head phân cụm theo nguyên mẫu với gán nhãn cân bằng (Sinkhorn)

Tài liệu lý thuyết và cách triển khai của kiến trúc `latent_proto`. Phần 1-7 nói về **bản thân
kiến trúc**, dùng được cho bất kỳ bộ mã hoá nào (văn bản, mã nguồn, ảnh). Phần 8 mới đối chiếu
với cài đặt trong repo này.

Nguồn gốc: họ phương pháp **tự gán nhãn bằng vận chuyển tối ưu** (optimal transport):

- SeLa: Asano, Rupprecht, Vedaldi, *Self-labelling via simultaneous clustering and representation
  learning*, ICLR 2020, arXiv:1911.05371.
- SwAV: Caron et al., *Unsupervised Learning of Visual Features by Contrasting Cluster Assignments*,
  NeurIPS 2020, arXiv:2006.09882. Mã: https://github.com/facebookresearch/swav
- Thuật toán Sinkhorn cho vận chuyển tối ưu có điều chuẩn entropy: Cuturi, *Sinkhorn Distances:
  Lightspeed Computation of Optimal Transport*, NeurIPS 2013, arXiv:1306.0895.

---

## 1. Ý tưởng trong một đoạn

Giữ một tập **K nguyên mẫu** (prototype) học được, mỗi nguyên mẫu là một vector trong không gian
biểu diễn. Mỗi mẫu được **gán mềm** vào các nguyên mẫu dựa trên độ tương đồng cosine. Nhãn đích
của phép gán **không lấy từ dữ liệu** mà được tự tính trong từng batch bằng thuật toán
Sinkhorn-Knopp, với ràng buộc **các nguyên mẫu phải được dùng đều nhau**. Mô hình học để dự đoán
chính phép gán cân bằng đó. Kết quả: bộ mã hoá học được một cách chia dữ liệu thành K cụm **mà
không cần nhãn**, và ràng buộc cân bằng ngăn nó rơi vào nghiệm tầm thường "mọi mẫu vào một cụm".

## 2. Bài toán nó giải quyết: sụp đổ khi tự phân cụm

Muốn học biểu diễn bằng phân cụm không nhãn, cách ngây thơ là: gán mỗi mẫu vào cụm gần nhất rồi
huấn luyện mô hình dự đoán cụm đó. Cách này có một **nghiệm tầm thường**: đưa mọi mẫu về cùng một
cụm, mất mát bằng 0, biểu diễn không mang thông tin gì. Các cách chống sụp đổ đã có:

| cách | cơ chế | nhược điểm |
|---|---|---|
| DeepCluster (arXiv:1807.05520) | k-means ngoại tuyến trên toàn bộ dữ liệu mỗi epoch, gán lại cụm rỗng | phải duyệt cả tập dữ liệu, không trực tuyến |
| **SeLa / SwAV** | **ràng buộc cân bằng** (equipartition): mỗi cụm nhận đúng 1/K khối lượng | cần batch đủ lớn so với K |
| DINO (arXiv:2104.14294) | căn giữa (centering) + làm sắc (sharpening) đầu ra thầy - trò | cần hai mạng, không có nguyên mẫu tường minh |

`latent_proto` thuộc dòng thứ hai.

## 3. Kiến trúc

```
      x ──► Bộ mã hoá f_θ ──► h (d_h chiều)
                              │
                              ├──► head nhiệm vụ chính (nếu dùng làm đầu ra phụ)
                              │
                              ▼
                   Head chiếu g_φ (MLP) ──► z, chuẩn hoá L2 (D chiều)
                              │
                              ▼
            Điểm s = zᵀC,  C = [c_1 … c_K] nguyên mẫu chuẩn hoá L2 (D × K)
                   │                              │
                   ▼                              ▼
       dự đoán p = softmax(s / τ)       đích q = Sinkhorn(s)   (dừng gradient)
                   │                              │
                   └────► L = - Σ_k q_k · log p_k ◄┘
```

| thành phần | vai trò | lưu ý |
|---|---|---|
| Bộ mã hoá `f_θ` | trích biểu diễn `h` | bất kỳ: Transformer, CNN… |
| Head chiếu `g_φ` | đưa `h` sang không gian phân cụm riêng | tách mục tiêu phân cụm khỏi `h` mà head chính đọc, để phân cụm không bóp méo hình học của `h` |
| Chuẩn hoá L2 cho `z` và `C` | điểm `s` thành cosine, nằm trong [-1, 1] | không chuẩn hoá thì điểm không bị chặn, Sinkhorn tràn số |
| Nguyên mẫu `C` | K "tâm cụm" học bằng gradient | tham số thường, không phải trung bình động |
| Sinkhorn | tạo đích gán cân bằng từ `s` | chạy không gradient |
| Nhiệt độ `τ` | độ sắc của phân phối dự đoán | chỉ áp vào nhánh dự đoán |

`D` (số chiều không gian phân cụm) và `K` (số nguyên mẫu) là **hai siêu tham số độc lập**. SwAV
dùng `D = 128`, `K = 3000` trên ImageNet.

## 4. Nền tảng lý thuyết: gán nhãn là một bài toán vận chuyển tối ưu

Với một batch B mẫu, ma trận điểm `S = Cᵀ Z` có kích thước `K × B`. Tìm ma trận gán mềm
`Q ∈ ℝ₊^{K×B}` sao cho tổng điểm lớn nhất, **dưới ràng buộc cân bằng**:

```
max_Q   Tr(Qᵀ S) + ε · H(Q)

với     Q · 1_B  = (1/K) · 1_K     (mỗi nguyên mẫu nhận đúng 1/K tổng khối lượng)
        Qᵀ · 1_K = (1/B) · 1_B     (mỗi mẫu được phân đúng 1/B khối lượng)
        H(Q) = - Σ_ij Q_ij log Q_ij   (entropy)
```

Tập ràng buộc là **đa diện vận chuyển** (transportation polytope). Số hạng entropy làm bài toán lồi
chặt và cho nghiệm dạng đóng:

```
Q* = Diag(u) · exp(S / ε) · Diag(v)
```

trong đó `u ∈ ℝ^K`, `v ∈ ℝ^B` là hai vector chuẩn hoá, tìm bằng **thuật toán Sinkhorn-Knopp**:
xen kẽ chuẩn hoá hàng và cột của `exp(S/ε)` cho tới khi thoả hai ràng buộc. Vài vòng lặp (SwAV
dùng 3) là đủ trong thực tế.

Lưu ý: vì bước chuẩn hoá cuối cùng là theo mẫu (cột), sau 3 vòng ràng buộc **mỗi mẫu** được thoả
chính xác, còn ràng buộc **mỗi nguyên mẫu** chỉ thoả **xấp xỉ**. Đo thử với B = 256, K = 8 và điểm
ngẫu nhiên: mỗi nguyên mẫu nhận 26-40 mẫu quanh mức lý tưởng 32. Mức xấp xỉ này đủ để chống sụp
đổ (đưa vào một đầu vào đã sụp hẳn, đích trả ra vẫn chia đều đúng 32 mẫu mỗi nguyên mẫu), và chính
SwAV chấp nhận nó để đổi lấy tốc độ.

**Vì sao chống sụp đổ.** Nghiệm "mọi mẫu vào một cụm" vi phạm ràng buộc hàng: một nguyên mẫu nhận
toàn bộ khối lượng thay vì 1/K. Nên cách duy nhất để giảm mất mát là gán **vừa tự tin vừa cân
bằng**, tức mô hình buộc phải tìm ra K nhóm phân biệt được trong dữ liệu.

**Vai trò của ε.** ε nhỏ cho phép gán gần như cứng (one-hot); ε lớn cho phép gán mềm, trải đều.
SwAV dùng ε = 0,05. ε quá nhỏ làm `exp(S/ε)` tràn số nếu không trừ giá trị lớn nhất trước.

**Phân biệt ε và τ.** ε điều khiển độ sắc của **đích** (trong Sinkhorn), τ điều khiển độ sắc của
**dự đoán** (trong softmax). Hai tham số đặt ở hai nơi khác nhau; không nhân τ vào điểm trước khi
đưa vào Sinkhorn.

## 5. Hàm mất mát: một góc nhìn hay hoán đổi hai góc nhìn

Mất mát cơ bản là cross-entropy giữa đích và dự đoán, trung bình trên batch:

```
L = - (1/B) · Σ_i Σ_k q_ik · log softmax(s_i / τ)_k
```

Có hai cách lấy đích:

| cách | đích `q` lấy từ | dùng ở | tính chất |
|---|---|---|---|
| **Một góc nhìn (self-labelling)** | điểm của **chính** mẫu đó | SeLa (ngoại tuyến), `latent_proto` (trực tuyến) | mô hình học phép chia cân bằng của hình học **sẵn có**; không tự học tính bất biến |
| **Hoán đổi (swapped prediction)** | điểm của **một phép biến đổi khác** của cùng mẫu | SwAV | dự đoán cụm của góc nhìn A từ góc nhìn B ⇒ biểu diễn phải **bất biến** với phép biến đổi |

Điểm quan trọng về lý thuyết: với **một góc nhìn**, đích được tính từ chính điểm mà mô hình đang
dự đoán, nên mục tiêu chỉ là "làm phép gán của mình vừa sắc vừa cân bằng". Mô hình sẽ chia dữ
liệu theo **trục dễ tách nhất** trong biểu diễn hiện có, chưa chắc là trục có nghĩa. Ở SwAV,
chính phép **hoán đổi giữa hai phép biến đổi** mới là thứ đưa ngữ nghĩa vào: cụm phải giữ nguyên
khi dữ liệu bị biến đổi. Với mã nguồn, các phép biến đổi giữ nghĩa có thể là đổi tên định danh,
chèn mã chết, đổi thứ tự câu lệnh độc lập.

Luôn **dừng gradient** qua đích `q`: gradient chỉ đi qua nhánh dự đoán.

## 6. Thuật toán và cài đặt tối giản (PyTorch)

### 6.1. Sinkhorn-Knopp

```python
@torch.no_grad()
def sinkhorn(scores, epsilon=0.05, n_iters=3):
    """scores: [B, K] điểm cosine. Trả về đích gán mềm [B, K], mỗi hàng tổng bằng 1."""
    Q = torch.exp((scores - scores.max()) / epsilon).t()   # [K, B]; trừ max để không tràn số
    Q = Q / Q.sum()                                         # tổng toàn ma trận = 1
    K, B = Q.shape
    for _ in range(n_iters):
        Q = Q / Q.sum(dim=1, keepdim=True) / K              # mỗi nguyên mẫu: tổng 1/K
        Q = Q / Q.sum(dim=0, keepdim=True) / B              # mỗi mẫu: tổng 1/B
    return (Q * B).t()                                      # mỗi mẫu: phân phối tổng 1
```

Phiên bản trong mã SwAV (`distributed_sinkhorn`) giống hệt về thuật toán, chỉ thêm `all_reduce`
để cộng tổng qua nhiều GPU khi huấn luyện phân tán.

### 6.2. Head nguyên mẫu

```python
class PrototypeHead(nn.Module):
    def __init__(self, d_in, d_proj=128, n_prototypes=64, hidden=None):
        super().__init__()
        hidden = hidden or d_in
        self.proj = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, d_proj))
        self.prototypes = nn.Parameter(torch.randn(n_prototypes, d_proj) * 0.02)

    def forward(self, h, freeze_prototypes=False):
        z = F.normalize(self.proj(h), dim=-1)                      # [B, D]
        C = F.normalize(self.prototypes, dim=-1)                   # [K, D]
        if freeze_prototypes:
            C = C.detach()
        return z @ C.t()                                           # [B, K] cosine, CHƯA chia τ


def prototype_loss(scores, tau=0.1, epsilon=0.05):
    q = sinkhorn(scores.detach(), epsilon)                         # đích, không gradient
    return -(q * F.log_softmax(scores / tau, dim=-1)).sum(dim=-1).mean()
```

### 6.3. Vòng huấn luyện khi dùng làm **đầu ra phụ**

```python
for step, (x, y) in enumerate(loader):
    h = encoder(x)
    loss_main = F.cross_entropy(main_head(h), y)
    scores = proto_head(h, freeze_prototypes=(step < freeze_steps))
    loss = loss_main + lam * prototype_loss(scores, tau, epsilon)
    loss.backward(); optimizer.step(); optimizer.zero_grad()
```

Khi dùng cho **học tự giám sát thuần**, bỏ `loss_main`, và nên dùng hoán đổi hai góc nhìn:
`loss = CE(q_A, p_B) + CE(q_B, p_A)` với `A`, `B` là hai phép biến đổi của cùng batch.

## 7. Siêu tham số và các chi tiết dễ cài sai

| tham số | ý nghĩa | giá trị tham chiếu (SwAV) |
|---|---|---|
| `K` | số nguyên mẫu (số cụm) | 3000 (ImageNet); nên chọn cỡ số nhóm tự nhiên của dữ liệu trở lên |
| `D` | số chiều không gian phân cụm | 128 |
| `τ` | nhiệt độ của dự đoán | 0,1 |
| `ε` | điều chuẩn entropy của Sinkhorn | 0,05 |
| số vòng Sinkhorn | | 3 |
| `freeze_prototypes_niters` | số bước đầu giữ nguyên mẫu cố định | 313 (khoảng 1 epoch) |
| hàng đợi (queue) | lưu đặc trưng các batch trước để Sinkhorn có đủ mẫu | bật khi batch nhỏ |

**Tám chi tiết dễ sai:**

1. **Không chuẩn hoá** `z` hoặc `C`: điểm không bị chặn, `exp(S/ε)` tràn số, và mô hình có thể
   giảm mất mát chỉ bằng cách phóng to vector thay vì học phân cụm.
2. **Không trừ giá trị lớn nhất** trước `exp`: với ε = 0,05, điểm 1,0 thành `exp(20)`; batch lớn
   hoặc nửa độ chính xác (fp16) là tràn.
3. **Nhân τ vào điểm trước Sinkhorn**: tương đương đổi ε thành `ε·τ` = 0,005, đích gần như cứng
   ngay từ đầu.
4. **Để gradient đi qua đích**: mô hình có thể "kéo" đích về phía dự đoán, mất tác dụng chống sụp.
5. **Batch nhỏ so với K**: ràng buộc cân bằng buộc mỗi nguyên mẫu nhận `B/K` mẫu mỗi batch. Nếu
   `B < K`, một số nguyên mẫu phải nhận "một phần mẫu", phép gán thành nhiễu. Cần `B ≫ K` hoặc
   hàng đợi.
6. **Nguyên mẫu di chuyển quá sớm**: lúc đầu `z` còn ngẫu nhiên, đích Sinkhorn thay đổi liên tục
   theo nguyên mẫu. Đóng băng nguyên mẫu vài trăm bước đầu để head chiếu ổn định trước.
7. **Đếm bước sai đơn vị**: bộ đếm "bước" để đóng băng nguyên mẫu phải đếm **bước tối ưu**, không
   phải lượt forward. Với các bộ tối ưu gọi forward hai lần mỗi bước (như SAM), đếm theo forward sẽ
   rút ngắn cửa sổ đóng băng đi một nửa.
8. **Tính lại đích ở lượt forward thứ hai của SAM**: ở điểm `w + e`, Sinkhorn cho một đích khác đích
   ở `w`. Nên tính đích một lần ở `w` rồi dùng lại, để gradient ở `w + e` đo đúng cùng một mục tiêu.

## 7.1. Chẩn đoán khi huấn luyện

Mất mát giảm **không** chứng minh head học được gì có ích. Nên theo dõi:

| đại lượng | dấu hiệu hỏng |
|---|---|
| số nguyên mẫu được dùng (argmax) trên một epoch | nhỏ hơn nhiều so với K ⇒ sụp một phần |
| tỉ lệ của nguyên mẫu đông nhất | vượt xa 1/K ⇒ mất cân bằng |
| entropy trung bình của `p` | rất cao ⇒ dự đoán không sắc; rất thấp quá sớm ⇒ học thuộc đích |
| NMI giữa cụm và **nhãn quan tâm** (vd loại lỗi) | gần 0 ⇒ cụm không trùng cấu trúc cần có |
| NMI giữa cụm và **nhãn nhiệm vụ chính** | cao ⇒ head phụ chỉ học lại cái head chính đã biết |

## 8. Cài đặt trong repo này (MultiVD)

### 8.1. Bản đồ mã

| thành phần | vị trí |
|---|---|
| Khai báo head chiếu + nguyên mẫu | `src/model.py:208-221` |
| Forward: chuẩn hoá, đóng băng nguyên mẫu, điểm cosine | `src/model.py:334-344` |
| Sinkhorn-Knopp | `src/model.py:398-415` |
| Mất mát (cross-entropy với đích Sinkhorn, chia τ) | `src/model.py:417-448` |
| Ghép `L_vul + λ · L_proto` | `src/train.py:52-58` |
| Chẩn đoán: số slot dùng, slot đông nhất, NMI với nhãn nhị phân và với CWE | `src/train.py:20-49` |
| Lượt forward thứ hai của SAM | `src/train.py:102-111` |
| Tham số dòng lệnh `--num_latent`, `--latent_temperature`, `--freeze_prototypes_steps` | `src/train_transfer.py:1811-1814`, `:1838-1840` |

### 8.2. Cấu hình

| | MultiVD | SwAV |
|---|---|---|
| Vai trò | **đầu ra phụ** ở Pha 1, cộng với mất mát phân loại lỗ hổng | mục tiêu tự giám sát duy nhất |
| Góc nhìn | **một** (không có phép biến đổi dữ liệu) | hai trở lên, hoán đổi dự đoán |
| Head chiếu | `Linear(H,H) → GELU → Linear(H,K)` | `Linear → BN → ReLU → Linear`, ra 128 chiều |
| `K` nguyên mẫu | 8 (`--num_latent`) | 3000 |
| `D` số chiều phân cụm | **= K = 8** (ma trận nguyên mẫu `K × K`) | 128, độc lập với K |
| `τ` | 0,1 (`--latent_temperature`) | 0,1 |
| `ε`, số vòng Sinkhorn | 0,05; 3 (cố định trong mã, không có cờ) | 0,05; 3 |
| Đóng băng nguyên mẫu | `--freeze_prototypes_steps`, mặc định 0 | 313 bước |
| Chuẩn hoá nguyên mẫu | trong forward (`F.normalize`) | chuẩn hoá lại trọng số sau mỗi bước tối ưu |
| Hàng đợi | không | có khi batch nhỏ |
| Trọng số λ | cờ `--lambda_cwe` (cấu hình từng dùng: 0,05) | không có |
| Ở Pha 2 | head đóng băng, không dùng | không áp dụng |

### 8.3. Những điểm phát hiện khi kiểm tra mã

- **`D` bị buộc bằng `K`.** `self.prototypes` có kích thước `(num_latent, num_latent)` và head
  chiếu ra `num_latent` chiều, nên không thể đổi số cụm mà giữ nguyên số chiều hay ngược lại.
  Muốn tách, thêm một cờ riêng cho `D`.
- **Bộ đếm `_step` đếm lượt forward ở chế độ train**, không đếm bước tối ưu. Khi bật SAM (hai lượt
  forward mỗi bước), cửa sổ `--freeze_prototypes_steps` bị rút ngắn một nửa. Vô hại khi cờ này ở
  mặc định 0.
- **Lượt forward thứ hai của SAM tính lại đích Sinkhorn** tại `w + e`, nên gradient ở đó ứng với
  một mục tiêu hơi khác (chi tiết số 8 ở mục 7).
- **Một góc nhìn, không phép biến đổi**: đúng như mục 5, head học phép chia cân bằng của biểu diễn
  sẵn có chứ không có nguồn tín hiệu buộc cụm mang ngữ nghĩa loại lỗ hổng.
- **Batch so với K**: batch Pha 1 là 16 với K = 8, tức trung bình 2 mẫu mỗi nguyên mẫu mỗi batch.
  Đủ để Sinkhorn chạy, nhưng đích dao động mạnh giữa các batch; không có hàng đợi để làm mượt.
- Thuật toán Sinkhorn và mất mát **khớp** với SwAV (trừ max trước `exp`, chuẩn hoá hàng rồi cột,
  3 vòng, τ áp riêng ở nhánh dự đoán, dừng gradient qua đích).

### 8.4. Trạng thái trong dự án (tóm tắt)

Đã bị loại khỏi cấu hình chốt ngày 31/08/2026. Số đo và lập luận đầy đủ ở `FACTS.md`, tóm tắt:
đóng góp riêng của head gần bằng 0 ở mọi khối; tự sập ở CodeBERT với nguồn `full` (val 0,3432);
thêm SAM ở Pha 2 làm hại (1/6 ô dương, ba fold sập về 0,45-0,54 trên một backbone); khi học λ tự
động thì λ hiệu dụng ≈ 1,05 chỉ vì ràng buộc cân bằng giữ mất mát cùng thang với cross-entropy nhị
phân, không phản ánh giá trị cho bài toán đích. Các điểm ở mục 8.3 (một góc nhìn, K nhỏ, batch nhỏ,
không hàng đợi) là những lý do lý thuyết có thể giải thích vì sao head không học được cấu trúc có
ích; chúng **chưa được kiểm chứng riêng** bằng thí nghiệm.

## 9. Tài liệu tham khảo

1. Y. M. Asano, C. Rupprecht, A. Vedaldi. *Self-labelling via simultaneous clustering and
   representation learning*. ICLR 2020. arXiv:1911.05371.
2. M. Caron, I. Misra, J. Mairal, P. Goyal, P. Bojanowski, A. Joulin. *Unsupervised Learning of
   Visual Features by Contrasting Cluster Assignments* (SwAV). NeurIPS 2020. arXiv:2006.09882.
3. M. Cuturi. *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*. NeurIPS 2013.
   arXiv:1306.0895.
4. M. Caron, P. Bojanowski, A. Joulin, M. Douze. *Deep Clustering for Unsupervised Learning of
   Visual Features* (DeepCluster). ECCV 2018. arXiv:1807.05520.
5. M. Caron et al. *Emerging Properties in Self-Supervised Vision Transformers* (DINO). ICCV 2021.
   arXiv:2104.14294.
