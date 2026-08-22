# SAM — mã gốc của Google và những gì phải giữ khi port sang PyTorch

Nguồn: https://github.com/google-research/sam
Bài báo: Foret, Kleiner, Mobahi, Neyshabur, *Sharpness-Aware Minimization for
Efficiently Improving Generalization*, arXiv:2010.01412 (ICLR 2021).

## Điều phải biết trước: repo gốc là **JAX/Flax**, không phải PyTorch

Thư mục chính là `sam_jax/`, chạy bằng `sam.sam_jax.train.py`, dùng checkpoint
FLAX. Toàn bộ codebase của dự án này là PyTorch. **Không thể dùng lại mã**; phải
port công thức. Bù lại, phần thuật toán rất ngắn nên port đúng được — chép lại
nguyên văn ở dưới để không phải nhớ.

## Mã gốc, nguyên văn

Chuẩn hoá gradient (`sam_jax/training_utils/flax_training.py`):

```python
def dual_vector(y: jnp.ndarray) -> jnp.ndarray:
  """Returns the solution of max_x y^T x s.t. ||x||_2 <= 1.

  Args:
    y: A pytree of numpy ndarray, vector y in the equation above.
  """
  gradient_norm = jnp.sqrt(sum(
      [jnp.sum(jnp.square(e)) for e in jax.tree_util.tree_leaves(y)]))
  normalized_gradient = jax.tree_map(lambda x: x / gradient_norm, y)
  return normalized_gradient
```

Bước leo và lần lấy gradient thứ hai:

```python
grad = dual_vector(grad)
noised_model = jax.tree_multimap(lambda a, b: a + rho * b,
                                 model, grad)
(_, (_, logits)), grad = jax.value_and_grad(
    forward_and_loss, has_aux=True)(noised_model)
```

## Bốn chi tiết dễ port sai

1. **Chuẩn L2 lấy TOÀN CỤC, không theo từng lớp.** `gradient_norm` cộng bình
   phương qua *mọi* tensor rồi mới chia. Chuẩn hoá từng lớp là một thuật toán
   khác và cho kết quả khác.

2. **ρ là độ dài TUYỆT ĐỐI của nhiễu loạn.** Vì gradient đã chuẩn hoá về chuẩn 1,
   nên `‖ε‖ = ρ` đúng bằng ρ. Nó **không** tỉ lệ theo `‖w‖`. Các giá trị ρ trong
   bài báo (0.05, 0.1…) là theo quy ước tuyệt đối này.

3. **Gradient thứ hai áp cho trọng số GỐC**, không phải cho điểm đã nhiễu loạn.
   Điểm nhiễu loạn chỉ dùng để *lấy* gradient rồi bỏ đi.

4. **Hai lượt forward-backward mỗi bước** → 2× thời gian huấn luyện ở giai đoạn
   nào dùng nó.

## Khi port vào dự án này còn phải quyết ba việc

- **Dùng ở Phase 1 hay Phase 2 hay cả hai.** §40 (Phase 1 đẩy CodeT5+ đi xa gấp
  1.86× CodeBERT) trỏ về Phase 1: làm nghiệm nguồn phẳng để Phase 2 phá ít hơn.
- **Chồng lấn với RecAdam.** Cả hai đều sửa bước cập nhật — RecAdam thêm lực kéo
  về điểm neo (`RecAdam.py:128`), SAM đổi chỗ lấy gradient. Ghép được nhưng phải
  quyết thứ tự và viết cẩn thận.
- **Dò ρ**, và mỗi giá trị ρ là một lượt huấn luyện tốn gấp đôi.

## Ghi chú về `src/measure_sharpness.py`

Script đo độ nhọn dùng `‖ε‖ = ρ·‖w‖` (**tương đối** theo chuẩn trọng số) chứ
không phải `‖ε‖ = ρ` như bài báo. Cố ý: các backbone có `‖w‖` khác nhau, nên đo
tương đối mới so được giữa chúng — đó là mục đích của phép đo.

Nhưng **hai quy ước này không được lẫn**. Script nhận cờ `--rho_mode` để chọn:

- `relative` (mặc định) — để so giữa các backbone
- `absolute` — đúng quy ước bài báo, để chọn ρ khi thực sự chạy SAM

Và nó in ra cả `‖ε‖` tuyệt đối lẫn ρ tương đối ở mọi dòng, nên chuyển đổi được.
