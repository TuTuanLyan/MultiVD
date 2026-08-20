# Hình kiến trúc

| File | Nội dung |
| --- | --- |
| `fig1_pipeline.svg/.png/.pdf` | Pipeline hai pha: Phase 1 multitask trên source, Phase 2 RecAdam trên target |
| `fig2_aux_heads.svg/.png/.pdf` | Bốn biến thể head phụ: `cwe`, `latent_bottleneck`, `latent_proto`, `none` |

Baseline không có trong hình — nó chỉ là backbone + `vul_head`, không có Phase 1.

## Nguồn sự thật

Hình được vẽ theo `src/model.py` (`TransferModel.forward`, `auxiliary_loss`) và
`run/fold-major.sh`, không theo trí nhớ. Nếu sửa kiến trúc thì phải sửa hình.

Các số Δ trong Hình 2 lấy từ `RESULT.md` §27 (n = 10 fold ghép cặp, seed 36 + 12,
CodeBERT, pooling `cls`).

## Dựng lại ảnh từ SVG

SVG là bản gốc; PNG và PDF sinh ra từ nó.

```bash
python -m pip install cairosvg      # nếu chưa có
python - <<'PY'
import cairosvg
for f in ("figures/fig1_pipeline", "figures/fig2_aux_heads"):
    cairosvg.svg2png(url=f + ".svg", write_to=f + ".png", scale=2.0, background_color="white")
    cairosvg.svg2pdf(url=f + ".svg", write_to=f + ".pdf", background_color="white")
PY
```

**Đừng dùng ImageMagick `convert`** để chuyển các SVG này. Bộ render SVG của nó bỏ
qua khối Unicode Latin Extended Additional, nên mọi dấu tiếng Việt kiểu `ạ ế ị ộ ừ`
biến mất trong khi `ê ô ó` vẫn hiện — lỗi im lặng, ảnh vẫn ra bình thường và chỉ
phát hiện được khi nhìn kỹ.
