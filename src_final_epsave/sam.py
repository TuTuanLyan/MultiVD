#!/usr/bin/env python3
"""SAM — Sharpness-Aware Minimization, port từ mã gốc của Google.

Foret, Kleiner, Mobahi, Neyshabur, ICLR 2021 (arXiv:2010.01412).
Mã gốc: https://github.com/google-research/sam (JAX/Flax, nên phải port chứ
không dùng lại được). Nguyên văn phần thuật toán chép ở docs/SAM_REFERENCE.md.

SAM cực tiểu   min_w  max_{||eps|| <= rho}  L(w + eps)
tức đi tìm cực tiểu mà CẢ LÂN CẬN đều có loss thấp, không chỉ điểm đó.

Mỗi bước hai lượt forward-backward:
  1. leo   eps = rho * g / ||g||   ->  w + eps   (điểm xấu nhất trong quả cầu)
  2. xuống lấy gradient TẠI w + eps, áp cho w GỐC, rồi khôi phục w

Bốn chi tiết phải giữ đúng như mã gốc — cả bốn đều dễ port sai:

  * `dual_vector` chuẩn hoá theo chuẩn L2 **TOÀN CỤC** qua mọi tensor, không phải
    từng lớp. Chuẩn hoá từng lớp là thuật toán khác.
  * rho là độ dài **TUYỆT ĐỐI** của nhiễu loạn: gradient đã chuẩn hoá về chuẩn 1
    nên ||eps|| = rho, KHÔNG tỉ lệ theo ||w||.
  * gradient lượt hai áp cho **trọng số gốc**; điểm nhiễu loạn chỉ để lấy gradient.
  * mọi thứ diễn ra TRƯỚC optimizer.step(), nên SAM ghép được với bất kỳ optimizer
    nào — ở đây là RecAdam hoặc AdamW, không cần sửa optimizer.

Lưu ý về RecAdam: RecAdam thêm một lực kéo về điểm neo SAU khi dùng gradient
(`RecAdam.py:128`), còn SAM chỉ đổi CHỖ lấy gradient. Hai thứ tác động vào hai
khâu khác nhau nên chồng lên nhau được, và lực kéo của RecAdam không đi qua bước
leo của SAM.
"""

import torch


@torch.no_grad()
def _grad_global_norm(params):
    """Chuẩn L2 toàn cục của gradient — `dual_vector` trong mã gốc."""
    total = None
    for p in params:
        if p.grad is None:
            continue
        s = (p.grad.detach() ** 2).sum()
        total = s if total is None else total + s
    if total is None:
        return None
    return torch.sqrt(total)


class SAMStep:
    """Bước leo/khôi phục của SAM quanh một lượt backward sẵn có.

    Dùng như sau, thay cho một lượt forward-backward đơn:

        sam.ascend(params)          # sau khi da backward lan 1
        <forward + backward lan 2>
        sam.restore()               # truoc optimizer.step()

    Giữ bản sao của eps chứ không sao chép cả trọng số: bộ nhớ thêm đúng bằng
    kích thước gradient, và khôi phục là phép trừ chính xác chứ không phải nạp lại.
    """

    def __init__(self, rho, variant="sam", eta=0.01):
        """variant="sam": ban goc Foret et al. — eps = rho*g/||g||, chuan L2 TOAN CUC.
        variant="asam": Kwon et al., ICML 2021 (arXiv:2102.11600) — chuan hoa THEO
        TUNG THAM SO qua T_w = |w| + eta, nen rho BAT BIEN theo thang trong so:

            eps = rho * T_w^2 * g / ||T_w * g||

        Ly do thu: trong du an nay rho toi uu lech 5 lan giua hai backbone
        (CodeBERT 0,02 · CodeT5+ 0,05-0,10) va hai cua so gan nhu khong giao nhau —
        dung trieu chung ma ASAM duoc thiet ke de chua. Neu ASAM dung thi MOT rho
        chay duoc cho ca ba backbone.

        Theo ban cai chinh thuc (SamsungLabs/ASAM): chi tham so co 'weight' trong
        ten moi duoc nhan T_w; bias va cac he so khac giu T_w = 1.
        """
        if rho <= 0:
            raise ValueError("rho phai duong")
        if variant not in ("sam", "asam"):
            raise ValueError(f"variant phai la sam hoac asam, nhan {variant!r}")
        self.rho = rho
        self.variant = variant
        self.eta = eta
        self._eps = []

    @torch.no_grad()
    def ascend(self, params, names=None):
        """Đi tới w + eps. Trả về False nếu gradient bằng 0 (bỏ qua bước).

        `names` chỉ dùng cho variant="asam" (cần biết tham số nào là 'weight').
        """
        if self.variant == "asam":
            return self._ascend_asam(params, names)
        self._eps = []
        norm = _grad_global_norm(params)
        if norm is None or not torch.isfinite(norm) or norm.item() == 0.0:
            return False
        scale = self.rho / norm
        for p in params:
            if p.grad is None:
                self._eps.append(None)
                continue
            e = p.grad.detach() * scale
            p.add_(e)
            self._eps.append(e)
        return True

    @torch.no_grad()
    def _ascend_asam(self, params, names):
        if names is None:
            raise ValueError("ASAM can `names` de biet tham so nao la 'weight'")
        self._eps = []
        tws = []
        for p, n in zip(params, names):
            if p.grad is None:
                tws.append(None); continue
            tws.append(p.detach().abs().add(self.eta) if "weight" in n else None)
        total = None
        for p, tw in zip(params, tws):
            if p.grad is None:
                continue
            g = p.grad.detach() if tw is None else p.grad.detach() * tw
            sq = (g ** 2).sum()
            total = sq if total is None else total + sq
        if total is None:
            return False
        norm = torch.sqrt(total)
        if not torch.isfinite(norm) or norm.item() == 0.0:
            return False
        scale = self.rho / norm
        for p, tw in zip(params, tws):
            if p.grad is None:
                self._eps.append(None); continue
            g = p.grad.detach()
            e = (g if tw is None else g * tw * tw) * scale
            p.add_(e)
            self._eps.append(e)
        return True

    @torch.no_grad()
    def restore(self, params):
        """Trừ đúng eps đã cộng, đưa trọng số về w trước khi optimizer bước."""
        for p, e in zip(params, self._eps):
            if e is not None:
                p.sub_(e)
        self._eps = []
