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

    def __init__(self, rho):
        if rho <= 0:
            raise ValueError("rho phai duong")
        self.rho = rho
        self._eps = []

    @torch.no_grad()
    def ascend(self, params):
        """Đi tới w + rho*g/||g||. Trả về False nếu gradient bằng 0 (bỏ qua bước)."""
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
    def restore(self, params):
        """Trừ đúng eps đã cộng, đưa trọng số về w trước khi optimizer bước."""
        for p, e in zip(params, self._eps):
            if e is not None:
                p.sub_(e)
        self._eps = []


@torch.no_grad()
def _tw_grad_global_norm(params, eta):
    """Chuan L2 toan cuc cua T_w * g, voi T_w = |w| + eta."""
    total = None
    for p in params:
        if p.grad is None:
            continue
        t = p.detach().abs().add_(eta)
        s = ((t * p.grad.detach()) ** 2).sum()
        total = s if total is None else total + s
    if total is None:
        return None
    return torch.sqrt(total)


class ASAMStep(SAMStep):
    """ASAM — Adaptive Sharpness-Aware Minimization.

    Kwon, Kim, Park, Choi, ICML 2021 (arXiv:2102.11600).

    Khac SAM DUNG MOT CHO: hinh dang cua vung nhieu loan.

        SAM   eps = rho * g / ||g||              -> qua cau ban kinh TUYET DOI
        ASAM  eps = rho * T_w^2 * g / ||T_w g||  -> ellipsoid ti le theo |w|

    voi T_w = diag(|w_1|, ..., |w_k|). Bai bao noi thang ly do: "appropriate rho
    for SAM is dependent on the scales of w on the training trajectory, whereas
    rho of ASAM is not." Do dung la benh da do duoc tren chinh du an nay ngay
    27/08 — cung rho=0.05, codebert ket o train loss = ln2 suot 13 epoch (val
    macro-F1 0.3333) trong khi t5p (0.6614) va unixcoder (0.6476) khong sao.

    HAI CHI TIET DE PORT SAI:

    * `eta` (mac dinh 0.01, dung so cua bai bao): T_w duoc thay bang T_w + eta*I
      cho on dinh so. Khong co no thi tham so nao gan 0 se co nhieu loan gan 0 va
      mau so co the ve 0.
    * **rho cua ASAM KHONG cung thang do voi rho cua SAM.** Vi ban kinh do bang
      don vi |w| chu khong phai khoang cach L2 tuyet doi, gia tri dung lon hon
      khoang mot bac. Bai bao quet {5e-5 ... 0.5, 1.0, 2.0} va chon rho=0.5 cho
      CIFAR-10, 1.0 cho CIFAR-100/ImageNet, trong khi SAM dung 0.05/0.1/0.05.
      Thi nghiem transformer duy nhat cua ho (IWSLT'14 DE-EN, Adam) dung rho=0.1
      cho SAM va 0.2 cho ASAM.
      => Bung rho=0.05 cua SAM vao ASAM la GAN NHU KHONG LAM GI. Neu thay ket qua
      trung khit voi nhanh khong-SAM thi kiem tra rho truoc khi ket luan.
    """

    def __init__(self, rho, eta=0.01):
        super().__init__(rho)
        if eta < 0:
            raise ValueError("eta khong duoc am")
        self.eta = eta

    @torch.no_grad()
    def ascend(self, params):
        self._eps = []
        norm = _tw_grad_global_norm(params, self.eta)
        if norm is None or not torch.isfinite(norm) or norm.item() == 0.0:
            return False
        scale = self.rho / norm
        for p in params:
            if p.grad is None:
                self._eps.append(None)
                continue
            t = p.detach().abs().add_(self.eta)
            e = (t * t) * p.grad.detach() * scale
            p.add_(e)
            self._eps.append(e)
        return True
