import torch


@torch.no_grad()
def _grad_global_norm(params):
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
    def __init__(self, rho, variant="sam", eta=0.01):
        if rho <= 0:
            raise ValueError("rho must be positive")
        if variant not in ("sam", "asam"):
            raise ValueError(f"variant must be sam or asam, got {variant!r}")
        self.rho = rho
        self.variant = variant
        self.eta = eta
        self._eps = []

    @torch.no_grad()
    def ascend(self, params, names=None):
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
            raise ValueError("ASAM needs `names` to know which parameters are 'weight'")
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
        for p, e in zip(params, self._eps):
            if e is not None:
                p.sub_(e)
        self._eps = []
