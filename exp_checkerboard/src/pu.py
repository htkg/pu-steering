"""Online PU classifier f_t: a logistic head on fixed Random Fourier Features (nnPU risk)."""

from dataclasses import dataclass, field

import torch


@torch.no_grad()
def _median_sigma(x: torch.Tensor, max_pairs: int = 20000, scale: float = 0.3) -> float:
    n = x.shape[0]
    if n < 2:
        return 1.0
    if n * (n - 1) // 2 <= max_pairs:
        d = torch.pdist(x)
    else:
        i, j = torch.randint(0, n, (max_pairs,)), torch.randint(0, n, (max_pairs,))
        m = i != j
        d = (x[i[m]] - x[j[m]]).norm(dim=-1)
    return float(torch.median(d).clamp_min(1e-6)) * scale


@dataclass
class OnlinePURFF:
    """Online PU logistic regression on fixed Random Fourier Features."""

    n_features: int = 256
    sigma: float | None = None  # RBF bandwidth; None -> median heuristic on first fit
    sigma_scale: float = 0.3
    lam: float = 1e-3  # L2 on the linear weights
    inner_steps: int = 5  # warm-started Adam steps per fit()
    lr: float = 0.1
    nn_pu: bool = True  # non-negative PU correction
    seed: int = 0

    _init: bool = field(default=False, init=False)

    def _ensure_init(self, x_ref: torch.Tensor) -> None:
        if self._init:
            return
        d = x_ref.shape[-1]
        sig = max(self.sigma if self.sigma is not None else _median_sigma(x_ref, scale=self.sigma_scale), 1e-6)
        g = torch.Generator().manual_seed(self.seed)
        # phi(x) = sqrt(2/D) cos(W x + b),  W ~ N(0, 1/sig^2),  b ~ U[0, 2pi]
        self.W = torch.randn(self.n_features, d, generator=g) / sig
        self.bph = 2.0 * torch.pi * torch.rand(self.n_features, generator=g)
        self.scale = (2.0 / self.n_features) ** 0.5
        self.w = torch.zeros(self.n_features)
        self.b = torch.zeros(())
        self._init = True

    def _phi(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * torch.cos(x @ self.W.T + self.bph)

    @staticmethod
    def _norm_w(w: torch.Tensor | None, n: int) -> torch.Tensor:
        if w is None:
            return torch.full((n,), 1.0 / max(1, n))
        w = torch.nan_to_num(w.detach().clamp(min=0.0), nan=0.0, posinf=0.0, neginf=0.0)
        s = w.sum()
        return w / s if s > 1e-12 else torch.full_like(w, 1.0 / max(1, len(w)))

    def fit(self, x_pos: torch.Tensor, x_u: torch.Tensor, w_u, pi: float, w_pos=None) -> None:
        self._ensure_init(torch.cat([x_pos, x_u], dim=0))
        Pp, Pu = self._phi(x_pos).detach(), self._phi(x_u).detach()
        wp, wu = self._norm_w(w_pos, x_pos.shape[0]), self._norm_w(w_u, x_u.shape[0])
        sp = torch.nn.functional.softplus

        w = self.w.clone().requires_grad_(True)
        b = self.b.clone().requires_grad_(True)
        opt = torch.optim.Adam([w, b], lr=self.lr)
        for _ in range(self.inner_steps):
            opt.zero_grad(set_to_none=True)
            gp, gu = Pp @ w + b, Pu @ w + b
            risk_pos = pi * (wp * sp(-gp)).sum()
            neg_part = (wu * sp(gu)).sum() - pi * (wp * sp(gp)).sum()
            if self.nn_pu:
                neg_part = torch.clamp(neg_part, min=0.0)
            (risk_pos + neg_part + 0.5 * self.lam * (w @ w)).backward()
            opt.step()
        self.w, self.b = w.detach(), b.detach()

    def decision(self, x: torch.Tensor) -> torch.Tensor:
        return self._phi(x) @ self.w + self.b

    def predict_proba_pos(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decision(x))
