from dataclasses import dataclass

import torch


def rbf_features(x: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    """Gaussian kernel features k(x,c) = exp(-||x-c||_2^2 / (2 sigma^2))."""
    x2 = (x**2).sum(dim=-1, keepdim=True)
    c2 = (centers**2).sum(dim=-1, keepdim=True).T
    dist2 = x2 - 2.0 * (x @ centers.T) + c2
    return torch.exp(-dist2 / (2.0 * sigma * sigma))


@torch.no_grad()
def median_heuristic_sigma(centers: torch.Tensor, scale: float = 0.4) -> float:
    """Return ``scale`` times the median pairwise center distance."""
    b = centers.shape[0]
    if b < 2:
        return 1.0
    D = torch.cdist(centers, centers, p=2)
    iu = torch.triu_indices(b, b, offset=1, device=centers.device)
    med = float(max(torch.median(D[iu[0], iu[1]]).item(), 1e-6))
    return med * scale


@dataclass
class KernelPULogistic:
    sigma: float | None = None
    sigma_scale: float = 0.4
    lam: float = 1e-3  # RKHS L2 regularization
    max_iter: int = 50
    device: str = "cpu"

    def fit(self, x_pos: torch.Tensor, x_u: torch.Tensor, w_u: torch.Tensor, pi: float) -> None:
        """uPU logistic fit. ``x_pos`` positives (style-B denoise), ``x_u`` unlabeled
        (content denoise) with per-particle weights ``w_u``, class prior ``pi``."""
        if x_u.shape[0] != w_u.shape[0]:
            raise ValueError("w_u must match x_u along dim 0.")
        if not (0.0 < pi < 1.0):
            raise ValueError("pi must be in (0,1).")

        centers = torch.cat([x_pos, x_u], dim=0).detach()
        b = centers.shape[0]
        sig = self.sigma if self.sigma is not None else median_heuristic_sigma(centers, scale=self.sigma_scale)

        Phi_p = rbf_features(x_pos, centers, sig)
        Phi_u = rbf_features(x_u, centers, sig)
        Kcc = rbf_features(centers, centers, sig)

        # Normalize the unlabeled weights and discard invalid values.
        w_u_norm = w_u.detach().clone().clamp_min(0.0)
        w_u_norm[torch.isnan(w_u_norm) | torch.isinf(w_u_norm)] = 0.0
        s = w_u_norm.sum()
        w_u_norm = w_u_norm / s if s >= 1e-12 else torch.full_like(w_u_norm, 1.0 / max(1, x_u.shape[0]))
        w_p_norm = torch.full((x_pos.shape[0],), 1.0 / max(1, x_pos.shape[0]), device=self.device)

        alpha = torch.zeros(b, device=self.device, requires_grad=True)
        bias = torch.zeros((), device=self.device, requires_grad=True)
        opt = torch.optim.LBFGS(
            [alpha, bias],
            lr=1.0,
            max_iter=self.max_iter,
            tolerance_grad=1e-7,
            tolerance_change=1e-9,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            softplus = torch.nn.functional.softplus
            g_p = Phi_p @ alpha + bias
            g_u = Phi_u @ alpha + bias
            # uPU risk: pi*E_p[loss(+)] + (E_u[loss(-)] - pi*E_p[loss(-)]) + RKHS reg.
            term_pos = float(pi) * (w_p_norm * softplus(-g_p)).sum()
            term_u = (w_u_norm * softplus(g_u)).sum()
            term_pos_neg = -float(pi) * (w_p_norm * softplus(g_p)).sum()
            reg = 0.5 * self.lam * (alpha @ (Kcc @ alpha))
            loss = term_pos + term_u + term_pos_neg + reg
            loss.backward()
            return loss

        opt.step(closure)  # type: ignore
        self.centers = centers.detach()
        self.alpha = alpha.detach()
        self.bias = bias.detach()
        self.sigma_used = sig

    def decision(self, x: torch.Tensor) -> torch.Tensor:
        """Raw logit g(x). Before the first fit (no centers yet) returns 0 (=> prob
        0.5); the SMC loop queries p_old at the first PU-on step before fitting."""
        if getattr(self, "centers", None) is None:
            return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        return rbf_features(x, self.centers, float(self.sigma_used)) @ self.alpha + self.bias

    def predict_proba_pos(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decision(x))
