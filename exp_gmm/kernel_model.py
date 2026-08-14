"""PU logistic classifier in RKHS with adaptive RBF bandwidth (median heuristic)."""

from dataclasses import dataclass

import torch


def rbf_features(x: torch.Tensor, centers: torch.Tensor, sigma: float) -> torch.Tensor:
    x2 = (x**2).sum(dim=-1, keepdim=True)
    c2 = (centers**2).sum(dim=-1, keepdim=True).T
    dist2 = x2 - 2.0 * (x @ centers.T) + c2
    return torch.exp(-dist2 / (2.0 * sigma * sigma))


@torch.no_grad()
def median_heuristic_sigma(centers: torch.Tensor, max_pairs: int = 50000, scale: float = 0.4) -> float:
    b = centers.shape[0]
    device = centers.device
    if b < 2:
        return 1.0

    num_all = b * (b - 1) // 2
    if num_all <= max_pairs:
        D = torch.cdist(centers, centers, p=2)
        iu = torch.triu_indices(b, b, offset=1, device=device)
        dists = D[iu[0], iu[1]]
    else:
        i = torch.randint(0, b, (max_pairs,), device=device)
        j = torch.randint(0, b, (max_pairs,), device=device)
        mask = i != j
        i = i[mask]
        j = j[mask]
        dists = torch.norm(centers[i] - centers[j], dim=-1)

    med = torch.median(dists).item()
    med = float(max(med, 1e-6))
    return med * scale


@dataclass
class KernelPULogistic:
    sigma: float | None = None
    sigma_scale: float = 0.4
    lam: float = 1e-3
    n_centers_pos: int = 256
    n_centers_u: int = 256
    max_iter: int = 50
    device: str = "cpu"

    def _choose_centers(self, x_pos: torch.Tensor, x_u: torch.Tensor) -> torch.Tensor:
        ncp = min(self.n_centers_pos, x_pos.shape[0])
        ncu = min(self.n_centers_u, x_u.shape[0])
        ip = torch.randperm(x_pos.shape[0], device=self.device)[:ncp]
        iu = torch.randperm(x_u.shape[0], device=self.device)[:ncu]
        return torch.cat([x_pos[ip], x_u[iu]], dim=0).detach()

    def fit(
        self,
        x_pos: torch.Tensor,
        x_u: torch.Tensor,
        w_u: torch.Tensor,
        pi: float,
        w_pos: torch.Tensor | None = None,
    ) -> None:
        if x_u.shape[0] != w_u.shape[0]:
            raise ValueError("w_u must have shape [M] matching x_u.")
        if not (0.0 < pi < 1.0):
            raise ValueError("pi must be in (0,1).")
        if w_pos is not None and x_pos.shape[0] != w_pos.shape[0]:
            raise ValueError("w_pos must have shape [N] matching x_pos.")

        centers = self._choose_centers(x_pos, x_u)
        b = centers.shape[0]

        sig = self.sigma
        if sig is None:
            sig = median_heuristic_sigma(centers, scale=self.sigma_scale)

        Phi_p = rbf_features(x_pos, centers, sig)
        Phi_u = rbf_features(x_u, centers, sig)

        w_u_norm = w_u.detach().clone()
        w_u_norm = torch.clamp(w_u_norm, min=0.0)
        w_u_norm[torch.isnan(w_u_norm) | torch.isinf(w_u_norm)] = 0.0
        s = w_u_norm.sum()
        if s < 1e-12:
            w_u_norm = torch.full_like(w_u_norm, 1.0 / max(1, x_u.shape[0]))
        else:
            w_u_norm = w_u_norm / s

        if w_pos is None:
            w_p_norm = torch.full((x_pos.shape[0],), 1.0 / max(1, x_pos.shape[0]), device=self.device)
        else:
            w_p_norm = w_pos.detach().clone()
            w_p_norm = torch.clamp(w_p_norm, min=0.0)
            w_p_norm[torch.isnan(w_p_norm) | torch.isinf(w_p_norm)] = 0.0
            s = w_p_norm.sum()
            if s < 1e-12:
                w_p_norm = torch.full_like(w_p_norm, 1.0 / max(1, x_pos.shape[0]))
            else:
                w_p_norm = w_p_norm / s

        Kcc = rbf_features(centers, centers, sig)

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
        Phi = rbf_features(x, self.centers, float(self.sigma_used))
        return Phi @ self.alpha + self.bias

    def predict_proba_pos(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.decision(x))
