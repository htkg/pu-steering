"""Small SMC helpers shared by the score and reference samplers."""

import torch

from .diffusion import clip01


def ess(w: torch.Tensor) -> float:
    """Kish effective sample size 1 / sum(w_norm^2). 0 if total mass is non-positive."""
    w = w.detach()
    s = float(w.sum())
    if s <= 0.0:
        return 0.0
    wn = w / (s + 1e-12)
    return float(1.0 / (wn.pow(2).sum() + 1e-12))


def solve_delta_for_ess(w_q: torch.Tensor, one_minus_p: torch.Tensor, target_ess: float, delta_max: float, max_iter: int = 30) -> float:
    """Largest tempering increment delta in [0, delta_max] keeping ESS(w_q*(1-p)^delta)
    >= target_ess. ESS is monotone non-increasing in delta, so bisect."""
    omp = clip01(one_minus_p)

    def ess_of(d: float) -> float:
        return ess(w_q * (omp**d))

    lo, hi = 0.0, float(delta_max)
    if ess_of(lo) <= target_ess:
        return 0.0
    if ess_of(hi) >= target_ess:
        return hi
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if ess_of(mid) > target_ess:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)
