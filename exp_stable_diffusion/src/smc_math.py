import math

import torch


def tweedie_x0hat_from_eps(x_t: torch.Tensor, abar: float, eps: torch.Tensor) -> torch.Tensor:
    """Tweedie x0-hat from an eps prediction: (x_t - sqrt(1-abar)*eps) / sqrt(abar)."""
    return (x_t - math.sqrt(1.0 - abar) * eps) / math.sqrt(abar)


def ess(w: torch.Tensor) -> float:
    """Effective sample size 1 / sum_i (w_i / sum_j w_j)^2 of non-negative linear
    weights (renormalized internally). 0.0 if total mass is non-positive."""
    w = w.detach()
    s = float(w.sum().item())
    if s <= 0.0:
        return 0.0
    wn = w / (s + 1e-12)
    return float(1.0 / (wn.pow(2).sum().item() + 1e-12))


def clip_weights(weights: torch.Tensor, clip: float | None) -> torch.Tensor:
    """Clipped importance sampling: clamp each weight's log-ratio to the geometric
    mean into [-clip, clip] (no particle carries more than exp(clip)x the mean).
    ``clip`` None or <= 0 is a no-op. Returns un-normalized clamped weights."""
    if clip is None or clip <= 0:
        return weights
    w = weights.detach().clamp_min(0.0)
    if w.sum() <= 0:
        return weights
    logw = torch.log(w + 1e-30)
    logw = logw - logw.mean()
    logw = logw.clamp(-float(clip), float(clip))
    return torch.exp(logw)


def systematic_resample_sorted(weights: torch.Tensor, generator: "torch.Generator | None" = None) -> torch.Tensor:
    """Systematic resampling with descending weight sort (Chopin et al. 2022).

    ``weights`` must be normalized (sum == 1). Returns an (N,) integer index
    tensor (with repeats) in the original particle ordering. Sorting before the
    cumsum + systematic grid reduces index-map variance; the last cumsum entry is
    forced to 1.0 for numerical safety.
    """
    N = weights.shape[0]
    order = torch.argsort(weights, descending=True)
    cumsum = torch.cumsum(weights[order], dim=0)
    cumsum[-1] = 1.0
    u0 = torch.rand(1, device=weights.device, dtype=weights.dtype, generator=generator) / N
    u = u0 + torch.arange(N, device=weights.device, dtype=weights.dtype) / N
    return order[torch.searchsorted(cumsum, u).clamp(max=N - 1)]


def solve_delta_for_ess(
    w_q: torch.Tensor,
    one_minus_p: torch.Tensor,
    target_ess: float,
    delta_max: float,
    max_iter: int = 30,
) -> float:
    """Binary-search the tempering increment delta in [0, delta_max] keeping the
    ESS of ``w_q * one_minus_p**delta`` ~ ``target_ess``. ESS is non-increasing in
    delta, so: ESS<=target at 0 -> 0; ESS>=target at delta_max -> delta_max; else
    bisect.
    """
    omp = one_minus_p.detach()

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
    return float(0.5 * (lo + hi))
