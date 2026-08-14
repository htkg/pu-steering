"""FID and KID from cached Inception-2048 activations (no images needed).

FID = standard pytorch-fid Fréchet distance (scipy sqrtm).
KID = textbook unbiased polynomial-kernel MMD^2 (Binkowski et al. 2018), averaged over random equal-size subsets.
Both consume the Inception activations cached at generation time.
"""

import numpy as np
import torch
from pytorch_fid.fid_score import calculate_frechet_distance


def _mean_cov(act: torch.Tensor):
    """Mean (d,) and covariance (d,d) of (N,d) activations, as numpy float64."""
    a = act.detach().to(torch.float64).cpu().numpy()
    return a.mean(axis=0), np.cov(a, rowvar=False)


def fid(act_x: torch.Tensor, act_y: torch.Tensor) -> float:
    """FID between two sets of (N,2048) Inception activations."""
    mu1, s1 = _mean_cov(act_x)
    mu2, s2 = _mean_cov(act_y)
    return float(calculate_frechet_distance(mu1, s1, mu2, s2))


def _weighted_mean_cov(act: torch.Tensor, w: torch.Tensor):
    """Weighted mean and unbiased weighted covariance of (N,d) activations.

    ``w`` (N,) >= 0, renormalized to sum 1. The covariance uses the reliability-
    weights correction 1 / (1 - sum w^2); for uniform w this is N/(N-1), so it
    matches _mean_cov (np.cov, ddof=1)."""
    a = act.detach().to(torch.float64).cpu().numpy()
    weights = w.detach().to(torch.float64).cpu().numpy().reshape(-1)
    weights = weights / weights.sum()
    mu = weights @ a
    dev = a - mu
    cov = (dev.T * weights) @ dev / (1.0 - (weights * weights).sum())
    return mu, cov


def weighted_fid(act_x: torch.Tensor, w_x: torch.Tensor, act_y: torch.Tensor) -> float:
    """FID of a weighted set ``act_x`` (importance weights ``w_x``) vs an unweighted
    ``act_y``, from closed-form weighted moments (no resampling/seed). Equals
    fid(act_x, act_y) when w_x is uniform."""
    mu1, s1 = _weighted_mean_cov(act_x, w_x)
    mu2, s2 = _mean_cov(act_y)
    return float(calculate_frechet_distance(mu1, s1, mu2, s2))


def _poly_kernel(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cubic polynomial kernel k(x,y) = (x^T y / d + 1)^3 (Binkowski et al. 2018)."""
    d = x.shape[1]
    return (x @ y.t() / d + 1.0) ** 3


def _kid_mmd2(x: torch.Tensor, y: torch.Tensor) -> float:
    """Unbiased MMD^2 with the cubic polynomial kernel (single subset)."""
    m, n = x.shape[0], y.shape[0]
    kxx, kyy, kxy = _poly_kernel(x, x), _poly_kernel(y, y), _poly_kernel(x, y)
    sum_xx = (kxx.sum() - kxx.diagonal().sum()) / (m * (m - 1))
    sum_yy = (kyy.sum() - kyy.diagonal().sum()) / (n * (n - 1))
    sum_xy = kxy.sum() / (m * n)
    return float((sum_xx + sum_yy - 2.0 * sum_xy).item())


def kid(act_x: torch.Tensor, act_y: torch.Tensor, subset_size: int = 1000, n_subsets: int = 100, seed: int = 0):
    """KID (mean, std) over n_subsets random subsets of size min(subset_size, n_x, n_y). Returns (None, None) if fewer than 2 usable samples."""
    nx, ny = act_x.shape[0], act_y.shape[0]
    s = int(min(subset_size, nx, ny))
    if s < 2:
        return None, None
    g = torch.Generator().manual_seed(int(seed))
    vals = []
    for _ in range(int(n_subsets)):
        ix = torch.randperm(nx, generator=g)[:s]
        iy = torch.randperm(ny, generator=g)[:s]
        vals.append(_kid_mmd2(act_x[ix], act_y[iy]))
    t = torch.tensor(vals, dtype=torch.float64)
    return float(t.mean().item()), float(t.std(unbiased=True).item())


def weighted_kid(act_x: torch.Tensor, w_x: torch.Tensor, act_y: torch.Tensor) -> float:
    """Unbiased weighted KID (poly-kernel MMD^2) of a weighted set ``act_x``
    (importance weights ``w_x``, renormalized to sum 1) vs an unweighted ``act_y``.

    Closed form over the full sets, no subsampling/seed. The weighted XX term drops
    the diagonal and normalizes by 1 - sum w^2 (same reliability factor as
    _weighted_mean_cov), so it matches the standard unbiased MMD^2 for uniform w_x."""
    x, y = act_x.double(), act_y.double()
    w = w_x.double().reshape(-1)
    w = w / w.sum()
    m = y.shape[0]
    kxx, kyy, kxy = _poly_kernel(x, x), _poly_kernel(y, y), _poly_kernel(x, y)
    xx = (w @ kxx @ w - (w * w * kxx.diagonal()).sum()) / (1.0 - (w * w).sum())  # weighted, no diagonal
    yy = (kyy.sum() - kyy.diagonal().sum()) / (m * (m - 1))  # standard unbiased
    xy = (w @ kxy).sum() / m  # weighted x, uniform y
    return float((xx + yy - 2.0 * xy).item())
