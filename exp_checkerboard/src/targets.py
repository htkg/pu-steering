"""Toy distributions (ring, MoG-40, checkerboard) and the avoided/kept class split."""

import math

import numpy as np
import torch

from .diffusion import Gaussian, GaussianMixture


def _iso(mean: torch.Tensor, std: float) -> Gaussian:
    return Gaussian(mean=mean, cov=(std**2) * torch.eye(mean.shape[0]))


def make_ring(n_comps: int = 10, radius: float = 6.0, std: float = 0.85) -> GaussianMixture:
    ang = torch.linspace(0, 2 * math.pi, n_comps + 1)[:-1]
    comps = [_iso(torch.tensor([radius * math.cos(a), radius * math.sin(a)]), std) for a in ang]
    return GaussianMixture(torch.ones(n_comps) / n_comps, tuple(comps))


def make_mog40(seed: int = 0, n_comps: int = 40, box: float = 40.0, std: float = 1.0) -> GaussianMixture:
    means = np.random.RandomState(seed).uniform(-box, box, size=(n_comps, 2)).astype(np.float32)
    comps = [_iso(torch.tensor(means[i]), std) for i in range(n_comps)]
    return GaussianMixture(torch.ones(n_comps) / n_comps, tuple(comps))


def sample_checkerboard(n: int, scale: float = 4.0, seed: int | None = None) -> torch.Tensor:
    rng = np.random.RandomState(seed)
    x1 = rng.rand(n) * 4 - 2
    x2 = rng.rand(n) - rng.randint(0, 2, n) * 2 + (np.floor(x1) % 2)
    return torch.tensor(np.stack([x1, x2], 1) * (scale / 2.0), dtype=torch.float32)


def split_gmm(gmm: GaussianMixture, avoid_idx) -> tuple[GaussianMixture, GaussianMixture, float]:
    """Split a base GMM into (pos = avoided, neg = kept, pi = p(avoided))."""
    avoid = {int(i) for i in avoid_idx}
    pos_w, neg_w, pos_c, neg_c = [], [], [], []
    for k, g in enumerate(gmm.comps):
        (pos_w if k in avoid else neg_w).append(gmm.weights[k])
        (pos_c if k in avoid else neg_c).append(g)
    pos_w, neg_w = torch.stack(pos_w), torch.stack(neg_w)
    pi = float(pos_w.sum())
    pos = GaussianMixture(pos_w / pos_w.sum(), tuple(pos_c))
    neg = GaussianMixture(neg_w / neg_w.sum(), tuple(neg_c))
    return pos, neg, pi


def ring_avoid_idx() -> list[int]:
    """Indices of three adjacent ring modes to avoid."""
    return [0, 1, 2]


def mog40_avoid_idx(gmm: GaussianMixture, k: int = 6) -> list[int]:
    """The k modes nearest to the mode closest to the cloud centroid (a local cluster)."""
    means = torch.stack([g.mean for g in gmm.comps])
    anchor = means[(means - means.mean(0)).pow(2).sum(-1).argmin()]
    return (means - anchor).pow(2).sum(-1).argsort()[:k].tolist()
