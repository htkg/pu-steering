"""GMM-setting metrics: leak (avoided mass left) and W2 to the kept target."""

import numpy as np
import ot
import torch

from .diffusion import GaussianMixture, sample_gmm


def gmm_post(x: torch.Tensor, pos: GaussianMixture, neg: GaussianMixture, pi: float) -> torch.Tensor:
    """Oracle posterior p(c|x) = pi p(x|c) / (pi p(x|c) + (1-pi) p(x|not c))."""
    pp = torch.exp(pos.logpdf(x))
    pn = torch.exp(neg.logpdf(x))
    return pi * pp / (pi * pp + (1.0 - pi) * pn + 1e-12)


def gmm_w2(x: torch.Tensor, neg: GaussianMixture, w: torch.Tensor | None = None, n_ref: int = 3000, seed: int = 7) -> float:
    """W2 between (optionally weighted) samples x and samples of the kept GMM neg."""
    ref = sample_gmm(neg, n_ref, generator=torch.Generator().manual_seed(seed)).numpy().astype(np.float64)
    xn = x.detach().numpy().astype(np.float64)
    a = np.ones(n_ref) / n_ref
    b = (w / (w.sum() + 1e-12)).detach().numpy().astype(np.float64) if w is not None else np.ones(xn.shape[0]) / xn.shape[0]
    return float(np.sqrt(ot.emd2(a, b, ot.dist(ref, xn))))
