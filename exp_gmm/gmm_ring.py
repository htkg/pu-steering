"""Ring-GMM construction, evaluation metrics, and logging utilities."""

import math
import sys

import numpy as np
import ot
import torch
from gmm_ddpm import Gaussian, GaussianMixture, sample_gmm


def make_ring_gmm(seed: int = 0, n_comps: int = 12, n_pos: int = 3) -> tuple[GaussianMixture, GaussianMixture, float]:
    """Generate a ring GMM.

    Returns ``(pos_gmm, neg_gmm, class_prior)`` where ``pos_gmm`` is the
    undesired/avoided class, ``neg_gmm`` is the desired/kept distribution, and
    ``class_prior`` is the prior mass of the undesired class.
    """
    rng = np.random.RandomState(seed)
    radius = 10.0
    z_spread = 1.0
    radius_scatter = 0.3
    rotation_noise = 0.3

    angles = np.linspace(0, 2 * math.pi, n_comps, endpoint=False)
    angles += rng.uniform(-0.2, 0.2, n_comps)
    radii = radius * (1 + rng.uniform(-radius_scatter, radius_scatter, n_comps))

    means = np.zeros((n_comps, 3))
    means[:, 0] = radii * np.cos(angles)
    means[:, 1] = radii * np.sin(angles)
    means[:, 2] = rng.uniform(-z_spread, z_spread, n_comps)
    means -= means.mean(axis=0)

    covs = np.zeros((n_comps, 3, 3))
    for i in range(n_comps):
        radial = np.array([math.cos(angles[i]), math.sin(angles[i]), 0.0])
        tangent = np.array([-math.sin(angles[i]), math.cos(angles[i]), 0.0])
        z_dir = np.array([0.0, 0.0, 1.0])

        rot = rng.uniform(0, rotation_noise)
        ra = rng.uniform(0, 2 * math.pi)
        c, s = math.cos(ra * rot), math.sin(ra * rot)
        rn = c * radial + s * tangent
        tn = -s * radial + c * tangent
        rn += rng.randn(3) * 0.1
        rn /= np.linalg.norm(rn)
        tn = tn - tn.dot(rn) * rn
        tn /= np.linalg.norm(tn)
        z_dir = np.cross(rn, tn)

        R = np.column_stack([rn, tn, z_dir])
        lam_r = rng.uniform(1.0, 2.5)
        lam_t = rng.uniform(0.3, 0.8)
        lam_z = rng.uniform(0.5, 2.0)
        covs[i] = R @ np.diag([lam_r, lam_t, lam_z]) @ R.T
        covs[i] = (covs[i] + covs[i].T) / 2

    weights = rng.dirichlet(np.ones(n_comps) * 5)
    pos_idxs = set(rng.choice(n_comps, n_pos, replace=False).tolist())

    pos_comps, neg_comps = [], []
    pos_w, neg_w = [], []
    for i in range(n_comps):
        comp = Gaussian(
            mean=torch.tensor(means[i], dtype=torch.float32),
            cov=torch.tensor(covs[i], dtype=torch.float32),
        )
        if i in pos_idxs:
            pos_comps.append(comp)
            pos_w.append(weights[i])
        else:
            neg_comps.append(comp)
            neg_w.append(weights[i])

    pos_w = torch.tensor(pos_w, dtype=torch.float32)
    pos_w /= pos_w.sum()
    neg_w = torch.tensor(neg_w, dtype=torch.float32)
    neg_w /= neg_w.sum()

    return (
        GaussianMixture(weights=pos_w, comps=tuple(pos_comps)),
        GaussianMixture(weights=neg_w, comps=tuple(neg_comps)),
        n_pos / n_comps,
    )


def compute_post(x: torch.Tensor, pos: GaussianMixture, neg: GaussianMixture, pi: float) -> torch.Tensor:
    """Compute oracle posterior P(pos|x)."""
    pp = torch.exp(pos.logpdf(x))
    np_ = torch.exp(neg.logpdf(x))
    return pi * pp / (pi * pp + (1 - pi) * np_ + 1e-12)


def wasserstein(x: torch.Tensor, neg: GaussianMixture, w: torch.Tensor | None = None, generator: torch.Generator | None = None) -> float:
    """Wasserstein-2 distance between samples and neg GMM reference."""
    ref = sample_gmm(neg, 3000, generator=generator).numpy().astype(np.float64)
    xn = x.detach().numpy().astype(np.float64)
    a = np.ones(3000, dtype=np.float64) / 3000
    if w is not None:
        b = (w / (w.sum() + 1e-12)).detach().numpy().astype(np.float64)
    else:
        b = np.ones(xn.shape[0], dtype=np.float64) / xn.shape[0]
    return float(np.sqrt(ot.emd2(a, b, ot.dist(ref, xn))))  # type: ignore


class Logger:
    """Tee stdout to a file."""

    def __init__(self, path: str):
        self.f = open(path, "w")  # noqa
        self.out = sys.stdout

    def write(self, msg: str):
        self.out.write(msg)
        self.f.write(msg)
        self.out.flush()
        self.f.flush()

    def flush(self):
        self.out.flush()
        self.f.flush()

    def close(self):
        self.f.close()
