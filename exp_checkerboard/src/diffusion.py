"""Gaussian mixtures and a cosine DDPM schedule; a GMM's noised score is closed-form."""

import math
from dataclasses import dataclass

import torch


@dataclass
class Gaussian:
    mean: torch.Tensor  # (d,)
    cov: torch.Tensor  # (d, d)

    def logpdf(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        L = torch.linalg.cholesky(self.cov)
        y = torch.cholesky_solve((x - self.mean).unsqueeze(-1), L).squeeze(-1)
        quad = ((x - self.mean) * y).sum(dim=-1)
        logdet = 2.0 * torch.log(torch.diag(L)).sum()
        return -0.5 * (d * math.log(2.0 * math.pi) + logdet + quad)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        L = torch.linalg.cholesky(self.cov)
        return -torch.cholesky_solve((x - self.mean).unsqueeze(-1), L).squeeze(-1)


@dataclass
class GaussianMixture:
    weights: torch.Tensor  # (K,), sum=1
    comps: tuple[Gaussian, ...]

    def logpdf(self, x: torch.Tensor) -> torch.Tensor:
        logs = [torch.log(self.weights[k]) + g.logpdf(x) for k, g in enumerate(self.comps)]
        return torch.logsumexp(torch.stack(logs, dim=-1), dim=-1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        comp_log = torch.stack([torch.log(self.weights[k]) + g.logpdf(x) for k, g in enumerate(self.comps)], dim=-1)
        resp = torch.softmax(comp_log, dim=-1)  # responsibilities p(k|x)
        out = torch.zeros_like(x)
        for k, g in enumerate(self.comps):
            out = out + resp[:, k : k + 1] * g.score(x)
        return out


def merge_gmm(pos: GaussianMixture, neg: GaussianMixture, class_prior: float) -> GaussianMixture:
    """Unconditional GMM p(x) = pi*p(x|c) + (1-pi)*p(x|not c)."""
    weights = torch.cat([class_prior * pos.weights, (1.0 - class_prior) * neg.weights], dim=0)
    return GaussianMixture(weights / weights.sum(), pos.comps + neg.comps)


def convolve_forward(gmm: GaussianMixture, abar_t: float) -> GaussianMixture:
    """Forward-noised marginal p(x_t): mean *= sqrt(abar), cov = abar*cov + (1-abar)*I."""
    out = []
    for g in gmm.comps:
        d = g.mean.shape[0]
        out.append(Gaussian(math.sqrt(abar_t) * g.mean, abar_t * g.cov + (1.0 - abar_t) * torch.eye(d)))
    return GaussianMixture(gmm.weights, tuple(out))


@torch.no_grad()
def sample_gmm(gmm: GaussianMixture, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    ks = torch.multinomial(gmm.weights, n, replacement=True, generator=generator)
    xs = []
    for k, g in enumerate(gmm.comps):
        nk = int((ks == k).sum())
        if nk == 0:
            continue
        d = g.mean.shape[0]
        L = torch.linalg.cholesky(g.cov)
        z = torch.empty(nk, d).normal_(generator=generator)
        xs.append(g.mean + z @ L.T)
    return torch.cat(xs, dim=0)


def clip01(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.clamp(p, eps, 1.0 - eps)


@dataclass
class DDPMSchedule:
    """Cosine VP schedule; all getters are 1-indexed by diffusion step t in [1, T]."""

    T: int
    cosine_s: float = 0.008
    max_beta: float = 0.999

    def __post_init__(self):
        ts = torch.linspace(0.0, float(self.T), self.T + 1)
        f = (ts / self.T + self.cosine_s) / (1.0 + self.cosine_s)
        abar_full = torch.cos(0.5 * math.pi * f).pow(2)
        abar_full = abar_full / abar_full[0]
        betas = torch.clamp(1.0 - abar_full[1:] / abar_full[:-1], 1e-12, self.max_beta)
        self.alphas = 1.0 - betas
        self.abar = torch.cumprod(self.alphas, dim=0)
        abar_prev = torch.cat([torch.ones(1), self.abar[:-1]], dim=0)
        self.beta_tilde = betas * (1.0 - abar_prev) / (1.0 - self.abar)
        self.betas = betas

    def alpha(self, t: int) -> float:
        return float(self.alphas[t - 1])

    def beta(self, t: int) -> float:
        return float(self.betas[t - 1])

    def abar_t(self, t: int) -> float:
        return float(self.abar[t - 1])

    def sigma2(self, t: int) -> float:
        return float(self.beta_tilde[t - 1])


def tweedie_x0hat(x_t: torch.Tensor, abar_t: float, score: torch.Tensor) -> torch.Tensor:
    """Posterior-mean x0 estimate from a score: (x_t + (1-abar)*score) / sqrt(abar)."""
    return (x_t + (1.0 - abar_t) * score) / math.sqrt(abar_t)


def ddpm_mu_from_score(sched: DDPMSchedule, x_t: torch.Tensor, t: int, score: torch.Tensor) -> torch.Tensor:
    """DDPM posterior mean mu_{t-1} from an (unconditional or guided) score."""
    abar, alpha, beta = sched.abar_t(t), sched.alpha(t), sched.beta(t)
    eps = -math.sqrt(1.0 - abar) * score
    return (1.0 / math.sqrt(alpha)) * (x_t - (beta / math.sqrt(1.0 - abar)) * eps)


@torch.no_grad()
def oracle_sample(sched: DDPMSchedule, gmm: GaussianMixture, M: int, seed: int) -> torch.Tensor:
    """Ancestral DDPM sampling from a GMM using its exact (analytic) score."""
    torch.manual_seed(seed)
    d = gmm.comps[0].mean.shape[0]
    x = torch.randn(M, d)
    for t in range(sched.T, 0, -1):
        score = convolve_forward(gmm, sched.abar_t(t)).score(x)
        mu = ddpm_mu_from_score(sched, x, t, score)
        x = mu + math.sqrt(max(sched.beta(t), 1e-12)) * torch.randn_like(x) if t > 1 else mu
    return x
