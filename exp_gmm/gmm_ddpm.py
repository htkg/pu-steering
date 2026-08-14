"""GMM distributions, VP-DDPM schedule, and sampling utilities."""

import math
from dataclasses import dataclass

import torch


@dataclass
class Gaussian:
    mean: torch.Tensor  # [d]
    cov: torch.Tensor  # [d,d]

    def logpdf(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        L = torch.linalg.cholesky(self.cov)
        y = torch.cholesky_solve((x - self.mean).unsqueeze(-1), L).squeeze(-1)
        quad = ((x - self.mean) * y).sum(dim=-1)
        logdet = 2.0 * torch.log(torch.diag(L)).sum()
        return -0.5 * (d * math.log(2.0 * math.pi) + logdet + quad)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        L = torch.linalg.cholesky(self.cov)
        y = torch.cholesky_solve((x - self.mean).unsqueeze(-1), L).squeeze(-1)
        return -y


@dataclass
class GaussianMixture:
    weights: torch.Tensor  # [K], sum=1
    comps: tuple[Gaussian, ...]

    def logpdf(self, x: torch.Tensor) -> torch.Tensor:
        logs = []
        for k, g in enumerate(self.comps):
            logs.append(torch.log(self.weights[k]) + g.logpdf(x))
        return torch.logsumexp(torch.stack(logs, dim=-1), dim=-1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        comp_log = []
        comp_score = []
        for k, g in enumerate(self.comps):
            comp_log.append(torch.log(self.weights[k]) + g.logpdf(x))
            comp_score.append(g.score(x))
        comp_log = torch.stack(comp_log, dim=-1)
        logZ = torch.logsumexp(comp_log, dim=-1)
        resp = torch.exp(comp_log - logZ.unsqueeze(-1))
        score = torch.zeros_like(comp_score[0])
        for k in range(resp.shape[-1]):
            score = score + resp[:, k : k + 1] * comp_score[k]
        return score


def merge_gmm(pos: GaussianMixture, neg: GaussianMixture, class_prior: float) -> GaussianMixture:
    """Merge pos/neg GMMs into an unconditional GMM weighted by class prior."""
    comps = pos.comps + neg.comps
    weights = torch.cat(
        [class_prior * pos.weights, (1.0 - class_prior) * neg.weights],
        dim=0,
    )
    weights = weights / weights.sum()
    return GaussianMixture(weights, comps)


@dataclass
class DDPMSchedule:
    T: int
    cosine_s: float = 0.008
    max_beta: float = 0.999
    device: str = "cpu"

    def __post_init__(self):
        ts = torch.linspace(0.0, float(self.T), self.T + 1, device=self.device)
        f = (ts / float(self.T) + self.cosine_s) / (1.0 + self.cosine_s)
        abar_full = torch.cos(0.5 * math.pi * f).pow(2)
        abar_full = abar_full / abar_full[0]

        abar_prev = abar_full[:-1]
        abar = abar_full[1:]

        betas = 1.0 - (abar / abar_prev)
        betas = torch.clamp(betas, min=1e-12, max=float(self.max_beta))

        alphas = 1.0 - betas
        abar_check = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.abar = abar_check

        abar_prev2 = torch.cat([torch.ones(1, device=self.device), self.abar[:-1]], dim=0)
        self.beta_tilde = betas * (1.0 - abar_prev2) / (1.0 - self.abar)

    def alpha(self, t: int) -> float:
        return float(self.alphas[t - 1].item())

    def beta(self, t: int) -> float:
        return float(self.betas[t - 1].item())

    def abar_t(self, t: int) -> float:
        return float(self.abar[t - 1].item())

    def abar_prev(self, t: int) -> float:
        if t == 1:
            return 1.0
        return float(self.abar[t - 2].item())

    def sigma2(self, t: int) -> float:
        return float(self.beta_tilde[t - 1].item())


def convolve_gaussian_forward(g: Gaussian, abar_t: float) -> Gaussian:
    d = g.mean.shape[0]
    mean_t = math.sqrt(abar_t) * g.mean
    cov_t = abar_t * g.cov + (1.0 - abar_t) * torch.eye(d, device=g.mean.device)
    return Gaussian(mean_t, cov_t)


def convolve_gmm_forward(gmm: GaussianMixture, abar_t: float) -> GaussianMixture:
    comps_t = tuple(convolve_gaussian_forward(g, abar_t) for g in gmm.comps)
    return GaussianMixture(gmm.weights, comps_t)


@torch.no_grad()
def sample_gmm(gmm: GaussianMixture, n: int, generator: torch.Generator | None = None) -> torch.Tensor:
    xs: list[torch.Tensor] = []
    ks: torch.Tensor = torch.multinomial(gmm.weights, n, replacement=True, generator=generator)
    K = len(gmm.comps)
    for k in range(K):
        mask: torch.Tensor = ks == k
        nk: int = int(mask.sum().item())
        if nk == 0:
            continue
        g = gmm.comps[k]
        d = g.mean.shape[0]
        L = torch.linalg.cholesky(g.cov)
        z = torch.empty(nk, d, device=g.mean.device).normal_(generator=generator)
        xs.append(g.mean + z @ L.T)
    return torch.cat(xs, dim=0)


def safe_clip01(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.clamp(p, eps, 1.0 - eps)


@dataclass
class OracleDiffusion:
    """Sample from p(x|not c) using the true negative score."""

    schedule: DDPMSchedule
    neg_x0_gmm: GaussianMixture
    M: int = 1024
    seed: int = 0

    def sample(self) -> torch.Tensor:
        torch.manual_seed(self.seed)
        d = self.neg_x0_gmm.comps[0].mean.shape[0]
        x_t = torch.randn(self.M, d)
        for t in range(self.schedule.T, 0, -1):
            abar = self.schedule.abar_t(t)
            alpha = self.schedule.alpha(t)
            beta = self.schedule.beta(t)
            with torch.no_grad():
                score = convolve_gmm_forward(self.neg_x0_gmm, abar).score(x_t)
            eps = -math.sqrt(1.0 - abar) * score
            mu = (1.0 / math.sqrt(alpha)) * (x_t - (beta / math.sqrt(1.0 - abar)) * eps)
            x_t = mu + math.sqrt(max(beta, 1e-12)) * torch.randn_like(x_t) if t > 1 else mu
        return x_t
