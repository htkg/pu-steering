"""Score-based avoidance: DNG baseline and Ours (weak DNG + online-PU SMC tilt)."""

import math
from dataclasses import dataclass

import torch

from .diffusion import DDPMSchedule, GaussianMixture, clip01, convolve_forward, ddpm_mu_from_score, merge_gmm, tweedie_x0hat
from .pu import OnlinePURFF
from .smc_util import ess, solve_delta_for_ess


@dataclass
class DNG:
    """Dynamic Negative Guidance: per-particle scale lam_t = lambda0 * p_t/(1-p_t),
    with p_t a running Bayesian posterior estimate p(c|x_t)."""

    schedule: DDPMSchedule
    pos_x0_gmm: GaussianMixture  # p(x0|c), avoided
    neg_x0_gmm: GaussianMixture  # p(x0|not c)
    class_prior: float
    M: int = 64
    lambda0: float = 5.0
    tau: float = 0.2
    pmin: float = 1e-3
    lam_max: float = 20.0
    seed: int = 0

    def __post_init__(self):
        torch.manual_seed(self.seed)
        self.uncond = merge_gmm(self.pos_x0_gmm, self.neg_x0_gmm, self.class_prior)

    def _eps(self, gmm: GaussianMixture, x_t: torch.Tensor, t: int) -> torch.Tensor:
        abar = self.schedule.abar_t(t)
        return -math.sqrt(1.0 - abar) * convolve_forward(gmm, abar).score(x_t)

    def sample(self) -> torch.Tensor:
        d = self.pos_x0_gmm.comps[0].mean.shape[0]
        x_t = torch.randn(self.M, d)
        p_t = clip01(torch.full((self.M,), float(self.class_prior)), eps=self.pmin)

        for t in range(self.schedule.T, 0, -1):
            abar, alpha, beta = self.schedule.abar_t(t), self.schedule.alpha(t), self.schedule.beta(t)
            with torch.no_grad():
                eps_u = self._eps(self.uncond, x_t, t)
                eps_f = self._eps(self.pos_x0_gmm, x_t, t)
            lam = torch.clamp(self.lambda0 * (p_t / (1.0 - p_t)), 0.0, self.lam_max)
            eps_g = eps_u - lam[:, None] * (eps_f - eps_u)
            mu = (1.0 / math.sqrt(alpha)) * (x_t - (beta / math.sqrt(1.0 - abar)) * eps_g)
            x_prev = mu + math.sqrt(max(beta, 1e-12)) * torch.randn_like(x_t) if t > 1 else mu

            # Update the posterior from the realized step.
            b = max(beta, 1e-12)
            mu_f = (1.0 / math.sqrt(alpha)) * (x_t - (beta / math.sqrt(1.0 - abar)) * eps_f)
            mu_u = (1.0 / math.sqrt(alpha)) * (x_t - (beta / math.sqrt(1.0 - abar)) * eps_u)
            d_f = (x_prev - mu_f).pow(2).sum(-1)
            d_u = (x_prev - mu_u).pow(2).sum(-1)
            p_t = clip01(p_t * torch.exp(-(self.tau / (2.0 * b)) * (d_f - d_u)), eps=self.pmin)
            x_t = x_prev
        return x_t


@dataclass
class PUSMCScore:
    """Ours (score setting): weak DNG proposal + online-PU ESS-adaptive tilt + SMC."""

    schedule: DDPMSchedule
    pos_x0_gmm: GaussianMixture
    neg_x0_gmm: GaussianMixture
    class_prior: float  # p(c) of the unconditional p(x)
    pu: OnlinePURFF
    M: int = 64
    pi_min: float = 0.01  # PU prior schedule floor
    pi_max: float = 0.15
    pi_decay: float = 3.0
    ess_target: float = 0.9
    ess_resample: float = 0.9
    resample_stop: int = 2
    dng_lambda0: float = 0.6  # weak DNG proposal scale
    dng_lam_max: float = 4.0
    tilt_max: float = 1.0  # cap on the SMC tempering exponent lam
    seed: int = 0

    def __post_init__(self):
        torch.manual_seed(self.seed)
        self.uncond = merge_gmm(self.pos_x0_gmm, self.neg_x0_gmm, self.class_prior)
        self.delta_max = 3.0 * self.tilt_max / self.schedule.T
        self.ess_min = self.ess_target * self.M

    def _score(self, gmm: GaussianMixture, x_t: torch.Tensor, t: int) -> torch.Tensor:
        return convolve_forward(gmm, self.schedule.abar_t(t)).score(x_t)

    def _x0hat(self, x_t, t, score):
        return tweedie_x0hat(x_t, self.schedule.abar_t(t), score)

    def _mu(self, x_t, t, score):
        return ddpm_mu_from_score(self.schedule, x_t, t, score)

    def _pi_t(self, t: int) -> float:
        frac = t / self.schedule.T
        return self.pi_min + (self.pi_max - self.pi_min) * math.exp(-self.pi_decay * (1.0 - frac))

    def _resample(self, x, w_u, w_q, w_c):
        if ess(w_q) >= self.ess_resample * self.M:
            return x, w_u, w_q, w_c
        wn = w_q / (w_q.sum() + 1e-12)
        N = wn.shape[0]
        cumsum = torch.cumsum(wn, dim=0)
        u = (torch.arange(N, dtype=wn.dtype) + torch.rand(N, dtype=wn.dtype)) / N
        idx = torch.searchsorted(cumsum, u).clamp(max=N - 1)
        return x[idx], w_u[idx] / (w_q[idx] + 1e-12), torch.ones(self.M), w_c[idx] / (w_q[idx] + 1e-12)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.neg_x0_gmm.comps[0].mean.shape[0]
        x_t = torch.randn(self.M, d)
        w_u = torch.ones(self.M)
        w_q = torch.ones(self.M)
        w_c = torch.ones(self.M)
        lam_t = 0.0

        T = self.schedule.T
        with torch.no_grad():
            x0_u = self._x0hat(x_t, T, self._score(self.uncond, x_t, T)).detach()
            x0_pos = self._x0hat(x_t, T, self._score(self.pos_x0_gmm, x_t, T)).detach()
        self.pu.fit(x0_pos, x0_u, w_u, pi=self._pi_t(T), w_pos=w_c)

        for t in range(T, 0, -1):
            sigma2 = self.schedule.sigma2(t)
            with torch.no_grad():
                x0_u_t = self._x0hat(x_t, t, self._score(self.uncond, x_t, t)).detach()
                p_t = clip01(self.pu.predict_proba_pos(x0_u_t))
            delta = solve_delta_for_ess(w_q, 1.0 - p_t, self.ess_min, self.delta_max)
            lam_prev = float(min(lam_t + delta, self.tilt_max))

            # Weak DNG proposal.
            mu = self._mu(x_t, t, self._score(self.uncond, x_t, t))
            mu_c = self._mu(x_t, t, self._score(self.pos_x0_gmm, x_t, t))
            scale = torch.clamp(self.dng_lambda0 * (p_t / (1.0 - p_t)), 0.0, self.dng_lam_max)[:, None]
            mu_hat = mu + scale * (mu - mu_c)

            if sigma2 <= 0.0:
                x_prev = mu_hat
                ratio_p, ratio_c = torch.ones(self.M), torch.ones(self.M)
            else:
                x_prev = mu_hat + math.sqrt(sigma2) * torch.randn_like(x_t)
                dpr = (x_prev - mu_hat).pow(2).sum(-1)
                ratio_p = torch.exp(-(0.5 / sigma2) * ((x_prev - mu).pow(2).sum(-1) - dpr))
                ratio_c = torch.exp(-(0.5 / sigma2) * ((x_prev - mu_c).pow(2).sum(-1) - dpr))
            w_u_prev, w_c_prev = w_u * ratio_p, w_c * ratio_c

            # Refit the classifier and update the SMC weight.
            t_prev = t - 1
            with torch.no_grad():
                if t_prev == 0:
                    x0_u_prev = x_prev.detach()
                    x0_pos_prev = self._x0hat(x_prev, 1, self._score(self.pos_x0_gmm, x_prev, 1)).detach()
                else:
                    x0_u_prev = self._x0hat(x_prev, t_prev, self._score(self.uncond, x_prev, t_prev)).detach()
                    x0_pos_prev = self._x0hat(x_prev, t_prev, self._score(self.pos_x0_gmm, x_prev, t_prev)).detach()
            self.pu.fit(x0_pos_prev, x0_u_prev, w_u_prev, pi=self._pi_t(t_prev), w_pos=w_c_prev)
            with torch.no_grad():
                one_minus_p_prev = 1.0 - clip01(self.pu.predict_proba_pos(x0_u_prev))

            w_q_prev = w_q * ratio_p * (clip01(one_minus_p_prev) ** lam_prev / clip01(1.0 - p_t) ** lam_t)

            if t > self.resample_stop:
                x_prev, w_u_prev, w_q_prev, w_c_prev = self._resample(x_prev, w_u_prev, w_q_prev, w_c_prev)
            x_t, w_u, w_q, w_c = x_prev, w_u_prev, w_q_prev, w_c_prev
            lam_t = lam_prev
        return x_t, w_q
