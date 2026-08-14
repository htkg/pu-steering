"""Reference-based avoidance: SafeDenoiser baseline and Ours (weak SafeDenoiser + online-PU SMC tilt)."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .diffusion import DDPMSchedule, clip01, ddpm_mu_from_score, tweedie_x0hat
from .pu import OnlinePURFF
from .smc_util import ess, solve_delta_for_ess


def safedenoiser_x0hat(ref: torch.Tensor, abar: float, x_t: torch.Tensor, x0hat: torch.Tensor, eta: float) -> torch.Tensor:
    """SafeDenoiser correction: push x0hat away from the reference (avoided) set,
    scaled by eta and the local reference mass beta(x_t)."""
    d = x_t.shape[-1]
    diff = x_t.unsqueeze(1) - (math.sqrt(abar) * ref).unsqueeze(0)
    log_qt = -0.5 * diff.pow(2).sum(-1) / (1.0 - abar)
    ref_denoiser = torch.softmax(log_qt, dim=-1) @ ref
    log_qt_full = log_qt - 0.5 * d * math.log(2.0 * math.pi * (1.0 - abar))
    beta_val = torch.exp(-math.log(ref.shape[0]) + torch.logsumexp(log_qt_full, dim=-1))
    return x0hat + eta * beta_val.unsqueeze(-1) * (x0hat - ref_denoiser)


@dataclass
class SafeDenoiser:
    schedule: DDPMSchedule
    score_fn: Callable[[torch.Tensor, int], torch.Tensor]  # callable(x_t, t) -> unconditional score
    ref: torch.Tensor  # (N, d) avoided reference samples
    M: int = 64
    eta: float = 1.0
    t_start: int = 100
    t_stop: int = 5
    seed: int = 0

    def sample(self) -> torch.Tensor:
        torch.manual_seed(self.seed)
        x_t = torch.randn(self.M, self.ref.shape[1])
        for t in range(self.schedule.T, 0, -1):
            abar, alpha, beta = self.schedule.abar_t(t), self.schedule.alpha(t), self.schedule.beta(t)
            with torch.no_grad():
                x0hat = tweedie_x0hat(x_t, abar, self.score_fn(x_t, t))
                if self.t_stop <= t <= self.t_start:
                    x0hat = safedenoiser_x0hat(self.ref, abar, x_t, x0hat, self.eta)
                eps = (x_t - math.sqrt(abar) * x0hat) / math.sqrt(1.0 - abar)
                mu = (1.0 / math.sqrt(alpha)) * (x_t - (beta / math.sqrt(1.0 - abar)) * eps)
            x_t = mu + math.sqrt(max(beta, 1e-12)) * torch.randn_like(x_t) if t > 1 else mu
        return x_t


@dataclass
class PUSMCRef:
    """Ours (reference setting): weak SafeDenoiser proposal + online-PU tilt + SMC."""

    schedule: DDPMSchedule
    score_fn: Callable[[torch.Tensor, int], torch.Tensor]  # callable(x_t, t) -> unconditional score
    ref: torch.Tensor  # (N, d) avoided reference samples = PU positives
    pu: OnlinePURFF
    M: int = 64
    pu_class_prior: float = 0.1
    ess_target: float = 0.9
    ess_resample: float = 0.9
    resample_stop: int = 2
    lam_max: float = 0.7
    sd_eta: float = 48.0  # weak SafeDenoiser proposal strength
    seed: int = 0

    def __post_init__(self):
        torch.manual_seed(self.seed)
        self.delta_max = 1.5 / self.schedule.T
        self.ess_min = self.ess_target * self.M

    def _x0hat(self, x_t, t):
        return tweedie_x0hat(x_t, self.schedule.abar_t(t), self.score_fn(x_t, t))

    def _resample(self, x, w_u, w_q):
        if ess(w_q) >= self.ess_resample * self.M:
            return x, w_u, w_q
        wn = w_q / (w_q.sum() + 1e-12)
        N = wn.shape[0]
        order = torch.argsort(wn, descending=True)  # systematic-sorted resampling
        cumsum = torch.cumsum(wn[order], dim=0)
        u = torch.rand(1, dtype=wn.dtype) / N + torch.arange(N, dtype=wn.dtype) / N
        idx = order[torch.searchsorted(cumsum, u).clamp(max=N - 1)]
        return x[idx], w_u[idx] / (w_q[idx] + 1e-12), torch.ones(self.M)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        x_t = torch.randn(self.M, self.ref.shape[1])
        w_u = torch.ones(self.M)
        w_q = torch.ones(self.M)
        lam_t = 0.0

        T = self.schedule.T
        with torch.no_grad():
            x0_u = self._x0hat(x_t, T).detach()
        self.pu.fit(self.ref, x0_u, w_u, pi=self.pu_class_prior)

        for t in range(T, 0, -1):
            abar, alpha, beta, sigma2 = (self.schedule.abar_t(t), self.schedule.alpha(t), self.schedule.beta(t), self.schedule.sigma2(t))
            with torch.no_grad():
                uncond_score = self.score_fn(x_t, t)
                x0_u_t = tweedie_x0hat(x_t, abar, uncond_score).detach()
                p_t = clip01(self.pu.predict_proba_pos(x0_u_t))
            delta = solve_delta_for_ess(w_q, 1.0 - p_t, self.ess_min, self.delta_max)
            lam_prev = float(min(lam_t + delta, self.lam_max))

            # SafeDenoiser proposal.
            x0_guided = safedenoiser_x0hat(self.ref, abar, x_t, x0_u_t, self.sd_eta)
            eps_g = (x_t - math.sqrt(abar) * x0_guided) / math.sqrt(1.0 - abar)
            mu_hat = (1.0 / math.sqrt(alpha)) * (x_t - (beta / math.sqrt(1.0 - abar)) * eps_g)
            mu = ddpm_mu_from_score(self.schedule, x_t, t, uncond_score)

            if sigma2 <= 0.0:
                x_prev = mu_hat
                ratio_p = torch.ones(self.M)
            else:
                x_prev = mu_hat + math.sqrt(sigma2) * torch.randn_like(x_t)
                dpr = (x_prev - mu_hat).pow(2).sum(-1)
                ratio_p = torch.exp(-(0.5 / sigma2) * ((x_prev - mu).pow(2).sum(-1) - dpr))
            w_u_prev = w_u * ratio_p

            t_prev = t - 1
            with torch.no_grad():
                x0_u_prev = x_prev.detach() if t_prev == 0 else tweedie_x0hat(x_prev, self.schedule.abar_t(t_prev), self.score_fn(x_prev, t_prev)).detach()
            self.pu.fit(self.ref, x0_u_prev, w_u_prev, pi=self.pu_class_prior)
            with torch.no_grad():
                one_minus_p_prev = 1.0 - clip01(self.pu.predict_proba_pos(x0_u_prev))

            w_q_prev = w_q * ratio_p * (clip01(one_minus_p_prev) ** lam_prev / clip01(1.0 - p_t) ** lam_t)

            if t > self.resample_stop:
                x_prev, w_u_prev, w_q_prev = self._resample(x_prev, w_u_prev, w_q_prev)
            x_t, w_u, w_q = x_prev, w_u_prev, w_q_prev
            lam_t = lam_prev
        return x_t, w_q
