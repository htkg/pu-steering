"""Samplers for the true-score negative-guidance experiment."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from gmm_ddpm import DDPMSchedule, GaussianMixture, convolve_gmm_forward, merge_gmm, safe_clip01
from kernel_model import KernelPULogistic
from tqdm import tqdm


@dataclass
class PUNegGuideDDPMSMC:
    """PU-SMC negative guidance.

    In this experiment, ``pos_x0_gmm`` is the undesired/avoided class ``c``
    and ``neg_x0_gmm`` is the desired/kept distribution ``not c``. The
    naming follows the PU-learning convention, not the generation goal.
    Requires the true positive score for PU training.
    """

    schedule: DDPMSchedule
    pos_x0_gmm: GaussianMixture  # undesired/avoided class c: p(x0 | c)
    neg_x0_gmm: GaussianMixture  # desired/kept class not c: p(x0 | not c)
    class_prior: float  # p(c), the prior mass of the undesired class
    pu: KernelPULogistic

    M: int = 1024
    pu_class_prior: float = 0.05  # PU prior pi
    ess_target: float = 0.8
    resample_at: str = ""  # comma-separated t values for scheduled resampling
    grad_clip: float = 1.0
    proposal: str = "grad"  # "grad" (classifier-gradient guidance) or "dng" (DNG guidance)
    dng_lambda0: float = 1.0
    dng_tau: float = 1.0
    dng_lam_max: float = 20.0
    dng_pmin: float = 1e-2
    dng_delta: float = 0.0
    wq_ratio: bool = True  # False: drop the proposal IS ratio from w_q
    lam_fixed: float = 0.0  # constant tempering exponent (0 = ESS-adaptive)
    tilt_max: float = 1.0  # cap on the tempering exponent lam
    tilt_start: float = 1.0  # fraction of T; grow lam only for t <= tilt_start*T
    lam_mode: str = "adaptive"  # "adaptive" | "linear" | "cos" (fixed schedules 0 -> lam_end)
    lam_end: float = 1.0  # terminal exponent for the fixed schedules
    lam_ramp: float = 1.0  # fraction of trajectory to reach lam_end (then plateau)
    ess_resample: float = 0.0  # >0: resample when ESS(w_q) < ess_resample*M
    resample_stop: int = 0  # no ESS-triggered resampling for t <= resample_stop
    resample_cooldown: int = 0  # min steps between ESS-triggered fires

    device: str = "cpu"
    seed: int = 0

    def __post_init__(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.uncond_x0_gmm = merge_gmm(self.pos_x0_gmm, self.neg_x0_gmm, self.class_prior)

        self.delta_max: float = 1.5 * self.tilt_max / self.schedule.T
        self.t_start_step: float = self.tilt_start * self.schedule.T
        self.ess_min: float = self.ess_target * float(self.M)
        self.resample_at_steps = {int(s) for s in self.resample_at.split(",") if s.strip()} if self.resample_at else set()
        self.resample_count: int = 0

    def _score_uncond(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        abar = self.schedule.abar_t(t)
        return convolve_gmm_forward(self.uncond_x0_gmm, abar).score(x_t)

    def _score_cond_c(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        abar = self.schedule.abar_t(t)
        return convolve_gmm_forward(self.pos_x0_gmm, abar).score(x_t)

    def _tweedie_x0hat(self, x_t: torch.Tensor, t: int, score: torch.Tensor) -> torch.Tensor:
        abar = self.schedule.abar_t(t)
        return (x_t + (1.0 - abar) * score) / math.sqrt(abar)

    def _ddpm_mu_from_score(self, x_t: torch.Tensor, t: int, score: torch.Tensor) -> torch.Tensor:
        alpha = self.schedule.alpha(t)
        beta = self.schedule.beta(t)
        abar = self.schedule.abar_t(t)
        eps = -math.sqrt(1.0 - abar) * score
        return (1.0 / math.sqrt(alpha)) * (x_t - (beta / math.sqrt(1.0 - abar)) * eps)

    def _ess(self, w: torch.Tensor) -> float:
        w = w.detach()
        s = float(w.sum().item())
        if s <= 0.0:
            return 0.0
        wn = w / (s + 1e-12)
        return float(1.0 / (wn.pow(2).sum().item() + 1e-12))

    def _resample(
        self, x_t: torch.Tensor, w_u: torch.Tensor, w_q: torch.Tensor, w_c: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.resample_count += 1
        wq = w_q.detach()
        wn = wq / (wq.sum() + 1e-12)

        N = wn.shape[0]
        cumsum = torch.cumsum(wn, dim=0)
        u = (torch.arange(N, device=wn.device, dtype=wn.dtype) + torch.rand(N, device=wn.device, dtype=wn.dtype)) / N
        idx = torch.searchsorted(cumsum, u).clamp(max=N - 1)

        return x_t[idx], w_u[idx] / (wq[idx] + 1e-12), torch.ones(self.M, device=x_t.device), w_c[idx] / (wq[idx] + 1e-12)

    def _solve_delta_for_ess(self, w_q, one_minus_p, target_ess, delta_max, max_iter=30):
        omp = safe_clip01(one_minus_p)

        def ess_of(delta):
            return self._ess(w_q * (omp**delta))

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

    def _lam_sched(self, t: int) -> float:
        s = (self.schedule.T - t + 1) / self.schedule.T
        s = min(1.0, s / max(self.lam_ramp, 1e-9))
        if self.lam_mode == "linear":
            return float(self.lam_end) * s
        return float(self.lam_end) * 0.5 * (1.0 - math.cos(math.pi * s))

    def sample(self, snapshot_callback=None) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.neg_x0_gmm.comps[0].mean.shape[0]
        x_t = torch.randn(self.M, d, device=self.device)
        w_u = torch.ones(self.M, device=self.device)
        w_q = torch.ones(self.M, device=self.device)
        w_c = torch.ones(self.M, device=self.device)
        lam_t = 0.0

        dng_p = safe_clip01(torch.full((self.M,), float(self.class_prior), device=self.device), eps=self.dng_pmin)
        need_pu = self.proposal == "grad" or self.tilt_max > 0.0 or self.lam_fixed > 0.0 or self.lam_mode in ("linear", "cos")
        last_fire_t = None

        t_init = self.schedule.T
        if need_pu:
            with torch.no_grad():
                x0hat_u = self._tweedie_x0hat(x_t, t_init, self._score_uncond(x_t, t_init)).detach()
                x0hat_pos = self._tweedie_x0hat(x_t, t_init, self._score_cond_c(x_t, t_init)).detach()
            self.pu.fit(x0hat_pos, x0hat_u, w_u, pi=self.pu_class_prior, w_pos=w_c)

        for t in tqdm(range(self.schedule.T, 0, -1), desc="PU-SMC (score)"):
            beta = float(self.schedule.beta(t))
            sigma2 = self.schedule.sigma2(t)
            one_minus_p_t = torch.ones_like(w_q)

            if need_pu:
                with torch.no_grad():
                    x0hat_u_t = self._tweedie_x0hat(x_t, t, self._score_uncond(x_t, t)).detach()
                    p_t = safe_clip01(self.pu.predict_proba_pos(x0hat_u_t))
                one_minus_p_t = 1.0 - p_t
            if self.lam_mode in ("linear", "cos"):
                lam_prev = self._lam_sched(t)
            elif self.lam_fixed > 0.0:
                lam_prev = float(self.lam_fixed)
            elif self.tilt_max > 0.0 and t <= self.t_start_step:
                delta = self._solve_delta_for_ess(w_q, one_minus_p_t, self.ess_min, self.delta_max)
                lam_prev = float(min(lam_t + delta, self.tilt_max))
            else:
                lam_prev = float(min(lam_t, self.tilt_max))

            uncond_score = self._score_uncond(x_t, t)
            mu = self._ddpm_mu_from_score(x_t, t, uncond_score)
            cond_score = self._score_cond_c(x_t, t)
            mu_c = self._ddpm_mu_from_score(x_t, t, cond_score)

            if self.proposal == "dng":
                lam_dng = torch.clamp(self.dng_lambda0 * (dng_p / (1.0 - dng_p)), 0.0, self.dng_lam_max)
                mu_hat = mu + lam_dng.unsqueeze(-1) * (mu - mu_c)
                var, add_noise = beta, t > 1
            else:
                x_req = x_t.detach().clone().requires_grad_(True)
                x0hat_req = self._tweedie_x0hat(x_req, t, self._score_uncond(x_req, t))
                p_req = safe_clip01(self.pu.predict_proba_pos(x0hat_req))
                grad_log_1mp = torch.autograd.grad(torch.log(1.0 - p_req).sum(), x_req)[0].detach()
                gnorm = grad_log_1mp.norm(dim=-1, keepdim=True)
                grad_log_1mp = grad_log_1mp * torch.clamp(self.grad_clip / (gnorm + 1e-12), max=1.0)
                grad_log_1mp = torch.nan_to_num(grad_log_1mp, nan=0.0, posinf=0.0, neginf=0.0)
                mu_hat = mu + (sigma2 * lam_prev) * grad_log_1mp
                var, add_noise = sigma2, sigma2 > 0.0

            if not add_noise:
                x_prev = mu_hat
                ratio_p_over_q = torch.ones(self.M, device=self.device)
                ratio_c_over_q = torch.ones(self.M, device=self.device)
            else:
                x_prev = mu_hat + math.sqrt(var) * torch.randn_like(x_t)
                dpr = (x_prev - mu_hat).pow(2).sum(dim=-1)
                ratio_p_over_q = torch.exp(-(0.5 / var) * ((x_prev - mu).pow(2).sum(dim=-1) - dpr))
                ratio_c_over_q = torch.exp(-(0.5 / var) * ((x_prev - mu_c).pow(2).sum(dim=-1) - dpr))
            w_u_prev = w_u * ratio_p_over_q
            w_c_prev = w_c * ratio_c_over_q

            if self.proposal == "dng":
                b = max(beta, 1e-12)
                d_c = (x_prev - mu_c).pow(2).sum(dim=-1)
                d_u = (x_prev - mu).pow(2).sum(dim=-1)
                dng_p = safe_clip01(dng_p * torch.exp(-(self.dng_tau / (2.0 * b)) * (d_c - d_u) + (self.dng_delta / (2.0 * b))), eps=self.dng_pmin)

            t_prev = t - 1
            if need_pu:
                with torch.no_grad():
                    if t_prev == 0:
                        x0hat_u_prev = x_prev.detach()
                        x0hat_pos_prev = self._tweedie_x0hat(x_prev, 1, self._score_cond_c(x_prev, 1)).detach()
                    else:
                        x0hat_u_prev = self._tweedie_x0hat(x_prev, t_prev, self._score_uncond(x_prev, t_prev)).detach()
                        x0hat_pos_prev = self._tweedie_x0hat(x_prev, t_prev, self._score_cond_c(x_prev, t_prev)).detach()
                self.pu.fit(x0hat_pos_prev, x0hat_u_prev, w_u_prev, pi=self.pu_class_prior, w_pos=w_c_prev)
                with torch.no_grad():
                    p_prev = safe_clip01(self.pu.predict_proba_pos(x0hat_u_prev))
                ratio_q = ratio_p_over_q if self.wq_ratio else torch.ones(self.M, device=self.device)
                w_q_prev = w_q * ratio_q * ((safe_clip01(1.0 - p_prev) ** lam_prev) / (safe_clip01(one_minus_p_t) ** lam_t))
            else:
                p_prev = dng_p
                w_q_prev = w_q * (ratio_p_over_q if self.wq_ratio else 1.0)

            if snapshot_callback is not None:
                snapshot_callback(
                    t=t - 1,
                    x_t=x_prev.detach().clone(),
                    x0hat=x_prev.detach().clone(),
                    pu_proba=p_prev.detach().clone(),
                    w_u=w_u_prev.detach().clone(),
                    w_q=w_q_prev.detach().clone(),
                    lam_t=lam_prev,
                )

            fire = t in self.resample_at_steps
            if not fire and self.ess_resample > 0.0 and t > self.resample_stop and (last_fire_t is None or last_fire_t - t >= self.resample_cooldown):
                fire = self._ess(w_q_prev) < self.ess_resample * float(self.M)
                if fire:
                    last_fire_t = t
            if fire:
                x_prev, w_u_prev, w_q_prev, w_c_prev = self._resample(x_prev, w_u_prev, w_q_prev, w_c_prev)

            x_t, w_u, w_q, w_c = x_prev, w_u_prev, w_q_prev, w_c_prev
            lam_t = lam_prev

        return x_t, w_q


@dataclass
class DNGDDPM:
    """Dynamic Negative Guidance. Requires true positive score."""

    schedule: DDPMSchedule
    pos_x0_gmm: GaussianMixture  # p(x0|c) undesired
    neg_x0_gmm: GaussianMixture  # p(x0|not c)
    class_prior: float

    M: int = 1024
    lambda0: float = 5.0
    tau: float = 0.2
    delta: float = 0.0
    pmin: float = 1e-6
    lam_max: float = 20.0

    device: str = "cpu"
    seed: int = 0

    def __post_init__(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.uncond_x0_gmm = merge_gmm(self.pos_x0_gmm, self.neg_x0_gmm, self.class_prior)

    def _score_uncond(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        abar = self.schedule.abar_t(t)
        return convolve_gmm_forward(self.uncond_x0_gmm, abar).score(x_t)

    def _score_forget(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        abar = self.schedule.abar_t(t)
        return convolve_gmm_forward(self.pos_x0_gmm, abar).score(x_t)

    def _eps_from_score(self, t: int, score: torch.Tensor) -> torch.Tensor:
        abar = self.schedule.abar_t(t)
        return -math.sqrt(1.0 - abar) * score

    def _mu_from_eps(self, x_t: torch.Tensor, t: int, eps: torch.Tensor) -> torch.Tensor:
        alpha = self.schedule.alpha(t)
        beta = self.schedule.beta(t)
        abar = self.schedule.abar_t(t)
        return (1.0 / math.sqrt(alpha)) * (x_t - (beta / math.sqrt(1.0 - abar)) * eps)

    def _compute_posterior(self, p_prev, x_prev, x_t, eps_u, eps_f, t):
        alpha = self.schedule.alpha(t)
        abar = self.schedule.abar_t(t)
        beta = max(float(self.schedule.beta(t)), 1e-12)

        mu_f = (1.0 / math.sqrt(alpha)) * (x_t - (beta / math.sqrt(1.0 - abar)) * eps_f)
        mu_u = (1.0 / math.sqrt(alpha)) * (x_t - (beta / math.sqrt(1.0 - abar)) * eps_u)

        d_f = (x_prev - mu_f).pow(2).sum(dim=-1)
        d_u = (x_prev - mu_u).pow(2).sum(dim=-1)

        log_mult = (-(float(self.tau) / (2.0 * beta)) * (d_f - d_u)) + (float(self.delta) / (2.0 * beta))
        p_new = p_prev * torch.exp(log_mult)
        return safe_clip01(p_new, eps=self.pmin)

    def sample(self) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.pos_x0_gmm.comps[0].mean.shape[0]
        x_t = torch.randn(self.M, d, device=self.device)

        p_t = torch.full((self.M,), float(self.class_prior), device=self.device)
        p_t = safe_clip01(p_t, eps=self.pmin)

        for t in tqdm(range(self.schedule.T, 0, -1), desc="DNG"):
            with torch.no_grad():
                score_u = self._score_uncond(x_t, t)
                score_f = self._score_forget(x_t, t)
                eps_u = self._eps_from_score(t, score_u)
                eps_f = self._eps_from_score(t, score_f)

            lam_t = float(self.lambda0) * (p_t / (1.0 - p_t))
            lam_t = torch.clamp(lam_t, min=0.0, max=self.lam_max)
            eps_guid = eps_u - lam_t[:, None] * (eps_f - eps_u)
            mu_guid = self._mu_from_eps(x_t, t, eps_guid)

            beta = float(self.schedule.beta(t))
            if t > 1:
                x_prev = mu_guid + math.sqrt(max(beta, 1e-12)) * torch.randn_like(x_t)
            else:
                x_prev = mu_guid

            with torch.no_grad():
                p_prev = self._compute_posterior(p_t, x_prev, x_t, eps_u, eps_f, t)

            x_t = x_prev
            p_t = p_prev

        return x_t, p_t


@dataclass
class NegCFG:
    """Negative Classifier-Free Guidance with fixed guidance scale."""

    schedule: DDPMSchedule
    pos_x0_gmm: GaussianMixture  # p(x0|c) undesired
    neg_x0_gmm: GaussianMixture  # p(x0|not c)
    class_prior: float

    M: int = 1024
    lam: float = 5.0

    device: str = "cpu"
    seed: int = 0

    def __post_init__(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.uncond_x0_gmm = merge_gmm(self.pos_x0_gmm, self.neg_x0_gmm, self.class_prior)

    def _score_uncond(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        abar = self.schedule.abar_t(t)
        return convolve_gmm_forward(self.uncond_x0_gmm, abar).score(x_t)

    def _score_forget(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        abar = self.schedule.abar_t(t)
        return convolve_gmm_forward(self.pos_x0_gmm, abar).score(x_t)

    def sample(self) -> torch.Tensor:
        d = self.pos_x0_gmm.comps[0].mean.shape[0]
        x_t = torch.randn(self.M, d, device=self.device)

        for t in tqdm(range(self.schedule.T, 0, -1), desc="NegCFG"):
            with torch.no_grad():
                score_u = self._score_uncond(x_t, t)
                score_f = self._score_forget(x_t, t)
                abar = self.schedule.abar_t(t)
                eps_u = -math.sqrt(1.0 - abar) * score_u
                eps_f = -math.sqrt(1.0 - abar) * score_f

            eps_guid = eps_u - self.lam * (eps_f - eps_u)
            alpha = self.schedule.alpha(t)
            beta = self.schedule.beta(t)
            mu = (1.0 / math.sqrt(alpha)) * (x_t - (beta / math.sqrt(1.0 - abar)) * eps_guid)

            if t > 1:
                x_t = mu + math.sqrt(max(float(beta), 1e-12)) * torch.randn_like(x_t)
            else:
                x_t = mu

        return x_t
