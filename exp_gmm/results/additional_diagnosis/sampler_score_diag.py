"""Diagnostic copy of PUNegGuideDDPMSMC (Case 1) — dynamics identical to
sampler_score.PUNegGuideDDPMSMC; the only changes are in snapshot_callback,
which also receives w_c (the classifier positive-term weight) and gets the
denoised x0hat_u (not x_prev) as x0hat."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from gmm_ddpm import DDPMSchedule, GaussianMixture, convolve_gmm_forward, merge_gmm, safe_clip01
from kernel_model import KernelPULogistic
from tqdm import tqdm


@dataclass
class PUNegGuideDDPMSMCDiag:
    schedule: DDPMSchedule
    pos_x0_gmm: GaussianMixture
    neg_x0_gmm: GaussianMixture
    class_prior: float
    pu: KernelPULogistic

    M: int = 1024
    pu_class_prior: float = 0.05
    ess_target: float = 0.8
    resample_at: str = ""
    grad_clip: float = 1.0
    proposal: str = "grad"
    dng_lambda0: float = 1.0
    dng_tau: float = 1.0
    dng_lam_max: float = 20.0
    dng_pmin: float = 1e-2
    dng_delta: float = 0.0
    wq_ratio: bool = True
    lam_fixed: float = 0.0
    tilt_max: float = 1.0
    tilt_start: float = 1.0
    lam_mode: str = "adaptive"
    lam_end: float = 1.0
    lam_ramp: float = 1.0
    ess_resample: float = 0.0
    resample_stop: int = 0
    resample_cooldown: int = 0

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

        for t in tqdm(range(self.schedule.T, 0, -1), desc="PU-SMC (diag)"):
            beta = float(self.schedule.beta(t))
            sigma2 = self.schedule.sigma2(t)
            one_minus_p_t = torch.ones_like(w_q)
            x0hat_u_prev = x_t

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
                    x0hat=(x0hat_u_prev if need_pu else x_prev).detach().clone(),
                    pu_proba=p_prev.detach().clone(),
                    w_u=w_u_prev.detach().clone(),
                    w_c=w_c_prev.detach().clone(),
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
