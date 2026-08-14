"""Samplers for the reference-based negative-guidance experiment."""

import math
from dataclasses import dataclass

import numpy as np
import torch
from gmm_ddpm import DDPMSchedule, GaussianMixture, convolve_gmm_forward, merge_gmm, safe_clip01, sample_gmm
from kernel_model import KernelPULogistic


@dataclass
class SafeDenoiserDDPM:
    """SafeDenoiser baseline.

    E_safe[x|xt] = E_data[x|xt] + eta * beta(xt) * (E_data[x|xt] - E_unsafe[x|xt])
    """

    schedule: DDPMSchedule
    pos_x0_gmm: GaussianMixture
    neg_x0_gmm: GaussianMixture
    class_prior: float

    M: int = 1024
    N_ref: int = 512
    eta: float = 1.0
    t_start: int = 100
    t_stop: int = 5

    device: str = "cpu"
    seed: int = 0

    def __post_init__(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.uncond_x0_gmm = merge_gmm(self.pos_x0_gmm, self.neg_x0_gmm, self.class_prior)
        self.x_unsafe = sample_gmm(self.pos_x0_gmm, self.N_ref).detach()

    def _score_uncond(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        abar = self.schedule.abar_t(t)
        return convolve_gmm_forward(self.uncond_x0_gmm, abar).score(x_t)

    def _tweedie_x0hat(self, x_t: torch.Tensor, t: int, score: torch.Tensor) -> torch.Tensor:
        abar = self.schedule.abar_t(t)
        return (x_t + (1.0 - abar) * score) / math.sqrt(abar)

    def _compute_safe_x0hat(self, x_t: torch.Tensor, x0hat: torch.Tensor, t: int) -> torch.Tensor:
        abar = self.schedule.abar_t(t)
        d = x_t.shape[-1]

        x_unsafe_scaled = math.sqrt(abar) * self.x_unsafe
        diff = x_t.unsqueeze(1) - x_unsafe_scaled.unsqueeze(0)
        log_qt = -0.5 * diff.pow(2).sum(dim=-1) / (1.0 - abar)

        weights_unsafe = torch.softmax(log_qt, dim=-1)
        x_unsafe_denoiser = torch.einsum("mn,nd->md", weights_unsafe, self.x_unsafe)

        log_qt_full = log_qt - 0.5 * d * math.log(2.0 * math.pi * (1.0 - abar))
        log_beta = -math.log(self.N_ref) + torch.logsumexp(log_qt_full, dim=-1)
        beta_val = torch.exp(log_beta)

        correction = self.eta * beta_val.unsqueeze(-1) * (x0hat - x_unsafe_denoiser)
        return x0hat + correction

    def sample(self) -> torch.Tensor:
        torch.manual_seed(self.seed)
        d = self.neg_x0_gmm.comps[0].mean.shape[0]
        x_t = torch.randn(self.M, d, device=self.device)

        for t in range(self.schedule.T, 0, -1):
            abar = self.schedule.abar_t(t)
            alpha = self.schedule.alpha(t)
            beta = self.schedule.beta(t)

            with torch.no_grad():
                score = self._score_uncond(x_t, t)
                x0hat = self._tweedie_x0hat(x_t, t, score)

                if self.t_stop <= t <= self.t_start:
                    x0hat = self._compute_safe_x0hat(x_t, x0hat, t)

                eps = (x_t - math.sqrt(abar) * x0hat) / math.sqrt(1.0 - abar)
                mu = (1.0 / math.sqrt(alpha)) * (x_t - (beta / math.sqrt(1.0 - abar)) * eps)

            x_t = mu + math.sqrt(max(beta, 1e-12)) * torch.randn_like(x_t) if t > 1 else mu

        return x_t


@dataclass
class PUNegGuideDDPMSMC:
    """PU-SMC negative guidance using reference particles.

    Gradient guidance in x0hat space (not x_t space). Positive training data
    for PU classifier is fixed reference samples (pos_mode="direct"). In this
    experiment, ``pos_x0_gmm`` is the undesired/avoided class and
    ``neg_x0_gmm`` is the desired/kept distribution.
    """

    schedule: DDPMSchedule
    pos_x0_gmm: GaussianMixture  # undesired/avoided class c
    neg_x0_gmm: GaussianMixture  # desired/kept class not c
    class_prior: float  # p(c), the prior mass of the undesired class

    pu: KernelPULogistic
    N_pos: int = 512
    M: int = 1024

    pu_class_prior: float = 0.05
    ess_target: float = 0.9
    ess_resample: float = 0.9
    resample_method: str = "stratified"  # "stratified" or "systematic_sorted"
    resample_stop: int = 100  # set to T to disable resampling
    guidance_stop: int = 0  # stop guidance for t <= this value

    grad_clip: float | None = 3.0
    lam_max: float = 1.0
    lam_schedule: str = "adaptive"  # "adaptive" or "cos"

    device: str = "cpu"
    seed: int = 0

    def __post_init__(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.uncond_x0_gmm = merge_gmm(self.pos_x0_gmm, self.neg_x0_gmm, self.class_prior)

        self.x0_pos = sample_gmm(self.pos_x0_gmm, self.N_pos).detach()

        self.delta_max: float = 1 / self.schedule.T * 1.5
        self.ess_min: float = self.ess_target * float(self.M)
        self.resample_count: int = 0

    def _score_uncond(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        abar = self.schedule.abar_t(t)
        return convolve_gmm_forward(self.uncond_x0_gmm, abar).score(x_t)

    def _tweedie_x0hat(self, x_t: torch.Tensor, t: int, score: torch.Tensor) -> torch.Tensor:
        abar = self.schedule.abar_t(t)
        return (x_t + (1.0 - abar) * score) / math.sqrt(abar)

    def _ddpm_mu_from_score(self, x_t: torch.Tensor, t: int, score: torch.Tensor) -> torch.Tensor:
        alpha = self.schedule.alpha(t)
        beta = self.schedule.beta(t)
        abar = self.schedule.abar_t(t)
        eps = -math.sqrt(1.0 - abar) * score
        return (1.0 / math.sqrt(alpha)) * (x_t - (beta / math.sqrt(1.0 - abar)) * eps)

    def _lam_at(self, t: int) -> float:
        frac = 1.0 - t / self.schedule.T
        return self.lam_max * 0.5 * (1.0 - math.cos(math.pi * frac))

    def _ess(self, w: torch.Tensor) -> float:
        w = w.detach()
        s = float(w.sum().item())
        if s <= 0.0:
            return 0.0
        wn = w / (s + 1e-12)
        return float(1.0 / (wn.pow(2).sum().item() + 1e-12))

    def _resample_idx(self, wn: torch.Tensor) -> torch.Tensor:
        N = wn.shape[0]

        if self.resample_method == "systematic_sorted":
            sorted_order = torch.argsort(wn, descending=True)
            sorted_weights = wn[sorted_order]
            cumsum = torch.cumsum(sorted_weights, dim=0)
            u0 = torch.rand(1, device=wn.device, dtype=wn.dtype) / N
            u = u0 + torch.arange(N, device=wn.device, dtype=wn.dtype) / N
            sorted_idx = torch.searchsorted(cumsum, u).clamp(max=N - 1)
            return sorted_order[sorted_idx]

        cumsum = torch.cumsum(wn, dim=0)
        u = (torch.arange(N, device=wn.device, dtype=wn.dtype) + torch.rand(N, device=wn.device, dtype=wn.dtype)) / N
        return torch.searchsorted(cumsum, u).clamp(max=N - 1)

    def _maybe_resample(
        self,
        x_t: torch.Tensor,
        w_u: torch.Tensor,
        w_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ess = self._ess(w_q)
        if ess >= self.ess_resample * float(self.M):
            return x_t, w_u, w_q

        self.resample_count += 1
        wq = w_q.detach()
        wn = wq / (wq.sum() + 1e-12)
        idx = self._resample_idx(wn)

        return x_t[idx], w_u[idx] / (wq[idx] + 1e-12), torch.ones(self.M, device=x_t.device)

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

    def sample(self, snapshot_callback=None) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.neg_x0_gmm.comps[0].mean.shape[0]
        x_t = torch.randn(self.M, d, device=self.device)
        w_u = torch.ones(self.M, device=self.device)
        w_q = torch.ones(self.M, device=self.device)
        if self.lam_schedule == "adaptive":
            lam_t = 0.0
        else:
            lam_t = self._lam_at(self.schedule.T)

        t_init = self.schedule.T
        with torch.no_grad():
            x0hat_u = self._tweedie_x0hat(x_t, t_init, self._score_uncond(x_t, t_init)).detach()
        self.pu.fit(self.x0_pos, x0hat_u, w_u, pi=float(self.pu_class_prior))

        for t in range(self.schedule.T, 0, -1):
            abar = self.schedule.abar_t(t)
            alpha = self.schedule.alpha(t)
            beta = self.schedule.beta(t)
            sigma2 = self.schedule.sigma2(t)

            with torch.no_grad():
                uncond_score = self._score_uncond(x_t, t)
                x0hat_u_t = self._tweedie_x0hat(x_t, t, uncond_score).detach()
                p_t = safe_clip01(self.pu.predict_proba_pos(x0hat_u_t))
                one_minus_p_t = 1.0 - p_t

            if self.lam_schedule == "adaptive":
                delta = self._solve_delta_for_ess(w_q, one_minus_p_t, self.ess_min, self.delta_max)
                lam_prev = float(min(lam_t + delta, self.lam_max))
            else:
                lam_prev = self._lam_at(t - 1)
            apply_guidance = t > self.guidance_stop
            if apply_guidance:
                x0hat_req = x0hat_u_t.detach().clone().requires_grad_(True)
                p_req = safe_clip01(self.pu.predict_proba_pos(x0hat_req))
                r_sum = torch.log(1.0 - p_req).sum()
                grad_x0hat = torch.autograd.grad(r_sum, x0hat_req)[0].detach()

                if self.grad_clip is not None:
                    gnorm = grad_x0hat.norm(dim=-1, keepdim=True)
                    grad_x0hat = grad_x0hat * torch.clamp(self.grad_clip / (gnorm + 1e-12), max=1.0)
                grad_x0hat = torch.nan_to_num(grad_x0hat, nan=0.0, posinf=0.0, neginf=0.0)

                x0hat_guided = x0hat_u_t + lam_prev * grad_x0hat
                eps_guided = (x_t - math.sqrt(abar) * x0hat_guided) / math.sqrt(1.0 - abar)
                mu_hat = (1.0 / math.sqrt(alpha)) * (x_t - (beta / math.sqrt(1.0 - abar)) * eps_guided)
                mu = self._ddpm_mu_from_score(x_t, t, uncond_score)
            else:
                mu = self._ddpm_mu_from_score(x_t, t, uncond_score)
                mu_hat = mu

            if sigma2 <= 0.0:
                x_prev = mu_hat
                ratio_p_over_q = torch.ones(self.M, device=self.device)
            else:
                z = torch.randn_like(x_t)
                x_prev = mu_hat + math.sqrt(sigma2) * z
                diff_un = x_prev - mu
                diff_pr = x_prev - mu_hat
                log_ratio = -(0.5 / sigma2) * (diff_un.pow(2).sum(dim=-1) - diff_pr.pow(2).sum(dim=-1))
                ratio_p_over_q = torch.exp(log_ratio)

            w_u_prev = w_u * ratio_p_over_q

            t_prev = t - 1
            with torch.no_grad():
                if t_prev == 0:
                    x0hat_u_prev = x_prev.detach()
                else:
                    x0hat_u_prev = self._tweedie_x0hat(x_prev, t_prev, self._score_uncond(x_prev, t_prev)).detach()
            self.pu.fit(self.x0_pos, x0hat_u_prev, w_u_prev, pi=float(self.pu_class_prior))

            with torch.no_grad():
                p_prev = safe_clip01(self.pu.predict_proba_pos(x0hat_u_prev))
                one_minus_p_prev = 1.0 - p_prev

            w_q_prev = w_q * ratio_p_over_q * ((safe_clip01(one_minus_p_prev) ** lam_prev) / (safe_clip01(one_minus_p_t) ** lam_t))

            if snapshot_callback is not None:
                snapshot_callback(
                    t=t - 1,
                    x_t=x_prev.detach().clone(),
                    x0hat=x0hat_u_prev.detach().clone(),
                    pu_proba=p_prev.detach().clone(),
                    w_u=w_u_prev.detach().clone(),
                    w_q=w_q_prev.detach().clone(),
                    lam_t=lam_prev,
                )

            if t > self.resample_stop:
                x_prev, w_u_prev, w_q_prev = self._maybe_resample(x_prev, w_u_prev, w_q_prev)

            x_t, w_u, w_q = x_prev, w_u_prev, w_q_prev
            lam_t = lam_prev

        return x_t, w_q


@dataclass
class SDPropPUSMC:
    """SafeDenoiser proposal + PU-SMC weight tracking.

    Uses SafeDenoiser as the proposal kernel and PU classifier for importance
    weight tracking. Lambda schedule is deterministic (not ESS-adaptive).
    """

    schedule: DDPMSchedule
    pos_x0_gmm: GaussianMixture
    neg_x0_gmm: GaussianMixture
    class_prior: float

    pu: KernelPULogistic
    N_pos: int = 512
    M: int = 1024

    eta: float = 50.0
    sd_t_stop: int = 0

    pu_class_prior: float = 0.05
    correction_clip: float = 0.0  # clip norm of SD correction (0=off)

    lam_schedule: str = "linear"  # "linear" or "cos"
    lam_target: float = 1.0

    ess_resample: float = 0.5
    resample_method: str = "stratified"  # "stratified" or "systematic_sorted"
    resample_stop: int = 100  # set to T to disable resampling

    device: str = "cpu"
    seed: int = 0

    def __post_init__(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.uncond_x0_gmm = merge_gmm(self.pos_x0_gmm, self.neg_x0_gmm, self.class_prior)

        self.x0_pos = sample_gmm(self.pos_x0_gmm, self.N_pos).detach()
        self.resample_count: int = 0

    def _lam_at(self, t: int) -> float:
        frac = 1.0 - t / self.schedule.T
        if self.lam_schedule == "cos":
            return self.lam_target * 0.5 * (1.0 - math.cos(math.pi * frac))
        return self.lam_target * frac

    def _score_uncond(self, x_t: torch.Tensor, t: int) -> torch.Tensor:
        abar = self.schedule.abar_t(t)
        return convolve_gmm_forward(self.uncond_x0_gmm, abar).score(x_t)

    def _tweedie_x0hat(self, x_t: torch.Tensor, t: int, score: torch.Tensor) -> torch.Tensor:
        abar = self.schedule.abar_t(t)
        return (x_t + (1.0 - abar) * score) / math.sqrt(abar)

    def _ddpm_mu_from_score(self, x_t: torch.Tensor, t: int, score: torch.Tensor) -> torch.Tensor:
        alpha = self.schedule.alpha(t)
        beta = self.schedule.beta(t)
        abar = self.schedule.abar_t(t)
        eps = -math.sqrt(1.0 - abar) * score
        return (1.0 / math.sqrt(alpha)) * (x_t - (beta / math.sqrt(1.0 - abar)) * eps)

    def _compute_safe_x0hat(self, x_t: torch.Tensor, x0hat: torch.Tensor, t: int) -> torch.Tensor:
        abar = self.schedule.abar_t(t)
        d = x_t.shape[-1]

        x_pos_scaled = math.sqrt(abar) * self.x0_pos
        diff = x_t.unsqueeze(1) - x_pos_scaled.unsqueeze(0)
        log_qt = -0.5 * diff.pow(2).sum(dim=-1) / (1.0 - abar)

        weights_unsafe = torch.softmax(log_qt, dim=-1)
        x_unsafe_denoiser = torch.einsum("mn,nd->md", weights_unsafe, self.x0_pos)

        log_qt_full = log_qt - 0.5 * d * math.log(2.0 * math.pi * (1.0 - abar))
        log_beta = -math.log(self.N_pos) + torch.logsumexp(log_qt_full, dim=-1)
        beta_val = torch.exp(log_beta)

        correction = self.eta * beta_val.unsqueeze(-1) * (x0hat - x_unsafe_denoiser)

        if self.correction_clip > 0:
            cnorm = correction.norm(dim=-1, keepdim=True)
            scale = torch.clamp(self.correction_clip / (cnorm + 1e-12), max=1.0)
            correction = correction * scale

        return x0hat + correction

    def _ess(self, w: torch.Tensor) -> float:
        w = w.detach()
        s = float(w.sum().item())
        if s <= 0.0:
            return 0.0
        wn = w / (s + 1e-12)
        return float(1.0 / (wn.pow(2).sum().item() + 1e-12))

    def _maybe_resample(
        self,
        x_t: torch.Tensor,
        w_u: torch.Tensor,
        w_q: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ess = self._ess(w_q)
        if ess >= self.ess_resample * float(self.M):
            return x_t, w_u, w_q

        self.resample_count += 1
        wq = w_q.detach()
        wn = wq / (wq.sum() + 1e-12)
        N = wn.shape[0]

        if self.resample_method == "systematic_sorted":
            sorted_order = torch.argsort(wn, descending=True)
            sorted_w = wn[sorted_order]
            cumsum = torch.cumsum(sorted_w, dim=0)
            u0 = torch.rand(1, device=wn.device, dtype=wn.dtype) / N
            u = u0 + torch.arange(N, device=wn.device, dtype=wn.dtype) / N
            sorted_idx = torch.searchsorted(cumsum, u).clamp(max=N - 1)
            idx = sorted_order[sorted_idx]
        else:
            cumsum = torch.cumsum(wn, dim=0)
            u = (torch.arange(N, device=wn.device, dtype=wn.dtype) + torch.rand(N, device=wn.device, dtype=wn.dtype)) / N
            idx = torch.searchsorted(cumsum, u).clamp(max=N - 1)

        return x_t[idx], w_u[idx] / (wq[idx] + 1e-12), torch.ones(self.M, device=x_t.device)

    def sample(self, snapshot_callback=None) -> tuple[torch.Tensor, torch.Tensor]:
        d = self.neg_x0_gmm.comps[0].mean.shape[0]
        x_t = torch.randn(self.M, d, device=self.device)
        w_u = torch.ones(self.M, device=self.device)
        w_q = torch.ones(self.M, device=self.device)
        lam_t = self._lam_at(self.schedule.T)

        t_init = self.schedule.T
        with torch.no_grad():
            x0hat_u = self._tweedie_x0hat(x_t, t_init, self._score_uncond(x_t, t_init)).detach()
        self.pu.fit(self.x0_pos, x0hat_u, w_u, pi=float(self.pu_class_prior))

        for t in range(self.schedule.T, 0, -1):
            abar = self.schedule.abar_t(t)
            alpha = self.schedule.alpha(t)
            beta = self.schedule.beta(t)
            sigma2 = self.schedule.sigma2(t)

            with torch.no_grad():
                uncond_score = self._score_uncond(x_t, t)
                x0hat_u_t = self._tweedie_x0hat(x_t, t, uncond_score).detach()
                p_t = safe_clip01(self.pu.predict_proba_pos(x0hat_u_t))
                one_minus_p_t = 1.0 - p_t

            lam_prev = self._lam_at(t - 1)
            with torch.no_grad():
                if t > self.sd_t_stop:
                    x0hat_safe = self._compute_safe_x0hat(x_t, x0hat_u_t, t)
                else:
                    x0hat_safe = x0hat_u_t

                eps_safe = (x_t - math.sqrt(abar) * x0hat_safe) / math.sqrt(1.0 - abar)
                mu_sd = (1.0 / math.sqrt(alpha)) * (x_t - (beta / math.sqrt(1.0 - abar)) * eps_safe)
                mu = self._ddpm_mu_from_score(x_t, t, uncond_score)

            if sigma2 <= 0.0:
                x_prev = mu_sd
                ratio_p_over_q = torch.ones(self.M, device=self.device)
            else:
                z = torch.randn_like(x_t)
                x_prev = mu_sd + math.sqrt(sigma2) * z
                diff_un = x_prev - mu
                diff_pr = x_prev - mu_sd
                log_ratio = -(0.5 / sigma2) * (diff_un.pow(2).sum(dim=-1) - diff_pr.pow(2).sum(dim=-1))
                ratio_p_over_q = torch.exp(log_ratio)

            w_u_prev = w_u * ratio_p_over_q

            t_prev = t - 1
            with torch.no_grad():
                if t_prev == 0:
                    x0hat_u_prev = x_prev.detach()
                else:
                    x0hat_u_prev = self._tweedie_x0hat(x_prev, t_prev, self._score_uncond(x_prev, t_prev)).detach()
            self.pu.fit(self.x0_pos, x0hat_u_prev, w_u_prev, pi=float(self.pu_class_prior))

            with torch.no_grad():
                p_prev = safe_clip01(self.pu.predict_proba_pos(x0hat_u_prev))
                one_minus_p_prev = 1.0 - p_prev

            w_q_prev = w_q * ratio_p_over_q * ((safe_clip01(one_minus_p_prev) ** lam_prev) / (safe_clip01(one_minus_p_t) ** lam_t))

            if snapshot_callback is not None:
                snapshot_callback(
                    t=t - 1,
                    x_t=x_prev.detach().clone(),
                    x0hat=x0hat_u_prev.detach().clone(),
                    pu_proba=p_prev.detach().clone(),
                    w_u=w_u_prev.detach().clone(),
                    w_q=w_q_prev.detach().clone(),
                    lam_t=lam_prev,
                )

            if t > self.resample_stop:
                x_prev, w_u_prev, w_q_prev = self._maybe_resample(x_prev, w_u_prev, w_q_prev)

            x_t, w_u, w_q = x_prev, w_u_prev, w_q_prev
            lam_t = lam_prev

        return x_t, w_q
