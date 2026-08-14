import math

import numpy as np
import torch
import torch.nn.functional as F
from improved_diffusion.gaussian_diffusion import GaussianDiffusion, _extract_into_tensor

from config import count_class


class UNetFeatureExtractor:
    """Extract bottleneck features from UNet via forward hook."""

    def __init__(self, model: torch.nn.Module, device: str = "cuda") -> None:
        self.model = model
        self.device = device
        self.features: torch.Tensor = torch.empty(0, device=device)
        self._hook = model.middle_block.register_forward_hook(self._hook_fn)  # type: ignore

    def _hook_fn(self, module: torch.nn.Module, input: tuple, output: torch.Tensor) -> None:
        self.features = output.detach()

    @torch.no_grad()
    def extract(self, x_t: torch.Tensor, t_tensor: torch.Tensor, pool: bool = True) -> torch.Tensor:
        with torch.amp.autocast("cuda"):
            _ = self.model(x_t.to(self.device), t_tensor.to(self.device))
        feat = self.features.float()  # (B, 256, 4, 4)
        if pool:
            return feat.mean(dim=(2, 3))  # (B, 256)
        return feat.flatten(1)  # (B, 4096)

    def close(self) -> None:
        self._hook.remove()


class LinearClassifierWS:
    """Linear classifier with warm-start support for L-BFGS. Supports loss types: bce, convex_pu, upu."""

    def __init__(self, dim: int = 256, lam: float = 1e-3, loss: str = "bce", device: str = "cuda") -> None:
        self.dim = dim
        self.lam = lam
        self.loss_type = loss
        self.device = device
        self.w: torch.Tensor | None = None
        self.bias: torch.Tensor | None = None

    def fit(self, feat_pos: torch.Tensor, feat_u: torch.Tensor, w_u: torch.Tensor, pi: float) -> None:
        feat_pos = feat_pos.to(self.device).detach()
        feat_u = feat_u.to(self.device).detach()

        w_u_norm = w_u.detach().to(self.device).clamp(min=0.0)
        s = w_u_norm.sum()
        w_u_norm = w_u_norm / (s + 1e-12) if s > 1e-12 else torch.ones_like(w_u_norm) / w_u_norm.shape[0]
        w_p_norm = torch.full((feat_pos.shape[0],), 1.0 / feat_pos.shape[0], device=self.device)

        if self.w is not None:
            w = self.w.clone().requires_grad_(True)
            bias = self.bias.clone().requires_grad_(True)  # type: ignore
            max_iter = 20  # warm start
        else:
            w = torch.zeros(self.dim, device=self.device, requires_grad=True)
            bias = torch.zeros((), device=self.device, requires_grad=True)
            max_iter = 100  # cold start

        opt = torch.optim.LBFGS([w, bias], lr=1.0, max_iter=max_iter, tolerance_grad=1e-7, tolerance_change=1e-9, line_search_fn="strong_wolfe")

        loss_type = self.loss_type
        lam_reg = self.lam

        def closure():
            opt.zero_grad(set_to_none=True)
            g_p = feat_pos @ w + bias
            g_u = feat_u @ w + bias

            if loss_type == "convex_pu":
                loss = -float(pi) * (w_p_norm * g_p).sum() + (w_u_norm * F.softplus(g_u)).sum()
            elif loss_type == "upu":
                loss_pos = float(pi) * (w_p_norm * F.softplus(-g_p)).sum()
                loss_u = (w_u_norm * F.softplus(g_u)).sum()
                loss_neg = -float(pi) * (w_p_norm * F.softplus(g_p)).sum()
                neg_part = loss_u + loss_neg
                loss = loss_pos + (neg_part if neg_part.item() >= 0 else -neg_part)
            elif loss_type == "bce":
                loss = (w_p_norm * F.softplus(-g_p)).sum() + (w_u_norm * F.softplus(g_u)).sum()
            else:
                raise ValueError(f"Unknown loss: {loss_type}")

            loss = loss + 0.5 * lam_reg * (w @ w)
            loss.backward()
            return loss

        opt.step(closure)
        self.w = w.detach()
        self.bias = bias.detach()

    @torch.no_grad()
    def predict_proba_pos(self, feat: torch.Tensor) -> torch.Tensor:
        if self.w is None:
            return torch.full((feat.shape[0],), 0.5, device=self.device)
        logit = feat.to(self.device) @ self.w + self.bias  # type: ignore
        return torch.sigmoid(logit)

    def reset(self) -> None:
        self.w = None
        self.bias = None


def median_heuristic(x: torch.Tensor, scale: float = 1.0) -> float:
    """Compute median of pairwise L2 distances."""
    D = torch.cdist(x, x)
    iu = torch.triu_indices(x.shape[0], x.shape[0], offset=1)
    dists = D[iu[0], iu[1]]
    return max(dists.median().item(), 1e-6) * scale


def safe_clip01(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.clamp(p, eps, 1.0 - eps)


def systematic_resample_sorted(weights: torch.Tensor) -> torch.Tensor:
    """Systematic resampling + weight sorting (Chopin et al. 2022)."""
    N = weights.shape[0]
    sorted_order = torch.argsort(weights, descending=True)
    sorted_weights = weights[sorted_order]
    cumsum = torch.cumsum(sorted_weights, dim=0)
    cumsum[-1] = 1.0
    u0 = torch.rand(1, device=weights.device, dtype=weights.dtype) / N
    u = u0 + torch.arange(N, device=weights.device, dtype=weights.dtype) / N
    sorted_indices = torch.searchsorted(cumsum, u).clamp(max=N - 1)
    return sorted_order[sorted_indices]


def run_trial(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    feat_extractor: UNetFeatureExtractor,
    eval_clf: torch.nn.Module,
    diag_feats: dict[str, torch.Tensor],
    *,
    x_ref: torch.Tensor,
    feat_ref: torch.Tensor,
    sd_sigma: float,
    loss_type: str = "bce",
    pu_lam: float = 1e-3,
    pi: float = 0.1,
    lam_max: float = 3.0,
    sd_scale: float = 0.3,
    is_mode: str = "none",
    is_tau: float = 1.0,
    ess_thresh: float = 0.9,
    resample_stop: int = 15,
    lam_schedule: str = "linear",
    target_class: str = "plane",
    target_class_idx: int = 0,
    M: int = 128,
    seed: int = 0,
    device: str = "cuda",
    use_denoised_feat: bool = False,
) -> dict:
    """Single trial: SD proposal + PU-SMC with resampling.
    When lam_max=0, PU training and resampling are skipped (SD-only mode).
    Returns dict with diagnostics and generated samples.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    T = diffusion.num_timesteps
    x_t = torch.randn(M, 3, 32, 32, device=device)

    w_q = torch.ones(M, device=device) / M
    w_u = torch.ones(M, device=device) / M
    lam_t = 0.0

    sd_stop = int(0.08 * T)
    pu_start_idx = int(0.6 * T)
    total_pu_steps = pu_start_idx

    use_pu = lam_max > 0
    clf = None
    if use_pu:
        clf = LinearClassifierWS(dim=256, lam=pu_lam, loss=loss_type, device=device)

    particle_ids = torch.arange(M, device=device)
    resample_count = 0
    resample_steps = []
    ess_trajectory = []
    w_u_ess_trajectory = []
    gap_trajectory = []
    is_ratio_stats = []

    for i in range(T - 1, -1, -1):
        t_tensor = torch.tensor([i] * M, device=device)

        with torch.no_grad(), torch.amp.autocast("cuda"):
            out = diffusion.p_mean_variance(model, x_t, t_tensor, clip_denoised=True)
            x0hat = out["pred_xstart"].float()
            mu_uncond = out["mean"].float()
            log_var = out["log_variance"].float()
            sigma2 = torch.exp(log_var[0, 0, 0, 0]).item()

        # PU classifier: feature extraction and prediction
        if use_pu:
            assert clf is not None
            feat_particles = feat_extractor.features.float().mean(dim=(2, 3))

            with torch.no_grad():
                if use_denoised_feat:
                    t0_p = torch.zeros(M, dtype=torch.long, device=device)
                    x0hat_clipped_p = x0hat.clamp(-1, 1)
                    with torch.amp.autocast("cuda"):
                        _ = model(x0hat_clipped_p, t0_p)
                    feat_for_clf = feat_extractor.features.float().mean(dim=(2, 3))
                    p_t = safe_clip01(clf.predict_proba_pos(feat_for_clf))
                else:
                    p_t = safe_clip01(clf.predict_proba_pos(feat_particles))
                one_minus_p_t = 1.0 - p_t

            # Lambda schedule
            train_pu = i < pu_start_idx
            if train_pu:
                frac = (pu_start_idx - i) / total_pu_steps
                if lam_schedule == "cosine":
                    lam_prev = lam_max * 0.5 * (1.0 - math.cos(math.pi * frac))
                else:  # linear
                    lam_prev = lam_max * frac
            else:
                lam_prev = 0.0
        else:
            train_pu = False
            lam_prev = 0.0

        # SD proposal
        apply_sd = sd_scale > 0 and i > sd_stop
        if apply_sd:
            with torch.no_grad():
                # SafeDenoiser-style guidance, following: https://github.com/MingyuKim87/Safe_Denoiser
                # repellency/repellency_methods_fast.py, RBFKernelRepellency.empirical_denoiser
                # kernel = exp(-cdist(...)/(2σ^2)); x_0_hat -= scale * (kernel @ x_ref)
                # (note: uses ||x-y||, not squared; matches original implementation)
                D = 3 * 32 * 32
                dists = torch.cdist(x0hat.reshape(M, D), x_ref.reshape(x_ref.shape[0], D))
                weights = torch.softmax(-dists / (2.0 * sd_sigma**2), dim=-1)
                unsafe_d = torch.mm(weights, x_ref.reshape(x_ref.shape[0], D)).reshape_as(x0hat)
                x0hat_safe = x0hat - sd_scale * unsafe_d
                coef1 = _extract_into_tensor(diffusion.posterior_mean_coef1, t_tensor, x_t.shape)
                coef2 = _extract_into_tensor(diffusion.posterior_mean_coef2, t_tensor, x_t.shape)
                mu_prop = coef1 * x0hat_safe + coef2 * x_t
        else:
            mu_prop = mu_uncond

        # Sample x_{t-1}
        noise = torch.randn_like(x_t)
        nonzero_mask = (t_tensor != 0).float().view(-1, 1, 1, 1)
        x_prev = mu_prop + nonzero_mask * torch.exp(0.5 * log_var) * noise

        # SD-only mode: skip IS, PU, resampling
        if not use_pu:
            x_t = x_prev
            ess_trajectory.append((i, float(M)))
            w_u_ess_trajectory.append((i, float(M)))
            continue

        # IS ratio
        if apply_sd and sigma2 > 1e-12 and i > 0:
            diff_un = (x_prev - mu_uncond).flatten(1)
            diff_pr = (x_prev - mu_prop).flatten(1)
            log_ratio = -(0.5 / sigma2) * (diff_un.pow(2).sum(1) - diff_pr.pow(2).sum(1))
            if is_tau != 1.0:
                log_ratio = log_ratio / is_tau
            log_ratio = log_ratio.clamp(-20, 20)
            is_ratio = torch.exp(log_ratio)
        else:
            is_ratio = torch.ones(M, device=device)

        # Track IS ratio stats for diagnostics
        if apply_sd and i > 0:
            is_ratio_stats.append((i, float(is_ratio.mean()), float(is_ratio.std()), float(is_ratio.min()), float(is_ratio.max())))

        # Ratio for w_q depends on IS mode
        if is_mode == "none":  # heuristic: disable IS (unstable in high-dim); resampling still guided by classifier weights
            ratio = torch.ones(M, device=device)
        else:
            ratio = is_ratio

        # w_u always uses IS ratio
        w_u_prev = w_u * is_ratio if (apply_sd and sigma2 > 1e-12 and i > 0) else w_u.clone()

        # Train PU classifier
        if train_pu:
            assert clf is not None
            with torch.no_grad():
                if use_denoised_feat:
                    t0_tensor = torch.zeros(M, dtype=torch.long, device=device)
                    x0hat_clipped = x0hat.clamp(-1, 1)
                    with torch.amp.autocast("cuda"):
                        _ = model(x0hat_clipped, t0_tensor)
                    feat_prev = feat_extractor.features.float().mean(dim=(2, 3))
                elif i > 0:
                    t_prev_t = torch.tensor([i - 1] * M, device=device)
                    with torch.amp.autocast("cuda"):
                        _ = diffusion.p_mean_variance(model, x_prev, t_prev_t, clip_denoised=True)
                    feat_prev = feat_extractor.features.float().mean(dim=(2, 3))
                else:
                    t0 = torch.zeros(M, dtype=torch.long, device=device)
                    with torch.amp.autocast("cuda"):
                        _ = model(x_prev, t0)
                    feat_prev = feat_extractor.features.float().mean(dim=(2, 3))

            clf.fit(feat_ref, feat_prev, w_u_prev, pi)

            with torch.no_grad():
                p_prev = safe_clip01(clf.predict_proba_pos(feat_prev))
                one_minus_p_prev = 1.0 - p_prev

                step_gap = (clf.predict_proba_pos(feat_ref).mean() - clf.predict_proba_pos(feat_prev).mean()).item()
                gap_trajectory.append((i, step_gap))
        else:
            one_minus_p_prev = torch.ones(M, device=device)

        # Weight update
        if train_pu and lam_t > 0:
            w_q_prev = w_q * ratio * (safe_clip01(one_minus_p_prev) ** lam_prev) / (safe_clip01(one_minus_p_t) ** lam_t + 1e-12)  # type: ignore
        else:
            w_q_prev = w_q * ratio

        # Normalize weights
        s_q = w_q_prev.sum()
        w_q_prev = w_q_prev / s_q if s_q > 1e-12 else torch.ones(M, device=device) / M
        s_u = w_u_prev.sum()
        w_u_prev = w_u_prev / s_u if s_u > 1e-12 else torch.ones(M, device=device) / M

        # ESS tracking
        ess_q = 1.0 / (w_q_prev.pow(2).sum().item() + 1e-12)
        ess_u = 1.0 / (w_u_prev.pow(2).sum().item() + 1e-12)
        ess_trajectory.append((i, ess_q))
        w_u_ess_trajectory.append((i, ess_u))

        # Resampling
        if i > resample_stop and ess_q < ess_thresh * M:
            idx = systematic_resample_sorted(w_q_prev)
            x_prev = x_prev[idx]

            w_u_prev = w_u_prev[idx] / (w_q_prev[idx] + 1e-12)
            s_u2 = w_u_prev.sum()
            w_u_prev = w_u_prev / s_u2 if s_u2 > 1e-12 else torch.ones(M, device=device) / M

            w_q_prev = torch.ones(M, device=device) / M
            particle_ids = particle_ids[idx]
            resample_steps.append(i)
            resample_count += 1

        x_t = x_prev
        w_u = w_u_prev
        w_q = w_q_prev
        lam_t = lam_prev

    # === Evaluation ===
    samples = x_t.cpu()
    n_target = count_class(eval_clf, samples, class_idx=target_class_idx, device=device)
    raw_rate = n_target / M

    n_unique = len(particle_ids.unique())

    # Per-class gap on diagnostic data
    gap = None
    if use_pu and diag_feats and clf is not None and clf.w is not None:
        per_class = {}
        for cls, feat in diag_feats.items():
            per_class[cls] = clf.predict_proba_pos(feat).cpu().mean().item()
        target_p = per_class.get(target_class, 0)
        rest = [v for k, v in per_class.items() if k != target_class]
        gap = target_p - float(np.mean(rest)) if rest else 0.0

    clf_w = None
    clf_bias = None
    if use_pu and clf is not None and clf.w is not None:
        clf_w = clf.w.cpu().numpy().tolist()
        clf_bias = float(clf.bias.item())  # type: ignore

    return {
        "raw_target_rate": raw_rate,
        "n_target": n_target,
        "n_unique": n_unique,
        "resample_count": resample_count,
        "resample_steps": resample_steps,
        "final_ess": 1.0 / (w_q.pow(2).sum().item() + 1e-12),
        "ess_min": float(min(e for _, e in ess_trajectory)) if ess_trajectory else M,
        "ess_trajectory": ess_trajectory,
        "w_u_ess_trajectory": w_u_ess_trajectory,
        "gap": gap,
        "gap_trajectory": gap_trajectory,
        "is_ratio_stats": is_ratio_stats,
        "classifier_w": clf_w,
        "classifier_bias": clf_bias,
        "samples": samples,
    }
