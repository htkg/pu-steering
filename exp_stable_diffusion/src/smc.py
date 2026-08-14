"""Ours: M-particle latent SMC over the SD UNet with an online kernel-PU tilt.

  * proposal: SLD(xxmax) (any baseline proposal works).
  * f_t: paired score-based PU. Positives = style-B Tweedie x0-hat, unlabeled =
    content Tweedie x0-hat of the same particles, both pooled to a latent feature.
  * classifier: KernelPULogistic (RBF, uPU), refit every step. Its logit is
    z-scored across particles before sigmoid -> p.
  * tilt: incremental SMC weight (1-p_new)^lam / (1-p_old)^lam_prev, lambda grown
    ESS-adaptively (or a linear ramp), weights clipped.
  * windowed ESS-gated systematic resampling; per-particle weights returned.

lam_max=0 reduces to the bare proposal.
"""

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from diffusers import DDPMScheduler, StableDiffusionPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.kernel_pu import KernelPULogistic
from src.proposals import Proposal
from src.sd import _decode, _encode
from src.smc_math import clip_weights, ess, solve_delta_for_ess, systematic_resample_sorted, tweedie_x0hat_from_eps


PU_DELTA = 1e-3  # two-sided clamp on p


@dataclass
class OursConfig:
    """Parameters for the SMC sampler."""

    lam_max: float  # terminal tempering strength (0 => bare proposal)
    M: int = 32  # particles per run
    num_steps: int = 50

    pi: float = 0.15  # PU class prior P(c)
    pu_warmup_frac: float = 0.7  # fraction of (late) steps with PU on
    pu_kernel_scale: float = 0.4  # RBF median-bw scale
    pu_lam_reg: float = 1e-2  # kernel RKHS L2 reg
    pu_standardize_logit: bool = True  # z-score the logit across particles before sigmoid
    feat_out_hw: int = 16  # pooled latent feature side (C*hw*hw dims)

    lam_adaptive: bool = True  # ESS-adaptive lambda vs linear ramp
    ess_target: float = 0.5  # adaptive-lambda target ESS fraction
    ess_resample: float = 0.7  # resample when ESS < ess_resample * M
    resample_start_frac: float = 0.3  # resample only after this fraction of steps
    resample_stop_frac: float = 0.3  # ...and before the last this fraction (window)
    resample_every: int = 5  # cadence within the window
    weight_clip: float | None = 2.0  # clipped-IS log-ratio bound

    seed: int = 0


def pooled_latent_feature(x0hat: torch.Tensor, out_hw: int = 16) -> torch.Tensor:
    """Pool a latent x0-hat estimate into one feature vector per particle.

    ``x0hat`` has shape (M, C, H, W): M particles, C latent channels (4 for SD
    v1.x), H x W spatial. Each channel is average-pooled to out_hw x out_hw and
    flattened, giving an (M, C * out_hw * out_hw) feature — the input to the
    kernel-PU classifier f_t.
    """
    return torch.nn.functional.adaptive_avg_pool2d(x0hat, out_hw).flatten(1)


@torch.no_grad()
def _eps(pipe: StableDiffusionPipeline, x: torch.Tensor, t, emb: torch.Tensor) -> torch.Tensor:
    return pipe.unet(x, t, encoder_hidden_states=emb.expand(x.shape[0], -1, -1)).sample


def _proba(clf: KernelPULogistic, feat: torch.Tensor, cfg: OursConfig) -> torch.Tensor:
    """P(c|feat). With pu_standardize_logit, z-score the logit across particles before
    sigmoid (+ two-sided delta clamp). p_old and p_new both route here, so the tilt
    ratio stays consistent."""
    if not cfg.pu_standardize_logit:
        return clf.predict_proba_pos(feat)
    logit = clf.decision(feat).float()
    z = (logit - logit.mean()) / logit.std().clamp_min(1e-6)
    p = torch.sigmoid(z)
    return p.clamp(PU_DELTA, 1.0 - PU_DELTA)


def _smc_run(
    pipe: StableDiffusionPipeline,
    scheduler: DDPMScheduler,
    emb_A: torch.Tensor,
    emb_B: torch.Tensor,
    emb_null: torch.Tensor | None,
    proposal: Proposal,
    cfg: OursConfig,
    abar: torch.Tensor,
    device: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """One SMC run -> (latents (M,C,H,W), per-particle weights (M,)).

    Not wrapped in torch.no_grad: the kernel-PU LBFGS fit needs autograd. The UNet
    forwards (_eps) are no_grad, so only the classifier parameters require grad.
    """
    g = torch.Generator(device=device).manual_seed(cfg.seed)
    scheduler.set_timesteps(cfg.num_steps, device=device)
    timesteps = scheduler.timesteps
    C = pipe.unet.config.in_channels
    S = pipe.unet.config.sample_size

    x = torch.randn(cfg.M, C, S, S, generator=g, device=device, dtype=dtype) * scheduler.init_noise_sigma
    w_q = torch.ones(cfg.M, device=device)
    lam_t = 0.0
    proposal.reset()

    n = len(timesteps)
    pu_start = round((1.0 - cfg.pu_warmup_frac) * n)  # PU on at/after this index
    resample_start = round(cfg.resample_start_frac * n)
    resample_stop = round((1.0 - cfg.resample_stop_frac) * n)
    delta_max = max(cfg.lam_max * 1.5 / n, 1e-3)
    resample_on = cfg.lam_max > 0.0
    clf: KernelPULogistic | None = None

    for i, t in enumerate(timesteps):
        pu_on = (cfg.lam_max > 0.0) and (i >= pu_start)
        abar_t = float(abar[t].item())

        eps_pos = _eps(pipe, x, t, emb_A)
        eps_style = _eps(pipe, x, t, emb_B)
        eps_uncond = _eps(pipe, x, t, emb_null) if proposal.needs_uncond else None  # type: ignore
        abar_next = float(abar[timesteps[i + 1]].item()) if i + 1 < n else 1.0
        ctx = {"i": i, "n": n, "t": int(t), "x": x, "abar": abar_t, "abar_next": abar_next}
        eps_guide = proposal.guided_eps(eps_pos, eps_style, eps_uncond, ctx=ctx)

        if pu_on:
            # Fit the paired PU classifier on Tweedie x0 estimates from the
            # content and avoided-style predictions.
            feat_u = pooled_latent_feature(tweedie_x0hat_from_eps(x, abar_t, eps_pos).float(), cfg.feat_out_hw)
            feat_pos = pooled_latent_feature(tweedie_x0hat_from_eps(x, abar_t, eps_style).float(), cfg.feat_out_hw)
            if clf is None:
                clf = KernelPULogistic(sigma_scale=cfg.pu_kernel_scale, lam=cfg.pu_lam_reg, device=device)

            # Evaluate the previous classifier before refitting it.
            p_old = _proba(clf, feat_u, cfg)
            one_minus_p_old = (1.0 - p_old).clamp(PU_DELTA, 1.0)

            if cfg.lam_adaptive:
                d = solve_delta_for_ess(w_q, one_minus_p_old, cfg.ess_target * cfg.M, delta_max)
                lam_new = min(lam_t + d, cfg.lam_max)
            else:
                frac = (i - pu_start + 1) / max(1, n - pu_start)
                lam_new = cfg.lam_max * float(frac)

            clf.fit(feat_pos, feat_u, w_q / (w_q.sum() + 1e-12), pi=cfg.pi)
            p_new = _proba(clf, feat_u, cfg)
            one_minus_p_new = (1.0 - p_new).clamp(PU_DELTA, 1.0)

            # Apply the incremental tempering ratio to the particle weights.
            w_q = w_q * (one_minus_p_new**lam_new) / (one_minus_p_old**lam_t)
            w_q = clip_weights(w_q, cfg.weight_clip)
            lam_t = lam_new

        x = scheduler.step(eps_guide, t, x, generator=g).prev_sample  # type: ignore

        # Resample only inside the configured ESS-gated window.
        in_window = pu_on and (resample_start <= i < resample_stop)
        on_cadence = ((i - resample_start) % max(1, cfg.resample_every)) == 0
        if resample_on and in_window and on_cadence and ess(w_q) < cfg.ess_resample * cfg.M:
            w_draw = clip_weights(w_q, cfg.weight_clip)
            w_norm = w_draw / (w_draw.sum() + 1e-12)
            x = x[systematic_resample_sorted(w_norm, generator=g)]
            w_q = torch.ones(cfg.M, device=device)

    # Keep the final weights for the post-hoc weighted metrics.
    return x, w_q


def generate_ours(
    pipe: StableDiffusionPipeline,
    scheduler: DDPMScheduler,
    obj: str,
    style: str,
    proposal: Proposal,
    cfg: OursConfig,
    n: int,
    seed_offset: int,
    seeds: list[int],
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ``n`` Ours images by tiling M-particle SMC runs over ``seeds``.

    Run seed = seed_offset + seed_val + 1000*pass. Each run's weights are normalized
    to sum 1 then scaled by 1/num_runs, so every run carries equal pooled mass (SMC
    weights are only relative within a run). Returns (images [0,1] (n,3,H,W) on CPU,
    weights (n,) float64 summing 1).
    Prompts: content = "a painting of {obj}", style B = "... in the style of {style}".
    """
    emb_A = _encode(pipe, f"a painting of {obj}", device, dtype)
    emb_B = _encode(pipe, f"a painting of {obj} in the style of {style}", device, dtype)
    emb_null = _encode(pipe, "", device, dtype) if proposal.needs_uncond else None
    abar = scheduler.alphas_cumprod.to(device)

    imgs_all: list[torch.Tensor] = []
    w_all: list[torch.Tensor] = []
    have, pass_idx = 0, 0
    while have < n:
        for seed_val in seeds:
            if have >= n:
                break
            run_cfg = replace(cfg, seed=seed_offset + seed_val + 1000 * pass_idx)
            x, w = _smc_run(pipe, scheduler, emb_A, emb_B, emb_null, proposal, run_cfg, abar, device, dtype)
            imgs = _decode(pipe, x, dtype)
            imgs_all.append(imgs)
            wr = w.detach().to(torch.float64).reshape(-1).cpu().clamp_min(0.0)
            s = float(wr.sum())
            wr = wr / s if s > 0.0 else torch.full((imgs.shape[0],), 1.0 / imgs.shape[0], dtype=torch.float64)
            w_all.append(wr)  # per-run, sums to 1 (scaled by 1/num_runs below)
            have += imgs.shape[0]
        pass_idx += 1

    images = torch.cat(imgs_all, 0)[:n]
    num_runs = len(w_all)
    weights = torch.cat([wr / num_runs for wr in w_all], 0)[:n]
    s = float(weights.sum())
    weights = weights / s if s > 0.0 else torch.full((weights.shape[0],), 1.0 / weights.shape[0], dtype=torch.float64)
    return images, weights
