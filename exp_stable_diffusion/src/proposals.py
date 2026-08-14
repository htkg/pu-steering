"""Negative-guidance proposals for baseline method generation.

A proposal maps the per-step UNet eps predictions to a guided eps that the DDPM
ancestral step follows; running it bare (no SMC tilt) is the baseline. Convention
(SD v1.x eps-prediction):
  eps_pos    = eps(x_t, t, "a painting of {A}")            content prompt
  eps_style  = eps(x_t, t, "... in the style of {B}")      avoided style B
  eps_uncond = eps(x_t, t, "")                             null (only if needed)

Stateful proposals (SLD, DNG) read a per-step ``ctx`` dict (i, t, x, abar) from the
sampler; stateless ones (NegCFG) ignore it. ``reset()`` clears per-run state.
"""

import abc
from dataclasses import dataclass

import torch


class Proposal(abc.ABC):
    @property
    @abc.abstractmethod
    def needs_uncond(self) -> bool:
        """True if ``guided_eps`` needs ``eps_uncond`` (an extra null-prompt forward)."""

    def reset(self) -> None:
        """Clear per-run state. Stateless proposals keep this default no-op."""

    @abc.abstractmethod
    def guided_eps(
        self,
        eps_pos: torch.Tensor,
        eps_style: torch.Tensor,
        eps_uncond: torch.Tensor | None = None,
        ctx: dict | None = None,
    ) -> torch.Tensor:
        """Guided eps from eps_pos / eps_style [/ eps_uncond]. ``ctx`` carries the
        per-step (i, t, x, abar) used by stateful proposals."""


@dataclass
class NegCFG(Proposal):
    """Negative-prompt CFG — anchor on the avoided style B, extrapolate to content:

        eps_guide = eps_style + s×(eps_pos - eps_style)
                  = s×eps_pos + (1 - s)×eps_style
                  = eps_pos - (s - 1)×(eps_style - eps_pos)

    s=1 → pos content only; s>1 pushes toward pos content and away from avoided style B. (scale = s.)

    anchor="content" replaces the raw eps_pos with the CFG content prediction
    eps_content = eps_uncond + cfg_scale×(eps_pos - eps_uncond), so eps_guide =
    eps_style + s×(eps_content - eps_style); s=1 then reproduces content-CFG (needs eps_uncond).
    """

    scale: float = 3.0  # stateless; inherits the base no-op reset()
    anchor: str = "style"  # "style": extrapolate to raw eps_pos; "content": to the CFG content prediction
    cfg_scale: float = 7.5  # CFG scale for the content anchor (anchor="content")

    def guided_eps(
        self,
        eps_pos: torch.Tensor,
        eps_style: torch.Tensor,
        eps_uncond: torch.Tensor | None = None,
        ctx: dict | None = None,
    ) -> torch.Tensor:
        _ = ctx  # unused
        if self.anchor == "content":
            if eps_uncond is None:
                raise ValueError("content anchor needs eps_uncond")
            eps_c = eps_uncond + self.cfg_scale * (eps_pos - eps_uncond)
            return eps_style + self.scale * (eps_c - eps_style)
        return eps_style + self.scale * (eps_pos - eps_style)

    @property
    def needs_uncond(self) -> bool:
        return self.anchor == "content"


# SLD — Safe Latent Diffusion (Schramowski et al., CVPR 2023).
# Presets are (δ, s_S, λ threshold, s_m, β_m). `max` is the published
# SafetyConfig.MAX preset; `xxmax` and `x3max` use stronger safety settings.
_SLD_CONFIGS = {
    #       δ   s_S   λ_thr   s_m  β_m
    "max": (0, 5000.0, 1.0, 0.5, 0.7),
    "xxmax": (0, 20000.0, 1.0, 0.7, 0.7),
    "x3max": (0, 20000.0, 1.0, 1.0, 0.8),
}


@dataclass
class SLD(Proposal):
    """Safe Latent Diffusion safety guidance (Schramowski et al., CVPR 2023).

    Standard CFG toward content A, MINUS a thresholded, warmup-gated, momentum-
    augmented safety term steering away from the avoided style B:

        scale = clamp(|eps_pos - eps_style|×s_S, max=1)            # element-wise
        gate  = where(eps_pos - eps_style ≥ λ_thr, 0, scale)
        γ_raw = gate×(eps_style - eps_uncond)
        γ_t   = γ_raw + s_m×ν_{t-1};  ν_t = β_m×ν_{t-1} + (1-β_m)×γ_t
        eps_guide = eps_uncond + s_g×(eps_pos - eps_uncond) - (γ_t if i ≥ δ else 0)

    The 5 knobs (δ, s_S, λ_thr, s_m, β_m) come from the ``config`` preset; s_g is the
    normal CFG scale. Needs eps_uncond (3 forwards/step). State = momentum ν.
    """

    config: str = "max"
    cfg_scale: float = 7.5

    def __post_init__(self):
        if self.config not in _SLD_CONFIGS:
            raise ValueError(f"SLD config must be one of {list(_SLD_CONFIGS)}; got {self.config!r}")
        self._mom = None  # momentum buffer ν (lazily shaped on the first step)

    def reset(self):
        self._mom = None

    @property
    def needs_uncond(self) -> bool:
        return True

    def guided_eps(
        self,
        eps_pos: torch.Tensor,
        eps_style: torch.Tensor,
        eps_uncond: torch.Tensor | None = None,
        ctx: dict | None = None,
    ) -> torch.Tensor:
        if eps_uncond is None:
            raise ValueError("SLD needs eps_uncond (needs_uncond=True)")
        delta, s_S, lam_thr, s_m, beta_m = _SLD_CONFIGS[self.config]
        i = 0 if ctx is None else int(ctx.get("i", 0))

        scale = torch.clamp(torch.abs(eps_pos - eps_style) * s_S, max=1.0)
        gate = torch.where(eps_pos - eps_style >= lam_thr, torch.zeros_like(scale), scale)
        gamma_raw = gate * (eps_style - eps_uncond)
        if self._mom is None:
            self._mom = torch.zeros_like(gamma_raw)  # ν_0 = 0
        gamma_t = gamma_raw + s_m * self._mom
        self._mom = beta_m * self._mom + (1.0 - beta_m) * gamma_t

        base = eps_uncond + self.cfg_scale * (eps_pos - eps_uncond)
        return base - gamma_t if i >= delta else base  # warmup: ν still updated above


@dataclass
class DNG(Proposal):
    """Dynamic Negative Guidance (Koulischer et al., ICLR 2025; repo
    FelixKoulischer/Dynamic-Negative-Guidance, ``compute_posterior_DNG``).

    The constant NegCFG scale becomes a per-particle scale from a posterior
    p_t = p(c | x_t):

        λ_t       = λ0×p_t / (1 - p_t)                             # per particle
        eps_guide = eps_pos - λ_t×(eps_style - eps_pos)           # SD role swap: base=eps_pos, forget=eps_style

    Staggered posterior recursion (Alg. 2): the means μ(x_t) are cached at step t,
    and the posterior is updated once the sampler has drawn x_{t-1} (it arrives as
    ctx["x"] on the next call):

        α_t   = ᾱ_t / ᾱ_{t-1},  β_t = 1 - α_t                      # sub-sampled DDPM step
        σ^2_t = (1 - ᾱ_{t-1}) / (1 - ᾱ_t) × β_t                  # DDPM posterior variance
        μ_f   = (x_t - (β_t/√(1-ᾱ_t))×eps_style) / √α_t          # undesired mean
        μ_u   = (x_t - (β_t/√(1-ᾱ_t))×eps_pos  ) / √α_t          # base mean
        d_f   = Σ_dims (x_{t-1} - μ_f)^2,  d_u = Σ_dims (x_{t-1} - μ_u)^2
        p_{t-1} = clamp(p_t×exp( ((-τ)×(d_f - d_u) + δ) / (2σ^2_t) ), p_min, p_max)

    ᾱ_{t-1} (the step's target) arrives as ctx["abar_next"].

    anchor="content" replaces the raw eps_pos base with the CFG content prediction
    eps_content = eps_uncond + cfg_scale×(eps_pos - eps_uncond) (needs eps_uncond, 3 forwards);
    on SD this base is what lets the dynamic scale actually reduce the avoided style.
    """

    lambda0: float = 15.0
    tau: float = 0.2
    delta: float = 0.004
    p0: float = 0.01
    p_min: float = 1e-6
    p_max: float = 0.99
    anchor: str = "pos"  # "pos": base=eps_pos; "content": base=CFG content prediction (needs eps_uncond)
    cfg_scale: float = 7.5  # CFG scale for the content anchor (anchor="content")

    def __post_init__(self):
        self._p = None  # per-particle posterior p_t, shape (M,)
        self._prev = None  # (mu_f, mu_u, sigma2_t) cached at the previous step

    def reset(self):
        self._p = None
        self._prev = None

    @property
    def needs_uncond(self) -> bool:
        return self.anchor == "content"

    def guided_eps(
        self,
        eps_pos: torch.Tensor,
        eps_style: torch.Tensor,
        eps_uncond: torch.Tensor | None = None,
        ctx: dict | None = None,
    ) -> torch.Tensor:
        if ctx is None or "x" not in ctx or ctx.get("abar") is None or ctx.get("abar_next") is None:
            raise ValueError("DNG needs ctx with 'x', 'abar', 'abar_next'")
        M = eps_pos.shape[0]
        if self.anchor == "content":
            if eps_uncond is None:
                raise ValueError("content anchor needs eps_uncond")
            eps_base = eps_uncond + self.cfg_scale * (eps_pos - eps_uncond)
        else:
            eps_base = eps_pos
        x = ctx["x"].float()  # current latent x_t (= the x_{t-1} drawn after the previous step)
        abar_t = float(ctx["abar"])  # ᾱ_t (current step)
        abar_next = float(ctx["abar_next"])  # ᾱ_{t-1} (step target; 1.0 at the last step)
        if self._p is None:
            self._p = torch.full((M,), float(self.p0), device=eps_pos.device, dtype=torch.float32)

        # Update the posterior for the previous transition.
        if self._prev is not None:
            mu_f, mu_u, sigma2 = self._prev
            d_f = ((x - mu_f) ** 2).flatten(1).sum(1)  # (M,) sum over latent dims
            d_u = ((x - mu_u) ** 2).flatten(1).sum(1)  # (M,)
            log_mult = (-self.tau * (d_f - d_u) + self.delta) / (2.0 * sigma2)
            self._p = (self._p * torch.exp(log_mult)).clamp(self.p_min, self.p_max)

        p = self._p
        lam_vec = self.lambda0 * p / (1.0 - p)
        lam_t = lam_vec.to(eps_pos.dtype).view(M, *([1] * (eps_pos.dim() - 1)))
        eps_guide = eps_base - lam_t * (eps_style - eps_base)

        # Cache the DDPM means and posterior variance for the next update.
        alpha_t = min(max(abar_t / max(abar_next, 1e-12), 1e-12), 1.0)
        beta_t = max(1.0 - alpha_t, 1e-12)
        sigma2 = max((1.0 - abar_next) / max(1.0 - abar_t, 1e-12) * beta_t, 1e-20)
        coef = beta_t / ((1.0 - abar_t) ** 0.5 + 1e-12)
        sqrt_alpha = alpha_t**0.5
        mu_f = (x - coef * eps_style.float()) / sqrt_alpha  # undesired (style B) mean
        mu_u = (x - coef * eps_base.float()) / sqrt_alpha  # base (content A) mean
        self._prev = (mu_f, mu_u, sigma2)
        return eps_guide
