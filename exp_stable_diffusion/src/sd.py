"""Stable Diffusion v1.4 loading + plain classifier-free-guidance sampling.

The reference pool is plain CFG generation toward a single prompt (no avoidance):
    eps_guide = eps_uncond + s * (eps_pos - eps_uncond)
DDPM ancestral sampling, M images per seed, consecutive seeds for reproducibility.
"""

import sys
from pathlib import Path

import torch
from diffusers import DDPMScheduler, StableDiffusionPipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.proposals import Proposal

MODEL = "CompVis/stable-diffusion-v1-4"


def load_pipeline(device: str = "cuda", dtype: torch.dtype = torch.float16) -> tuple[StableDiffusionPipeline, DDPMScheduler]:
    """Load SD v1.4 (safety checker off) + a DDPM scheduler."""
    pipe = StableDiffusionPipeline.from_pretrained(MODEL, torch_dtype=dtype, safety_checker=None).to(device)
    pipe.set_progress_bar_config(disable=True)
    scheduler: DDPMScheduler = DDPMScheduler.from_config(pipe.scheduler.config)  # type: ignore
    return pipe, scheduler


@torch.no_grad()
def _encode(pipe: StableDiffusionPipeline, prompt: str, device: str, dtype: torch.dtype) -> torch.Tensor:
    tok = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    return pipe.text_encoder(tok.input_ids.to(device))[0].to(dtype)


@torch.no_grad()
def _decode(pipe: StableDiffusionPipeline, latents: torch.Tensor, dtype: torch.dtype, batch: int = 16) -> torch.Tensor:
    """VAE-decode latents -> images in [0,1], (N,3,H,W) on CPU."""
    scaling = getattr(pipe.vae.config, "scaling_factor", 0.18215)
    imgs = []
    for j in range(0, latents.shape[0], batch):
        chunk = latents[j : j + batch].to(dtype) / scaling
        dec = pipe.vae.decode(chunk).sample
        imgs.append((dec / 2 + 0.5).clamp(0, 1).float().cpu())
    return torch.cat(imgs, 0)


@torch.no_grad()
def generate_cfg(
    pipe: StableDiffusionPipeline,
    scheduler: DDPMScheduler,
    prompt: str,
    n: int,
    seed0: int,
    M: int = 32,
    num_steps: int = 50,
    cfg_scale: float = 7.5,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Generate ``n`` images of ``prompt`` by plain CFG, M per seed from seed0.

    Each seed s draws M latents with its own generator and runs the full DDPM
    reverse process; seeds are consecutive so the pool is fixed/reproducible.
    Returns decoded images in [0,1], (n,3,H,W) on CPU.
    """
    emb = _encode(pipe, prompt, device, dtype)
    emb_null = _encode(pipe, "", device, dtype)
    C = pipe.unet.config.in_channels
    S = pipe.unet.config.sample_size

    out, have, s = [], 0, seed0
    while have < n:
        g = torch.Generator(device=device).manual_seed(s)
        scheduler.set_timesteps(num_steps, device=device)
        x = torch.randn(M, C, S, S, generator=g, device=device, dtype=dtype)
        x = x * scheduler.init_noise_sigma

        for t in scheduler.timesteps:
            e_c = pipe.unet(x, t, encoder_hidden_states=emb.expand(M, -1, -1)).sample
            e_u = pipe.unet(x, t, encoder_hidden_states=emb_null.expand(M, -1, -1)).sample
            e = e_u + cfg_scale * (e_c - e_u)
            x = scheduler.step(e, t, x, generator=g).prev_sample  # type: ignore
        out.append(_decode(pipe, x, dtype))
        have += x.shape[0]
        s += 1
    return torch.cat(out, 0)[:n]


@torch.no_grad()
def generate_method(
    pipe: StableDiffusionPipeline,
    scheduler: DDPMScheduler,
    obj: str,
    style: str,
    proposal: Proposal,
    n: int,
    seed_offset: int,
    seeds: list[int],
    M: int = 32,
    num_steps: int = 50,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Generate ``n`` images of a bare-proposal baseline (no SMC tilt).

    Each run is M particles guided by ``proposal`` (which maps eps_pos / eps_style
    [/ eps_uncond] -> a guided eps). Runs cycle over ``seeds`` with run seed =
    ``seed_offset + seed_val + 1000×pass`` until n images are collected, so shards
    with different seed_offset draw disjoint, reproducible samples.
    Prompts: content = "a painting of {obj}", style B = "... in the style of {style}".
    Returns decoded images in [0,1], (n,3,H,W) on CPU.
    """
    emb_A = _encode(pipe, f"a painting of {obj}", device, dtype)
    emb_B = _encode(pipe, f"a painting of {obj} in the style of {style}", device, dtype)
    emb_null = _encode(pipe, "", device, dtype) if proposal.needs_uncond else None
    C = pipe.unet.config.in_channels
    S = pipe.unet.config.sample_size
    abar = scheduler.alphas_cumprod.to(device)  # ᾱ_t for stateful proposals (DNG)

    out, have, pass_idx = [], 0, 0
    while have < n:
        for seed_val in seeds:
            if have >= n:
                break
            proposal.reset()  # clear per-run state (no-op for stateless NegCFG)
            g = torch.Generator(device=device).manual_seed(seed_offset + seed_val + 1000 * pass_idx)
            scheduler.set_timesteps(num_steps, device=device)
            x = torch.randn(M, C, S, S, generator=g, device=device, dtype=dtype)
            x = x * scheduler.init_noise_sigma

            ts = scheduler.timesteps
            for i, t in enumerate(ts):
                eps_pos = pipe.unet(x, t, encoder_hidden_states=emb_A.expand(M, -1, -1)).sample
                eps_style = pipe.unet(x, t, encoder_hidden_states=emb_B.expand(M, -1, -1)).sample
                eps_uncond = pipe.unet(x, t, encoder_hidden_states=emb_null.expand(M, -1, -1)).sample if emb_null is not None else None
                # Stateful proposals use the per-step context; DNG also needs the cumulative alpha at the target timestep.
                abar_next = float(abar[ts[i + 1]]) if i + 1 < len(ts) else 1.0
                ctx = {"i": i, "n": num_steps, "t": int(t), "x": x, "abar": float(abar[t]), "abar_next": abar_next}
                eps_guide = proposal.guided_eps(eps_pos, eps_style, eps_uncond, ctx=ctx)
                x = scheduler.step(eps_guide, t, x, generator=g).prev_sample  # type: ignore
            out.append(_decode(pipe, x, dtype))
            have += x.shape[0]
        pass_idx += 1
    return torch.cat(out, 0)[:n]
