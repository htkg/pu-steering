"""
Unconditional CIFAR-10 image generation using improved-diffusion.
Single-GPU script (no MPI required).
"""

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "improved-diffusion"))
from improved_diffusion.script_util import create_model_and_diffusion, model_and_diffusion_defaults


def make_grid(images: np.ndarray, nrow: int = 8) -> np.ndarray:
    """Arrange images into a grid. images: (N, H, W, C) uint8 numpy array."""
    n, h, w, c = images.shape
    ncol = nrow
    nrow_actual = (n + ncol - 1) // ncol
    grid = np.zeros((nrow_actual * h, ncol * w, c), dtype=np.uint8)
    for idx in range(n):
        r = idx // ncol
        col = idx % ncol
        grid[r * h : (r + 1) * h, col * w : (col + 1) * w] = images[idx]
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from CIFAR-10 diffusion model")
    parser.add_argument("--model_path", type=str, default=os.path.join(os.path.dirname(__file__), "improved-diffusion", "cifar10_uncond_50M_500K.pt"))
    # https://openaipublic.blob.core.windows.net/diffusion/march-2021/cifar10_uncond_50M_500K.pt
    # adopted from https://github.com/openai/improved-diffusion/blob/main/README.md
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--timestep_respacing", type=str, default="250", help="Number of respaced timesteps (e.g. '250', 'ddim50')")
    parser.add_argument("--use_ddim", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save results (default: results/uncond)")
    args = parser.parse_args()

    # Seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # CIFAR-10 unconditional model hyperparameters (from README)
    model_config = model_and_diffusion_defaults()
    model_config.update(
        image_size=32,
        num_channels=128,
        num_res_blocks=3,
        learn_sigma=True,
        dropout=0.3,
        diffusion_steps=4000,
        noise_schedule="cosine",
        timestep_respacing=args.timestep_respacing,
    )

    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**model_config)

    print(f"Loading checkpoint: {args.model_path}")
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(__file__), "results", "uncond")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.num_samples} samples...")
    all_images = []
    num_generated = 0
    sample_fn = diffusion.ddim_sample_loop if args.use_ddim else diffusion.p_sample_loop

    while num_generated < args.num_samples:
        bs = min(args.batch_size, args.num_samples - num_generated)
        sample = sample_fn(
            model,
            (bs, 3, 32, 32),
            clip_denoised=True,
            model_kwargs={},
        )
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        sample = sample.permute(0, 2, 3, 1).cpu().numpy()
        all_images.append(sample)
        num_generated += bs
        print(f"  Generated {num_generated}/{args.num_samples} samples")

    arr = np.concatenate(all_images, axis=0)[: args.num_samples]

    # Save as npz
    npz_path = os.path.join(args.output_dir, "samples.npz")
    np.savez(npz_path, arr)
    print(f"Saved npz: {npz_path}")

    # Save grid image
    grid = make_grid(arr, nrow=8)
    grid_path = os.path.join(args.output_dir, "grid.png")
    Image.fromarray(grid).save(grid_path)
    print(f"Saved grid: {grid_path}")

    print("Done!")


if __name__ == "__main__":
    main()
