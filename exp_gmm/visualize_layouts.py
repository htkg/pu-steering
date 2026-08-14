"""Visualize ring GMM layouts.

Shows 4 different GMM layouts (seed 0-3) with pos/neg mode coloring.

Usage:
    uv run python visualize_layouts.py
"""

import matplotlib.pyplot as plt
import torch
from gmm_ddpm import sample_gmm
from gmm_ring import make_ring_gmm

VIS_SEED = 42

N_GT = 3000
SEEDS = list(range(4))
NCOLS = 4
NROWS = 1


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="plots")
    args = parser.parse_args()
    import os

    os.makedirs(args.outdir, exist_ok=True)

    _, axes = plt.subplots(NROWS, NCOLS, figsize=(5 * NCOLS, 5 * NROWS))
    axes = axes.flatten()

    for i, seed in enumerate(SEEDS):
        ax = axes[i]
        pos, neg, pi = make_ring_gmm(seed=seed)

        n_pos = int(N_GT * pi)
        n_neg = N_GT - n_pos
        with torch.no_grad():
            vis_gen = torch.Generator().manual_seed(VIS_SEED + seed)
            x_pos = sample_gmm(pos, n_pos, generator=vis_gen).numpy()
            x_neg = sample_gmm(neg, n_neg, generator=vis_gen).numpy()

        ax.scatter(x_neg[:, 0], x_neg[:, 1], s=6, alpha=0.3, color="tab:blue", label="neg (desired)", zorder=2)
        ax.scatter(x_pos[:, 0], x_pos[:, 1], s=6, alpha=0.3, color="tab:red", label="pos (avoid)", zorder=3)

        ax.set_title(f"Layout {seed}  (pi={pi:.2f})", fontsize=12)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
        if i == 0:
            ax.legend(fontsize=9, loc="upper right")

        xmax = max(abs(x_pos[:, 0]).max(), abs(x_neg[:, 0]).max(), abs(x_pos[:, 1]).max(), abs(x_neg[:, 1]).max())
        ax.set_xlim(-xmax * 1.15, xmax * 1.15)
        ax.set_ylim(-xmax * 1.15, xmax * 1.15)

    plt.tight_layout()
    out = os.path.join(args.outdir, "gmm_layouts.pdf")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
