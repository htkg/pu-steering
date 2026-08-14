"""Create a four-panel GMM comparison plot.

Layout:
    Oracle  |  PU Steering (ON_TILT headline config of run_score.py)
    NegCFG  |  DNG

All panels colored by oracle posterior p(c|x) with coolwarm colormap.
Usage:
    uv run python visualize_score_paper_plot.py --layout 0 --seed 0
    uv run python visualize_score_paper_plot.py --layout 0 --seed 0 --outdir plots
"""

import argparse
import os

os.environ["TQDM_DISABLE"] = "1"

import matplotlib.pyplot as plt
import torch
from gmm_ddpm import DDPMSchedule, OracleDiffusion, sample_gmm
from gmm_ring import compute_post, make_ring_gmm
from kernel_model import KernelPULogistic
from sampler_score import DNGDDPM, NegCFG, PUNegGuideDDPMSMC

T = 100
M = 1024
N_GT = 3000
VIS_SEED = 42

PANEL_ORDER = [
    ("Oracle", 0, 0),
    ("PU Steering", 0, 1),
    ("NegCFG", 1, 0),
    ("DNG", 1, 1),
]


FONT_SIZE = 16


def plot_paper(pos, neg, pi, results, layout, seed, outdir="plots"):
    vis_gen = torch.Generator().manual_seed(VIS_SEED)
    x_pos_gt = sample_gmm(pos, int(N_GT * pi), generator=vis_gen).numpy()
    x_neg_gt = sample_gmm(neg, N_GT - int(N_GT * pi), generator=vis_gen).numpy()
    xmax = (
        max(
            abs(x_pos_gt[:, 0]).max(),
            abs(x_neg_gt[:, 0]).max(),
            abs(x_pos_gt[:, 1]).max(),
            abs(x_neg_gt[:, 1]).max(),
        )
        * 1.15
    )

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    plt.rcParams.update({"font.size": FONT_SIZE})

    sc_ref = None
    for name, row, col in PANEL_ORDER:
        r = results[name]
        x = r["x"].numpy()
        post = r["post"].numpy()

        ax = axes[row, col]
        ax.scatter(x_pos_gt[:, 0], x_pos_gt[:, 1], s=2, alpha=0.2, color="#CCCCCC", zorder=1)
        ax.scatter(x_neg_gt[:, 0], x_neg_gt[:, 1], s=2, alpha=0.2, color="#CCCCCC", zorder=1)
        sc = ax.scatter(x[:, 0], x[:, 1], s=12, alpha=0.7, c=post, cmap="coolwarm", vmin=0, vmax=1, zorder=5)
        sc_ref = sc

        if name in ("Oracle", "PU Steering"):
            ax.set_title(f"{name}", fontsize=FONT_SIZE)
        else:
            str_title = name + r"  ($\eta=1$)"
            ax.set_title(str_title, fontsize=FONT_SIZE)
        ax.set_xlim(-xmax, xmax)
        ax.set_ylim(-xmax, xmax)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.15)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    fig.subplots_adjust(right=0.87, hspace=0.18, wspace=0.12)
    assert sc_ref is not None
    cax = fig.add_axes((0.90, 0.15, 0.025, 0.7))
    cb = fig.colorbar(sc_ref, cax=cax, label=r"true posterior $p(c \mid x_0)$")
    cb.ax.tick_params(labelsize=FONT_SIZE - 1)
    cb.set_label(r"true posterior $p(c \mid x_0)$", fontsize=FONT_SIZE - 1)

    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"gmm_paper_L{layout}s{seed}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    out = os.path.join(outdir, f"gmm_paper_L{layout}s{seed}.pdf")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="plots")
    args = parser.parse_args()

    layout, seed = args.layout, args.seed
    sched = DDPMSchedule(T=T)
    pos, neg, pi = make_ring_gmm(seed=layout)

    print(f"Running GMM paper plot: L{layout}s{seed} ...")

    xo = OracleDiffusion(schedule=sched, neg_x0_gmm=neg, M=M, seed=seed).sample()

    dng = DNGDDPM(schedule=sched, pos_x0_gmm=pos, neg_x0_gmm=neg, class_prior=pi, M=M, lambda0=1, tau=1, pmin=1e-2, lam_max=20, seed=seed)
    xd, _ = dng.sample()

    cfg = NegCFG(schedule=sched, pos_x0_gmm=pos, neg_x0_gmm=neg, class_prior=pi, M=M, lam=1.0, seed=seed)
    xc = cfg.sample()

    pu = KernelPULogistic(sigma=None, sigma_scale=0.2, lam=1e-4, n_centers_pos=128, n_centers_u=64, max_iter=100)
    pusmc = PUNegGuideDDPMSMC(
        schedule=sched,
        pos_x0_gmm=pos,
        neg_x0_gmm=neg,
        class_prior=pi,
        pu=pu,
        M=M,
        pu_class_prior=pi,
        ess_target=0.6,
        seed=seed,
        proposal="dng",
        dng_lambda0=0.6,
        dng_tau=1.0,
        dng_lam_max=4.0,
        wq_ratio=False,
        tilt_max=3.0,
        resample_at="40",
    )
    xp, _ = pusmc.sample()

    results = {
        "Oracle": {"x": xo, "post": compute_post(xo, pos, neg, pi)},
        "PU Steering": {"x": xp, "post": compute_post(xp, pos, neg, pi)},
        "NegCFG": {"x": xc, "post": compute_post(xc, pos, neg, pi)},
        "DNG": {"x": xd, "post": compute_post(xd, pos, neg, pi)},
    }

    plot_paper(pos, neg, pi, results, layout, seed, args.outdir)
    print("Done.")


if __name__ == "__main__":
    main()
