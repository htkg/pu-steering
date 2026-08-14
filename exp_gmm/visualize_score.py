"""Visualize Case 1 (true-score) results: comparison + classifier snapshots.

Usage:
    uv run python visualize_score.py --mode OFF
    uv run python visualize_score.py --mode ON
    uv run python visualize_score.py --mode ON_TILT
    uv run python visualize_score.py --mode OFF --layout 0 --seed 0
"""

import argparse
import os
from typing import Any

os.environ["TQDM_DISABLE"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import torch
from gmm_ddpm import DDPMSchedule, GaussianMixture, sample_gmm
from gmm_ring import compute_post, make_ring_gmm
from kernel_model import KernelPULogistic
from sampler_score import DNGDDPM, NegCFG, PUNegGuideDDPMSMC

T = 100
M = 1024
N_GT = 3000
SNAPSHOT_TIMES = [99, 75, 50, 25, 10, 1, 0]
VIS_SEED = 42


def plot_comparison(pos: GaussianMixture, neg: GaussianMixture, pi: float, results: dict, mode: str, layout: int, seed: int, outdir: str = "plots"):
    """2-row comparison: top=oracle posterior, bottom=method-specific coloring."""
    methods = list(results.keys())
    ncols = len(methods)
    fig, axes = plt.subplots(2, ncols, figsize=(5.5 * ncols, 10))

    vis_gen = torch.Generator().manual_seed(VIS_SEED)
    x_pos_gt = sample_gmm(pos, int(N_GT * pi), generator=vis_gen).numpy()
    x_neg_gt = sample_gmm(neg, N_GT - int(N_GT * pi), generator=vis_gen).numpy()
    xmax = max(abs(x_pos_gt[:, 0]).max(), abs(x_neg_gt[:, 0]).max(), abs(x_pos_gt[:, 1]).max(), abs(x_neg_gt[:, 1]).max()) * 1.15

    for col, name in enumerate(methods):
        r = results[name]
        x = r["x"].numpy()
        post = r["post"].numpy()

        for row in range(2):
            ax = axes[row, col]
            ax.scatter(x_pos_gt[:, 0], x_pos_gt[:, 1], s=2, alpha=0.08, color="#CCCCCC", zorder=1)
            ax.scatter(x_neg_gt[:, 0], x_neg_gt[:, 1], s=2, alpha=0.08, color="#CCCCCC", zorder=1)

            if row == 0:
                sc = ax.scatter(x[:, 0], x[:, 1], s=12, alpha=0.7, c=post, cmap="coolwarm", vmin=0, vmax=1, zorder=5)
                pm = post.mean()
                leak = (post > 0.5).mean()
                title = f"{name}\npm={pm:.4f} leak={leak:.4f}"
                if "w" in r:
                    wn = r["w"].numpy()
                    wn = wn / (wn.sum() + 1e-12)
                    w_pm = (wn * post).sum()
                    title += f"\nw_pm={w_pm:.4f}"
                ax.set_title(title, fontsize=10)
                if col == ncols - 1:
                    plt.colorbar(sc, ax=ax, shrink=0.7, label="P(pos|x)")
            else:
                if name == "GT":
                    ax.scatter(x_neg_gt[:, 0], x_neg_gt[:, 1], s=6, alpha=0.3, color="tab:blue", zorder=2)
                    ax.scatter(x_pos_gt[:, 0], x_pos_gt[:, 1], s=6, alpha=0.3, color="tab:red", zorder=3)
                    ax.set_title("pos (red) / neg (blue)", fontsize=10)
                elif "w" in r:
                    wn = r["w"].numpy()
                    wn = wn / (wn.sum() + 1e-12)
                    sc = ax.scatter(x[:, 0], x[:, 1], s=12, alpha=0.7, c=wn, cmap="viridis", zorder=5)
                    ess = 1.0 / ((wn**2).sum() + 1e-12)
                    ax.set_title(f"weights (ESS={ess:.0f})", fontsize=10)
                    plt.colorbar(sc, ax=ax, shrink=0.7, label="weight")
                else:
                    sc = ax.scatter(x[:, 0], x[:, 1], s=12, alpha=0.7, c=post, cmap="coolwarm", vmin=0, vmax=1, zorder=5)
                    ax.set_title("oracle posterior", fontsize=10)

            ax.set_xlim(-xmax, xmax)
            ax.set_ylim(-xmax, xmax)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.15)

    fig.suptitle(f"Case 1: True-score — Resample {mode} — L{layout}s{seed}", fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(outdir, f"score_{mode}_comparison_L{layout}s{seed}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_classifier_snapshots(pos: GaussianMixture, neg: GaussianMixture, pi: float, snapshots: list, mode: str, layout: int, seed: int, outdir: str = "plots"):
    """Classifier snapshots at selected timesteps."""
    times = [s for s in snapshots if s["t"] in SNAPSHOT_TIMES]
    if not times:
        return

    nrows = len(times)
    fig, axes = plt.subplots(nrows, 4, figsize=(20, 4.2 * nrows))
    if nrows == 1:
        axes = axes[np.newaxis, :]

    vis_gen = torch.Generator().manual_seed(VIS_SEED)
    x_pos_gt = sample_gmm(pos, int(N_GT * pi), generator=vis_gen).numpy()
    x_neg_gt = sample_gmm(neg, N_GT - int(N_GT * pi), generator=vis_gen).numpy()
    xmax = max(abs(x_pos_gt[:, 0]).max(), abs(x_neg_gt[:, 0]).max(), abs(x_pos_gt[:, 1]).max(), abs(x_neg_gt[:, 1]).max()) * 1.15

    for row, s in enumerate(times):
        t = s["t"]
        x0h = s["x0hat"].numpy()
        pu_p = s["pu_proba"].numpy()
        w_q = s["w_q"].numpy()
        lam = s["lam_t"]

        oracle_post = compute_post(s["x0hat"], pos, neg, pi).numpy()
        wn = w_q / (w_q.sum() + 1e-12)
        ess = 1.0 / ((wn**2).sum() + 1e-12)

        bg_kwargs = {"s": 2, "alpha": 0.08, "color": "#CCCCCC", "zorder": 1}

        ax = axes[row, 0]
        ax.scatter(x_pos_gt[:, 0], x_pos_gt[:, 1], **bg_kwargs)
        ax.scatter(x_neg_gt[:, 0], x_neg_gt[:, 1], **bg_kwargs)
        sc = ax.scatter(x0h[:, 0], x0h[:, 1], s=10, alpha=0.7, c=pu_p, cmap="coolwarm", vmin=0, vmax=1, zorder=5)
        ax.set_title(f"t={t}  PU pred (mean={pu_p.mean():.3f})", fontsize=10)
        plt.colorbar(sc, ax=ax, shrink=0.7)

        ax = axes[row, 1]
        ax.scatter(x_pos_gt[:, 0], x_pos_gt[:, 1], **bg_kwargs)
        ax.scatter(x_neg_gt[:, 0], x_neg_gt[:, 1], **bg_kwargs)
        sc = ax.scatter(x0h[:, 0], x0h[:, 1], s=10, alpha=0.7, c=oracle_post, cmap="coolwarm", vmin=0, vmax=1, zorder=5)
        ax.set_title(f"Oracle post (mean={oracle_post.mean():.3f})", fontsize=10)
        plt.colorbar(sc, ax=ax, shrink=0.7)

        ax = axes[row, 2]
        ax.scatter(oracle_post, pu_p, s=3, alpha=0.3, zorder=2)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, zorder=1)
        corr = np.corrcoef(oracle_post, pu_p)[0, 1]
        ax.set_title(f"Calibration (rho={corr:.3f})", fontsize=10)
        ax.set_xlabel("Oracle posterior")
        ax.set_ylabel("PU prediction")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")

        ax = axes[row, 3]
        ax.scatter(x_pos_gt[:, 0], x_pos_gt[:, 1], **bg_kwargs)
        ax.scatter(x_neg_gt[:, 0], x_neg_gt[:, 1], **bg_kwargs)
        sc = ax.scatter(x0h[:, 0], x0h[:, 1], s=10, alpha=0.7, c=wn, cmap="viridis", zorder=5)
        ax.set_title(f"Weights (ESS={ess:.0f}, lam={lam:.3f})", fontsize=10)
        plt.colorbar(sc, ax=ax, shrink=0.7)

        for c in range(4):
            if c != 2:
                axes[row, c].set_xlim(-xmax, xmax)
                axes[row, c].set_ylim(-xmax, xmax)
                axes[row, c].set_aspect("equal")
            axes[row, c].grid(True, alpha=0.15)

    fig.suptitle(f"Case 1: Classifier Snapshots — Resample {mode} — L{layout}s{seed}", fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(outdir, f"score_{mode}_classifier_L{layout}s{seed}.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["OFF", "ON", "ON_TILT"], required=True)
    parser.add_argument("--layout", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="plots")
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    mode, layout, seed = args.mode, args.layout, args.seed
    sched = DDPMSchedule(T=T)
    pos, neg, pi = make_ring_gmm(seed=layout)
    results = {}

    gt_gen = torch.Generator().manual_seed(VIS_SEED)
    x_pos = sample_gmm(pos, int(M * pi), generator=gt_gen)
    x_neg = sample_gmm(neg, M - int(M * pi), generator=gt_gen)
    x_gt = torch.cat([x_pos, x_neg], dim=0)
    results["GT"] = {"x": x_gt, "post": compute_post(x_gt, pos, neg, pi)}

    from gmm_ddpm import OracleDiffusion

    xo = OracleDiffusion(schedule=sched, neg_x0_gmm=neg, M=M, seed=seed).sample()
    results["Oracle"] = {"x": xo, "post": compute_post(xo, pos, neg, pi)}

    dng = DNGDDPM(schedule=sched, pos_x0_gmm=pos, neg_x0_gmm=neg, class_prior=pi, M=M, lambda0=1, tau=1, pmin=1e-2, lam_max=20, seed=seed)
    xd, _ = dng.sample()
    results["DNG"] = {"x": xd, "post": compute_post(xd, pos, neg, pi)}

    cfg = NegCFG(schedule=sched, pos_x0_gmm=pos, neg_x0_gmm=neg, class_prior=pi, M=M, lam=1.0, seed=seed)
    xc = cfg.sample()
    results["NegCFG(1.0)"] = {"x": xc, "post": compute_post(xc, pos, neg, pi)}

    pu = KernelPULogistic(sigma=None, sigma_scale=0.2, lam=1e-4, n_centers_pos=128, n_centers_u=64, max_iter=100)
    common: dict[str, Any] = {
        "schedule": sched,
        "pos_x0_gmm": pos,
        "neg_x0_gmm": neg,
        "class_prior": pi,
        "pu": pu,
        "M": M,
        "pu_class_prior": pi,
        "ess_target": 0.6,
        "seed": seed,
    }
    if mode == "OFF":
        pusmc = PUNegGuideDDPMSMC(
            **common,
            grad_clip=0.5,
            tilt_max=3.0,
            tilt_start=0.6,
        )
    elif mode == "ON":
        pusmc = PUNegGuideDDPMSMC(
            **common,
            proposal="dng",
            dng_lambda0=0.6,
            dng_tau=1.0,
            dng_lam_max=4.0,
            lam_fixed=2.0,
            tilt_max=0.0,
            resample_at="40",
        )
    else:
        pusmc = PUNegGuideDDPMSMC(
            **common,
            proposal="dng",
            dng_lambda0=0.6,
            dng_tau=1.0,
            dng_lam_max=4.0,
            wq_ratio=False,
            tilt_max=3.0,
            resample_at="40",
        )

    snapshots = []

    def callback(**kwargs):
        snapshots.append(kwargs)

    xp, wp = pusmc.sample(snapshot_callback=callback)
    label = f"Ours ({mode})"
    results[label] = {"x": xp, "post": compute_post(xp, pos, neg, pi), "w": wp}

    print(f"Running visualization for Case 1 {mode}, L{layout}s{seed}...")
    plot_comparison(pos, neg, pi, results, mode, layout, seed, outdir)
    plot_classifier_snapshots(pos, neg, pi, snapshots, mode, layout, seed, outdir)
    print("Done.")


if __name__ == "__main__":
    main()
