"""Visualize Case 2 (reference-based) results: comparison + classifier snapshots.

Usage:
    uv run python visualize_ref.py --mode OFF --proposal grad
    uv run python visualize_ref.py --mode ON --proposal grad
    uv run python visualize_ref.py --mode OFF --proposal sd
    uv run python visualize_ref.py --mode ON --proposal sd
    uv run python visualize_ref.py --mode ON --proposal grad --layout 0 --seed 0
"""

import argparse
import os

os.environ["TQDM_DISABLE"] = "1"

import matplotlib.pyplot as plt
import numpy as np
import torch

from gmm_ddpm import DDPMSchedule, GaussianMixture, OracleDiffusion, sample_gmm
from gmm_ring import compute_post, make_ring_gmm
from kernel_model import KernelPULogistic
from sampler_ref import PUNegGuideDDPMSMC, SafeDenoiserDDPM, SDPropPUSMC

T = 100
M = 1024
N_GT = 3000
SNAPSHOT_TIMES = [99, 75, 50, 25, 10, 1, 0]
VIS_SEED = 42


def plot_comparison(
    pos: GaussianMixture, neg: GaussianMixture, pi: float, results: dict, mode: str, proposal: str, layout: int, seed: int, outdir: str = "plots"
):
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

    fig.suptitle(f"Case 2: Reference-based — Resample {mode} — {proposal} — L{layout}s{seed}", fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(outdir, f"ref_{mode}_{proposal}_comparison_L{layout}s{seed}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_classifier_snapshots(
    pos: GaussianMixture, neg: GaussianMixture, pi: float, snapshots: list, mode: str, proposal: str, layout: int, seed: int, outdir: str = "plots"
):
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

    fig.suptitle(f"Case 2: Classifier Snapshots — {mode} — {proposal} — L{layout}s{seed}", fontsize=14, y=1.01)
    plt.tight_layout()
    out = os.path.join(outdir, f"ref_{mode}_{proposal}_classifier_L{layout}s{seed}.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["OFF", "ON"], required=True)
    parser.add_argument("--proposal", choices=["grad", "sd"], required=True)
    parser.add_argument("--layout", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="plots")
    args = parser.parse_args()

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    mode, proposal, layout, seed = args.mode, args.proposal, args.layout, args.seed
    sched = DDPMSchedule(T=T)
    pos, neg, pi = make_ring_gmm(seed=layout)
    results = {}

    gt_gen = torch.Generator().manual_seed(VIS_SEED)
    x_pos = sample_gmm(pos, int(M * pi), generator=gt_gen)
    x_neg = sample_gmm(neg, M - int(M * pi), generator=gt_gen)
    x_gt = torch.cat([x_pos, x_neg], dim=0)
    results["GT"] = {"x": x_gt, "post": compute_post(x_gt, pos, neg, pi)}

    xo = OracleDiffusion(schedule=sched, neg_x0_gmm=neg, M=M, seed=seed).sample()
    results["Oracle"] = {"x": xo, "post": compute_post(xo, pos, neg, pi)}

    for eta in [30, 50] if mode == "ON" else [50, 100]:
        sd = SafeDenoiserDDPM(schedule=sched, pos_x0_gmm=pos, neg_x0_gmm=neg, class_prior=pi, M=M, N_ref=512, eta=eta, t_start=T, t_stop=5, seed=seed)
        xsd = sd.sample()
        results[f"SD(eta={eta})"] = {"x": xsd, "post": compute_post(xsd, pos, neg, pi)}

    snapshots = []

    def callback(**kwargs):
        snapshots.append(kwargs)

    if proposal == "grad":
        if mode == "OFF":
            pu = KernelPULogistic(sigma=None, sigma_scale=0.3, lam=1e-4, n_centers_pos=128, n_centers_u=64, max_iter=100)
            sampler = PUNegGuideDDPMSMC(
                schedule=sched,
                pos_x0_gmm=pos,
                neg_x0_gmm=neg,
                class_prior=pi,
                pu=pu,
                N_pos=512,
                M=M,
                pu_class_prior=0.05,
                grad_clip=3.0,
                ess_target=0.9,
                ess_resample=0.9,
                resample_stop=100,
                lam_schedule="adaptive",
                lam_max=1.0,
                seed=seed,
            )
        else:
            pu = KernelPULogistic(sigma=None, sigma_scale=0.3, lam=1e-2, n_centers_pos=128, n_centers_u=64, max_iter=100)
            sampler = PUNegGuideDDPMSMC(
                schedule=sched,
                pos_x0_gmm=pos,
                neg_x0_gmm=neg,
                class_prior=pi,
                pu=pu,
                N_pos=512,
                M=M,
                pu_class_prior=0.25,
                grad_clip=3.0,
                ess_target=0.9,
                ess_resample=0.95,
                resample_stop=20,
                guidance_stop=20,
                resample_method="systematic_sorted",
                lam_schedule="cos",
                lam_max=2.0,
                seed=seed,
            )
        label = f"Grad ({mode})"
    else:
        if mode == "OFF":
            pu = KernelPULogistic(sigma=None, sigma_scale=0.3, lam=1e-4, n_centers_pos=128, n_centers_u=64, max_iter=100)
            sampler = SDPropPUSMC(
                schedule=sched,
                pos_x0_gmm=pos,
                neg_x0_gmm=neg,
                class_prior=pi,
                pu=pu,
                N_pos=512,
                M=M,
                eta=30,
                sd_t_stop=0,
                pu_class_prior=0.18,
                correction_clip=0.7,
                lam_schedule="linear",
                lam_target=1.0,
                ess_resample=0.5,
                resample_stop=100,
                seed=seed,
            )
        else:
            pu = KernelPULogistic(sigma=None, sigma_scale=0.3, lam=1e-2, n_centers_pos=128, n_centers_u=64, max_iter=100)
            sampler = SDPropPUSMC(
                schedule=sched,
                pos_x0_gmm=pos,
                neg_x0_gmm=neg,
                class_prior=pi,
                pu=pu,
                N_pos=512,
                M=M,
                eta=30,
                sd_t_stop=20,
                pu_class_prior=0.25,
                correction_clip=0.0,
                lam_schedule="cos",
                lam_target=2.0,
                ess_resample=0.97,
                resample_method="systematic_sorted",
                resample_stop=20,
                seed=seed,
            )
        label = f"SDProp ({mode})"

    xp, wp = sampler.sample(snapshot_callback=callback)
    results[label] = {"x": xp, "post": compute_post(xp, pos, neg, pi), "w": wp}

    print(f"Running visualization for Case 2 {mode} {proposal}, L{layout}s{seed}...")
    plot_comparison(pos, neg, pi, results, mode, proposal, layout, seed, outdir)
    plot_classifier_snapshots(pos, neg, pi, snapshots, mode, proposal, layout, seed, outdir)
    print("Done.")


if __name__ == "__main__":
    main()
