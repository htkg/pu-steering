"""Plot the archived 80-seed Rate-FID/KL/MPD trade-off curves.

Generates:
  - Single-class: per-class 1x3 + combined 4x3 grid (80 seeds)
  - Multi-class: vehicles 1x3 (80 seeds)

The values are embedded below and are not read from ``results/``.

Usage:
  python plot_tradeoff.py
  python plot_tradeoff.py --outdir plots
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text

# SafeDenoiser sweep (80 seeds)
SD = {
    "Plane": {
        "scale": [0.0, 0.2, 0.3, 0.5, 0.7, 1.0],
        "rate": [8.7, 5.9, 4.8, 2.9, 2.2, 2.0],
        "FID": [7.02, 12.06, 20.75, 45.96, 73.68, 110.36],
        "KL": [0.096, 0.089, 0.127, 0.279, 0.459, 0.665],
        "MPD": [0.222, 0.21, 0.2, 0.175, 0.161, 0.148],
    },
    "Frog": {
        "scale": [0.0, 0.2, 0.3, 0.5, 0.7, 1.0],
        "rate": [9.7, 4.1, 2.5, 0.9, 0.3, 0.1],
        "FID": [7.64, 21.02, 41.53, 93.59, 146.34, 206.8],
        "KL": [0.107, 0.104, 0.181, 0.503, 1.086, 2.236],
        "MPD": [0.222, 0.2, 0.183, 0.162, 0.149, 0.135],
    },
    "Cat": {
        "scale": [0.0, 0.2, 0.3, 0.5, 0.7, 1.0],
        "rate": [8.3, 7.6, 7.2, 6.1, 4.7, 2.4],
        "FID": [6.21, 12.24, 20.63, 44.53, 72.8, 114.77],
        "KL": [0.09, 0.113, 0.15, 0.282, 0.491, 0.885],
        "MPD": [0.222, 0.215, 0.207, 0.183, 0.167, 0.149],
    },
    "Ship": {
        "scale": [0.0, 0.2, 0.3, 0.5, 0.7, 1.0],
        "rate": [11.3, 5.6, 3.6, 1.8, 1.0, 0.7],
        "FID": [8.3, 13.09, 23.68, 55.77, 93.76, 150.82],
        "KL": [0.125, 0.098, 0.13, 0.296, 0.494, 0.762],
        "MPD": [0.222, 0.208, 0.194, 0.169, 0.154, 0.139],
    },
}

# Ours class-specific best reg (80 seeds)
# Plane: pu_lam=0.001, Frog: pu_lam=1.0, Cat: pu_lam=0.001, Ship: pu_lam=1.0
Ours = {
    "Plane": {
        "lam": [0, 0.5, 1.0, 1.5, 2.0],
        "rate": [4.8, 3.9, 3.3, 3.5, 2.5],
        "FID": [20.75, 22.61, 24.74, 24.65, 26.06],
        "KL": [0.127, 0.139, 0.167, 0.148, 0.173],
        "MPD": [0.2, 0.204, 0.202, 0.197, 0.202],
    },
    "Frog": {
        "lam": [0, 0.5, 1.0, 1.5, 2.0],
        "rate": [2.5, 2.3, 1.7, 2.0, 1.1],
        "FID": [41.53, 43.28, 46.74, 49.25, 50.48],
        "KL": [0.181, 0.196, 0.219, 0.249, 0.249],
        "MPD": [0.183, 0.184, 0.178, 0.183, 0.184],
    },
    "Cat": {
        "lam": [0, 0.5, 1.0, 1.5, 2.0],
        "rate": [7.2, 6.6, 6.1, 5.7, 5.5],
        "FID": [20.63, 21.97, 23.5, 24.95, 26.66],
        "KL": [0.15, 0.155, 0.173, 0.163, 0.197],
        "MPD": [0.207, 0.199, 0.207, 0.203, 0.21],
    },
    "Ship": {
        "lam": [0, 0.5, 1.0, 1.5, 2.0],
        "rate": [3.6, 3.5, 3.0, 2.5, 1.5],
        "FID": [23.68, 24.2, 26.05, 27.44, 30.43],
        "KL": [0.13, 0.136, 0.143, 0.133, 0.176],
        "MPD": [0.194, 0.195, 0.196, 0.196, 0.193],
    },
}

# Vehicles multi-class (80 seeds)
SD_vehicles = {
    "scale": [0.0, 0.2, 0.3, 0.5, 0.7, 1.0],
    "rate": [41.4, 31.9, 27.6, 20.7, 15.7, 13.5],
    "FID": [30.93, 24.01, 25.8, 39.27, 60.35, 92.72],
    "KL": [0.538, 0.391, 0.346, 0.309, 0.342, 0.525],
    "MPD": [0.2221, 0.217, 0.2124, 0.1898, 0.1743, 0.1579],
}

Ours_vehicles = {
    "lam": [0.5, 1.0, 1.5, 2.0],
    "label": ["$\\lambda$=0.5", "$\\lambda$=1.0", "$\\lambda$=1.5", "$\\lambda$=2.0"],
    "rate": [25.9, 19.9, 18.1, 20.9],
    "FID": [25.01, 22.4, 21.86, 24.76],
    "KL": [0.326, 0.256, 0.24, 0.273],
    "MPD": [0.2089, 0.2175, 0.2131, 0.2070],
}

classes = ["Plane", "Frog", "Cat", "Ship"]
metrics = ["FID", "KL", "MPD"]


def plot_single_class(outdir: str) -> None:
    """Per-class (1x3) + combined (4x3) tradeoff plots."""

    for cls in classes:
        fig, axes = plt.subplots(1, 3, figsize=(5.5 * 3, 4.5))
        fig.suptitle(f"{cls}: Rate vs Quality Tradeoff (80 seeds, 10240 samples)", fontsize=13)

        for ax, metric in zip(axes, metrics):
            texts = []
            objects = []

            (line_sd,) = ax.plot(SD[cls]["rate"], SD[cls][metric], "o-", color="#1f77b4", label="SafeDenoiser", markersize=6, linewidth=2)
            objects.extend(ax.plot(SD[cls]["rate"], SD[cls][metric], "o", color="#1f77b4", markersize=6, alpha=0))
            for x, y, s in zip(SD[cls]["rate"], SD[cls][metric], SD[cls]["scale"]):
                texts.append(ax.text(x, y, f"$\\eta$={s}", fontsize=9, color="#1f77b4", fontweight="medium"))

            (line_ours,) = ax.plot(Ours[cls]["rate"], Ours[cls][metric], "x--", color="#d62728", label="Ours", markersize=8, linewidth=2, markeredgewidth=2.5)
            for x, y, lam in zip(Ours[cls]["rate"], Ours[cls][metric], Ours[cls]["lam"]):
                if lam == 0:
                    continue
                texts.append(ax.text(x, y, f"$\\lambda$={lam}", fontsize=8.5, color="#d62728", fontweight="medium"))

            adjust_text(
                texts,
                ax=ax,
                add_objects=[line_sd, line_ours],
                force_text=(0.5, 0.8),
                force_objects=(0.8, 1.0),
                expand=(1.5, 1.5),
                lim=500,
                arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.4},
            )

            ax.set_xlabel("Target Class Rate (%)", fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(f"Rate vs {metric}", fontsize=12)
            ax.legend(fontsize=9, loc="best")
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()

        plt.tight_layout()
        path = os.path.join(outdir, f"tradeoff_{cls.lower()}_80s.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")

    # Combined 4x3
    fig, axes = plt.subplots(4, 3, figsize=(5.5 * 3, 20))
    fig.suptitle("SafeDenoiser vs Ours: Rate-Quality Tradeoff\n(80 seeds, 10240 samples, class-specific best regularization)", fontsize=14, y=0.99)

    for row, cls in enumerate(classes):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            texts = []

            (line_sd,) = ax.plot(SD[cls]["rate"], SD[cls][metric], "o-", color="#1f77b4", label="SafeDenoiser", markersize=5, linewidth=1.8)
            for x, y, s in zip(SD[cls]["rate"], SD[cls][metric], SD[cls]["scale"]):
                texts.append(ax.text(x, y, f"$\\eta$={s}", fontsize=8, color="#1f77b4"))

            (line_ours,) = ax.plot(Ours[cls]["rate"], Ours[cls][metric], "x--", color="#d62728", label="Ours", markersize=6, linewidth=1.8, markeredgewidth=2)
            for x, y, lam in zip(Ours[cls]["rate"], Ours[cls][metric], Ours[cls]["lam"]):
                if lam == 0:
                    continue
                texts.append(ax.text(x, y, f"$\\lambda$={lam}", fontsize=7.5, color="#d62728"))

            adjust_text(
                texts,
                ax=ax,
                add_objects=[line_sd, line_ours],
                force_text=(0.5, 0.8),
                force_objects=(0.8, 1.0),
                expand=(1.5, 1.5),
                lim=500,
                arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.3},
            )

            ax.set_xlabel("Rate (%)", fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.set_title(f"{cls} — {metric}", fontsize=11, fontweight="medium")
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()
            if row == 0 and col == 0:
                ax.legend(fontsize=8)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    path = os.path.join(outdir, "tradeoff_all_80s.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_vehicles(outdir: str) -> None:
    """Vehicles multi-class tradeoff plot."""

    fig, axes = plt.subplots(1, 3, figsize=(5.5 * 3, 4.5))
    fig.suptitle(
        "Vehicles Avoidance (plane, car, ship, truck): Rate vs Quality Tradeoff\n(80 seeds, 10240 samples, BCE + denoised features)", fontsize=13, y=1.02
    )

    for ax, metric in zip(axes, metrics):
        texts = []

        (line_sd,) = ax.plot(SD_vehicles["rate"], SD_vehicles[metric], "o-", color="#1f77b4", label="SafeDenoiser", markersize=7, linewidth=2.2)
        for x, y, s in zip(SD_vehicles["rate"], SD_vehicles[metric], SD_vehicles["scale"]):
            texts.append(ax.text(x, y, f"$\\eta$={s}", fontsize=9, color="#1f77b4", fontweight="medium"))

        (line_ours,) = ax.plot(
            Ours_vehicles["rate"], Ours_vehicles[metric], "x--", color="#d62728", label="Ours", markersize=9, linewidth=2.2, markeredgewidth=2.5
        )
        for x, y, label in zip(Ours_vehicles["rate"], Ours_vehicles[metric], Ours_vehicles["label"]):
            texts.append(ax.text(x, y, label, fontsize=9, color="#d62728", fontweight="medium"))

        adjust_text(
            texts,
            ax=ax,
            add_objects=[line_sd, line_ours],
            force_text=(0.5, 0.8),
            force_objects=(0.8, 1.0),
            expand=(1.5, 1.5),
            lim=500,
            arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.4},
        )

        ax.set_xlabel("Vehicles Rate (%)", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f"Rate vs {metric}", fontsize=12)
        ax.legend(fontsize=10, loc="best")
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    plt.tight_layout()
    path = os.path.join(outdir, "tradeoff_vehicles_80s.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    plot_single_class(args.outdir)
    plot_vehicles(args.outdir)
    print("\nDone.")


if __name__ == "__main__":
    main()
