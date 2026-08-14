"""Plot Rate-quality trade-offs with uniform ``pu_lam`` across all classes.

Reads locally generated ``results/*/summary.json`` files. Generates two sets of plots:
  - pu_lam=0.001: per-class 1x2 + combined 4x2
  - pu_lam=1.0:   per-class 1x2 + combined 4x2

Usage:
  python plots/uniform_pulam/plot_tradeoff_uniform_pulam.py
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.axes import Axes

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "..", "results")

classes = ["Plane", "Frog", "Cat", "Ship"]
cls_keys = {"Plane": "plane", "Frog": "frog", "Cat": "cat", "Ship": "ship"}
scales = ["0.0", "0.2", "0.3", "0.5", "0.7", "1.0"]
lams = ["0.5", "1.0", "1.5", "2.0"]
metrics = ["FID", "KL", "MPD"]


def load_summary(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_sd_data() -> dict:
    sd = {}
    for cls_name, cls_key in cls_keys.items():
        d: dict[str, list] = {"scale": [], "rate": [], "FID": [], "KL": [], "MPD": []}
        for sc in scales:
            s = load_summary(os.path.join(RESULTS_DIR, f"{cls_key}_sd_{sc}", "summary.json"))
            if s is None:
                print(f"WARNING: missing {cls_key}_sd_{sc}", file=sys.stderr)
                continue
            d["scale"].append(float(sc))
            d["rate"].append(round(s["target_rate"] * 100, 1))
            d["FID"].append(round(s["fid"], 2))
            d["KL"].append(round(s["kl"], 3))
            d["MPD"].append(round(s["mpd"], 4))
        sd[cls_name] = d
    return sd


def load_ours_data(pu_lam: str) -> dict:
    default_pl = {"plane": "0.001", "cat": "0.001", "frog": "1.0", "ship": "1.0"}
    suffix_map = {"plane": "pl1.0", "cat": "pl1.0", "frog": "pl0.001", "ship": "pl0.001"}

    ours = {}
    for cls_name, cls_key in cls_keys.items():
        sd03 = load_summary(os.path.join(RESULTS_DIR, f"{cls_key}_sd_0.3", "summary.json"))
        d: dict[str, list] = {"lam": [0.0], "rate": [], "FID": [], "KL": [], "MPD": []}
        if sd03:
            d["rate"].append(round(sd03["target_rate"] * 100, 1))
            d["FID"].append(round(sd03["fid"], 2))
            d["KL"].append(round(sd03["kl"], 3))
            d["MPD"].append(round(sd03["mpd"], 4))

        for lam in lams:
            if pu_lam == default_pl[cls_key]:
                path = os.path.join(RESULTS_DIR, f"{cls_key}_ours_lam{lam}", "summary.json")
            else:
                path = os.path.join(RESULTS_DIR, f"{cls_key}_ours_lam{lam}_{suffix_map[cls_key]}", "summary.json")
            s = load_summary(path)
            d["lam"].append(float(lam))
            if s is None:
                print(f"WARNING: missing {path}", file=sys.stderr)
                d["rate"].append(None)
                d["FID"].append(None)
                d["KL"].append(None)
                d["MPD"].append(None)
            else:
                d["rate"].append(round(s["target_rate"] * 100, 1))
                d["FID"].append(round(s["fid"], 2))
                d["KL"].append(round(s["kl"], 3))
                d["MPD"].append(round(s["mpd"], 4))
        ours[cls_name] = d
    return ours


def _add_annotations(ax: Axes, xs: list, ys: list, labels: list, color: str, fontsize: float) -> list:
    """Add text annotations and return the Text objects for adjust_text."""
    texts = []
    for x, y, label in zip(xs, ys, labels):
        if x is None or y is None:
            continue
        texts.append(ax.text(x, y, label, fontsize=fontsize, color=color, fontweight="medium"))
    return texts


def plot_per_class(sd: dict, ours: dict, pu_lam: str, outdir: str) -> None:
    for cls in classes:
        fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 4.5))
        fig.suptitle(f"{cls}: Rate vs Quality (80 seeds, pu_lam={pu_lam})", fontsize=14)

        for ax, metric in zip(axes, metrics):
            ours_r = [v for v in ours[cls]["rate"] if v is not None]
            ours_m = [v for v in ours[cls][metric] if v is not None]

            (line_sd,) = ax.plot(sd[cls]["rate"], sd[cls][metric], "o-", color="#1f77b4", label="SafeDenoiser", markersize=7, linewidth=2)
            (line_ours,) = ax.plot(ours_r, ours_m, "x--", color="#d62728", label="Ours", markersize=9, linewidth=2, markeredgewidth=2.5)

            texts = []
            for x, y, s in zip(sd[cls]["rate"], sd[cls][metric], sd[cls]["scale"]):
                texts.append(ax.text(x, y, f"$\\eta$={s}", fontsize=8.5, color="#1f77b4", fontweight="medium"))
            for x, y, lam in zip(ours[cls]["rate"], ours[cls][metric], ours[cls]["lam"]):
                if x is None or y is None or lam == 0:
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
                arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.5},
            )

            ax.set_xlabel("Target Class Rate (%)", fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(f"Rate vs {metric}", fontsize=12)
            ax.legend(fontsize=10, loc="best")
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()

        plt.tight_layout()
        path = os.path.join(outdir, f"tradeoff_{cls.lower()}_pl{pu_lam}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {path}")


def plot_combined(sd: dict, ours: dict, pu_lam: str, outdir: str) -> None:
    fig, axes = plt.subplots(4, len(metrics), figsize=(5.5 * len(metrics), 20))
    fig.suptitle(f"SafeDenoiser vs Ours: Rate-Quality Tradeoff\n(80 seeds, 10240 samples, pu_lam={pu_lam} for all classes)", fontsize=15, y=0.99)

    for row, cls in enumerate(classes):
        for col, metric in enumerate(metrics):
            ax = axes[row, col]

            ours_r = [v for v in ours[cls]["rate"] if v is not None]
            ours_m = [v for v in ours[cls][metric] if v is not None]

            (line_sd,) = ax.plot(sd[cls]["rate"], sd[cls][metric], "o-", color="#1f77b4", label="SafeDenoiser", markersize=6, linewidth=1.8)
            (line_ours,) = ax.plot(ours_r, ours_m, "x--", color="#d62728", label="Ours", markersize=7, linewidth=1.8, markeredgewidth=2)

            texts = []
            for x, y, s in zip(sd[cls]["rate"], sd[cls][metric], sd[cls]["scale"]):
                texts.append(ax.text(x, y, f"$\\eta$={s}", fontsize=7.5, color="#1f77b4", fontweight="medium"))
            for i, (x, y, lam) in enumerate(zip(ours[cls]["rate"], ours[cls][metric], ours[cls]["lam"])):
                if x is None or y is None or lam == 0:
                    continue
                texts.append(ax.text(x, y, f"$\\lambda$={lam}", fontsize=7.5, color="#d62728", fontweight="medium"))

            adjust_text(
                texts,
                ax=ax,
                add_objects=[line_sd, line_ours],
                force_text=(0.5, 0.8),
                force_objects=(0.8, 1.0),
                expand=(1.5, 1.5),
                lim=500,
                arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.5},
            )

            ax.set_xlabel("Rate (%)", fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.set_title(f"{cls} — {metric}", fontsize=11, fontweight="medium")
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()
            if row == 0 and col == 0:
                ax.legend(fontsize=9)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    path = os.path.join(outdir, f"tradeoff_all_pl{pu_lam}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    outdir = SCRIPT_DIR
    os.makedirs(outdir, exist_ok=True)

    sd = load_sd_data()

    for pu_lam in ["0.001", "1.0"]:
        print(f"\n=== pu_lam={pu_lam} ===")
        ours = load_ours_data(pu_lam)
        plot_per_class(sd, ours, pu_lam, outdir)
        plot_combined(sd, ours, pu_lam, outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
