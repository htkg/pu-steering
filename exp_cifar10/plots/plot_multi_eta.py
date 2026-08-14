"""Plot Rate-quality trade-offs for multiple SafeDenoiser base strengths.

Reads ``summary_short.json`` from ``results/``.
SD curve in blue, Ours(eta=0.2) in green, Ours(eta=0.3) in red, Ours(eta=0.5) in purple.

Usage:
  python plots/plot_multi_eta.py
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adjustText import adjust_text

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(SCRIPT_DIR)  # exp_cifar10
RESULTS = os.path.join(BASE, "results")
OUTDIR = os.path.join(SCRIPT_DIR, "plots_multi_eta")
os.makedirs(OUTDIR, exist_ok=True)

classes = {
    "Plane": ("plane", "0.001"),
    "Cat": ("cat", "0.001"),
    "Frog": ("frog", "1.0"),
    "Ship": ("ship", "1.0"),
    "Vehicles": ("vehicles", "1.0"),
}
scales = ["0.0", "0.2", "0.3", "0.5", "0.7", "1.0"]
lams = ["0.5", "1.0", "1.5", "2.0"]
metrics = ["FID", "KL", "MPD"]

SD_COLOR = "#1f77b4"
OURS_COLORS = {"0.2": "#2ca02c", "0.3": "#d62728", "0.5": "#9467bd"}
OURS_LABELS = {"0.2": "Ours ($\\eta$=0.2)", "0.3": "Ours ($\\eta$=0.3)", "0.5": "Ours ($\\eta$=0.5)"}


def load(path: str) -> dict | None:
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def get_sd_data(cls_key: str) -> dict:
    d: dict[str, list] = {"rate": [], "FID": [], "KL": [], "MPD": [], "scale": []}
    for sc in scales:
        s = load(os.path.join(RESULTS, f"{cls_key}_sd_{sc}", "summary_short.json"))
        if s is None or s.get("fid") is None:
            continue
        d["scale"].append(float(sc))
        d["rate"].append(round(s["target_rate"] * 100, 1))
        d["FID"].append(round(s["fid"], 2))
        d["KL"].append(round(s["kl"], 3))
        d["MPD"].append(round(s["mpd"], 4))
    return d


def get_ours_data(cls_key: str, sd_scale: str, sd_tag: str) -> dict:
    d: dict[str, list] = {"lam": [0.0], "rate": [], "FID": [], "KL": [], "MPD": []}
    # lam=0 is the SD baseline at this eta
    sd_base = load(os.path.join(RESULTS, f"{cls_key}_sd_{sd_scale}", "summary_short.json"))
    if sd_base and sd_base.get("fid") is not None:
        d["rate"].append(round(sd_base["target_rate"] * 100, 1))
        d["FID"].append(round(sd_base["fid"], 2))
        d["KL"].append(round(sd_base["kl"], 3))
        d["MPD"].append(round(sd_base["mpd"], 4))
    else:
        d["rate"].append(None)
        d["FID"].append(None)
        d["KL"].append(None)
        d["MPD"].append(None)

    for lam in lams:
        if sd_scale == "0.3":
            path = os.path.join(RESULTS, f"{cls_key}_ours_lam{lam}", "summary_short.json")
        else:
            path = os.path.join(RESULTS, f"{cls_key}_ours_lam{lam}_sd{sd_tag}", "summary_short.json")
        s = load(path)
        d["lam"].append(float(lam))
        if s is None or s.get("fid") is None:
            d["rate"].append(None)
            d["FID"].append(None)
            d["KL"].append(None)
            d["MPD"].append(None)
        else:
            d["rate"].append(round(s["target_rate"] * 100, 1))
            d["FID"].append(round(s["fid"], 2))
            d["KL"].append(round(s["kl"], 3))
            d["MPD"].append(round(s["mpd"], 4))
    return d


def filter_none(rates: list, vals: list, labels: list) -> tuple[list, list, list]:
    r, v, l = [], [], []
    for ri, vi, li in zip(rates, vals, labels):
        if ri is not None and vi is not None:
            r.append(ri)
            v.append(vi)
            l.append(li)
    return r, v, l


def plot_class(cls_name: str, cls_key: str, pu_lam: str) -> None:
    sd = get_sd_data(cls_key)
    ours_data = {}
    for sd_scale, sd_tag in [("0.2", "02"), ("0.3", "03"), ("0.5", "05")]:
        ours_data[sd_scale] = get_ours_data(cls_key, sd_scale, sd_tag)

    fig, axes = plt.subplots(1, len(metrics), figsize=(15, 3.5))

    for ax, metric in zip(axes, metrics):
        texts = []
        lines = []

        # SD curve
        (line_sd,) = ax.plot(sd["rate"], sd[metric], "o-", color=SD_COLOR, label="SafeDenoiser", markersize=6, linewidth=2, zorder=3)
        lines.append(line_sd)
        for x, y, s in zip(sd["rate"], sd[metric], sd["scale"]):
            texts.append(ax.text(x, y, f"$\\eta$={s}", fontsize=9, color=SD_COLOR, fontweight="medium"))

        # Ours curves per eta
        for sd_scale in ["0.2", "0.3", "0.5"]:
            od = ours_data[sd_scale]
            color = OURS_COLORS[sd_scale]
            label = OURS_LABELS[sd_scale]
            lam_labels = [f"$\\lambda$={l}" if l > 0 else f"$\\eta$={sd_scale}" for l in od["lam"]]
            r, m, ll = filter_none(od["rate"], od[metric], lam_labels)
            if not r:
                continue
            (line_o,) = ax.plot(r, m, "x--", color=color, label=label, markersize=10, linewidth=1.8, markeredgewidth=2, zorder=4)
            lines.append(line_o)
            for x, y, lbl in zip(r, m, ll):
                if lbl.startswith("$\\eta$"):
                    continue
                texts.append(ax.text(x, y, lbl, fontsize=8.5, color=color, fontweight="medium"))

        adjust_text(
            texts,
            ax=ax,
            add_objects=lines,
            force_text=(0.5, 0.8),
            force_objects=(0.8, 1.0),
            expand=(1.5, 1.5),
            lim=500,
            arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.4},
        )

        ax.set_xlabel(f"Target Class ({cls_name}) Rate [%]", fontsize=11)

        if metric == "MPD":
            ax.set_ylabel("average pairwise LPIPS", fontsize=11)
        else:
            ax.set_ylabel(metric, fontsize=11)
        if metric == "KL" and cls_name == "Vehicles":
            ax.legend(fontsize=9, loc="lower left")
        else:
            ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        ax.invert_xaxis()

    plt.tight_layout()
    path = os.path.join(OUTDIR, f"tradeoff_{cls_key}_multi_eta.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def plot_combined() -> None:
    fig, axes = plt.subplots(len(classes), len(metrics), figsize=(5.5 * len(metrics), 5 * len(classes)))
    fig.suptitle("SD vs Ours (multi-$\\eta$): Rate-Quality Tradeoff\n(80 seeds, 10240 samples)", fontsize=14, y=0.99)

    for row, (cls_name, (cls_key, pu_lam)) in enumerate(classes.items()):
        sd = get_sd_data(cls_key)
        ours_data = {}
        for sd_scale, sd_tag in [("0.2", "02"), ("0.3", "03"), ("0.5", "05")]:
            ours_data[sd_scale] = get_ours_data(cls_key, sd_scale, sd_tag)

        for col, metric in enumerate(metrics):
            ax = axes[row, col]
            texts = []
            lines = []

            (line_sd,) = ax.plot(sd["rate"], sd[metric], "o-", color=SD_COLOR, label="SafeDenoiser", markersize=5, linewidth=1.8, zorder=3)
            lines.append(line_sd)
            for x, y, s in zip(sd["rate"], sd[metric], sd["scale"]):
                texts.append(ax.text(x, y, f"$\\eta$={s}", fontsize=8, color=SD_COLOR))

            for sd_scale in ["0.2", "0.3", "0.5"]:
                od = ours_data[sd_scale]
                color = OURS_COLORS[sd_scale]
                label = OURS_LABELS[sd_scale]
                lam_labels = [f"$\\lambda$={l}" if l > 0 else "" for l in od["lam"]]
                r, m, ll = filter_none(od["rate"], od[metric], lam_labels)
                if not r:
                    continue
                (line_o,) = ax.plot(r, m, "x--", color=color, label=label, markersize=6, linewidth=1.5, markeredgewidth=2, zorder=4)
                lines.append(line_o)
                for x, y, lbl in zip(r, m, ll):
                    if not lbl:
                        continue
                    texts.append(ax.text(x, y, lbl, fontsize=7.5, color=color))

            adjust_text(
                texts,
                ax=ax,
                add_objects=lines,
                force_text=(0.5, 0.8),
                force_objects=(0.8, 1.0),
                expand=(1.5, 1.5),
                lim=500,
                arrowprops={"arrowstyle": "-", "color": "gray", "lw": 0.3},
            )

            ax.set_xlabel("Rate (%)", fontsize=10)
            ax.set_ylabel(metric, fontsize=10)
            ax.set_title(f"{cls_name} — {metric} (pu_lam={pu_lam})", fontsize=11, fontweight="medium")
            ax.grid(True, alpha=0.3)
            ax.invert_xaxis()
            if row == 0 and col == 0:
                ax.legend(fontsize=8, loc="best")

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    path = os.path.join(OUTDIR, "tradeoff_all_multi_eta.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def main() -> None:
    for cls_name, (cls_key, pu_lam) in classes.items():
        plot_class(cls_name, cls_key, pu_lam)
    plot_combined()
    print("\nDone.")


if __name__ == "__main__":
    main()
