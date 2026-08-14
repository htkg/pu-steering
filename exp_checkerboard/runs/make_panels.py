"""Render figures/row_{i}_*.png from the cached particles/weights (no sampling)."""

import argparse
import math
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from scipy.stats import gaussian_kde

HERE = Path(__file__).resolve().parent
RES_DIR = HERE.parent / "results"
FIG_DIR = HERE.parent / "figures"
RED = "#ff3b54"


def draw_density(ax, X, W, lim, bw):
    """Draw a weighted KDE heatmap."""
    gx = np.linspace(-lim, lim, 260)
    XX, YY = np.meshgrid(gx, gx)
    w = (W / W.sum()).numpy().astype(np.float64)
    Z = gaussian_kde(X.T, bw_method=bw, weights=w)(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    ax.contourf(XX, YY, Z, levels=18, cmap="viridis")
    ax.contour(XX, YY, Z, levels=7, colors="white", linewidths=0.3, alpha=0.30)


def overlay_avoid(ax, geom, engine):
    for g in geom:
        if engine == "gmm":
            mean, cov = g
            vals, vecs = np.linalg.eigh(cov)
            ang = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
            w_, h_ = 2 * 2.0 * np.sqrt(np.maximum(vals, 1e-9))  # 2-sigma
            ax.add_patch(Ellipse(mean, w_, h_, angle=ang, fill=False, ec=RED, lw=1.6, ls=(0, (4, 2)), alpha=0.95))
        else:
            (cx, cy), cell = g
            ax.add_patch(Rectangle((cx * cell, cy * cell), cell, cell, fill=False, ec=RED, lw=1.6, ls=(0, (4, 2)), alpha=0.95))


def render_row(i, st):
    cols, lim = st["cols"], st["lim"]
    fig, axes = plt.subplots(1, len(cols), figsize=(2.85 * len(cols), 3.55))
    for c, col in enumerate(cols):
        ax = axes[c]
        ax.set_facecolor("black")
        draw_density(ax, col["X"].to(torch.float64).numpy(), col["W"].to(torch.float64), lim, st["bw"])
        overlay_avoid(ax, st["geom"], st["engine"])
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        txt = f"W2 {col['w2']:.2f}" if col["kind"] == "target" else f"leak {col['leak']:.3f}\nW2 {col['w2']:.2f}"
        ec = {"ours": "#39d98a", "good": "#bbbbbb"}.get(col["role"], "#ffb347")
        ax.text(
            0.035,
            0.965,
            txt,
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9.5,
            color="white",
            bbox={"boxstyle": "round,pad=0.25", "fc": "black", "ec": ec, "alpha": 0.7, "lw": 1.0},
        )
        ax.set_title(col["label"], fontsize=10.5, pad=6)
        if c == 0:
            ax.set_ylabel(st["name"], fontsize=13, fontweight="bold")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"row_{i}_{st['name'].replace(' - ', '_').replace(' ', '')}.png"
    fig.savefig(str(out), dpi=145, facecolor="white", metadata={"Software": None})
    print(f"[saved] {out}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Render the demo from cached particles/weights.")
    ap.add_argument("row", nargs="?", type=int, help="setting index 0/1/2 (default: all)")
    a = ap.parse_args()
    paths = [RES_DIR / f"row_{a.row}.pt"] if a.row is not None else sorted(RES_DIR.glob("row_*.pt"))
    for p in paths:
        i = int(p.stem.split("_")[1])
        render_row(i, torch.load(p, map_location="cpu", weights_only=False))


if __name__ == "__main__":
    main()
