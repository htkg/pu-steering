"""Plot the checker SafeDenoiser eta sweep (leak vs W2) from a 3-column 'eta leak W2' file."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle


def read_points(path):
    pts = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        eta, leak, w2 = line.split()[:3]
        pts.append((float(eta), float(leak), float(w2)))
    return sorted(pts)


def main():
    p = argparse.ArgumentParser(description="Plot the SafeDenoiser eta sweep (leak vs W2).")
    p.add_argument("data", help="3-column file: eta leak W2 (one per line)")
    p.add_argument("--out", default="figures/checker_safed_sweep.png")
    p.add_argument("--ours", nargs=2, type=float, metavar=("LEAK", "W2"), default=None)
    p.add_argument("--target", nargs=2, type=float, metavar=("LEAK", "W2"), default=None)
    a = p.parse_args()

    pts = read_points(a.data)
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    ax.plot([leak for _, leak, _ in pts], [w2 for _, _, w2 in pts], "-o", color="#ffb347", label="SafeDenoiser (sweep eta)", zorder=2)
    for eta, leak, w2 in pts:
        ax.annotate(f"{eta:g}", (leak, w2), fontsize=14, color="gray", xytext=(4, 4), textcoords="offset points")
    if a.target:
        ax.scatter([a.target[0]], [a.target[1]], marker=MarkerStyle("*"), s=300, color="#888", label="Target", zorder=3)
    if a.ours:
        ax.scatter([a.ours[0]], [a.ours[1]], marker=MarkerStyle("D"), s=140, color="#39d98a", label="Ours", zorder=3)
    ax.set_xlabel("leak  (avoided mass left)", fontsize=18)
    ax.set_ylabel("W2 to target", fontsize=18)
    ax.set_title("ref - checkerboard: SafeDenoiser strength sweep", fontsize=18)
    ax.tick_params(labelsize=15)
    ax.legend(fontsize=16)

    # Leave room for the eta labels.
    xa = [leak for _, leak, _ in pts] + [p[0] for p in (a.target, a.ours) if p]
    ya = [w2 for _, _, w2 in pts] + [p[1] for p in (a.target, a.ours) if p]
    xr, yr = max(xa) - min(xa), max(ya) - min(ya)
    ax.set_xlim(min(xa) - 0.06 * xr, max(xa) + 0.10 * xr)
    ax.set_ylim(min(ya) - 0.06 * yr, max(ya) + 0.12 * yr)
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=145, facecolor="white", metadata={"Software": None}, bbox_inches="tight", pad_inches=0.08)
    print(f"[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
