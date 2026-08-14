"""Pareto frontier (baselines + Ours) from frontier.json — two panels:
  (a) avoided-style rate p(c) vs FID (quality), (b) p(c) vs CSD-Fréchet (style dist).

Series: Ours (unweighted filled diamond + weighted ×, M=128/64), SLD, NegCFG, DNG, and the
oracle floor (gold star). NegCFG and DNG are drawn as connected strength sweeps; the DNG
sweep starts at the content (no-avoidance) point.

Conventions on Stable Diffusion:
  * NegCFG is standard negative prompting: eps_guide = eps_style + s×(eps_pos - eps_style),
    anchored on the avoided-style prediction (raw eps_pos, no extra CFG).
  * DNG uses a content-CFG base, the same base SLD keeps:
    eps_base = eps_uncond + 7.5×(eps_pos - eps_uncond). The plotted DNG points
    use this content-anchored variant and sweep λ0.

Config (argv[1] JSON, the same assemble config): {dataset, out}.
Output: results/dataset/{tag}/frontier.png (+ .pdf)
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OURS_COL = {128: "#d62728", 64: "#7d1c1c"}  # diamond (unweighted, main result)
OURS_W_COL = {128: "#f3a0a0", 64: "#b87070"}  # X (weighted SMC estimate)
SLD_COL, NEG_COL, DNG_COL, CONTENT_COL = "#1f77b4", "#ff7f0e", "#2ca02c", "black"
DNG_LABEL_L = {100, 150, 200, 240, 260, 300}  # which λ0 to annotate on the DNG sweep
NEG_LABEL_S = {1.5, 2.0, 4.0, 6.0, 7.5}  # NegCFG s to annotate (2 from each end + the elbow)
SLD_LBL = {"sld_max": "max", "sld_xxmax": "xxmax", "sld_x3max": "x3max"}
EDGE = {"edgecolors": "black", "linewidths": 0.5}
ALPHA = 0.9
plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def m_of(name: str) -> int:
    return 128 if "m128" in name else 64


def y_of(r: dict, ykey: str, weighted: bool = False) -> float:
    suf = "_weighted" if weighted else ("_unweighted" if r.get("weighted") else "")
    return r[("fid" if ykey == "fid" else "csd_frechet") + suf]


def x_of(r: dict, weighted: bool = False) -> float:
    return r["pc_weighted"] if weighted else (r["pc_unweighted"] if r.get("weighted") else r["pc"])


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: python {Path(__file__).name} <config.json>")
    arg = sys.argv[1]
    c = json.loads(Path(arg).read_text() if Path(arg).exists() else arg)
    ds = Path(c.get("dataset", "results/dataset/cafe_terrace/dataset.pt"))
    fj = ds.parent / "frontier.json"
    out = Path(c.get("out", ds.parent / "frontier.png"))

    F = json.loads(fj.read_text())
    rows = {r["method"]: r for r in F["rows"]}
    floor = rows.get("floor", {"pc": 0.238, "fid": 18.52, "csd_frechet": 0.009})
    content = rows.get("unconditional_content")
    sld = [m for m in rows if m.startswith("sld")]
    negcfg = sorted((m for m in rows if m.startswith("negcfg_s")), key=lambda m: float(m[len("negcfg_s") :].replace("p", ".")))
    dng = sorted((m for m in rows if m.startswith("dng_c_l")), key=lambda m: int(m[len("dng_c_l") :]))
    ours = [m for m in rows if m.startswith(("ou_", "ow_"))]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0))
    for ax, ykey, ylab, ptitle in [
        (axes[0], "fid", "FID  (Inception-2048, ↓ better)", "(a) rate vs quality"),
        (axes[1], "csd_frechet", "Style FD  (CSD-768, ↓ better)", "(b) rate vs style distance"),
    ]:
        # content anchor (no avoidance) — also the origin of the DNG sweep
        if content:
            cx, cy = content["pc"], content[ykey]
            ax.scatter(cx, cy, marker="P", s=150, c=CONTENT_COL, zorder=6, edgecolors="white", linewidths=0.6)
            ax.annotate("content", (cx, cy), textcoords="offset points", xytext=(7, 3), fontsize=10)
        # NegCFG sweep (standard negative prompting)
        nxy = [(rows[m]["pc"], rows[m][ykey]) for m in negcfg]
        ax.plot([p[0] for p in nxy], [p[1] for p in nxy], "-", c=NEG_COL, lw=1.1, alpha=0.6, zorder=2)
        for m, (x, y) in zip(negcfg, nxy):
            ax.scatter(x, y, marker="s", s=85, c=NEG_COL, zorder=3, alpha=ALPHA, **EDGE)
            s = float(m[len("negcfg_s") :].replace("p", "."))
            if s in NEG_LABEL_S:
                if ykey == "fid":  # panel (a): labels above, elbow below
                    dx, dy, ha, va = (0, -13, "center", "top") if s == 4.0 else (0, 10, "center", "bottom")
                else:  # panel (b): the points bunch up — spread them (s2 left/s6 right, s1.5 down/s7.5 up)
                    dx, dy, ha, va = {
                        1.5: (3, -13, "center", "top"),
                        2.0: (-10, 4, "right", "bottom"),
                        4.0: (0, -13, "center", "top"),
                        6.0: (11, 5, "left", "bottom"),
                        7.5: (0, 11, "center", "bottom"),
                    }[s]
                ax.annotate(f"s={s:g}", (x, y), textcoords="offset points", xytext=(dx, dy), ha=ha, va=va, fontsize=8.5, color="#a85000")
        # DNG sweep (content-CFG base), starting at content
        dxy = [(rows[m]["pc"], rows[m][ykey]) for m in dng]
        ax.plot(
            ([content["pc"]] if content else []) + [p[0] for p in dxy],
            ([content[ykey]] if content else []) + [p[1] for p in dxy],
            "-",
            c=DNG_COL,
            lw=1.3,
            zorder=4,
        )
        for m, (x, y) in zip(dng, dxy):
            ax.scatter(x, y, marker="v", s=95, c=DNG_COL, zorder=5, alpha=ALPHA, **EDGE)
            L = int(m[len("dng_c_l") :])
            if L in DNG_LABEL_L:
                ax.annotate(f"λ0={L}", (x, y), textcoords="offset points", xytext=(5, 5), fontsize=8.5, color="#1a661a")
        # Ours (unweighted diamond + weighted X)
        for m in ours:
            mm = m_of(m)
            ax.scatter(x_of(rows[m], True), y_of(rows[m], ykey, True), marker="X", s=110, c=OURS_W_COL[mm], zorder=6, alpha=ALPHA, **EDGE)
            ax.scatter(x_of(rows[m]), y_of(rows[m], ykey), marker="D", s=150, c=OURS_COL[mm], zorder=8 if mm == 128 else 7, alpha=ALPHA, **EDGE)
        # SLD markers and labels.
        for m in sld:
            sx, sy = rows[m]["pc"], rows[m][ykey]
            ax.scatter(sx, sy, marker="o", s=90, c=SLD_COL, zorder=10, alpha=ALPHA, **EDGE)
            ax.annotate(
                SLD_LBL.get(m, m),
                (sx, sy),
                textcoords="offset points",
                xytext=(0, -42),
                ha="center",
                va="top",
                fontsize=9,
                color="#124a6e",
                zorder=11,
                arrowprops={"arrowstyle": "-", "color": "#124a6e", "lw": 0.5, "shrinkA": 0, "shrinkB": 4},
            )
        ax.scatter(floor["pc"], floor[ykey], marker="*", s=640, c="gold", zorder=9, alpha=ALPHA, **EDGE)
        ax.set_xlabel("Avoided-style rate   (↓ better)")
        ax.set_ylabel(ylab)
        ax.set_title(ptitle)
        ax.grid(alpha=0.3, linewidth=0.6)
        ax.margins(0.10)

    mk = lambda marker, col, label, ms=11, ls="": Line2D([], [], ls=ls, marker=marker, ms=ms, c=col, mfc=col, mec="black", mew=0.5, label=label)
    handles = [
        mk("D", OURS_COL[128], "Ours M=128"),
        mk("D", OURS_COL[64], "Ours M=64"),
        mk("X", OURS_W_COL[128], "Ours (weighted)"),
        mk("o", SLD_COL, "SLD"),
        mk("s", NEG_COL, "NegCFG", ls="-"),
        mk("v", DNG_COL, "DNG", ls="-"),
        mk("P", CONTENT_COL, "content"),
        mk("*", "gold", "oracle floor", ms=18),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, framealpha=0.95, borderpad=0.7, columnspacing=1.5, handletextpad=0.4, bbox_to_anchor=(0.5, -0.03))
    fig.suptitle("Avoiding Van-Gogh style without naming the artist  (SD v1.4, N=2048)", fontsize=16, y=1.0)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    # Keep generated metadata deterministic.
    fig.savefig(str(out), dpi=200, bbox_inches="tight", metadata={"Software": None})
    fig.savefig(str(out.with_suffix(".pdf")), bbox_inches="tight", metadata={"CreationDate": None, "Creator": None, "Producer": None})
    print(f"[saved] {out} (+pdf)", flush=True)


if __name__ == "__main__":
    main()
