"""Merge per-seed 50-trial logs into summary files.

The script merges score-based runs and the NegCFG and DNG sweep logs.

Usage:
    uv run python results/50trials/merge.py
"""

from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
SEEDS = list(range(10))


def parse_kv(line: str) -> dict:
    """Parse 'key=val/wass' or 'key=val' pairs from a log line."""
    d = {}
    for p in line.split():
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        if "/" in v:
            pm, wass = v.split("/")
            d[k + "_pm"] = float(pm)
            d[k + "_wass"] = float(wass)
        else:
            try:
                d[k] = float(v.rstrip("s"))
            except ValueError:
                d[k] = v
    return d


def merge_score(variant: str = "score_ON"):
    trials = []
    for s in SEEDS:
        log = HERE / f"{variant}/seed_{s}/{variant}_log.txt"
        with open(log) as f:
            for line in f:
                if not line.startswith("  L"):
                    continue
                d = parse_kv(line)
                trials.append(d)

    n = len(trials)
    orc_pm = [t["Orc_pm"] for t in trials]
    orc_w = [t["Orc_wass"] for t in trials]
    dng_pm = [t["DNG_pm"] for t in trials]
    dng_w = [t["DNG_wass"] for t in trials]
    raw_pm = [t["raw_pm"] for t in trials]
    raw_w = [t["raw_wass"] for t in trials]
    w_pm = [t["w_pm"] for t in trials]
    w_w = [t["w_wass"] for t in trials]

    with open(HERE / f"{variant}_merged.txt", "w") as f:
        for s in SEEDS:
            log = HERE / f"{variant}/seed_{s}/{variant}_log.txt"
            with open(log) as lf:
                for line in lf:
                    if line.startswith("  L"):
                        f.write(line.rstrip().replace("elapsed=", "").rstrip() + "\n")

        win_r = sum(1 for a, b in zip(raw_pm, dng_pm) if a < b)
        win_w = sum(1 for a, b in zip(w_pm, dng_pm) if a < b)
        gap = np.mean(dng_pm) - np.mean(orc_pm)
        f.write(f"\nSUMMARY: {variant.replace('score_', 'Score ')} ({n} trials, seeds 0-9)\n")
        f.write(f"  Oracle  : post_mean={np.mean(orc_pm):.4f} ± {np.std(orc_pm):.4f}  Wasserstein={np.mean(orc_w):.1f} ± {np.std(orc_w):.1f}\n")
        f.write(f"  DNG     : post_mean={np.mean(dng_pm):.4f} ± {np.std(dng_pm):.4f}  Wasserstein={np.mean(dng_w):.1f} ± {np.std(dng_w):.1f}\n")
        f.write(f"  Ours(raw): post_mean={np.mean(raw_pm):.4f} ± {np.std(raw_pm):.4f}  Wasserstein={np.mean(raw_w):.1f} ± {np.std(raw_w):.1f}\n")
        f.write(f"  Ours(weighted): post_mean={np.mean(w_pm):.4f} ± {np.std(w_pm):.4f}  Wasserstein={np.mean(w_w):.1f} ± {np.std(w_w):.1f}\n")
        f.write(f"  Win vs DNG: raw {win_r}/{n}, w {win_w}/{n}\n")
        f.write(f"  Gap fill: raw {(np.mean(dng_pm) - np.mean(raw_pm)) / gap * 100:.0f}%, w {(np.mean(dng_pm) - np.mean(w_pm)) / gap * 100:.0f}%\n")
    print(f"{variant}_merged.txt: {n} trials")


def merge_negcfg():
    LAMBDAS = [0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    data = {lam: [] for lam in LAMBDAS}
    for s in SEEDS:
        log = HERE / f"negcfg/seed_{s}/tradeoff_negcfg_log.txt"
        cur_lam = None
        with open(log) as f:
            for line in f:
                line = line.strip()
                if line.startswith("--- NegCFG lam="):
                    cur_lam = float(line.split("=")[1].split()[0])
                elif line.startswith("L") and cur_lam is not None:
                    data[cur_lam].append(parse_kv(line))

    with open(HERE / "negcfg_merged.txt", "w") as f:
        for lam in LAMBDAS:
            trials = data[lam]
            pms = [t["pm"] for t in trials]
            ws = [t["wass"] for t in trials]
            f.write(f"  lambda={lam:<4} post_mean={np.mean(pms):.4f} ± {np.std(pms):.4f}  Wasserstein={np.mean(ws):.1f} ± {np.std(ws):.1f}\n")
    print(f"negcfg_merged.txt: {len(data[0.0])} trials/lambda")


def merge_dng():
    LAMBDA0S = [0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 5.0, 10.0, 20.0]
    for tau in [0.2, 1.0]:
        data = {lam: [] for lam in LAMBDA0S}
        for s in SEEDS:
            log = HERE / f"dng/tau{tau}/seed_{s}/tradeoff_dng_tau{tau}_log.txt"
            cur_lam = None
            with open(log) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("--- DNG"):
                        for p in line.split():
                            if p.startswith("lambda0="):
                                cur_lam = float(p.split("=")[1])
                    elif line.startswith("L") and cur_lam is not None:
                        data[cur_lam].append(parse_kv(line))

        with open(HERE / f"dng_tau{tau}_merged.txt", "w") as f:
            for lam in LAMBDA0S:
                trials = data[lam]
                pms = [t["pm"] for t in trials]
                ws = [t["wass"] for t in trials]
                f.write(f"  lambda0={lam:<5} post_mean={np.mean(pms):.4f} ± {np.std(pms):.4f}  Wasserstein={np.mean(ws):.1f} ± {np.std(ws):.1f}\n")
        print(f"dng_tau{tau}_merged.txt: {len(data[0.5])} trials/lambda0")


if __name__ == "__main__":
    for v in ["score_OFF", "score_ON", "score_ON_TILT", "score_ON_GRAD40", "score_ON_GRAD48"]:
        if (HERE / v).is_dir():
            merge_score(v)
    merge_negcfg()
    merge_dng()
