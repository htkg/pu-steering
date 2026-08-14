"""NegCFG guidance strength tradeoff (Case 1: true-score).

Sweeps NegCFG lambda to show the leak vs wasserstein tradeoff.
Higher lambda → lower leak but worse distribution quality (higher wasserstein).
Oracle is the ideal target that NegCFG cannot reach.

Usage:
    uv run python run_tradeoff_negcfg.py
    uv run python run_tradeoff_negcfg.py --layouts 5 --seeds 0 42 123
"""

import argparse
import os
import sys
import time

os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import torch

from gmm_ddpm import DDPMSchedule, OracleDiffusion
from gmm_ring import Logger, compute_post, make_ring_gmm, wasserstein
from sampler_score import NegCFG

T = 100
M = 1024
LAMBDAS = [0.0, 0.1, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]


def main():
    parser = argparse.ArgumentParser(description="NegCFG tradeoff sweep")
    parser.add_argument("--layouts", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42])
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    layouts = list(range(args.layouts))
    seeds = args.seeds
    n_trials = len(layouts) * len(seeds)

    out_dir = args.outdir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "tradeoff_negcfg_log.txt")
    log = Logger(log_path)
    sys.stdout = log

    sched = DDPMSchedule(T=T)

    print("=" * 100)
    print(f"NegCFG Tradeoff Sweep — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Lambdas: {LAMBDAS}")
    print(f"Ring GMM (radius=10), T={T}, M={M}")
    print(f"{len(layouts)} layouts x {len(seeds)} seeds = {n_trials} trials")
    print("=" * 100)

    print("\n--- Oracle ---")
    orc_results = []
    for ls in layouts:
        pos, neg, pi = make_ring_gmm(seed=ls)
        neg_centers = torch.stack([g.mean for g in neg.comps])
        for seed in seeds:
            wass_seed = (ls + 1) * (seed + 1)
            xo = OracleDiffusion(schedule=sched, neg_x0_gmm=neg, M=M, seed=seed).sample()
            op = compute_post(xo, pos, neg, pi)
            orc_results.append(
                {
                    "pm": float(op.mean()),
                    "leak": float((op > 0.5).float().mean()),
                    "wass": wasserstein(xo, neg, generator=torch.Generator().manual_seed(wass_seed)),
                    "mc": float((torch.cdist(xo, neg_centers).min(0).values < 5).float().mean()),
                }
            )
    print(
        f"  pm={np.mean([r['pm'] for r in orc_results]):.4f}  "
        f"leak={np.mean([r['leak'] for r in orc_results]):.4f}  "
        f"wass={np.mean([r['wass'] for r in orc_results]):.1f}"
    )

    summary = []
    for lam in LAMBDAS:
        print(f"\n--- NegCFG lam={lam} ---")
        all_r = []
        for ls in layouts:
            pos, neg, pi = make_ring_gmm(seed=ls)
            neg_centers = torch.stack([g.mean for g in neg.comps])
            for seed in seeds:
                wass_seed = (ls + 1) * (seed + 1)
                t0 = time.time()
                cfg = NegCFG(schedule=sched, pos_x0_gmm=pos, neg_x0_gmm=neg, class_prior=pi, M=M, lam=lam, seed=seed)
                xc = cfg.sample()
                elapsed = time.time() - t0
                cp = compute_post(xc, pos, neg, pi)
                r = {
                    "pm": float(cp.mean()),
                    "leak": float((cp > 0.5).float().mean()),
                    "wass": wasserstein(xc, neg, generator=torch.Generator().manual_seed(wass_seed)),
                    "mc": float((torch.cdist(xc, neg_centers).min(0).values < 5).float().mean()),
                }
                print(f"  L{ls}s{seed:>3}: pm={r['pm']:.4f} leak={r['leak']:.4f} wass={r['wass']:.1f} mc={r['mc']:.2f} elapsed={elapsed:.1f}s")
                all_r.append(r)

        s = {
            "lam": lam,
            "pm": np.mean([r["pm"] for r in all_r]),
            "pm_std": np.std([r["pm"] for r in all_r]),
            "leak": np.mean([r["leak"] for r in all_r]),
            "wass": np.mean([r["wass"] for r in all_r]),
            "wass_std": np.std([r["wass"] for r in all_r]),
            "mc": np.min([r["mc"] for r in all_r]),
        }
        summary.append(s)
        print(f"  → pm={s['pm']:.4f}±{s['pm_std']:.4f}  wass={s['wass']:.1f}±{s['wass_std']:.1f}  mc={s['mc']:.2f}")

    print(f"\n{'=' * 100}")
    print("TRADEOFF TABLE: NegCFG lambda sweep")
    print(f"{'=' * 100}")
    print(f"  {'Method':<20} | {'post_mean':>10} {'±':>5} | {'leak':>7} | {'wass':>10} {'±':>7} | {'mc':>5}")
    print(f"  {'-' * 75}")

    orc_pm = np.mean([r["pm"] for r in orc_results])
    orc_wass = np.mean([r["wass"] for r in orc_results])
    print(
        f"  {'Oracle':<20} | {orc_pm:>10.4f}       | {np.mean([r['leak'] for r in orc_results]):>7.4f} | "
        f"{orc_wass:>10.1f}         | {np.mean([r['mc'] for r in orc_results]):>5.2f}"
    )
    print(f"  {'-' * 75}")

    for s in summary:
        print(
            f"  {'NegCFG lam=' + str(s['lam']):<20} | {s['pm']:>10.4f} {s['pm_std']:>5.3f} | {s['leak']:>7.4f} | "
            f"{s['wass']:>10.1f} {s['wass_std']:>7.1f} | {s['mc']:>5.2f}"
        )

    print("\n  Higher lambda → lower post_mean (better avoidance) but higher wasserstein (worse distribution)")
    print("  Oracle achieves both low post_mean AND low wasserstein — NegCFG cannot reach this.")

    print(f"\nDone — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = log.out
    log.close()
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
