"""DNG guidance strength tradeoff (Case 1: true-score).

Sweeps DNG lambda0 and tau to show the leak vs wasserstein tradeoff.

Usage:
    uv run python run_tradeoff_dng.py --tau 1.0 --seeds 0
    uv run python run_tradeoff_dng.py --tau 0.2 --layouts 5 --seeds 0 42 123
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
from sampler_score import DNGDDPM

T = 100
M = 1024
LAMBDA0S = [0.5, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 5.0, 10.0, 20.0]


def main():
    parser = argparse.ArgumentParser(description="DNG tradeoff sweep")
    parser.add_argument("--tau", type=float, required=True)
    parser.add_argument("--layouts", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    tau = args.tau
    layouts = list(range(args.layouts))
    seeds = args.seeds
    n_trials = len(layouts) * len(seeds)

    out_dir = args.outdir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    log_path = os.path.join(out_dir, f"tradeoff_dng_tau{tau}_log.txt")
    log = Logger(log_path)
    sys.stdout = log

    sched = DDPMSchedule(T=T)

    print("=" * 100)
    print(f"DNG Tradeoff Sweep — tau={tau}")
    print(f"Lambda0s: {LAMBDA0S}")
    print(f"Ring GMM (radius=10), T={T}, M={M}")
    print(f"{len(layouts)} layouts x {len(seeds)} seeds = {n_trials} trials")
    print("=" * 100)

    print("\n--- Oracle ---")
    orc_results = []
    for ls in layouts:
        pos, neg, pi = make_ring_gmm(seed=ls)
        for seed in seeds:
            wass_seed = (ls + 1) * (seed + 1)
            xo = OracleDiffusion(schedule=sched, neg_x0_gmm=neg, M=M, seed=seed).sample()
            op = compute_post(xo, pos, neg, pi)
            orc_results.append({"pm": float(op.mean()), "wass": wasserstein(xo, neg, generator=torch.Generator().manual_seed(wass_seed))})
    print(f"  pm={np.mean([r['pm'] for r in orc_results]):.4f}  wass={np.mean([r['wass'] for r in orc_results]):.1f}")

    summary = []
    for lam0 in LAMBDA0S:
        print(f"\n--- DNG lambda0={lam0} tau={tau} ---")
        all_r = []
        for ls in layouts:
            pos, neg, pi = make_ring_gmm(seed=ls)
            for seed in seeds:
                wass_seed = (ls + 1) * (seed + 1)
                t0 = time.time()
                dng = DNGDDPM(
                    schedule=sched,
                    pos_x0_gmm=pos,
                    neg_x0_gmm=neg,
                    class_prior=pi,
                    M=M,
                    lambda0=lam0,
                    tau=tau,
                    pmin=1e-2,
                    lam_max=20,
                    seed=seed,
                )
                xd, _ = dng.sample()
                elapsed = time.time() - t0
                dp = compute_post(xd, pos, neg, pi)
                r = {
                    "pm": float(dp.mean()),
                    "wass": wasserstein(xd, neg, generator=torch.Generator().manual_seed(wass_seed)),
                }
                print(f"  L{ls}s{seed:>3}: pm={r['pm']:.4f} wass={r['wass']:.1f} elapsed={elapsed:.1f}s")
                all_r.append(r)

        s = {
            "lam0": lam0,
            "pm": np.mean([r["pm"] for r in all_r]),
            "pm_std": np.std([r["pm"] for r in all_r]),
            "wass": np.mean([r["wass"] for r in all_r]),
            "wass_std": np.std([r["wass"] for r in all_r]),
        }
        summary.append(s)
        print(f"  → pm={s['pm']:.4f}±{s['pm_std']:.4f}  wass={s['wass']:.1f}±{s['wass_std']:.1f}")

    print(f"\n{'=' * 100}")
    print(f"TRADEOFF TABLE: DNG lambda0 sweep (tau={tau})")
    print(f"{'=' * 100}")
    print(f"  {'Method':<25} | {'post_mean':>10} {'±':>6} | {'wass':>10} {'±':>8}")
    print(f"  {'-' * 70}")

    orc_pm = np.mean([r["pm"] for r in orc_results])
    orc_wass = np.mean([r["wass"] for r in orc_results])
    print(f"  {'Oracle':<25} | {orc_pm:>10.4f}        | {orc_wass:>10.1f}")
    print(f"  {'-' * 70}")

    for s in summary:
        label = f"DNG lam0={s['lam0']}"
        print(f"  {label:<25} | {s['pm']:>10.4f} {s['pm_std']:>6.4f} | {s['wass']:>10.1f} {s['wass_std']:>8.1f}")

    print("\nDone")
    sys.stdout = log.out
    log.close()
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
