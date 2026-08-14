"""DNG T (diffusion steps) ablation, Case 1 true-score. Only T varies.

DNG config fixed to the main-table setting (lambda0=1, tau=1, pmin=1e-2,
lam_max=20, M=1024). Oracle is recomputed at each T for a matched reference.

Usage:
    uv run python results/additional_diagnosis/run_dng_T.py --T 100 --layouts 5 --seeds 0 1 2
    uv run python results/additional_diagnosis/run_dng_T.py --T 10 25 50 100 200 500 1000 --layouts 5 --seeds 0
"""

import argparse
import os
import sys
import time

os.environ["TQDM_DISABLE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
from gmm_ddpm import DDPMSchedule, OracleDiffusion
from gmm_ring import Logger, compute_post, make_ring_gmm, wasserstein
from sampler_score import DNGDDPM

M = 1024
LAMBDA0 = 1.0
TAU = 1.0
TS = [10, 25, 50, 100, 200, 500, 1000]  # current = 100


def main():
    parser = argparse.ArgumentParser(description="DNG T ablation")
    parser.add_argument("--T", type=int, nargs="+", default=TS)
    parser.add_argument("--layouts", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    layouts = list(range(args.layouts))
    seeds = args.seeds
    n_trials = len(layouts) * len(seeds)

    out_dir = args.outdir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    tag = "_".join(str(t) for t in args.T)
    log_path = os.path.join(out_dir, f"dng_T_{tag}_log.txt")
    log = Logger(log_path)
    sys.stdout = log

    print("=" * 100)
    print(f"DNG T ablation — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Ts: {args.T} (current=100), DNG lambda0={LAMBDA0} tau={TAU} pmin=1e-2 lam_max=20")
    print(f"Ring GMM (radius=10), M={M}, {len(layouts)} layouts x {len(seeds)} seeds = {n_trials} trials")
    print("=" * 100)

    summary = []
    for T in args.T:
        sched = DDPMSchedule(T=T)
        print(f"\n--- T={T} ---")
        orc_r, dng_r = [], []
        for ls in layouts:
            pos, neg, pi = make_ring_gmm(seed=ls)
            for seed in seeds:
                wass_seed = (ls + 1) * (seed + 1)

                xo = OracleDiffusion(schedule=sched, neg_x0_gmm=neg, M=M, seed=seed).sample()
                op = compute_post(xo, pos, neg, pi)
                orc_r.append({"pm": float(op.mean()), "wass": wasserstein(xo, neg, generator=torch.Generator().manual_seed(wass_seed))})

                t0 = time.time()
                dng = DNGDDPM(schedule=sched, pos_x0_gmm=pos, neg_x0_gmm=neg, class_prior=pi, M=M, lambda0=LAMBDA0, tau=TAU, pmin=1e-2, lam_max=20, seed=seed)
                xd, _ = dng.sample()
                elapsed = time.time() - t0
                dp = compute_post(xd, pos, neg, pi)
                r = {
                    "pm": float(dp.mean()),
                    "lk": float((dp > 0.5).float().mean()),
                    "wass": wasserstein(xd, neg, generator=torch.Generator().manual_seed(wass_seed)),
                }
                dng_r.append(r)
                print(f"  L{ls}s{seed:>3}: DNG pm={r['pm']:.4f} lk={r['lk']:.3f} wass={r['wass']:.1f} elapsed={elapsed:.1f}s")

        s = {
            "T": T,
            "orc_pm": np.mean([r["pm"] for r in orc_r]),
            "orc_wass": np.mean([r["wass"] for r in orc_r]),
            "pm": np.mean([r["pm"] for r in dng_r]),
            "pm_std": np.std([r["pm"] for r in dng_r]),
            "lk": np.mean([r["lk"] for r in dng_r]),
            "wass": np.mean([r["wass"] for r in dng_r]),
            "wass_std": np.std([r["wass"] for r in dng_r]),
        }
        summary.append(s)
        print(
            f"  → Oracle pm={s['orc_pm']:.4f} wass={s['orc_wass']:.1f} | DNG pm={s['pm']:.4f}±{s['pm_std']:.4f} lk={s['lk']:.3f} wass={s['wass']:.1f}±{s['wass_std']:.1f}"
        )

    print(f"\n{'=' * 100}")
    print(f"DNG T ABLATION TABLE (lambda0={LAMBDA0}, tau={TAU})")
    print(f"{'=' * 100}")
    print(f"  {'T':>6} | {'orc_pm':>8} {'orc_wass':>9} | {'dng_pm':>8} {'±':>6} | {'dng_wass':>9} {'±':>7} | {'dng_lk':>7}")
    print(f"  {'-' * 78}")
    for s in summary:
        print(
            f"  {s['T']:>6} | {s['orc_pm']:>8.4f} {s['orc_wass']:>9.1f} | {s['pm']:>8.4f} {s['pm_std']:>6.4f} | {s['wass']:>9.1f} {s['wass_std']:>7.1f} | {s['lk']:>7.3f}"
        )

    print(f"\nDone — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = log.out
    log.close()
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
