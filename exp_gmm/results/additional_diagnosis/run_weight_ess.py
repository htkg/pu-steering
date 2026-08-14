"""ESS of the classifier-training importance weights along the SMC path.

The PU classifier is fit with w_u (unlabeled term) and w_c (positive/conditional-c
term). As guidance pushes the path away from c these can degenerate. Here we log
ESS/M of w_u, w_c, and (for reference) the SMC weight w_q at every step, binned by
diffusion time, reporting the mean and the worst (min) step. M=1024.

Usage:
    uv run python results/additional_diagnosis/run_weight_ess.py --mode ON_TILT --layouts 5 --seeds 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import os
import sys
import time

os.environ["TQDM_DISABLE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from gmm_ddpm import DDPMSchedule
from gmm_ring import Logger, compute_post, make_ring_gmm
from mode_configs import ESS_TARGET, MODE_DESC, MODE_KW, make_pu
from sampler_score_diag import PUNegGuideDDPMSMCDiag

T = 100
M = 1024


def _essf(w):
    w = w.detach().clamp(min=0.0)
    s = float(w.sum())
    if s <= 0:
        return 0.0
    wn = w / (s + 1e-12)
    return float(1.0 / (wn.pow(2).sum() + 1e-12)) / w.shape[0]


def _bin(t):
    f = t / T
    return "early" if f > 0.66 else ("mid" if f > 0.33 else "late")


def main():
    parser = argparse.ArgumentParser(description="Classifier-weight ESS along the path")
    parser.add_argument("--mode", choices=["OFF", "ON", "ON_TILT"], required=True)
    parser.add_argument("--layouts", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    layouts = list(range(args.layouts))
    seeds = args.seeds
    n_trials = len(layouts) * len(seeds)

    out_dir = args.outdir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    log = Logger(os.path.join(out_dir, f"weight_ess_{args.mode}_log.txt"))
    sys.stdout = log

    sched = DDPMSchedule(T=T)
    print("=" * 90)
    print(f"Classifier-weight ESS along path ({args.mode}) — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{args.mode}: {MODE_DESC[args.mode]}")
    print(f"M={M}, T={T}, {len(layouts)} layouts x {len(seeds)} seeds = {n_trials} trials")
    print("=" * 90)

    rec = []  # per-step {bin, u, c, q}
    for ls in layouts:
        pos, neg, pi = make_ring_gmm(seed=ls)
        for seed in seeds:
            steps = []

            def cb(t, x_t, x0hat, pu_proba, w_u, w_c, w_q, lam_t, _steps=steps):
                _steps.append({"bin": _bin(t), "u": _essf(w_u), "c": _essf(w_c), "q": _essf(w_q)})

            pu = make_pu()
            pusmc = PUNegGuideDDPMSMCDiag(
                schedule=sched,
                pos_x0_gmm=pos,
                neg_x0_gmm=neg,
                class_prior=pi,
                pu=pu,
                M=M,
                pu_class_prior=pi,
                ess_target=ESS_TARGET,
                seed=seed,
                **MODE_KW[args.mode],
            )
            xp, _wp = pusmc.sample(snapshot_callback=cb)
            pm = float(compute_post(xp, pos, neg, pi).mean())
            rec.extend(steps)
            mu = np.mean([s["u"] for s in steps])
            mc = np.mean([s["c"] for s in steps])
            mq = np.mean([s["q"] for s in steps])
            print(f"  L{ls}s{seed:>3}: raw_pm={pm:.4f} | ESS/M mean  w_u={mu:.3f} w_c={mc:.3f} w_q={mq:.3f} | min w_c={min(s['c'] for s in steps):.3f}")

    print(f"\n{'=' * 90}")
    print("ESS/M of classifier weights, by diffusion time (mean over steps x trials; min = worst step)")
    print(f"{'=' * 90}")
    print(f"  {'bin':>8} | {'w_u mean':>9} {'w_u min':>8} | {'w_c mean':>9} {'w_c min':>8} | {'w_q mean':>9} {'w_q min':>8}")
    print(f"  {'-' * 74}")
    for label in ["overall", "early", "mid", "late"]:
        sub = rec if label == "overall" else [r for r in rec if r["bin"] == label]
        um = np.mean([r["u"] for r in sub])
        un = np.min([r["u"] for r in sub])
        cm = np.mean([r["c"] for r in sub])
        cn = np.min([r["c"] for r in sub])
        qm = np.mean([r["q"] for r in sub])
        qn = np.min([r["q"] for r in sub])
        print(f"  {label:>8} | {um:>9.3f} {un:>8.3f} | {cm:>9.3f} {cn:>8.3f} | {qm:>9.3f} {qn:>8.3f}")

    print(f"\nDone — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = log.out
    log.close()


if __name__ == "__main__":
    main()
