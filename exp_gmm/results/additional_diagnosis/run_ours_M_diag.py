"""Ours M ablation + classifier (f_t) diagnostics, Case 1 true-score.

Same M sweep as results/ablation (Oracle recomputed at each M), plus per-step
diagnostics of the PU classifier against the exact oracle posterior on the same
x0hat (via snapshot_callback):
  - AUC        : f_t separates pos/neg region (label = oracle p>0.5)
  - prob-RMSE  : calibration of f_t to the true posterior value
  - Brier      : (p_pu - label)^2
  - grad-cos   : cos(grad log(1-p_pu), grad log(1-p_true)) in x0hat space
Stratified by diffusion time (early t/T>0.66, mid 0.33-0.66, late <0.33).

Usage:
    uv run python results/additional_diagnosis/run_ours_M_diag.py --mode ON_TILT --M 1024 --layouts 5 --seeds 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import os
import sys
import time

os.environ["TQDM_DISABLE"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch
import torch.nn.functional as F
from gmm_ddpm import DDPMSchedule, OracleDiffusion, safe_clip01
from gmm_ring import Logger, compute_post, make_ring_gmm, wasserstein
from mode_configs import ESS_TARGET, MODE_DESC, MODE_KW, make_pu
from sampler_score_diag import PUNegGuideDDPMSMCDiag

T = 100
MS = [64, 256, 1024, 4096]


def _rankdata(a):
    a = np.asarray(a, dtype=float)
    order = a.argsort()
    sa = a[order]
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0  # 1-based average rank
        i = j + 1
    return ranks


def _auc(scores, labels):
    labels = np.asarray(labels)
    npos = int(labels.sum())
    nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    r = _rankdata(scores)
    return (r[labels == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def _bin(t):
    f = t / T
    return "early" if f > 0.66 else ("mid" if f > 0.33 else "late")


def run_ours(pu, sched, pos, neg, pi, neg_centers, mode, M, seed, wass_seed, diag):
    def cb(t, x_t, x0hat, pu_proba, w_u, w_c, w_q, lam_t):
        p_pu = pu_proba.detach().cpu().numpy()
        p_true_t = compute_post(x0hat, pos, neg, pi)
        p_true = p_true_t.detach().cpu().numpy()
        label = (p_true > 0.5).astype(float)
        with torch.enable_grad():
            xh = x0hat.detach().clone().requires_grad_(True)
            g_pu = torch.autograd.grad(torch.log(1.0 - safe_clip01(pu.predict_proba_pos(xh))).sum(), xh)[0]
            xh2 = x0hat.detach().clone().requires_grad_(True)
            g_tr = torch.autograd.grad(torch.log(1.0 - safe_clip01(compute_post(xh2, pos, neg, pi))).sum(), xh2)[0]
        gcos = float(F.cosine_similarity(g_pu, g_tr, dim=-1).mean())
        diag.append(
            {
                "bin": _bin(t),
                "auc": _auc(p_pu, label),
                "rmse": float(np.sqrt(np.mean((p_pu - p_true) ** 2))),
                "brier": float(np.mean((p_pu - label) ** 2)),
                "gcos": gcos,
            }
        )

    t0 = time.time()
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
        **MODE_KW[mode],
    )
    xp, wp = pusmc.sample(snapshot_callback=cb)
    p = compute_post(xp, pos, neg, pi)
    wn = wp / (wp.sum() + 1e-12)
    return {
        "raw_pm": float(p.mean()),
        "w_pm": float((wn * p).sum()),
        "raw_wass": wasserstein(xp, neg, generator=torch.Generator().manual_seed(wass_seed)),
        "w_wass": wasserstein(xp, neg, wp, generator=torch.Generator().manual_seed(wass_seed)),
        "ess": float(1.0 / (wn.pow(2).sum() + 1e-12)),
        "rs": pusmc.resample_count,
        "mc": float((torch.cdist(xp, neg_centers).min(0).values < 5).float().mean()),
        "time": time.time() - t0,
    }


def _agg(rows, key, subset=None):
    v = [r[key] for r in rows if (subset is None or r["bin"] == subset)]
    v = [x for x in v if not (isinstance(x, float) and np.isnan(x))]
    return (np.mean(v), np.std(v)) if v else (float("nan"), float("nan"))


def main():
    parser = argparse.ArgumentParser(description="Ours M ablation + f_t diagnostics")
    parser.add_argument("--mode", choices=["OFF", "ON", "ON_TILT"], required=True)
    parser.add_argument("--M", type=int, nargs="+", default=MS)
    parser.add_argument("--layouts", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    layouts = list(range(args.layouts))
    seeds = args.seeds
    n_trials = len(layouts) * len(seeds)

    out_dir = args.outdir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    tag = "_".join(str(m) for m in args.M)
    log_path = os.path.join(out_dir, f"ours_M_diag_{args.mode}_{tag}_log.txt")
    log = Logger(log_path)
    sys.stdout = log

    sched = DDPMSchedule(T=T)
    print("=" * 100)
    print(f"Ours M ablation + f_t diagnostics ({args.mode}) — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{args.mode}: {MODE_DESC[args.mode]}")
    print(f"class_prior = pu_class_prior = true layout pi (calib), ess_target={ESS_TARGET}, kernel PU ss=0.2 lam=1e-4")
    print(f"Ms: {args.M}, T={T}, {len(layouts)} layouts x {len(seeds)} seeds = {n_trials} trials")
    print("=" * 100)

    perf = []
    diag_by_M = {}
    for M in args.M:
        print(f"\n--- M={M} ---")
        orc_r, ours_r, diag = [], [], []
        for ls in layouts:
            pos, neg, pi = make_ring_gmm(seed=ls)
            neg_centers = torch.stack([g.mean for g in neg.comps])
            for seed in seeds:
                wass_seed = (ls + 1) * (seed + 1)
                xo = OracleDiffusion(schedule=sched, neg_x0_gmm=neg, M=M, seed=seed).sample()
                op = compute_post(xo, pos, neg, pi)
                orc_r.append({"pm": float(op.mean()), "wass": wasserstein(xo, neg, generator=torch.Generator().manual_seed(wass_seed))})
                pu = make_pu()
                r = run_ours(pu, sched, pos, neg, pi, neg_centers, args.mode, M, seed, wass_seed, diag)
                ours_r.append(r)
                print(
                    f"  L{ls}s{seed:>3}: Orc={float(op.mean()):.4f} raw={r['raw_pm']:.4f}/{r['raw_wass']:.1f} w={r['w_pm']:.4f}/{r['w_wass']:.1f} ESS/M={r['ess'] / M:.2f} rs={r['rs']} t={r['time']:.1f}s"
                )
        diag_by_M[M] = diag
        s = {
            "M": M,
            "opm": np.mean([x["pm"] for x in orc_r]),
            "ow": np.mean([x["wass"] for x in orc_r]),
            "rpm": np.mean([x["raw_pm"] for x in ours_r]),
            "rpm_s": np.std([x["raw_pm"] for x in ours_r]),
            "wpm": np.mean([x["w_pm"] for x in ours_r]),
            "wpm_s": np.std([x["w_pm"] for x in ours_r]),
            "rw": np.mean([x["raw_wass"] for x in ours_r]),
            "ww": np.mean([x["w_wass"] for x in ours_r]),
            "ess": np.mean([x["ess"] for x in ours_r]) / M,
            "rs": np.mean([x["rs"] for x in ours_r]),
            "tm": np.mean([x["time"] for x in ours_r]),
        }
        perf.append(s)

    print(f"\n{'=' * 100}")
    print(f"M ABLATION — performance ({args.mode})")
    print(f"{'=' * 100}")
    print(
        f"  {'M':>6} | {'orc_pm':>8} {'orc_wass':>8} | {'raw_pm':>8} {'±':>6} | {'raw_wass':>8} | {'w_pm':>8} {'±':>6} | {'w_wass':>8} | {'ESS/M':>6} | {'rs':>4} | {'s/trial':>8}"
    )
    print(f"  {'-' * 110}")
    for s in perf:
        print(
            f"  {s['M']:>6} | {s['opm']:>8.4f} {s['ow']:>8.1f} | {s['rpm']:>8.4f} {s['rpm_s']:>6.4f} | {s['rw']:>8.1f} | {s['wpm']:>8.4f} {s['wpm_s']:>6.4f} | {s['ww']:>8.1f} | {s['ess']:>6.2f} | {s['rs']:>4.1f} | {s['tm']:>8.1f}"
        )

    print(f"\n{'=' * 100}")
    print("f_t CLASSIFIER DIAGNOSTICS vs oracle posterior (higher AUC/gcos better; lower RMSE/Brier better)")
    print(f"{'=' * 100}")
    print(f"  {'M':>6} | {'time':>6} | {'AUC':>12} | {'prob-RMSE':>12} | {'Brier':>12} | {'grad-cos':>12}")
    print(f"  {'-' * 78}")
    for M in args.M:
        d = diag_by_M[M]
        for label, sub in [("overall", None), ("early", "early"), ("mid", "mid"), ("late", "late")]:
            auc_m, _ = _agg(d, "auc", sub)
            rmse_m, _ = _agg(d, "rmse", sub)
            br_m, _ = _agg(d, "brier", sub)
            gc_m, gc_s = _agg(d, "gcos", sub)
            print(f"  {M:>6} | {label:>6} | {auc_m:>12.4f} | {rmse_m:>12.4f} | {br_m:>12.4f} | {gc_m:>7.3f}±{gc_s:.2f}")

    print(f"\nDone — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = log.out
    log.close()
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
