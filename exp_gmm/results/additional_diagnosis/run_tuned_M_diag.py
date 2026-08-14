"""M ablation + f_t diagnostics + classifier-weight ESS for the Ours ON_GRAD
variant (grad proposal, cos tilt 0->lam_end with plateau, ESS-fired resampling with
cooldown). Same metrics as run_ours_M_diag.py / run_weight_ess.py, one pass.

Usage:
    uv run python results/additional_diagnosis/run_tuned_M_diag.py --variant cd48 --M 1024 --layouts 5 --seeds 0 1 2 3 4 5 6 7 8 9
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
from mode_configs import MODE_KW, make_pu
from sampler_score_diag import PUNegGuideDDPMSMCDiag

T = 100
MS = [64, 256, 1024, 4096]

VARIANTS = {"cd40": MODE_KW["ON_GRAD40"], "cd48": MODE_KW["ON_GRAD48"]}


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
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
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


def _essf(w):
    w = w.detach().clamp(min=0.0)
    s = float(w.sum())
    if s <= 0:
        return 0.0
    wn = w / (s + 1e-12)
    return float(1.0 / (wn.pow(2).sum() + 1e-12)) / w.shape[0]


def run_ours(pu, sched, pos, neg, pi, neg_centers, kw, M, seed, wass_seed, diag, wrec):
    def cb(t, x_t, x0hat, pu_proba, w_u, w_c, w_q, lam_t):
        p_pu = pu_proba.detach().cpu().numpy()
        p_true = compute_post(x0hat, pos, neg, pi).detach().cpu().numpy()
        label = (p_true > 0.5).astype(float)
        with torch.enable_grad():
            xh = x0hat.detach().clone().requires_grad_(True)
            g_pu = torch.autograd.grad(torch.log(1.0 - safe_clip01(pu.predict_proba_pos(xh))).sum(), xh)[0]
            xh2 = x0hat.detach().clone().requires_grad_(True)
            g_tr = torch.autograd.grad(torch.log(1.0 - safe_clip01(compute_post(xh2, pos, neg, pi))).sum(), xh2)[0]
        gcos = float(F.cosine_similarity(g_pu, g_tr, dim=-1).mean())
        b = _bin(t)
        diag.append(
            {
                "bin": b,
                "auc": _auc(p_pu, label),
                "rmse": float(np.sqrt(np.mean((p_pu - p_true) ** 2))),
                "brier": float(np.mean((p_pu - label) ** 2)),
                "gcos": gcos,
            }
        )
        wrec.append({"bin": b, "u": _essf(w_u), "c": _essf(w_c), "q": _essf(w_q)})

    t0 = time.time()
    pusmc = PUNegGuideDDPMSMCDiag(
        schedule=sched,
        pos_x0_gmm=pos,
        neg_x0_gmm=neg,
        class_prior=pi,
        pu=pu,
        M=M,
        pu_class_prior=pi,
        seed=seed,
        **kw,
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
    parser = argparse.ArgumentParser(description="Ours ON_GRAD: M ablation + diagnostics")
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--M", type=int, nargs="+", default=MS)
    parser.add_argument("--layouts", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    kw = VARIANTS[args.variant]
    layouts = list(range(args.layouts))
    seeds = args.seeds
    n_trials = len(layouts) * len(seeds)

    out_dir = args.outdir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    tag = "_".join(str(m) for m in args.M)
    log_path = os.path.join(out_dir, f"tuned_M_diag_{args.variant}_{tag}_log.txt")
    log = Logger(log_path)
    sys.stdout = log

    sched = DDPMSchedule(T=T)
    print("=" * 100)
    print(f"Ours ON_GRAD ({args.variant}) — M ablation + diagnostics — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"kwargs: {kw}")
    print(f"calib pi, wq_ratio=True, T={T}, Ms {args.M}, {len(layouts)} layouts x {len(seeds)} seeds = {n_trials} trials")
    print("=" * 100)

    perf = []
    diag_by_M, wrec_by_M = {}, {}
    for M in args.M:
        print(f"\n--- M={M} ---")
        orc_r, ours_r, diag, wrec = [], [], [], []
        for ls in layouts:
            pos, neg, pi = make_ring_gmm(seed=ls)
            neg_centers = torch.stack([g.mean for g in neg.comps])
            for seed in seeds:
                wass_seed = (ls + 1) * (seed + 1)
                xo = OracleDiffusion(schedule=sched, neg_x0_gmm=neg, M=M, seed=seed).sample()
                op = compute_post(xo, pos, neg, pi)
                orc_r.append({"pm": float(op.mean()), "wass": wasserstein(xo, neg, generator=torch.Generator().manual_seed(wass_seed))})
                pu = make_pu()
                r = run_ours(pu, sched, pos, neg, pi, neg_centers, kw, M, seed, wass_seed, diag, wrec)
                ours_r.append(r)
                print(
                    f"  L{ls}s{seed:>3}: Orc={float(op.mean()):.4f} raw={r['raw_pm']:.4f}/{r['raw_wass']:.1f} ESS/M={r['ess'] / M:.2f} rs={r['rs']} t={r['time']:.1f}s"
                )
        diag_by_M[M], wrec_by_M[M] = diag, wrec
        perf.append(
            {
                "M": M,
                "opm": np.mean([x["pm"] for x in orc_r]),
                "ow": np.mean([x["wass"] for x in orc_r]),
                "rpm": np.mean([x["raw_pm"] for x in ours_r]),
                "rpm_s": np.std([x["raw_pm"] for x in ours_r]),
                "rw": np.mean([x["raw_wass"] for x in ours_r]),
                "rw_s": np.std([x["raw_wass"] for x in ours_r]),
                "ess": np.mean([x["ess"] for x in ours_r]) / M,
                "rs": np.mean([x["rs"] for x in ours_r]),
                "rs_s": np.std([x["rs"] for x in ours_r]),
                "mc": np.mean([x["mc"] for x in ours_r]),
                "tm": np.mean([x["time"] for x in ours_r]),
            }
        )

    print(f"\n{'=' * 100}")
    print(f"M ABLATION — performance (tuned {args.variant})")
    print(f"{'=' * 100}")
    print(
        f"  {'M':>6} | {'orc_pm':>8} {'orc_wass':>8} | {'raw_pm':>8} {'±':>6} | {'raw_wass':>8} {'±':>4} | {'ESS/M':>6} | {'rs':>8} | {'mc':>5} | {'s/trial':>8}"
    )
    print(f"  {'-' * 100}")
    for s in perf:
        print(
            f"  {s['M']:>6} | {s['opm']:>8.4f} {s['ow']:>8.1f} | {s['rpm']:>8.4f} {s['rpm_s']:>6.4f} | {s['rw']:>8.1f} {s['rw_s']:>4.1f} | {s['ess']:>6.2f} | {s['rs']:>4.1f}±{s['rs_s']:.1f} | {s['mc']:>5.2f} | {s['tm']:>8.1f}"
        )

    print(f"\n{'=' * 100}")
    print("f_t CLASSIFIER DIAGNOSTICS vs oracle posterior")
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

    print(f"\n{'=' * 100}")
    print("ESS/M of classifier weights (w_u unlabeled, w_c positive, w_q SMC; min = worst step)")
    print(f"{'=' * 100}")
    print(f"  {'M':>6} | {'bin':>8} | {'w_u mean':>9} {'w_u min':>8} | {'w_c mean':>9} {'w_c min':>8} | {'w_q mean':>9} {'w_q min':>8}")
    print(f"  {'-' * 90}")
    for M in args.M:
        rec = wrec_by_M[M]
        for label in ["overall", "early", "mid", "late"]:
            sub = rec if label == "overall" else [r for r in rec if r["bin"] == label]
            um = np.mean([r["u"] for r in sub])
            un = np.min([r["u"] for r in sub])
            cm = np.mean([r["c"] for r in sub])
            cn = np.min([r["c"] for r in sub])
            qm = np.mean([r["q"] for r in sub])
            qn = np.min([r["q"] for r in sub])
            print(f"  {M:>6} | {label:>8} | {um:>9.3f} {un:>8.3f} | {cm:>9.3f} {cn:>8.3f} | {qm:>9.3f} {qn:>8.3f}")

    print(f"\nDone — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = log.out
    log.close()
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
