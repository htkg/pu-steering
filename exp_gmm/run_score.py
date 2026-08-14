"""Case 1: True-score negative guidance experiments.

Usage:
    uv run python run_score.py --mode OFF
    uv run python run_score.py --mode ON
    uv run python run_score.py --mode ON_TILT
    uv run python run_score.py --mode ON_GRAD48
    uv run python run_score.py --mode ON --layouts 5 --seeds 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import os
import sys
import time
from typing import Any

os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import torch
from gmm_ddpm import DDPMSchedule, OracleDiffusion
from gmm_ring import Logger, compute_post, make_ring_gmm, wasserstein
from kernel_model import KernelPULogistic
from sampler_score import DNGDDPM, NegCFG, PUNegGuideDDPMSMC

T = 100
M = 1024


def main():
    parser = argparse.ArgumentParser(description="Case 1: True-score experiments")
    parser.add_argument("--mode", choices=["OFF", "ON", "ON_TILT", "ON_GRAD40", "ON_GRAD48"], required=True)
    parser.add_argument("--layouts", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42])
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--save-particles", type=str, default=None, metavar="DIR", help="Save per-trial particle .pt files to DIR")
    args = parser.parse_args()

    mode = args.mode
    layouts = list(range(args.layouts))
    seeds = args.seeds
    n_trials = len(layouts) * len(seeds)

    out_dir = args.outdir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    save_particles = args.save_particles
    if save_particles:
        os.makedirs(save_particles, exist_ok=True)
    log_path = os.path.join(out_dir, f"score_{mode}_log.txt")
    log = Logger(log_path)
    sys.stdout = log

    print("=" * 110)
    print(f"Case 1: True-score — Resample {mode}")
    if mode == "OFF":
        print("Ours: no-resample, grad proposal, PU pi=class_prior (calib), tilt=3.0 start=0.6, gc=0.5, ss=0.2, lam=1e-4")
    elif mode == "ON":
        print("Ours: DNG(l0=0.6) proposal, w_q = IS ratio x tilt (lam=2.0 fixed), resample at t=40, calib, ss=0.2, lam=1e-4")
    elif mode == "ON_TILT":
        print("Ours: DNG(l0=0.6) proposal, w_q = tilt only (ESS-adaptive, tilt_max=3.0), resample at t=40, calib, ss=0.2, lam=1e-4")
    elif mode == "ON_GRAD40":
        print("Ours: grad proposal, w_q = IS ratio x tilt (cos 0->2, plateau 0.5T), ESS-fired resample (er=0.9, cooldown=40, stop=40), calib, ss=0.2, lam=1e-4")
    else:
        print("Ours: grad proposal, w_q = IS ratio x tilt (cos 0->2, plateau 0.5T), ESS-fired resample (er=0.9, cooldown=48, stop=38), calib, ss=0.2, lam=1e-4")
    print(f"Ring GMM (radius=10), T={T}, M={M}")
    print(f"{len(layouts)} layouts x {len(seeds)} seeds = {n_trials} trials")
    print("=" * 110)

    sched = DDPMSchedule(T=T)
    all_r = []

    for ls in layouts:
        pos, neg, pi = make_ring_gmm(seed=ls)
        neg_centers = torch.stack([g.mean for g in neg.comps])

        for seed in seeds:
            r = {"ls": ls, "seed": seed}
            wass_seed = (ls + 1) * (seed + 1)

            t0 = time.time()
            xo = OracleDiffusion(schedule=sched, neg_x0_gmm=neg, M=M, seed=seed).sample()
            op = compute_post(xo, pos, neg, pi)
            r["orc_pm"] = float(op.mean())
            r["orc_lk"] = float((op > 0.5).float().mean())
            r["orc_wass"] = wasserstein(xo, neg, generator=torch.Generator().manual_seed(wass_seed))
            r["orc_time"] = time.time() - t0

            t0 = time.time()
            dng = DNGDDPM(schedule=sched, pos_x0_gmm=pos, neg_x0_gmm=neg, class_prior=pi, M=M, lambda0=1, tau=1, pmin=1e-2, lam_max=20, seed=seed)
            xd, _dng_p = dng.sample()
            dp = compute_post(xd, pos, neg, pi)
            r["dng_pm"] = float(dp.mean())
            r["dng_lk"] = float((dp > 0.5).float().mean())
            r["dng_wass"] = wasserstein(xd, neg, generator=torch.Generator().manual_seed(wass_seed))
            r["dng_time"] = time.time() - t0

            t0 = time.time()
            cfg = NegCFG(schedule=sched, pos_x0_gmm=pos, neg_x0_gmm=neg, class_prior=pi, M=M, lam=2.0, seed=seed)
            xc = cfg.sample()
            cp_ = compute_post(xc, pos, neg, pi)
            r["cfg_pm"] = float(cp_.mean())
            r["cfg_lk"] = float((cp_ > 0.5).float().mean())
            r["cfg_wass"] = wasserstein(xc, neg, generator=torch.Generator().manual_seed(wass_seed))
            r["cfg_time"] = time.time() - t0

            t0 = time.time()
            pu = KernelPULogistic(sigma=None, sigma_scale=0.2, lam=1e-4, n_centers_pos=128, n_centers_u=64, max_iter=100)
            common: dict[str, Any] = {
                "schedule": sched,
                "pos_x0_gmm": pos,
                "neg_x0_gmm": neg,
                "class_prior": pi,
                "pu": pu,
                "M": M,
                "pu_class_prior": pi,
                "ess_target": 0.6,
                "seed": seed,
            }
            if mode == "OFF":
                pusmc = PUNegGuideDDPMSMC(
                    **common,
                    grad_clip=0.5,
                    tilt_max=3.0,
                    tilt_start=0.6,
                )
            elif mode == "ON":
                pusmc = PUNegGuideDDPMSMC(
                    **common,
                    proposal="dng",
                    dng_lambda0=0.6,
                    dng_tau=1.0,
                    dng_lam_max=4.0,
                    lam_fixed=2.0,
                    tilt_max=0.0,
                    resample_at="40",
                )
            elif mode == "ON_TILT":
                pusmc = PUNegGuideDDPMSMC(
                    **common,
                    proposal="dng",
                    dng_lambda0=0.6,
                    dng_tau=1.0,
                    dng_lam_max=4.0,
                    wq_ratio=False,
                    tilt_max=3.0,
                    resample_at="40",
                )
            elif mode == "ON_GRAD40":
                pusmc = PUNegGuideDDPMSMC(
                    **common,
                    grad_clip=0.5,
                    lam_mode="cos",
                    lam_end=2.0,
                    lam_ramp=0.5,
                    tilt_max=0.0,
                    ess_resample=0.9,
                    resample_stop=40,
                    resample_cooldown=40,
                )
            else:
                pusmc = PUNegGuideDDPMSMC(
                    **common,
                    grad_clip=0.5,
                    lam_mode="cos",
                    lam_end=2.0,
                    lam_ramp=0.5,
                    tilt_max=0.0,
                    ess_resample=0.9,
                    resample_stop=38,
                    resample_cooldown=48,
                )

            xp, wp = pusmc.sample()
            p = compute_post(xp, pos, neg, pi)
            wn = wp / (wp.sum() + 1e-12)

            r["raw_pm"] = float(p.mean())
            r["raw_lk"] = float((p > 0.5).float().mean())
            r["w_pm"] = float((wn * p).sum())
            r["w_lk"] = float((wn * (p > 0.5).float()).sum())
            r["raw_wass"] = wasserstein(xp, neg, generator=torch.Generator().manual_seed(wass_seed))
            r["w_wass"] = wasserstein(xp, neg, wp, generator=torch.Generator().manual_seed(wass_seed))
            r["ess"] = float(1.0 / (wn.pow(2).sum() + 1e-12))
            r["rs"] = pusmc.resample_count
            r["mc"] = float((torch.cdist(xp, neg_centers).min(0).values < 5).float().mean())
            r["ours_time"] = time.time() - t0

            wl = "W" if r["raw_pm"] < r["dng_pm"] else "L"
            print(
                f"  L{ls}s{seed:>3}: Orc={r['orc_pm']:.4f}/{r['orc_wass']:.1f} "
                f"DNG={r['dng_pm']:.4f}/{r['dng_wass']:.1f} "
                f"CFG={r['cfg_pm']:.4f}/{r['cfg_wass']:.1f} "
                f"raw={r['raw_pm']:.4f}/{r['raw_wass']:.1f} "
                f"w={r['w_pm']:.4f}/{r['w_wass']:.1f} "
                f"mc={r['mc']:.2f} ESS={r['ess']:.0f} rs={r['rs']} "
                f"elapsed={r['ours_time']:.1f}s {wl}"
            )
            all_r.append(r)

            if save_particles:
                tag = f"L{ls}_s{seed}"
                torch.save({"x": xo, "post": op}, os.path.join(save_particles, f"{tag}_oracle.pt"))
                torch.save({"x": xd, "post": dp, "p_bad": _dng_p}, os.path.join(save_particles, f"{tag}_dng.pt"))
                torch.save({"x": xc, "post": cp_}, os.path.join(save_particles, f"{tag}_negcfg.pt"))
                torch.save({"x": xp, "w": wp, "post": p, "resample_count": pusmc.resample_count}, os.path.join(save_particles, f"{tag}_ours.pt"))

    n = len(all_r)
    print(f"\n{'=' * 110}")
    print(f"SUMMARY: Resample {mode} ({n} trials)")
    print(f"{'=' * 110}")

    for name, kp, kl, kw in [
        ("Oracle", "orc_pm", "orc_lk", "orc_wass"),
        ("DNG", "dng_pm", "dng_lk", "dng_wass"),
        ("NegCFG", "cfg_pm", "cfg_lk", "cfg_wass"),
    ]:
        pms = [r[kp] for r in all_r]
        lks = [r[kl] for r in all_r]
        ws = [r[kw] for r in all_r]
        print(f"  {name:<8}: post_mean={np.mean(pms):.4f} ± {np.std(pms):.4f} leak={np.mean(lks):.4f} Wasserstein={np.mean(ws):.1f} ± {np.std(ws):.1f}")

    rpms = [r["raw_pm"] for r in all_r]
    wpms = [r["w_pm"] for r in all_r]
    rlks = [r["raw_lk"] for r in all_r]
    wlks = [r["w_lk"] for r in all_r]
    rws = [r["raw_wass"] for r in all_r]
    wws = [r["w_wass"] for r in all_r]
    dpms = [r["dng_pm"] for r in all_r]
    opms = [r["orc_pm"] for r in all_r]
    mcs = [r["mc"] for r in all_r]
    esss = [r["ess"] for r in all_r]
    rss = [r["rs"] for r in all_r]

    print(f"  Ours (raw): post_mean={np.mean(rpms):.4f} ± {np.std(rpms):.4f} leak={np.mean(rlks):.4f} Wasserstein={np.mean(rws):.1f} ± {np.std(rws):.1f}")
    print(f"  Ours (weighted): post_mean={np.mean(wpms):.4f} ± {np.std(wpms):.4f} leak={np.mean(wlks):.4f} Wasserstein={np.mean(wws):.1f} ± {np.std(wws):.1f}")
    print(
        f"  mode_coverage={np.mean(mcs):.3f} ± {np.std(mcs):.3f} ESS={np.mean(esss):.0f} ± {np.std(esss):.0f} resamples={np.mean(rss):.1f} ± {np.std(rss):.1f}"
    )

    print(f"\n  Win rates vs DNG ({n} trials):")
    print(f"    raw_pm: {sum(1 for a, b in zip(rpms, dpms) if a < b)}/{n}")
    print(f"    w_pm:   {sum(1 for a, b in zip(wpms, dpms) if a < b)}/{n}")

    orc_gap = np.mean(dpms) - np.mean(opms)
    print(f"\n  Oracle-DNG gap: {orc_gap:.4f}")
    if orc_gap > 1e-6:
        print(f"  Raw fills {(np.mean(dpms) - np.mean(rpms)) / orc_gap * 100:.0f}% of gap")
        print(f"  Weighted fills {(np.mean(dpms) - np.mean(wpms)) / orc_gap * 100:.0f}% of gap")

    print("\nDone")
    sys.stdout = log.out
    log.close()
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
