"""Evaluate the fair Ours (PUNegGuideDDPMSMC) over layouts x seeds; print JSON of pm/wass."""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("TQDM_DISABLE", "1")
import torch
from gmm_ddpm import DDPMSchedule, OracleDiffusion
from gmm_ring import compute_post, make_ring_gmm, wasserstein
from kernel_model import KernelPULogistic
from sampler_score import PUNegGuideDDPMSMC

T = 100


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layouts", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--M", type=int, default=1024)
    p.add_argument("--oracle", action="store_true")  # also run the Oracle at the same M
    p.add_argument("--pu-class-prior", type=float, default=0.02)
    p.add_argument("--calib-pi", action="store_true")  # set pu_class_prior=true pi
    p.add_argument("--tilt-max", type=float, default=1.0)
    p.add_argument("--tilt-start", type=float, default=1.0)  # warm-up: grow lam only for t<=start*T
    p.add_argument("--ess-target", type=float, default=0.8)
    p.add_argument("--resample-at", default="")
    p.add_argument("--grad-clip", type=float, default=3.0)
    p.add_argument("--proposal", choices=["grad", "dng"], default="grad")
    p.add_argument("--dng-l0", type=float, default=1.0)
    p.add_argument("--dng-tau", type=float, default=1.0)
    p.add_argument("--dng-lam-max", type=float, default=20.0)
    p.add_argument("--pu-sigma-scale", type=float, default=0.3)
    p.add_argument("--pu-sigma", type=float, default=None)
    p.add_argument("--pu-lam", type=float, default=1e-4)
    p.add_argument("--wq-tilt-only", action="store_true")  # exclude the proposal IS ratio from w_q
    p.add_argument("--lam-fixed", type=float, default=0.0)  # constant tempering exponent (0 = ESS-adaptive ramp)
    p.add_argument("--lam-mode", choices=["adaptive", "linear", "cos"], default="adaptive")
    p.add_argument("--lam-end", type=float, default=1.0)
    p.add_argument("--lam-ramp", type=float, default=1.0)
    p.add_argument("--ess-resample", type=float, default=0.0)
    p.add_argument("--resample-stop", type=int, default=0)
    p.add_argument("--resample-cooldown", type=int, default=0)
    p.add_argument("--tag", default="")
    p.add_argument("--dump", action="store_true")  # also print per-trial records
    a = p.parse_args()

    sched = DDPMSchedule(T=T)
    M = a.M
    raw_pm = []
    raw_w = []
    w_pm = []
    w_w = []
    rss = []
    mcs = []
    orc_pm = []
    orc_w = []
    for ls in range(a.layouts):
        pos, neg, pi = make_ring_gmm(seed=ls)
        neg_centers = torch.stack([g.mean for g in neg.comps])
        pu_cp = pi if a.calib_pi else a.pu_class_prior
        for seed in a.seeds:
            wass_seed = (ls + 1) * (seed + 1)
            if a.oracle:
                xo = OracleDiffusion(schedule=sched, neg_x0_gmm=neg, M=M, seed=seed).sample()
                orc_pm.append(float(compute_post(xo, pos, neg, pi).mean()))
                orc_w.append(wasserstein(xo, neg, generator=torch.Generator().manual_seed(wass_seed)))
            pu = KernelPULogistic(sigma=a.pu_sigma, sigma_scale=a.pu_sigma_scale, lam=a.pu_lam, n_centers_pos=128, n_centers_u=64, max_iter=100)
            smp = PUNegGuideDDPMSMC(
                schedule=sched,
                pos_x0_gmm=pos,
                neg_x0_gmm=neg,
                class_prior=pi,
                pu=pu,
                M=M,
                pu_class_prior=pu_cp,
                ess_target=a.ess_target,
                resample_at=a.resample_at,
                grad_clip=a.grad_clip,
                tilt_max=a.tilt_max,
                tilt_start=a.tilt_start,
                proposal=a.proposal,
                dng_lambda0=a.dng_l0,
                dng_tau=a.dng_tau,
                dng_lam_max=a.dng_lam_max,
                wq_ratio=not a.wq_tilt_only,
                lam_fixed=a.lam_fixed,
                lam_mode=a.lam_mode,
                lam_end=a.lam_end,
                lam_ramp=a.lam_ramp,
                ess_resample=a.ess_resample,
                resample_stop=a.resample_stop,
                resample_cooldown=a.resample_cooldown,
                seed=seed,
            )
            xp, wp = smp.sample()
            post = compute_post(xp, pos, neg, pi)
            wn = wp / (wp.sum() + 1e-12)
            raw_pm.append(float(post.mean()))
            w_pm.append(float((wn * post).sum()))
            raw_w.append(wasserstein(xp, neg, generator=torch.Generator().manual_seed(wass_seed)))
            w_w.append(wasserstein(xp, neg, wp, generator=torch.Generator().manual_seed(wass_seed)))
            rss.append(smp.resample_count)
            mcs.append(float((torch.cdist(xp, neg_centers).min(0).values < 5).float().mean()))
            if a.dump:
                d = {
                    "trial": 1,
                    "tag": a.tag,
                    "ls": ls,
                    "seed": seed,
                    "raw_pm": round(raw_pm[-1], 5),
                    "raw_wass": round(raw_w[-1], 4),
                    "w_pm": round(w_pm[-1], 5),
                    "w_wass": round(w_w[-1], 4),
                    "mc": round(mcs[-1], 4),
                    "rs": rss[-1],
                }
                if a.oracle:
                    d.update(orc_pm=round(orc_pm[-1], 5), orc_wass=round(orc_w[-1], 4))
                print(json.dumps(d), flush=True)

    n = len(raw_pm)
    mean = lambda v: round(sum(v) / n, 4)
    rec = {
        "tag": a.tag,
        "n": n,
        "M": M,
        "calib": a.calib_pi,
        "pu_cp": a.pu_class_prior,
        "tilt_max": a.tilt_max,
        "tilt_start": a.tilt_start,
        "ess_target": a.ess_target,
        "resample_at": a.resample_at,
        "grad_clip": a.grad_clip,
        "proposal": a.proposal,
        "dng_l0": a.dng_l0,
        "wq_tilt_only": a.wq_tilt_only,
        "lam_fixed": a.lam_fixed,
        "lam_mode": a.lam_mode,
        "lam_end": a.lam_end,
        "lam_ramp": a.lam_ramp,
        "ess_resample": a.ess_resample,
        "resample_stop": a.resample_stop,
        "resample_cooldown": a.resample_cooldown,
        "sig_scale": a.pu_sigma_scale,
        "pu_sigma": a.pu_sigma,
        "pu_lam": a.pu_lam,
        "raw_pm": mean(raw_pm),
        "raw_wass": mean(raw_w),
        "w_pm": mean(w_pm),
        "w_wass": mean(w_w),
        "mc": round(sum(mcs) / n, 3),
        "rs": round(sum(rss) / n, 1),
    }
    if a.oracle:
        rec.update(orc_pm=mean(orc_pm), orc_wass=mean(orc_w))
    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
