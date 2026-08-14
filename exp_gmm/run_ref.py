"""Case 2: Reference-based negative guidance experiments.

Usage:
    uv run python run_ref.py --mode OFF --proposal grad
    uv run python run_ref.py --mode OFF --proposal sd
    uv run python run_ref.py --mode ON --proposal grad
    uv run python run_ref.py --mode ON --proposal sd
    uv run python run_ref.py --mode OFF --proposal both
    uv run python run_ref.py --mode ON --layouts 5 --seeds 0 7 42
"""

import argparse
import os
import sys
import time

os.environ["TQDM_DISABLE"] = "1"

import numpy as np
import torch

from gmm_ddpm import DDPMSchedule, GaussianMixture, OracleDiffusion
from gmm_ring import Logger, compute_post, make_ring_gmm, wasserstein
from kernel_model import KernelPULogistic
from sampler_ref import PUNegGuideDDPMSMC, SafeDenoiserDDPM, SDPropPUSMC

T = 100
M = 1024


def run_baselines(
    sched: DDPMSchedule,
    pos: GaussianMixture,
    neg: GaussianMixture,
    pi: float,
    neg_centers: torch.Tensor,
    seed: int,
    sd_etas: list[float],
    wass_seed: int,
):
    """Run Oracle + SafeDenoiser baselines. Returns dict."""
    r = {}

    oracle = OracleDiffusion(schedule=sched, neg_x0_gmm=neg, M=M, seed=seed)
    xo = oracle.sample()
    op = compute_post(xo, pos, neg, pi)
    r["orc_pm"] = float(op.mean())
    r["orc_wass"] = wasserstein(xo, neg, generator=torch.Generator().manual_seed(wass_seed))

    for eta in sd_etas:
        sd = SafeDenoiserDDPM(schedule=sched, pos_x0_gmm=pos, neg_x0_gmm=neg, class_prior=pi, M=M, N_ref=512, eta=eta, t_start=T, t_stop=5, seed=seed)
        xsd = sd.sample()
        sp = compute_post(xsd, pos, neg, pi)
        r[f"sd{eta}_pm"] = float(sp.mean())
        r[f"sd{eta}_wass"] = wasserstein(xsd, neg, generator=torch.Generator().manual_seed(wass_seed))
        r[f"sd{eta}_mc"] = float((torch.cdist(xsd, neg_centers).min(0).values < 5).float().mean())

    return r


def run_grad_proposal(
    sched: DDPMSchedule, pos: GaussianMixture, neg: GaussianMixture, pi: float, neg_centers: torch.Tensor, seed: int, mode: str, wass_seed: int
):
    """Run PUNegGuideDDPMSMC (classifier gradient proposal)."""
    if mode == "OFF":
        pu = KernelPULogistic(sigma=None, sigma_scale=0.3, lam=1e-4, n_centers_pos=128, n_centers_u=64, max_iter=100)
        pusmc = PUNegGuideDDPMSMC(
            schedule=sched,
            pos_x0_gmm=pos,
            neg_x0_gmm=neg,
            class_prior=pi,
            pu=pu,
            N_pos=512,
            M=M,
            pu_class_prior=0.05,
            grad_clip=3.0,
            ess_target=0.9,
            ess_resample=0.9,
            resample_stop=100,
            lam_schedule="adaptive",
            lam_max=1.0,
            seed=seed,
        )
    else:
        pu = KernelPULogistic(sigma=None, sigma_scale=0.3, lam=1e-2, n_centers_pos=128, n_centers_u=64, max_iter=100)
        pusmc = PUNegGuideDDPMSMC(
            schedule=sched,
            pos_x0_gmm=pos,
            neg_x0_gmm=neg,
            class_prior=pi,
            pu=pu,
            N_pos=512,
            M=M,
            pu_class_prior=0.25,
            grad_clip=3.0,
            ess_target=0.9,
            ess_resample=0.95,
            resample_stop=20,
            guidance_stop=20,
            resample_method="systematic_sorted",
            lam_schedule="cos",
            lam_max=2.0,
            seed=seed,
        )

    xp, wp = pusmc.sample()
    p = compute_post(xp, pos, neg, pi)
    wn = wp / (wp.sum() + 1e-12)

    return {
        "raw_pm": float(p.mean()),
        "raw_lk": float((p > 0.5).float().mean()),
        "w_pm": float((wn * p).sum()),
        "w_lk": float((wn * (p > 0.5).float()).sum()),
        "raw_wass": wasserstein(xp, neg, generator=torch.Generator().manual_seed(wass_seed)),
        "w_wass": wasserstein(xp, neg, wp, generator=torch.Generator().manual_seed(wass_seed)),
        "ess": float(1.0 / (wn.pow(2).sum() + 1e-12)),
        "rs": pusmc.resample_count,
        "mc": float((torch.cdist(xp, neg_centers).min(0).values < 5).float().mean()),
    }


def run_sd_proposal(
    sched: DDPMSchedule, pos: GaussianMixture, neg: GaussianMixture, pi: float, neg_centers: torch.Tensor, seed: int, mode: str, wass_seed: int
):
    """Run SDPropPUSMC (SafeDenoiser proposal)."""
    if mode == "OFF":
        pu = KernelPULogistic(sigma=None, sigma_scale=0.3, lam=1e-4, n_centers_pos=128, n_centers_u=64, max_iter=100)
        sdprop = SDPropPUSMC(
            schedule=sched,
            pos_x0_gmm=pos,
            neg_x0_gmm=neg,
            class_prior=pi,
            pu=pu,
            N_pos=512,
            M=M,
            eta=30,
            sd_t_stop=0,
            pu_class_prior=0.18,
            correction_clip=0.7,
            lam_schedule="linear",
            lam_target=1.0,
            ess_resample=0.5,
            resample_stop=100,
            seed=seed,
        )
    else:
        pu = KernelPULogistic(sigma=None, sigma_scale=0.3, lam=1e-2, n_centers_pos=128, n_centers_u=64, max_iter=100)
        sdprop = SDPropPUSMC(
            schedule=sched,
            pos_x0_gmm=pos,
            neg_x0_gmm=neg,
            class_prior=pi,
            pu=pu,
            N_pos=512,
            M=M,
            eta=30,
            sd_t_stop=20,
            pu_class_prior=0.25,
            correction_clip=0.0,
            lam_schedule="cos",
            lam_target=2.0,
            ess_resample=0.97,
            resample_method="systematic_sorted",
            resample_stop=20,
            seed=seed,
        )

    xp, wp = sdprop.sample()
    p = compute_post(xp, pos, neg, pi)
    wn = wp / (wp.sum() + 1e-12)

    return {
        "raw_pm": float(p.mean()),
        "raw_lk": float((p > 0.5).float().mean()),
        "w_pm": float((wn * p).sum()),
        "w_lk": float((wn * (p > 0.5).float()).sum()),
        "raw_wass": wasserstein(xp, neg, generator=torch.Generator().manual_seed(wass_seed)),
        "w_wass": wasserstein(xp, neg, wp, generator=torch.Generator().manual_seed(wass_seed)),
        "ess": float(1.0 / (wn.pow(2).sum() + 1e-12)),
        "rs": sdprop.resample_count,
        "mc": float((torch.cdist(xp, neg_centers).min(0).values < 5).float().mean()),
    }


def print_summary(name: str, results: list[dict]):
    """Print summary statistics for a list of result dicts."""
    rpms = [r["raw_pm"] for r in results]
    wpms = [r["w_pm"] for r in results]
    rws = [r["raw_wass"] for r in results]
    wws = [r["w_wass"] for r in results]
    mcs = [r["mc"] for r in results]
    esss = [r["ess"] for r in results]
    rss = [r["rs"] for r in results]

    print(f"  {name} (raw): post_mean={np.mean(rpms):.4f} ± {np.std(rpms):.4f} Wasserstein={np.mean(rws):.1f} ± {np.std(rws):.1f}")
    print(f"  {name} (weighted): post_mean={np.mean(wpms):.4f} ± {np.std(wpms):.4f} Wasserstein={np.mean(wws):.1f} ± {np.std(wws):.1f}")
    print(f"  mode_coverage={np.mean(mcs):.3f} ESS={np.mean(esss):.0f} ± {np.std(esss):.0f} resamples={np.mean(rss):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Case 2: Reference-based experiments")
    parser.add_argument("--mode", choices=["OFF", "ON"], required=True)
    parser.add_argument("--proposal", choices=["grad", "sd", "both"], default="both")
    parser.add_argument("--layouts", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 42])
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    mode = args.mode
    proposal = args.proposal
    layouts = list(range(args.layouts))
    seeds = args.seeds
    n_trials = len(layouts) * len(seeds)

    run_grad = proposal in ("grad", "both")
    run_sd = proposal in ("sd", "both")

    out_dir = args.outdir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, f"ref_{mode}_{proposal}_log.txt")
    log = Logger(log_path)
    sys.stdout = log

    sd_etas = [30.0, 50.0] if mode == "ON" else [50.0, 100.0]

    print("=" * 110)
    print(f"Case 2: Reference-based — Resample {mode} — Proposal: {proposal} — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    if run_grad:
        if mode == "OFF":
            print("Grad: cp=0.05, gc=3.0, adaptive lam, rs_stop=100")
        else:
            print("Grad ON: cp=0.25, cos lam_max=2.0, gs=20, rs_stop=20, systematic_sorted")
    if run_sd:
        if mode == "OFF":
            print("SD: eta=30, cc=0.7, cp=0.18, linear lam=1.0, rs_stop=100")
        else:
            print("SD: eta=30, sd_s=20, cp=0.25, cos lam=2.0, rs_stop=20, systematic_sorted")
    print(f"SafeDenoiser baselines: eta={sd_etas}")
    print(f"Ring GMM (radius=10), T={T}, M={M}")
    print(f"{len(layouts)} layouts x {len(seeds)} seeds = {n_trials} trials")
    print("=" * 110)

    sched = DDPMSchedule(T=T)
    all_base = []
    all_grad = []
    all_sd = []

    for ls in layouts:
        pos, neg, pi = make_ring_gmm(seed=ls)
        neg_centers = torch.stack([g.mean for g in neg.comps])

        for seed in seeds:
            wass_seed = (ls + 1) * (seed + 1)
            t0 = time.time()
            base = run_baselines(sched, pos, neg, pi, neg_centers, seed, sd_etas, wass_seed=wass_seed)

            line = f"  L{ls}s{seed:>3}: Orc={base['orc_pm']:.4f}"
            for eta in sd_etas:
                line += f" SD{eta}={base[f'sd{eta}_pm']:.4f}"

            if run_grad:
                t0 = time.time()
                gr = run_grad_proposal(sched, pos, neg, pi, neg_centers, seed, mode, wass_seed=wass_seed)
                grad_elapsed = time.time() - t0
                line += f" | Grad: raw={gr['raw_pm']:.4f} w={gr['w_pm']:.4f} mc={gr['mc']:.2f} ESS={gr['ess']:.0f} rs={gr['rs']} elapsed={grad_elapsed:.1f}s"
                all_grad.append(gr)

            if run_sd:
                t0 = time.time()
                sd = run_sd_proposal(sched, pos, neg, pi, neg_centers, seed, mode, wass_seed=wass_seed)
                sd_elapsed = time.time() - t0
                line += f" | SDProp: raw={sd['raw_pm']:.4f} w={sd['w_pm']:.4f} mc={sd['mc']:.2f} ESS={sd['ess']:.0f} rs={sd['rs']} elapsed={sd_elapsed:.1f}s"
                all_sd.append(sd)

            print(line)
            all_base.append(base)

    n = len(all_base)
    print(f"\n{'=' * 110}")
    print(f"SUMMARY: Resample {mode} — Proposal: {proposal} ({n} trials)")
    print(f"{'=' * 110}")

    orc_pms = [r["orc_pm"] for r in all_base]
    print(f"  Oracle:   post_mean={np.mean(orc_pms):.4f} ± {np.std(orc_pms):.4f}")
    for eta in sd_etas:
        pms = [r[f"sd{eta}_pm"] for r in all_base]
        ws = [r[f"sd{eta}_wass"] for r in all_base]
        print(f"  SD(eta={eta}): post_mean={np.mean(pms):.4f} ± {np.std(pms):.4f} Wasserstein={np.mean(ws):.1f} ± {np.std(ws):.1f}")

    if run_grad and all_grad:
        print()
        print_summary("Grad", all_grad)

        sd_ref_eta = sd_etas[0]
        sd_ref_pms = [r[f"sd{sd_ref_eta}_pm"] for r in all_base]
        grad_rpms = [r["raw_pm"] for r in all_grad]
        grad_wpms = [r["w_pm"] for r in all_grad]
        print(f"\n  Win rates vs SD(eta={sd_ref_eta}):")
        print(f"    raw_pm: {sum(1 for a, b in zip(grad_rpms, sd_ref_pms) if a < b)}/{n}")
        print(f"    w_pm:   {sum(1 for a, b in zip(grad_wpms, sd_ref_pms) if a < b)}/{n}")

    if run_sd and all_sd:
        print()
        print_summary("SDProp", all_sd)

        sd_ref_eta = sd_etas[0]
        sd_ref_pms = [r[f"sd{sd_ref_eta}_pm"] for r in all_base]
        sd_rpms = [r["raw_pm"] for r in all_sd]
        sd_wpms = [r["w_pm"] for r in all_sd]
        print(f"\n  Win rates vs SD(eta={sd_ref_eta}):")
        print(f"    raw_pm: {sum(1 for a, b in zip(sd_rpms, sd_ref_pms) if a < b)}/{n}")
        print(f"    w_pm:   {sum(1 for a, b in zip(sd_wpms, sd_ref_pms) if a < b)}/{n}")

    print(f"\nDone — {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sys.stdout = log.out
    log.close()
    print(f"Log: {log_path}")


if __name__ == "__main__":
    main()
