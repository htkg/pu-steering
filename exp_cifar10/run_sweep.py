"""Run a sequential single-class sweep.

The sweep covers six SafeDenoiser scales and four ``lam_max`` values. It runs
all seeds sequentially within each configuration, computes FID/KL/MPD, and
prints a summary table.

Usage:
  python run_sweep.py --target_class plane --seeds 8
  python run_sweep.py --target_class frog --seeds 8 --pu_lam 1.0
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np
import torch
from config import (
    CIFAR10_CLASSES,
    RESULTS_DIR,
    get_all_predictions,
    load_diag_images,
    load_eval_classifier,
    load_model_and_diffusion,
    load_reference_images,
    prepare_ref_images,
)
from evaluate import compute_fid, compute_kl, compute_lpips_mpd, save_pngs, save_seed_results
from improved_diffusion.gaussian_diffusion import GaussianDiffusion
from sampler import UNetFeatureExtractor, median_heuristic, run_trial

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)


def run_single_config(
    target_class: str,
    target_class_idx: int,
    lam_max: float,
    sd_scale: float,
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    feat_extractor: UNetFeatureExtractor,
    eval_clf: torch.nn.Module,
    diag_feats: dict[str, torch.Tensor],
    ref_dir: str,
    n_ref: int = 200,
    M: int = 128,
    n_seeds: int = 8,
    device: str = "cuda",
    pu_lam: float = 1e-3,
) -> dict:
    """Run one configuration and return summary dict."""

    prefix = f"{target_class}_" if target_class != "plane" else ""
    if lam_max == 0:
        name = f"{prefix}sd_{sd_scale}"
    else:
        name = f"{prefix}ours_lam{lam_max}"
    out_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    all_stats = []
    all_samples = []
    total_target = 0

    for seed in range(n_seeds):
        t0 = time.time()

        ref_seed = seed + 10000
        x_ref = load_reference_images(target_class, n_ref, seed=ref_seed).to(device)
        with torch.no_grad():
            t0_t = torch.zeros(n_ref, dtype=torch.long, device=device)
            feat_ref = feat_extractor.extract(x_ref, t0_t, pool=True)
        sd_sigma = median_heuristic(x_ref.reshape(n_ref, -1), scale=1.0)

        result = run_trial(
            model,
            diffusion,
            feat_extractor,
            eval_clf,
            diag_feats,
            x_ref=x_ref,
            feat_ref=feat_ref,
            sd_sigma=sd_sigma,
            pu_lam=pu_lam,
            lam_max=lam_max,
            sd_scale=sd_scale,
            target_class=target_class,
            target_class_idx=target_class_idx,
            M=M,
            seed=seed,
            device=device,
        )

        elapsed = time.time() - t0
        result["seed"] = seed
        result["time"] = elapsed
        result["ref_seed"] = ref_seed
        result["sd_sigma"] = float(sd_sigma)

        samples = result["samples"]
        save_seed_results(seed, result, samples, out_dir)
        all_samples.append(samples)
        total_target += result["n_target"]

        rs_steps = result["resample_steps"]
        gap_str = f"gap={result['gap']:+.3f}" if result["gap"] is not None else "gap=N/A"
        print(
            f"  [seed {seed}] raw={result['raw_target_rate']:.1%} "
            f"uniq={result['n_unique']}/{M} rs={result['resample_count']} "
            f"ESS={result['final_ess']:.1f} {gap_str} t={elapsed:.0f}s"
        )
        sys.stdout.flush()

        all_stats.append(
            {
                "seed": seed,
                "raw_target_rate": result["raw_target_rate"],
                "n_target": result["n_target"],
                "n_unique": result["n_unique"],
                "resample_count": result["resample_count"],
                "resample_steps": rs_steps,
                "final_ess": result["final_ess"],
                "ess_min": result["ess_min"],
                "gap": result["gap"],
                "sd_sigma": float(sd_sigma),
                "time": elapsed,
            }
        )

    # Save PNGs
    total_samples = M * n_seeds
    all_tensor = torch.cat(all_samples, dim=0)
    png_dir = os.path.join(out_dir, "png")
    save_pngs(all_tensor, png_dir)

    # Metrics
    raws = [s["raw_target_rate"] for s in all_stats]
    uniqs = [s["n_unique"] for s in all_stats]

    preds = get_all_predictions(eval_clf, all_tensor, device=device)
    kl = compute_kl(preds, total_samples, target_class=target_class)
    fid = compute_fid(png_dir, ref_dir)
    mpd = compute_lpips_mpd(all_tensor)

    shutil.rmtree(png_dir, ignore_errors=True)

    summary = {
        "config": {"lam_max": lam_max, "sd_scale": sd_scale, "pu_lam": pu_lam},
        "target_class": target_class,
        "N_ref": n_ref,
        "M": M,
        "n_seeds": n_seeds,
        "total_target": total_target,
        "total_samples": total_samples,
        "target_rate": total_target / total_samples,
        "raw_mean": float(np.mean(raws)),
        "raw_std": float(np.std(raws)),
        "uniq_mean": float(np.mean(uniqs)),
        "uniq_std": float(np.std(uniqs)),
        "kl": kl,
        "fid": fid,
        "mpd": mpd,
        "per_seed": all_stats,
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  => {target_class} rate={summary['target_rate']:.1%} FID={fid} KL={kl:.3f} MPD={mpd:.4f} uniq={np.mean(uniqs):.0f}/{M}")
    sys.stdout.flush()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_class", type=str, required=True, choices=CIFAR10_CLASSES)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--M", type=int, default=128)
    parser.add_argument("--n_ref", type=int, default=200)
    parser.add_argument("--pu_lam", type=float, default=1e-3)
    args = parser.parse_args()

    target_class = args.target_class
    device = "cuda"

    sd_scales = [0.0, 0.2, 0.3, 0.5, 0.7, 1.0]
    lam_maxs = [0.5, 1.0, 1.5, 2.0]

    print("=" * 70)
    print(f"Sweep: target={target_class}")
    print(f"SD scales: {sd_scales}")
    print(f"Ours lam_max: {lam_maxs}")
    print(f"M={args.M}, N_ref={args.n_ref}, seeds={args.seeds}, pu_lam={args.pu_lam}")
    print("=" * 70)
    sys.stdout.flush()

    model, diffusion = load_model_and_diffusion(device=device)
    eval_clf = load_eval_classifier(device=device)
    feat_extractor = UNetFeatureExtractor(model, device=device)

    print("Loading diagnostic images (500/class)...")
    diag_images = load_diag_images(n_per_class=500, seed=777)
    diag_feats = {}
    for cls, imgs in diag_images.items():
        t0 = torch.zeros(imgs.shape[0], dtype=torch.long, device=device)
        diag_feats[cls] = feat_extractor.extract(imgs.to(device), t0, pool=True)
    del diag_images
    torch.cuda.empty_cache()

    target_class_idx = CIFAR10_CLASSES.index(target_class)
    ref_dir = prepare_ref_images(target_class, RESULTS_DIR)

    class_results = {}

    # SD sweep
    for scale in sd_scales:
        tag = f"SD {scale}"
        print(f"\n--- {target_class} / {tag} ---")
        sys.stdout.flush()
        s = run_single_config(
            target_class,
            target_class_idx,
            lam_max=0.0,
            sd_scale=scale,
            model=model,
            diffusion=diffusion,
            feat_extractor=feat_extractor,
            eval_clf=eval_clf,
            diag_feats={},
            ref_dir=ref_dir,
            n_ref=args.n_ref,
            M=args.M,
            n_seeds=args.seeds,
            device=device,
            pu_lam=args.pu_lam,
        )
        class_results[tag] = s

    # Ours sweep
    for lam in lam_maxs:
        tag = f"Ours lam{lam}"
        print(f"\n--- {target_class} / {tag} ---")
        sys.stdout.flush()
        s = run_single_config(
            target_class,
            target_class_idx,
            lam_max=lam,
            sd_scale=0.3,
            model=model,
            diffusion=diffusion,
            feat_extractor=feat_extractor,
            eval_clf=eval_clf,
            diag_feats=diag_feats,
            ref_dir=ref_dir,
            n_ref=args.n_ref,
            M=args.M,
            n_seeds=args.seeds,
            device=device,
            pu_lam=args.pu_lam,
        )
        class_results[tag] = s

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {target_class}")
    print(f"{'=' * 70}")
    print(f"{'Method':<20} {'Rate':>7} {'FID':>7} {'KL':>7} {'MPD':>7} {'Uniq':>8}")
    print("-" * 60)
    for tag, s in class_results.items():
        fid_str = f"{s['fid']:.1f}" if s["fid"] else "  -"
        print(f"{tag:<20} {s['target_rate']:>6.1%} {fid_str:>7} {s['kl']:>6.3f} {s['mpd']:>6.4f} {s['uniq_mean']:>4.0f}/{s['M']}")

    # Save combined results
    def clean(obj: object) -> object:
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return obj

    out_path = os.path.join(RESULTS_DIR, f"sweep_{target_class}_results.json")
    with open(out_path, "w") as f:
        json.dump(clean(class_results), f, indent=2)
    print(f"\nResults saved to {out_path}")
    sys.stdout.flush()

    feat_extractor.close()


if __name__ == "__main__":
    main()
