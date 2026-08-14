"""Generate and save samples for one seed.

The script writes ``seed_NNN/samples.pt`` and ``seed_NNN/log.json``. Different
seeds can be launched in separate processes; metric computation is handled by
``evaluate.py``.

Single-class examples:
  python generate.py --seed 0 --lam_max 1.0 --target_class plane --out_dir results/my_exp
  python generate.py --seed 0 --lam_max 0   --sd_scale 0.3 --target_class plane --out_dir results/sd_0.3

Multi-class example:
  python generate.py --seed 0 --lam_max 1.0 --group vehicles --out_dir results/vehicles_ours
  python generate.py --seed 0 --lam_max 0   --sd_scale 0.3 --group vehicles --out_dir results/vehicles_sd
"""

import argparse
import os
import sys
import time

import torch

from config import (
    CIFAR10_CLASSES,
    GROUPS,
    count_group,
    load_diag_images,
    load_eval_classifier,
    load_model_and_diffusion,
    load_reference_images,
    load_reference_images_multi,
)
from evaluate import save_seed_results
from sampler import UNetFeatureExtractor, median_heuristic, run_trial

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lam_max", type=float, default=1.0)
    p.add_argument("--sd_scale", type=float, default=0.3)
    p.add_argument("--target_class", type=str, default=None, choices=CIFAR10_CLASSES)
    p.add_argument("--group", type=str, default=None, choices=list(GROUPS.keys()))
    p.add_argument("--n_ref", type=int, default=200, help="Reference images (per class for group mode)")
    p.add_argument("--M", type=int, default=128)
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--pu_lam", type=float, default=1e-3, help="L2 regularization for PU classifier")
    p.add_argument("--resample_stop", type=int, default=15)
    p.add_argument("--is_mode", type=str, default="none", choices=["none", "full"])
    p.add_argument("--is_tau", type=float, default=1.0, help="Temperature for IS ratio (tau>1 smooths)")
    p.add_argument("--use_denoised_feat", action="store_true", help="Use x0hat features at t=0 for classifier")
    args = p.parse_args()

    if args.target_class is None and args.group is None:
        print("Error: specify --target_class or --group")
        sys.exit(1)

    device = "cuda"
    seed = args.seed
    M = args.M
    is_multi = args.group is not None
    group = ""
    group_classes = []

    os.makedirs(args.out_dir, exist_ok=True)

    model, diffusion = load_model_and_diffusion(device=device)
    eval_clf = load_eval_classifier(device=device)
    feat_extractor = UNetFeatureExtractor(model, device=device)

    # Load diagnostic features for gap computation
    diag_feats = {}
    if args.lam_max > 0:
        diag_images = load_diag_images(n_per_class=500, seed=777)
        for cls, imgs in diag_images.items():
            t0 = torch.zeros(imgs.shape[0], dtype=torch.long, device=device)
            diag_feats[cls] = feat_extractor.extract(imgs.to(device), t0, pool=True)
        del diag_images
        torch.cuda.empty_cache()

    # Load reference images
    ref_seed = seed + 10000
    if is_multi:
        group = args.group
        assert group is not None
        group_classes = GROUPS[group]
        n_cls = len(group_classes)
        x_ref = load_reference_images_multi(group, n_per_class=args.n_ref, seed=ref_seed).to(device)
        N_ref = args.n_ref * n_cls
        # Use first class in group as nominal target for run_trial internals
        target_class = group_classes[0]
        target_class_idx = CIFAR10_CLASSES.index(target_class)
    else:
        target_class = args.target_class
        assert target_class is not None
        target_class_idx = CIFAR10_CLASSES.index(target_class)
        x_ref = load_reference_images(target_class, args.n_ref, seed=ref_seed).to(device)
        N_ref = args.n_ref

    with torch.no_grad():
        t0_t = torch.zeros(N_ref, dtype=torch.long, device=device)
        feat_ref = feat_extractor.extract(x_ref, t0_t, pool=True)
    sd_sigma = median_heuristic(x_ref.reshape(N_ref, -1), scale=1.0)

    t0 = time.time()
    result = run_trial(
        model,
        diffusion,
        feat_extractor,
        eval_clf,
        diag_feats,
        x_ref=x_ref,
        feat_ref=feat_ref,
        sd_sigma=sd_sigma,
        pu_lam=args.pu_lam,
        lam_max=args.lam_max,
        sd_scale=args.sd_scale,
        is_mode=args.is_mode,
        is_tau=args.is_tau,
        resample_stop=args.resample_stop,
        target_class=target_class,
        target_class_idx=target_class_idx,
        M=M,
        seed=seed,
        device=device,
        use_denoised_feat=args.use_denoised_feat,
    )
    elapsed = time.time() - t0

    samples = result["samples"]
    result["seed"] = seed
    result["time"] = elapsed
    result["ref_seed"] = ref_seed
    result["sd_sigma"] = float(sd_sigma)

    extra = {}
    if is_multi:
        # Recompute group rate
        n_group = count_group(eval_clf, samples, group, device=device)
        group_rate = n_group / M
        extra = {
            "group": group,
            "group_classes": group_classes,
            "n_per_class": args.n_ref,
            "N_ref": N_ref,
            "n_group": n_group,
            "group_rate": group_rate,
        }

    save_seed_results(seed, result, samples, args.out_dir, extra_log=extra if extra else None)

    if is_multi:
        print(
            f"[seed {seed}] group_rate={group_rate:.1%} ({n_group}/{M}) "  # type: ignore
            f"uniq={result['n_unique']}/{M} rs={result['resample_count']} "
            f"ESS={result['final_ess']:.1f} t={elapsed:.0f}s"
        )
    else:
        gap_str = f"gap={result['gap']:+.3f}" if result["gap"] is not None else "gap=N/A"
        gt = result.get("gap_trajectory", [])
        tgap_str = f"tgap={gt[-1][1]:.3f}" if gt else ""
        print(
            f"[seed {seed}] rate={result['raw_target_rate']:.1%} "
            f"uniq={result['n_unique']}/{M} rs={result['resample_count']} "
            f"ESS={result['final_ess']:.1f} {gap_str} {tgap_str} t={elapsed:.0f}s"
        )

    feat_extractor.close()


if __name__ == "__main__":
    main()
