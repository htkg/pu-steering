"""Aggregate per-seed samples and compute evaluation metrics.

CLI usage:
  python evaluate.py --out_dir results/my_exp --target_class plane
  python evaluate.py --out_dir results/my_exp --group vehicles
"""

import argparse
import json
import os
import shutil
import subprocess
import sys

import lpips
import numpy as np
import torch
from PIL import Image

from config import (
    CIFAR10_CLASSES,
    GROUPS,
    RESULTS_DIR,
    get_all_predictions,
    group_class_indices,
    load_eval_classifier,
    prepare_ref_images,
    prepare_ref_images_multi,
    tensor_to_uint8,
)


def compute_fid(gen_dir: str, ref_dir: str, device: str = "cuda") -> float | None:
    """Compute FID using pytorch-fid CLI."""
    result = subprocess.run(
        [sys.executable, "-m", "pytorch_fid", gen_dir, ref_dir, "--device", device, "--batch-size", "256"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if result.returncode != 0:
        print(f"FID error: {result.stderr}")
        return None
    for line in result.stdout.strip().split("\n"):
        if "FID" in line:
            try:
                return float(line.split(":")[-1].strip())
            except ValueError:
                pass
    print(f"FID parse error: {result.stdout}")
    return None


def compute_kl(preds: np.ndarray, n_total: int, target_class: str = "plane") -> float:
    """KL(target || sampled) where target = uniform over non-target classes."""
    target = np.array([0.0 if c == target_class else 1 / 9 for c in CIFAR10_CLASSES])
    sampled = np.clip(np.array([(preds == i).sum() / n_total for i in range(10)]), 1e-10, None)
    return sum(target[i] * np.log(target[i] / sampled[i]) for i in range(10) if target[i] > 0)


def compute_kl_multi(preds: np.ndarray, n_total: int, group_name: str) -> float:
    """KL(target || sampled) where target = uniform over non-group classes."""
    group_idx = set(group_class_indices(group_name))
    n_non_group = 10 - len(group_idx)
    target = np.array([0.0 if i in group_idx else 1.0 / n_non_group for i in range(10)])
    sampled = np.clip(np.array([(preds == i).sum() / n_total for i in range(10)]), 1e-10, None)
    return sum(target[i] * np.log(target[i] / sampled[i]) for i in range(10) if target[i] > 0)


def compute_lpips_mpd(samples_tensor: torch.Tensor, n_sub: int = 256, seed: int = 42) -> float:
    """Mean pairwise LPIPS distance (AlexNet)."""
    loss_fn = lpips.LPIPS(net="alex").cuda()
    loss_fn.eval()

    torch.manual_seed(seed)
    N = samples_tensor.shape[0]
    idx = torch.randperm(N)[:n_sub] if N > n_sub else torch.arange(N)
    imgs = samples_tensor[idx].cuda()
    n = len(imgs)

    dists = []
    with torch.no_grad():
        for i in range(n):
            for j_start in range(i + 1, n, 64):
                j_end = min(j_start + 64, n)
                ref = imgs[i : i + 1].expand(j_end - j_start, -1, -1, -1)
                d = loss_fn(ref, imgs[j_start:j_end])
                dists.extend(d.squeeze().cpu().tolist() if d.numel() > 1 else [d.item()])
    del loss_fn
    torch.cuda.empty_cache()
    return float(np.mean(dists))


def save_pngs(samples_tensor: torch.Tensor, png_dir: str) -> int:
    """Save samples as PNGs for FID computation."""
    os.makedirs(png_dir, exist_ok=True)
    imgs = tensor_to_uint8(samples_tensor)
    for j, img in enumerate(imgs):
        Image.fromarray(img).save(os.path.join(png_dir, f"{j:06d}.png"))
    return len(imgs)


def save_seed_results(seed: int, result: dict, samples_tensor: torch.Tensor, out_dir: str, extra_log: dict | None = None) -> str:
    """Save per-seed outputs: samples.pt, classifier.pt, log.json."""
    seed_dir = os.path.join(out_dir, f"seed_{seed:03d}")
    os.makedirs(seed_dir, exist_ok=True)

    torch.save(samples_tensor, os.path.join(seed_dir, "samples.pt"))

    if result.get("classifier_w") is not None:
        torch.save({"w": torch.tensor(result["classifier_w"]), "bias": torch.tensor(result["classifier_bias"])}, os.path.join(seed_dir, "classifier.pt"))

    skip_keys = {"samples", "classifier_w", "classifier_bias"}
    log = {}
    for k, v in result.items():
        if k in skip_keys:
            continue
        elif isinstance(v, (int, float, str, bool, type(None))):
            log[k] = v
        elif isinstance(v, list):
            log[k] = [[float(x) for x in item] if isinstance(item, (tuple, list)) else (float(item) if isinstance(item, (int, float)) else item) for item in v]
        else:
            log[k] = str(v)

    if extra_log:
        log.update(extra_log)

    with open(os.path.join(seed_dir, "log.json"), "w") as f:
        json.dump(log, f, indent=2)

    return seed_dir


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate seed results and compute metrics")
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--target_class", type=str, default=None, choices=CIFAR10_CLASSES)
    p.add_argument("--group", type=str, default=None, choices=list(GROUPS.keys()))
    p.add_argument("--M", type=int, default=128)
    args = p.parse_args()

    if args.target_class is None and args.group is None:
        print("Error: specify --target_class or --group")
        sys.exit(1)

    device = "cuda"
    out_dir = args.out_dir
    M = args.M
    is_multi = args.group is not None

    # Discover seeds
    all_samples = []
    all_stats = []
    seed_dirs = sorted([d for d in os.listdir(out_dir) if d.startswith("seed_")])

    if not seed_dirs:
        print(f"No seed directories found in {out_dir}")
        sys.exit(1)

    for sd in seed_dirs:
        samples_path = os.path.join(out_dir, sd, "samples.pt")
        log_path = os.path.join(out_dir, sd, "log.json")
        if not os.path.exists(samples_path):
            print(f"  WARNING: missing {samples_path}")
            continue
        all_samples.append(torch.load(samples_path, map_location="cpu", weights_only=True))
        if os.path.exists(log_path):
            with open(log_path) as f:
                all_stats.append(json.load(f))

    n_seeds = len(all_samples)
    if n_seeds == 0:
        print("No samples found.")
        sys.exit(1)

    all_tensor = torch.cat(all_samples, dim=0)
    total_samples = all_tensor.shape[0]

    print(f"Evaluating: {out_dir}")
    print(f"  {n_seeds} seeds, {total_samples} samples")
    sys.stdout.flush()

    # Save PNGs for FID
    png_dir = os.path.join(out_dir, "png")
    save_pngs(all_tensor, png_dir)
    print(f"  PNGs saved to {png_dir}")

    # Load classifier and get predictions
    eval_clf = load_eval_classifier(device=device)
    preds = get_all_predictions(eval_clf, all_tensor, device=device)

    if is_multi:
        group = args.group
        group_idx = set(group_class_indices(group))
        n_target = sum(1 for p in preds if p in group_idx)
        rate = n_target / total_samples
        print(f"  Group rate: {rate:.1%} ({n_target}/{total_samples})")

        # Per-class breakdown
        print("  Per-class:")
        for i in range(10):
            n = int((preds == i).sum())
            marker = "*" if i in group_idx else " "
            print(f"    {marker}{CIFAR10_CLASSES[i]:6s}: {n:4d} ({n / total_samples:.1%})")

        kl = compute_kl_multi(preds, total_samples, group)
        ref_dir = prepare_ref_images_multi(group, RESULTS_DIR)
    else:
        target_class = args.target_class
        assert target_class is not None
        target_class_idx = CIFAR10_CLASSES.index(target_class)
        n_target = int((preds == target_class_idx).sum())
        rate = n_target / total_samples
        print(f"  Rate: {rate:.1%} ({n_target}/{total_samples})")

        kl = compute_kl(preds, total_samples, target_class=target_class)
        ref_dir = prepare_ref_images(target_class, RESULTS_DIR)

    print(f"  KL: {kl:.3f}")

    fid = compute_fid(png_dir, ref_dir)
    print(f"  FID: {fid}")

    shutil.rmtree(png_dir, ignore_errors=True)

    mpd = compute_lpips_mpd(all_tensor)
    print(f"  MPD: {mpd:.4f}")

    # Summary
    raws = [s.get("raw_target_rate", s.get("group_rate", 0)) for s in all_stats]
    uniqs = [s.get("n_unique", M) for s in all_stats]

    # Recover config from first log
    config = {}
    if all_stats:
        for key in [
            "loss_type",
            "pu_lam",
            "pi",
            "lam_max",
            "sd_scale",
            "is_mode",
            "ess_thresh",
            "resample_stop",
            "resample_algo",
            "lam_schedule",
            "target_class",
            "target_class_idx",
            "group",
            "group_classes",
        ]:
            if key in all_stats[0]:
                config[key] = all_stats[0][key]

    summary = {
        "config": config,
        "M": M,
        "n_seeds": n_seeds,
        "total_samples": total_samples,
        "n_target": n_target,
        "target_rate": float(rate),
        "raw_mean": float(np.mean(raws)),
        "raw_std": float(np.std(raws)),
        "uniq_mean": float(np.mean(uniqs)),
        "uniq_std": float(np.std(uniqs)),
        "kl": kl,
        "fid": fid,
        "mpd": mpd,
        "per_seed": all_stats,
    }

    if is_multi:
        summary["group"] = args.group
        summary["group_classes"] = GROUPS[args.group]
    else:
        summary["target_class"] = args.target_class

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  => rate={rate:.1%} FID={fid} KL={kl:.3f} MPD={mpd:.4f}")
    print(f"  Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
