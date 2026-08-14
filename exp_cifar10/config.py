"""Paths, constants, model loading, and data utilities for CIFAR-10 experiments.

Supports both single-class and multi-class (group) avoidance.
"""

import os
import sys

import numpy as np
import torch
from torch import nn
from torchvision import datasets, models, transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "improved-diffusion"))
from improved_diffusion.gaussian_diffusion import GaussianDiffusion
from improved_diffusion.script_util import create_model_and_diffusion

MODEL_PATH = os.path.join(SCRIPT_DIR, "improved-diffusion", "cifar10_uncond_50M_500K.pt")
EVAL_CLF_PATH = os.path.join(SCRIPT_DIR, "cifar10_classifier", "models", "cifar10_classifier.pt")
CIFAR10_DATA_ROOT = os.path.join(SCRIPT_DIR, "cifar10_classifier", "data")
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

CIFAR10_CLASSES = [
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# Multi-class group definitions
GROUPS = {
    "vehicles": ["plane", "car", "ship", "truck"],
    "animals": ["bird", "cat", "deer", "dog", "frog", "horse"],
}


def load_model_and_diffusion(timestep_respacing: str = "250", device: str = "cuda") -> tuple[nn.Module, GaussianDiffusion]:
    """Load improved-diffusion UNet + SpacedDiffusion."""
    model, diffusion = create_model_and_diffusion(
        image_size=32,
        class_cond=False,
        learn_sigma=True,
        sigma_small=False,
        num_channels=128,
        num_res_blocks=3,
        num_heads=4,
        num_heads_upsample=-1,
        attention_resolutions="16,8",
        dropout=0.3,
        diffusion_steps=4000,
        noise_schedule="cosine",
        timestep_respacing=timestep_respacing,
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=False,
        use_scale_shift_norm=True,
    )
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(ckpt)
    model.to(device).eval()
    return model, diffusion


def load_eval_classifier(device: str = "cuda") -> nn.Module:
    """Load evaluation ResNet-18 classifier for CIFAR-10."""
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()  # type: ignore
    m.fc = nn.Linear(512, 10)
    m.load_state_dict(torch.load(EVAL_CLF_PATH, map_location="cpu", weights_only=True))
    m.to(device).eval()
    return m


def _get_cifar10_dataset(train: bool = True) -> datasets.CIFAR10:
    """Get CIFAR-10 dataset via torchvision (no transform, raw PIL)."""
    return datasets.CIFAR10(root=CIFAR10_DATA_ROOT, train=train, download=False)


def load_reference_images(target_class: str, n_ref: int, seed: int = 42) -> torch.Tensor:
    """Load n_ref images of target_class as tensor [-1,1], shape (N,3,32,32).

    Uses the CIFAR-10 training set, selecting images of the target class
    randomly with the given seed.
    """
    ds = _get_cifar10_dataset(train=True)
    class_idx = CIFAR10_CLASSES.index(target_class)

    # Collect indices of the target class
    target_indices = [i for i, (_, label) in enumerate(ds) if label == class_idx]  # type: ignore
    rng = np.random.RandomState(seed)
    chosen = rng.choice(len(target_indices), n_ref, replace=False)

    imgs = []
    for idx in chosen:
        img = ds[target_indices[idx]][0]  # PIL Image
        arr = np.array(img).astype(np.float32) / 127.5 - 1.0
        imgs.append(arr)
    return torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2))


def load_reference_images_multi(group_name: str, n_per_class: int = 200, seed: int = 42) -> torch.Tensor:
    """Load reference images from all classes in a group. Returns (N_total, 3, 32, 32) in [-1,1]."""
    ds = _get_cifar10_dataset(train=True)
    rng = np.random.RandomState(seed)
    all_imgs = []

    for cls_name in GROUPS[group_name]:
        cls_idx = CIFAR10_CLASSES.index(cls_name)
        target_indices = [i for i, (_, label) in enumerate(ds) if label == cls_idx]  # type: ignore
        chosen = rng.choice(len(target_indices), min(n_per_class, len(target_indices)), replace=False)
        for c in chosen:
            img = ds[target_indices[c]][0]
            arr = np.array(img).astype(np.float32) / 127.5 - 1.0
            all_imgs.append(arr)

    return torch.from_numpy(np.stack(all_imgs).transpose(0, 3, 1, 2))


def load_diag_images(n_per_class: int = 500, seed: int = 777) -> dict[str, torch.Tensor]:
    """Load diagnostic images from test set for gap evaluation."""
    ds = _get_cifar10_dataset(train=False)
    rng = np.random.RandomState(seed)

    class_indices: dict[int, list[int]] = {i: [] for i in range(10)}
    for idx, (_, label) in enumerate(ds):  # type: ignore
        class_indices[label].append(idx)

    diag = {}
    for cls_idx, cls_name in enumerate(CIFAR10_CLASSES):
        candidates = class_indices[cls_idx]
        chosen = rng.choice(len(candidates), min(n_per_class, len(candidates)), replace=False)
        imgs = []
        for c in chosen:
            img = ds[candidates[c]][0]
            arr = np.array(img).astype(np.float32) / 127.5 - 1.0
            imgs.append(arr)
        diag[cls_name] = torch.from_numpy(np.stack(imgs).transpose(0, 3, 1, 2))
    return diag


def count_class(eval_clf: nn.Module, samples: torch.Tensor, class_idx: int = 0, device: str = "cuda") -> int:
    """Count samples classified as class_idx by the eval classifier."""
    norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    count = 0
    bs = 256
    for start in range(0, samples.shape[0], bs):
        batch = samples[start : start + bs].to(device)
        with torch.no_grad():
            preds = eval_clf(norm((batch + 1) / 2)).argmax(1)
        count += (preds == class_idx).sum().item()
    return count


def get_all_predictions(eval_clf: nn.Module, samples: torch.Tensor, device: str = "cuda") -> np.ndarray:
    """Get class predictions for all samples."""
    norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    all_preds = []
    bs = 256
    for start in range(0, samples.shape[0], bs):
        batch = samples[start : start + bs].to(device)
        with torch.no_grad():
            preds = eval_clf(norm((batch + 1) / 2)).argmax(1)
        all_preds.append(preds.cpu())
    return torch.cat(all_preds).numpy()


def tensor_to_uint8(x: torch.Tensor) -> np.ndarray:
    """Convert tensor [-1,1] to uint8 numpy [0,255], (N,H,W,C)."""
    x = ((x + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    return x.permute(0, 2, 3, 1).cpu().numpy()


def group_class_indices(group_name: str) -> list[int]:
    """Return list of CIFAR-10 class indices for a group."""
    return [CIFAR10_CLASSES.index(c) for c in GROUPS[group_name]]


def count_group(eval_clf: nn.Module, samples: torch.Tensor, group_name: str, device: str = "cuda") -> int:
    """Count samples classified as any class in the group."""
    norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    target_set = set(group_class_indices(group_name))
    count = 0
    bs = 256
    for start in range(0, samples.shape[0], bs):
        batch = samples[start : start + bs].to(device)
        with torch.no_grad():
            preds = eval_clf(norm((batch + 1) / 2)).argmax(1)
        count += sum(1 for p in preds.cpu().tolist() if p in target_set)
    return count


def prepare_ref_images(target_class: str, out_dir: str) -> str:
    """Save CIFAR-10 training images excluding target_class as PNGs for FID reference."""
    ref_dir = os.path.join(out_dir, f"ref_no_{target_class}")
    if os.path.exists(ref_dir) and len(os.listdir(ref_dir)) > 40000:
        return ref_dir

    os.makedirs(ref_dir, exist_ok=True)
    ds = _get_cifar10_dataset(train=True)
    class_idx = CIFAR10_CLASSES.index(target_class)

    count = 0
    for i, (img, label) in enumerate(ds):  # type: ignore
        if label == class_idx:
            continue
        img.save(os.path.join(ref_dir, f"{i:06d}.png"))
        count += 1

    print(f"Reference dir ready: {ref_dir} ({count} images)")
    return ref_dir


def prepare_ref_images_multi(group_name: str, out_dir: str) -> str:
    """Save CIFAR-10 training images EXCLUDING the group as PNGs for FID reference."""
    ref_dir = os.path.join(out_dir, f"ref_no_{group_name}")
    if os.path.exists(ref_dir) and len(os.listdir(ref_dir)) > 20000:
        return ref_dir

    os.makedirs(ref_dir, exist_ok=True)
    ds = _get_cifar10_dataset(train=True)
    exclude = set(group_class_indices(group_name))

    count = 0
    for i, (img, label) in enumerate(ds):  # type: ignore
        if label in exclude:
            continue
        img.save(os.path.join(ref_dir, f"{i:06d}.png"))
        count += 1

    print(f"Reference dir ready: {ref_dir} ({count} images)")
    return ref_dir
