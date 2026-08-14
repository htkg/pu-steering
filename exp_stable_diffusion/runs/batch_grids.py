"""Per-run particle grids: the first 64 particles of each SMC run (contiguous block
of M cached images), tiled 8x8 — visualizes within-run particle degeneracy after
resampling (contrast with the random grids of sample_grids.py).

Config (argv[1] JSON, optional): {dataset, method, M, n, ncol}.
Output: results/dataset/{tag}/grids/ours_batches/ours_batch{k}_{n}.png
"""

import json
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "{}"
    c: dict = json.loads(Path(arg).read_text() if Path(arg).exists() else arg)
    ds = Path(c.get("dataset", "results/dataset/cafe_terrace/dataset.pt"))
    name = c.get("method", "ou_m128_e5_l16")
    M = int(c.get("M", 128))
    n = int(c.get("n", 64))
    ncol = int(c.get("ncol", 8))

    path = ds.parent / "methods" / f"{name}_shard0.pt"
    out_dir = ds.parent / "grids" / "ours_batches"
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = torch.load(path, map_location="cpu")["images"]  # (N,3,H,W) uint8, N = runs*M
    for k in range(imgs.shape[0] // M):
        x = imgs[k * M : k * M + n].float() / 255.0
        out = out_dir / f"ours_batch{k}_{n}.png"
        save_image(x, out, nrow=ncol)
        print(f"[grid] {name} run {k}: particles {k * M}..{k * M + n - 1} -> {out}", flush=True)


if __name__ == "__main__":
    main()
