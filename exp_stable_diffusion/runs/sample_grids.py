"""Sample-image grids: 64 random cached images per method (seed 0) tiled into an
8x8 grid. Loads cached uint8 images only (no GPU).

By default shows one representative from each family: Ours (ou_m128_e5_l16),
SLD (sld_xxmax), NegCFG (negcfg_s4), and DNG (dng_c_l200). Override the list
with ``grid_methods`` in the optional config.

Config (argv[1] JSON, optional): {dataset, grid_methods, n, ncol, seed}.
Output: results/dataset/{tag}/grids/sample_{method}_{n}.png
"""

import json
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

GRID_METHODS = ["ou_m128_e5_l16", "sld_xxmax", "negcfg_s4", "dng_c_l200"]


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "{}"
    c: dict = json.loads(Path(arg).read_text() if Path(arg).exists() else arg)
    ds = Path(c.get("dataset", "results/dataset/cafe_terrace/dataset.pt"))
    methods = c.get("grid_methods", GRID_METHODS)
    n = int(c.get("n", 64))
    ncol = int(c.get("ncol", 8))
    seed = int(c.get("seed", 0))

    mdir = ds.parent / "methods"
    out_dir = ds.parent / "grids"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name in methods:
        paths = sorted(mdir.glob(f"{name}_shard*.pt"))
        if not paths:
            print(f"[skip] {name}: no shards", flush=True)
            continue
        imgs = torch.cat([torch.load(p, map_location="cpu")["images"] for p in paths], 0)  # (N,3,H,W) uint8
        N = imgs.shape[0]
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(N, generator=g)[:n]
        x = imgs[idx].float() / 255.0  # float only on the picked subset
        del imgs
        out = out_dir / f"sample_{name}_{n}.png"
        save_image(x, out, nrow=ncol)
        print(f"[grid] {name}: N={N} random{n} -> {out}", flush=True)


if __name__ == "__main__":
    main()
