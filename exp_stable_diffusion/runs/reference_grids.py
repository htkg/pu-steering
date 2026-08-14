"""Reference-image grids: 64 random images from each of the two context references,
unconditional content (floor pool, parts 1-7) and the avoided style (Van Gogh,
parts 0+1), tiled into 8x8 grids. Like sample_grids.py but for the reference
distributions rather than the methods. Loads cached uint8 images only (no GPU).

Config (argv[1] JSON, optional): {dataset, n, ncol, seed}.
Output: results/dataset/{tag}/grids/reference_{content,avoided_style}_{n}.png
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
    n = int(c.get("n", 64))
    ncol = int(c.get("ncol", 8))
    seed = int(c.get("seed", 0))
    d = ds.parent
    out_dir = d / "grids"
    out_dir.mkdir(parents=True, exist_ok=True)

    refs = {
        "content": [d / f"floor_part{p}.pt" for p in range(1, 8)],  # unconditional content
        "avoided_style": [d / f"vangogh_part{p}.pt" for p in (0, 1)],  # explicit Van Gogh
    }
    for name, paths in refs.items():
        paths = [p for p in paths if p.exists()]
        if not paths:
            print(f"[skip] {name}: no shards", flush=True)
            continue
        imgs = torch.cat([torch.load(p, map_location="cpu")["images"] for p in paths], 0)  # (N,3,H,W) uint8
        N = imgs.shape[0]
        g = torch.Generator().manual_seed(seed)
        idx = torch.randperm(N, generator=g)[:n]
        x = imgs[idx].float() / 255.0
        del imgs
        out = out_dir / f"reference_{name}_{n}.png"
        save_image(x, out, nrow=ncol)
        print(f"[grid] {name}: N={N} random{n} -> {out}", flush=True)


if __name__ == "__main__":
    main()
