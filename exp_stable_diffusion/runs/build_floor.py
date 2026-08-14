"""N-matched oracle floor: a second independent not-c reference vs the first.

The build_dataset.py "reference halves" floor is N-mismatched (1024 vs 1024 while
methods are 2048 vs 2048). Here we take the separate floor pool (floor_part*.pt,
new seeds from ds_floor*.json), drop the top `drop_top_frac` by p(c) with the same
cached head to get an independent not-c set refB (2048), and report floor = FID /
KID / CSD-Fréchet of refB vs the existing reference refA. Both are N=2048, so it is
directly comparable to the methods.

Written to results/dataset/{tag}/floor.json (the frontier step reads it, never
overwrites it).

Config (argv[1] JSON): {tag, drop_top_frac, kid_subset, kid_n}.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import fid, kid


def load_concat(paths, key: str) -> torch.Tensor:
    return torch.cat([torch.load(p, map_location="cpu")[key] for p in paths], 0)


def mean_pc(csd: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-row calibrated p(c) = sigmoid(csd^T w + b)."""
    return torch.sigmoid(csd.double() @ w.double() + float(b))


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: python {Path(__file__).name} <config.json>")
    arg = sys.argv[1]
    c: dict = json.loads(Path(arg).read_text() if Path(arg).exists() else arg)
    tag = c.get("tag", "cafe_terrace")
    frac = float(c.get("drop_top_frac", 0.75))
    ksub = int(c.get("kid_subset", 1024))
    kn = int(c.get("kid_n", 100))
    d: Path = Path("results/dataset") / tag

    # reference refA: the not-c set from the main content pool (cached in dataset.pt).
    idx = torch.load(d / "dataset.pt", map_location="cpu")
    w, b = idx["head_w"], idx["head_b"]
    ref_idx = idx["reference_idx"]
    cshards = [d / s for s in idx["content_shards"]]
    refA_inc = load_concat(cshards, "incep").double()[ref_idx]
    refA_csd = load_concat(cshards, "csd")[ref_idx]
    refA_pc = float(mean_pc(refA_csd, w, b).mean())

    # reference refB: independent not-c set from the fresh floor pool (new seeds).
    fshards = sorted(d.glob("floor_part*.pt"))
    assert fshards, f"no floor_part*.pt in {d} (generate ds_floor*.json first)"
    f_inc = load_concat(fshards, "incep").double()
    f_csd = load_concat(fshards, "csd")
    n_pool = f_inc.shape[0]
    f_pc = mean_pc(f_csd, w, b)
    n_keep = round(n_pool * (1.0 - frac))
    keep = torch.argsort(f_pc)[:n_keep]  # lowest p(c)
    refB_inc, refB_csd = f_inc[keep], f_csd[keep]
    refB_pc = float(f_pc[keep].mean())
    print(f"[build_floor] floor pool N={n_pool} -> refB N={n_keep} refB_pc={refB_pc:.3f} (refA_pc={refA_pc:.3f}, expected to be close)", flush=True)

    out = {
        "n": int(n_keep),
        "pc": refB_pc,
        "refA_pc": refA_pc,
        "fid": fid(refB_inc, refA_inc),
        "kid": kid(refB_inc, refA_inc, subset_size=ksub, n_subsets=kn)[0],
        "csd_frechet": fid(refB_csd, refA_csd),
        "note": "N-matched 2048-vs-2048; refB = independent floor pool (new seeds)",
    }
    print(f"[floor N-matched {n_keep}v{refA_inc.shape[0]}] FID={out['fid']:.2f} KID={out['kid']:.5f} CSD-Frechet={out['csd_frechet']:.3f}", flush=True)
    (d / "floor.json").write_text(json.dumps(out, indent=2))
    print(f"[saved] {d}/floor.json", flush=True)


if __name__ == "__main__":
    main()
