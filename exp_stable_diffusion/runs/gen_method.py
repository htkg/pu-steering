"""Generate one shard of a method and cache {images, incep, csd}.

Two kinds:
  * `kind:"proposal"` (default): a bare negative-guidance baseline (NegCFG / SLD /
    DNG), no SMC tilt.
  * `kind:"ours"`: the proposal + online kernel-PU SMC tilt at `lam_max`, with
    `ours_kwargs` knobs; the cache also stores per-particle `weights`.
FID / p(c) are recomputed post-hoc from the features, not here.

Config (argv[1] JSON): {tag, name, kind, proposal, proposal_kwargs, shard,
seed_block, n, object, seeds, M, num_steps, dataset_cache [, lam_max, ours_kwargs]}.
Output: results/dataset/{tag}/methods/{name}_shard{shard}.pt
"""

import dataclasses
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.csd import CSDStyleEmbedder
from src.inception import inception_features
from src.sd import generate_method, load_pipeline
from src.smc import OursConfig, generate_ours

from src.proposals import DNG, SLD, NegCFG

DEVICE = "cuda"
STYLE_B = "Van Gogh"  # the avoided style B (method prompt suffix)
PROPOSALS = {"negcfg": NegCFG, "sld": SLD, "dng": DNG}


def _u8(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).cpu()


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: python {Path(__file__).name} <config.json>")
    arg = sys.argv[1]
    c: dict = json.loads(Path(arg).read_text() if Path(arg).exists() else arg)

    tag = c.get("tag", "cafe_terrace")
    name = c["name"]
    kind = c.get("kind", "proposal")
    if kind not in ("proposal", "ours"):
        sys.exit(f"kind={kind!r} not supported (have proposal | ours)")
    pname = c["proposal"]
    if pname not in PROPOSALS:
        sys.exit(f"proposal {pname!r} not supported yet (have {sorted(PROPOSALS)})")
    proposal = PROPOSALS[pname](**c.get("proposal_kwargs", {}))

    shard = int(c.get("shard", 0))
    seed_offset = int(c["seed_block"]) + shard * 100_000
    n = int(c.get("n", 1024))
    M = int(c.get("M", 32))
    num_steps = int(c.get("num_steps", 50))

    base = Path(c["dataset_cache"]).parent if c.get("dataset_cache") else Path("results/dataset") / tag
    mdir = base / "methods"
    mdir.mkdir(parents=True, exist_ok=True)
    out = mdir / f"{name}_shard{shard}.pt"

    pipe, scheduler = load_pipeline(DEVICE)
    csd = CSDStyleEmbedder(DEVICE).load()
    print(
        f"[gen_method] {name} kind={kind} proposal={pname} kwargs={c.get('proposal_kwargs', {})} shard={shard} n={n} M={M} seed_offset={seed_offset}",
        flush=True,
    )

    weights = None
    if kind == "ours":
        fields = {f.name for f in dataclasses.fields(OursConfig)}
        knobs = {k: v for k, v in c.get("ours_kwargs", {}).items() if k in fields}
        ours_cfg = OursConfig(lam_max=float(c["lam_max"]), M=M, num_steps=num_steps, **knobs)
        print(f"[gen_method] ours lam_max={ours_cfg.lam_max} knobs={knobs}", flush=True)
        imgs, weights = generate_ours(pipe, scheduler, c["object"], STYLE_B, proposal, ours_cfg, n, seed_offset, c["seeds"], device=DEVICE)
    else:
        imgs = generate_method(pipe, scheduler, c["object"], STYLE_B, proposal, n, seed_offset, c["seeds"], M=M, num_steps=num_steps, device=DEVICE)

    blob = {
        "images": _u8(imgs),
        "incep": inception_features(imgs, DEVICE),  # (n, 2048) for FID
        "csd": csd.embed(imgs).cpu().float(),  # (n, 768) for p(c)
        "name": name,
        "kind": kind,
        "proposal": pname,
        "proposal_kwargs": c.get("proposal_kwargs", {}),
        "shard": shard,
        "seed_offset": seed_offset,
        "n": int(imgs.shape[0]),
    }
    if kind == "ours" and weights is not None:
        blob["weights"] = weights.reshape(-1)[:n]  # (n,) float64 SMC importance weights
        blob["M"] = M  # particles per run (weights are stored in run-blocks of M)
        blob["lam_max"] = float(c["lam_max"])
        blob["ours_kwargs"] = c.get("ours_kwargs", {})
    torch.save(blob, out)
    wmsg = f" weights={tuple(blob['weights'].shape)}" if kind == "ours" else ""
    print(f"[saved] {out.name}  n={imgs.shape[0]}{wmsg} -> {out}", flush=True)


if __name__ == "__main__":
    main()
