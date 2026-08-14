"""Generate one shard of the reference pool and cache it as a single tensor file.

Three kinds (config field ``kind``):
  content : plain CFG samples of "a painting of {subject}"        -> content_part{P}.pt
  vangogh : "... in the style of Vincent van Gogh" (avoidance c)  -> vangogh_part{P}.pt
  calib   : CSD calibration set (Van-Gogh positives; other-painter -> calib.pt
            + plain negatives), used later to fit the p(c) head.

Each shard caches images (uint8) + Inception-2048 + CSD-768, so FID and p(c) are
recomputed later from the cache (no image re-generation). Run shards in parallel
with distinct ``part`` indices; seeds are fixed per (kind, part) for reproducibility.

Config (argv[1] JSON): {tag, subject, kind, part, n, [seed_block], [out_kind]}.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.csd import CSDStyleEmbedder
from src.inception import inception_features
from src.sd import generate_cfg, load_pipeline

DEVICE = "cuda"
# Distinct seed block per kind so content / vangogh / calib pools never overlap.
KIND_SEED_BLOCK = {"content": 0, "vangogh": 5_000_000, "calib": 8_000_000}
# The "other painter" negative for CSD calibration (≠ Van Gogh).
OTHER_PAINTER = "Rembrandt"


def _u8(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(0, 1) * 255).round().to(torch.uint8).cpu()


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: python {Path(__file__).name} <config.json>")
    arg = sys.argv[1]
    raw = Path(arg).read_text() if Path(arg).exists() else arg
    c: dict = json.loads(raw)

    tag = c.get("tag", "cafe_terrace")
    subject = c["subject"]
    kind = c.get("kind", "content")
    part = int(c.get("part", 0))
    n = int(c.get("n", 1024))
    seed_block = int(c.get("seed_block", KIND_SEED_BLOCK[kind]))
    seed0 = seed_block + part * 100_000
    out_kind = c.get("out_kind", kind)  # prefix of the saved shard path: {out_kind}_part{part}.pt

    outdir: Path = Path("results/dataset") / tag
    outdir.mkdir(parents=True, exist_ok=True)

    pipe, scheduler = load_pipeline(DEVICE)
    csd = CSDStyleEmbedder(DEVICE).load()
    print(f"[load] SD v1.4 + CSD; kind={kind} part={part} n={n} seed0={seed0}", flush=True)

    def gen(prompt, count, seed):
        return generate_cfg(pipe, scheduler, prompt, count, seed, device=DEVICE)

    if kind == "calib":
        pos_prompt = f"a painting of {subject} in the style of Vincent van Gogh"
        neg_prompts = [f"a painting of {subject} in the style of {OTHER_PAINTER}", f"a painting of {subject}"]
        pos = gen(pos_prompt, n, seed0)
        neg = torch.cat([gen(neg_prompts[0], n // 2, seed0 + 10_000), gen(neg_prompts[1], n // 2, seed0 + 20_000)], 0)
        path = outdir / "calib.pt"
        torch.save(
            {
                "pos_csd": csd.embed(pos).cpu().float(),
                "neg_csd": csd.embed(neg).cpu().float(),
                "pos_images": _u8(pos),
                "neg_images": _u8(neg),
                "pos_prompt": pos_prompt,
                "neg_prompts": neg_prompts,
                "seed0_pos": seed0,
                "seed0_neg_other": seed0 + 10_000,
                "seed0_neg_plain": seed0 + 20_000,
                "n_pos": pos.shape[0],
                "n_neg": neg.shape[0],
            },
            path,
        )
        print(f"[saved] calib.pt  n_pos={pos.shape[0]} n_neg={neg.shape[0]} -> {path}", flush=True)
        return

    prompt = f"a painting of {subject}" if kind == "content" else f"a painting of {subject} in the style of Vincent van Gogh"
    imgs = gen(prompt, n, seed0)
    path = outdir / f"{out_kind}_part{part}.pt"
    torch.save(
        {
            "images": _u8(imgs),
            "incep": inception_features(imgs, DEVICE),  # (n, 2048) for FID
            "csd": csd.embed(imgs).cpu().float(),  # (n, 768) for p(c)
            "seed0": seed0,
            "prompt": prompt,
            "n": n,
        },
        path,
    )
    print(f"[saved] {out_kind}_part{part}.pt  n={n} -> {path}", flush=True)


if __name__ == "__main__":
    main()
