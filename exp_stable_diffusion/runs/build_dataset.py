"""Build the not-c reference set and all base-dataset stats from CACHED shards.

Consumes the gen_pool shards (content / vangogh / calib) — no image regeneration,
no Inception re-run:
  - p(c)   : fit a CSD logistic head on calib pos/neg, apply to content & vangogh.
  - keep   : reference = drop the top `drop_top_frac` of content by p(c).
  - FID/KID: from the cached Inception-2048 activations.

Writes results/dataset/{tag}/dataset.pt (light index) + base_info.json (numbers).

Config (argv[1] JSON): {tag, drop_top_frac, kid_subset, kid_n}.
"""

import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import fid, kid
from src.pc_head import LogisticHead

DEVICE = "cuda"


def load_concat(paths, key: str) -> torch.Tensor:
    return torch.cat([torch.load(p, map_location="cpu")[key] for p in paths], 0)


def fid_kid(act_x, act_y, ksub: int, kn: int):
    return fid(act_x, act_y), kid(act_x, act_y, subset_size=ksub, n_subsets=kn)[0]


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: python {Path(__file__).name} <config.json>")
    arg = sys.argv[1]
    raw = Path(arg).read_text() if Path(arg).exists() else arg
    c: dict = json.loads(raw)
    tag = c.get("tag", "cafe_terrace")
    drop = float(c.get("drop_top_frac", 0.75))
    ksub = int(c.get("kid_subset", 256))
    kn = int(c.get("kid_n", 50))
    d: Path = Path("results/dataset") / tag

    cparts = sorted(d.glob("content_part*.pt"))
    assert cparts, f"no content shards in {d}"
    images = load_concat(cparts, "images")  # (N,3,H,W) uint8
    incep = load_concat(cparts, "incep").double()  # (N,2048)
    csd = load_concat(cparts, "csd")  # (N,768)
    N = images.shape[0]
    print(f"[cache] content N={N} from {len(cparts)} shards", flush=True)

    # check distinct seed block per shard, and no duplicate images.
    seeds = [torch.load(p, map_location="cpu").get("seed0") for p in cparts]
    print("[verify] shard seed0:", list(zip([p.name for p in cparts], seeds)), flush=True)
    assert len(set(seeds)) == len(seeds), f"SEED COLLISION across shards: {seeds}"
    hashes = {hashlib.md5(images[i].numpy().tobytes()).hexdigest() for i in range(N)}
    print(
        f"[verify] images: {N} total, {len(hashes)} unique -> {'OK (no duplicates)' if len(hashes) == N else 'WARN ' + str(N - len(hashes)) + ' DUPLICATES'}",
        flush=True,
    )

    # p(c): fit the CSD logistic head on the calibration set, apply to content.
    cal = torch.load(d / "calib.pt", map_location="cpu")
    head = LogisticHead(csd.shape[1], l2=1e-2, device=DEVICE)
    head.fit(cal["pos_csd"], cal["neg_csd"])
    pc = head.predict_proba(csd).cpu()
    pos_pc = float(head.predict_proba(cal["pos_csd"]).mean())
    neg_pc = float(head.predict_proba(cal["neg_csd"]).mean())

    # reference = drop the top `drop` fraction by p(c) (keep the lowest-p(c) tail).
    order = torch.argsort(pc)  # ascending
    n_keep = max(2, round((1.0 - drop) * N))
    ref_idx = order[:n_keep]
    incep_ref = incep[ref_idx]
    pc_ref = pc[ref_idx]

    q = torch.quantile(pc.float(), torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0]))
    s, _ = torch.sort(pc.float())
    drops = " ".join(f"d{int(f * 100)}={float(s[: max(1, N - round(f * N))].mean()):.3f}" for f in [0.0, 0.25, 0.5, 0.7, 0.8])

    print(f"\n===== p(c) (CSD calib pos={pos_pc:.3f} neg={neg_pc:.3f}) =====", flush=True)
    print(f"  content (pre-removal)         p(c)={float(pc.mean()):.3f} std={float(pc.std()):.3f}", flush=True)
    print(f"  DIST min={q[0]:.3f} p25={q[2]:.3f} med={q[3]:.3f} p75={q[4]:.3f} p90={q[5]:.3f} p95={q[6]:.3f} max={q[7]:.3f}", flush=True)
    print(f"  DROP-TOP kept-mean: {drops}", flush=True)
    print(f"  reference (drop top {drop:.0%}, keep {n_keep}) p(c)={float(pc_ref.mean()):.3f}", flush=True)

    vg_path = sorted(d.glob("vangogh_part*.pt"))
    vg = None
    if vg_path:
        vg_incep = load_concat(vg_path, "incep").double()
        vg_csd = load_concat(vg_path, "csd")
        pc_vg = head.predict_proba(vg_csd).cpu()
        print(f"  avoidance_target (Van Gogh)   p(c)={float(pc_vg.mean()):.3f}", flush=True)
        vg = (vg_incep, pc_vg)

    print(f"\n===== FID / KID (from cached Inception-2048; ksub={ksub} kn={kn}) =====", flush=True)
    # sanity check only: splits the 2048 reference into halves (1024 vs 1024), so this
    # FID is inflated by the small N. The real N-matched floor is build_floor.py.
    half = n_keep // 2
    f_floor, k_floor = fid_kid(incep_ref[:half], incep_ref[half : 2 * half], ksub, kn)
    print(f"  FLOOR (reference halves, N=1024 sanity-check)  FID={f_floor:.1f} KID={k_floor:.5f}", flush=True)
    f_cr, k_cr = fid_kid(incep, incep_ref, ksub, kn)
    print(f"  content      vs reference       FID={f_cr:.1f} KID={k_cr:.5f}", flush=True)
    res = {
        "n_content": N,
        "n_reference": n_keep,
        "drop_top_frac": drop,
        "calib_pos_pc": pos_pc,
        "calib_neg_pc": neg_pc,
        "content_pc": float(pc.mean()),
        "reference_pc": float(pc_ref.mean()),
        "fid_floor": f_floor,
        "kid_floor": k_floor,
        "fid_content_vs_ref": f_cr,
        "kid_content_vs_ref": k_cr,
    }
    if vg is not None:
        f_vc, k_vc = fid_kid(vg[0], incep, ksub, kn)
        f_vr, k_vr = fid_kid(vg[0], incep_ref, ksub, kn)
        print(f"  avoidance_target vs content     FID={f_vc:.1f} KID={k_vc:.5f}", flush=True)
        print(f"  avoidance_target vs reference   FID={f_vr:.1f} KID={k_vr:.5f}", flush=True)
        res.update({"vangogh_pc": float(vg[1].mean()), "fid_vg_vs_content": f_vc, "fid_vg_vs_ref": f_vr, "kid_vg_vs_content": k_vc, "kid_vg_vs_ref": k_vr})

    # dataset.pt is a light index (the shards hold the data): only derived/small things.
    torch.save(
        {
            "tag": tag,
            "drop_top_frac": drop,
            "content_pc": pc,
            "reference_idx": ref_idx,
            "head_w": head.w.cpu(),
            "head_b": head.b.cpu(),
            "calib_pos_csd": cal["pos_csd"],
            "calib_neg_csd": cal["neg_csd"],
            "content_shards": [p.name for p in cparts],
            "vangogh_shard": (vg_path[0].name if vg_path else None),
            "calib_shard": "calib.pt",
            "n_content": int(N),
            "n_reference": int(n_keep),
        },
        d / "dataset.pt",
    )
    (d / "base_info.json").write_text(json.dumps(res, indent=2))
    print(f"\n[saved] {d}/dataset.pt (light index) + base_info.json", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
