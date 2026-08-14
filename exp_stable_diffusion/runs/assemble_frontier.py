"""Assemble the frontier from cached method shards and reference features.

The script recomputes p(c), FID, KID, and CSD-Fréchet distance without image
generation. Ours is reported with both equal particle weights and its SMC
weights; baselines have uniform weights. The floor row is read from floor.json.

Config (argv[1] JSON): {dataset, methods, kid_subset_size, kid_n_subsets, content_n}.
Output: a printed table and results/dataset/{tag}/frontier.json.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.metrics import fid, kid, weighted_fid, weighted_kid


def load_concat(paths: list[Path], key: str) -> torch.Tensor:
    return torch.cat([torch.load(p, map_location="cpu")[key] for p in paths], 0)


def _pc(csd: torch.Tensor, w: torch.Tensor, b: torch.Tensor, iw: torch.Tensor | None = None) -> float:
    """Mean calibrated p(c). ``iw`` = SMC importance weights (sum=1) for the weighted
    mean; None -> plain unweighted mean."""
    p = torch.sigmoid(csd.double() @ w.double() + float(b))
    return float(p.mean()) if iw is None else float((iw.double().reshape(-1) * p).sum())


def _kid(act_x: torch.Tensor, act_y: torch.Tensor, ksub: int, kn: int) -> float:
    """Plain (unweighted) KID mean, for baselines / the unweighted rows."""
    m, _ = kid(act_x, act_y, subset_size=ksub, n_subsets=kn)
    return float("nan") if m is None else m


def _run_ess(wr: torch.Tensor, m: int) -> torch.Tensor:
    """Return the absolute ESS for each run block of ``m`` particles."""
    nb = wr.numel() // m
    blk = wr[: nb * m].reshape(nb, m)
    blk = blk / blk.sum(1, keepdim=True).clamp_min(1e-12)
    return 1.0 / blk.pow(2).sum(1)


def _load_method(mdir: Path, name: str):
    """Concatenate a method's shards -> (incep, csd, weights|None, n, shards, ess_frac).

    ``weights`` is the pooled SMC importance weights (sum=1) when every shard cached
    them (the ours path), else None. Each shard's weights already sum to 1, so every
    shard is given equal mass 1/num_shards before concat. ``ess_frac`` is the per-run
    (batch of M) ESS fraction averaged over runs and shards; None for baselines."""
    paths = sorted(mdir.glob(f"{name}_shard*.pt"))
    if not paths:
        return None
    blobs = [torch.load(p, map_location="cpu") for p in paths]
    incep = torch.cat([d["incep"] for d in blobs], 0)
    csd = torch.cat([d["csd"] for d in blobs], 0)
    weights, ess_frac, ess_mean, ess_std = None, None, None, None
    if all("weights" in d for d in blobs):
        ns = len(blobs)
        m = int(blobs[0].get("M") or (128 if "m128" in name else 64))  # particles per run
        ws, fr = [], []
        for d in blobs:
            wr = d["weights"].to(torch.float64).reshape(-1).clamp_min(0.0)
            fr.append(_run_ess(wr, m))
            s = wr.sum()
            wr = wr / s if float(s) > 0.0 else torch.full_like(wr, 1.0 / max(1, wr.numel()))
            ws.append(wr / ns)
        weights = torch.cat(ws, 0)
        weights = weights / weights.sum()
        all_ess = torch.cat(fr)  # per-run ESS pooled over shards
        ess_frac = float((all_ess / m).mean())  # mean ESS/M (fraction)
        ess_mean, ess_std = float(all_ess.mean()), float(all_ess.std())  # absolute ESS
    return incep, csd, weights, int(incep.shape[0]), [p.name for p in paths], ess_frac, ess_mean, ess_std


def main():
    if len(sys.argv) < 2:
        sys.exit(f"usage: python {Path(__file__).name} <config.json>")
    arg = sys.argv[1]
    c: dict = json.loads(Path(arg).read_text() if Path(arg).exists() else arg)
    ds = Path(c.get("dataset", "results/dataset/cafe_terrace/dataset.pt"))
    methods = c.get("methods", [])
    ksub = int(c.get("kid_subset_size", 1024))
    kn = int(c.get("kid_n_subsets", 100))
    content_n = int(c.get("content_n", 2048))
    d = ds.parent
    mdir = d / "methods"

    # Reconstruct the not-c reference from the indexed content shards.
    idx = torch.load(ds, map_location="cpu")
    w, b, ref_idx = idx["head_w"], idx["head_b"], idx["reference_idx"]
    cshards = [d / s for s in idx["content_shards"]]
    content_inc = load_concat(cshards, "incep").double()
    content_csd = load_concat(cshards, "csd")
    ref_inc, ref_csd = content_inc[ref_idx], content_csd[ref_idx]
    ref_pc = _pc(ref_csd, w, b)
    print(f"[assemble] dataset={ds} N_ref={ref_idx.numel()} ref_pc={ref_pc:.3f}", flush=True)

    rows = []
    # floor row = the N-matched oracle floor from build_floor.py (step 3).
    floor_path = d / "floor.json"
    if floor_path.exists():
        fl = json.loads(floor_path.read_text())
        rows.append({"method": "floor", "n": fl["n"], "pc": fl["pc"], "fid": fl["fid"], "kid": fl["kid"], "csd_frechet": fl["csd_frechet"]})
    else:
        print("[warn] no floor.json (run build_floor.py); skipping floor row", flush=True)

    print(f"\n{'method':22s} {'N':>5s} {'p(c)':>7s} {'FID':>8s} {'KID':>9s} {'CSD-Fre':>8s} {'w?':>3s}", flush=True)
    if rows:
        fl = rows[0]
        print(f"{'floor':22s} {fl['n']:5d} {fl['pc']:7.3f} {fl['fid']:8.2f} {fl['kid']:9.5f} {fl['csd_frechet']:8.3f} {'-':>3s}", flush=True)

    def anchor(name: str, m_inc: torch.Tensor, m_csd: torch.Tensor):
        """Add a context anchor row (off the methods list), plain metrics vs reference."""
        pc, f_, k_, cf = _pc(m_csd, w, b), fid(m_inc, ref_inc), _kid(m_inc, ref_inc, ksub, kn), fid(m_csd, ref_csd)
        nn = int(m_inc.shape[0])
        print(f"{name:22s} {nn:5d} {pc:7.3f} {f_:8.2f} {k_:9.5f} {cf:8.3f} {'-':>3s}", flush=True)
        rows.append({"method": name, "n": nn, "weighted": False, "anchor": True, "pc": pc, "fid": f_, "kid": k_, "csd_frechet": cf})

    # avoided style (explicit Van Gogh): the canonical part0+part1 (2048) vs reference.
    vg_shards = [d / f"vangogh_part{p}.pt" for p in (0, 1) if (d / f"vangogh_part{p}.pt").exists()]
    if vg_shards:
        anchor("avoided_style", load_concat(vg_shards, "incep").double(), load_concat(vg_shards, "csd"))
    # unconditional content (no avoidance): floor pool parts 1-7, deterministic random uncond_n.
    uc_parts = [d / f"floor_part{p}.pt" for p in range(1, 8) if (d / f"floor_part{p}.pt").exists()]
    if uc_parts:
        uc_inc, uc_csd = load_concat(uc_parts, "incep").double(), load_concat(uc_parts, "csd")
        un = int(c.get("uncond_n", 2048))
        g = torch.Generator().manual_seed(int(c.get("uncond_seed", 0)))
        sel = torch.randperm(uc_inc.shape[0], generator=g)[:un]
        anchor("unconditional_content", uc_inc[sel], uc_csd[sel])

    for name in methods:
        if name == "content":
            m_inc, m_csd = content_inc[:content_n], content_csd[:content_n]
            pc, f_, k_, cf = _pc(m_csd, w, b), fid(m_inc, ref_inc), _kid(m_inc, ref_inc, ksub, kn), fid(m_csd, ref_csd)
            n = int(m_inc.shape[0])
            print(f"{name:22s} {n:5d} {pc:7.3f} {f_:8.2f} {k_:9.5f} {cf:8.3f} {'-':>3s}", flush=True)
            rows.append({"method": name, "n": n, "weighted": False, "pc": pc, "fid": f_, "kid": k_, "csd_frechet": cf})
            continue
        got = _load_method(mdir, name)
        if got is None:
            print(f"{name:22s}  -- no cached shards --", flush=True)
            continue
        m_inc, m_csd, iw, n, shards, ess_frac, ess_mean, ess_std = got

        # Equal-weight metrics describe the cached output images.
        pc_u, fid_u = _pc(m_csd, w, b), fid(m_inc, ref_inc)
        kid_u, csd_u = _kid(m_inc, ref_inc, ksub, kn), fid(m_csd, ref_csd)
        row = {
            "method": name,
            "n": n,
            "weighted": iw is not None,
            "shards": shards,
            "pc_unweighted": pc_u,
            "fid_unweighted": fid_u,
            "kid_unweighted": kid_u,
            "csd_frechet_unweighted": csd_u,
        }

        if iw is None:
            # Baselines have uniform weights.
            row.update(pc=pc_u, fid=fid_u, kid=kid_u, csd_frechet=csd_u)
            print(f"{name:22s} {n:5d} {pc_u:7.3f} {fid_u:8.2f} {kid_u:9.5f} {csd_u:8.3f} {'-':>3s}", flush=True)
        else:
            # Ours also reports the closed-form weighted SMC estimate.
            pc_w = _pc(m_csd, w, b, iw=iw)
            fid_w = weighted_fid(m_inc, iw, ref_inc)
            csd_w = weighted_fid(m_csd, iw, ref_csd)
            kid_w = weighted_kid(m_inc, iw, ref_inc)
            # Report the per-run effective sample size fraction.
            row.update(
                ess_per_m=ess_frac,
                ess_mean=ess_mean,
                ess_std=ess_std,
                pc=pc_w,
                fid=fid_w,
                kid=kid_w,
                csd_frechet=csd_w,
                pc_weighted=pc_w,
                fid_weighted=fid_w,
                kid_weighted=kid_w,
                csd_frechet_weighted=csd_w,
            )
            print(f"{name:22s} {n:5d} {pc_w:7.3f} {fid_w:8.2f} {kid_w:9.5f} {csd_w:8.3f} {'W':>3s}", flush=True)
            print(f"{'  └ unweighted':22s} {n:5d} {pc_u:7.3f} {fid_u:8.2f} {kid_u:9.5f} {csd_u:8.3f}  (ESS/M={ess_frac:.2f})", flush=True)
        rows.append(row)

    out = d / "frontier.json"
    out.write_text(
        json.dumps({"dataset": str(ds), "ref_pc": ref_pc, "n_ref": int(ref_idx.numel()), "kid_subset_size": ksub, "kid_n_subsets": kn, "rows": rows}, indent=2)
    )
    print(f"\n[saved] {out}", flush=True)


if __name__ == "__main__":
    main()
