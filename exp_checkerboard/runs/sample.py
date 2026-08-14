"""Run the samplers and cache pooled particles + SMC weights to results/row_{i}.pt."""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import checker as C
from src.diffusion import DDPMSchedule, oracle_sample, sample_gmm
from src.metrics import gmm_post, gmm_w2
from src.pu import OnlinePURFF
from src.samplers_ref import PUSMCRef, SafeDenoiser
from src.samplers_score import DNG, PUSMCScore
from src.targets import make_mog40, make_ring, mog40_avoid_idx, ring_avoid_idx, sample_checkerboard, split_gmm

SCHED = DDPMSchedule(T=100)
HERE = Path(__file__).resolve().parent
RES_DIR = HERE.parent / "results"


def pu(sigma_scale=0.3, inner_steps=5, n_features=256, sigma=None):
    return OnlinePURFF(n_features=n_features, sigma=sigma, sigma_scale=sigma_scale, lam=1e-4, inner_steps=inner_steps, lr=0.1)


def pool(raws, post_fn, w2_fn):
    """Pool per-run (x, w) into one particle set X (N,2) and weights W (N,), then the
    SMC-weighted leak / W2. Each run's weights are normalized then scaled by 1/n_runs."""
    xs, ws, nr = [], [], len(raws)
    for x, w in raws:
        if w is None:
            wn = torch.full((x.shape[0],), 1.0 / x.shape[0])
        else:
            wt = w.detach().clamp(min=0)
            s = float(wt.sum())
            wn = wt / s if s > 1e-12 else torch.full((x.shape[0],), 1.0 / x.shape[0])
        xs.append(x.detach())
        ws.append(wn / nr)
    X = torch.cat(xs, 0)
    W = torch.cat(ws, 0)
    W = W / W.sum()
    return X, W, float((W * post_fn(X)).sum()), w2_fn(X, W)


def gmm_setting(name, gmm, avoid_idx, ours, lim, bw):
    pos, neg, pi = split_gmm(gmm, avoid_idx)
    geom = [(g.mean.numpy(), g.cov.numpy()) for g in pos.comps]
    post_fn = lambda X: gmm_post(X, pos, neg, pi)
    w2_fn = lambda X, W: gmm_w2(X, neg, w=W)

    def run(kind, hp, seeds, M):
        raws = []
        for s in range(seeds):
            if kind == "target":
                x, w = sample_gmm(neg, M, generator=torch.Generator().manual_seed(s)), None
            elif kind == "uncond":
                x, w = oracle_sample(SCHED, gmm, M, seed=s), None
            elif kind == "dng":
                x, w = DNG(SCHED, pos, neg, pi, M=M, lambda0=hp["l0"], seed=s).sample(), None
            else:
                o = ours
                p = pu(sigma_scale=o["sig"], sigma=o.get("sig_abs"), inner_steps=o["inner"], n_features=o["feat"])
                x, w = PUSMCScore(
                    SCHED,
                    pos,
                    neg,
                    pi,
                    p,
                    M=M,
                    pi_min=o["pi"],
                    pi_max=o["pi"],
                    ess_target=o["etg"],
                    ess_resample=o["esr"],
                    resample_stop=o["rstop"],
                    dng_lambda0=o["l0"],
                    tilt_max=o["tilt"],
                    seed=s,
                ).sample()
            raws.append((x, w))
        return pool(raws, post_fn, w2_fn)

    return {"name": name, "engine": "gmm", "lim": lim, "bw": bw, "geom": geom, "run": run, "baseline": "dng", "ours_hp": {}}


def checker_setting(checker_ckpt=None, lim=4.4, bw=0.10):
    sfn = C.load_checker_score(SCHED, checker_ckpt)
    ref_pos = C.sample_cells(C.AVOID_CELLS, 2048, seed=0).detach()
    geom = [((cx, cy), C.CELL) for (cx, cy) in C.AVOID_CELLS]
    post_fn = lambda X: C.checker_post(X)
    w2_fn = lambda X, W: C.checker_w2(X, w=W)

    def run(kind, hp, seeds, M):
        raws = []
        for s in range(seeds):
            if kind == "target":
                x, w = C.sample_cells(C.KEEP_CELLS, M, seed=1000 + s), None
            elif kind == "uncond":
                x, w = sample_checkerboard(M, scale=C.SCALE, seed=2000 + s), None
            elif kind == "safed":
                x, w = SafeDenoiser(SCHED, sfn, ref_pos, M=M, eta=hp["eta"], seed=s).sample(), None
            else:
                x, w = PUSMCRef(SCHED, sfn, ref_pos, pu(sigma_scale=0.2, inner_steps=12), M=M, lam_max=0.7, sd_eta=48.0, seed=s).sample()
            raws.append((x, w))
        return pool(raws, post_fn, w2_fn)

    return {"name": "ref - checkerboard", "engine": "checker", "lim": lim, "bw": bw, "geom": geom, "run": run, "baseline": "safed", "ours_hp": {}}


OURS_RING = {"pi": 0.2, "sig": 0.15, "sig_abs": None, "l0": 0.3, "inner": 20, "feat": 256, "esr": 0.4, "etg": 0.6, "rstop": 2, "tilt": 4.0}
OURS_MOG = {"pi": 0.15, "sig": 0.3, "sig_abs": 1.5, "l0": 0.3, "inner": 40, "feat": 1024, "esr": 0.4, "etg": 0.6, "rstop": 2, "tilt": 3.0}


def build_settings(checker_ckpt=None):
    mog = make_mog40(seed=0)
    return [
        gmm_setting("score - ring", make_ring(), ring_avoid_idx(), OURS_RING, lim=10.5, bw=0.13),
        gmm_setting("score - MoG-40", mog, mog40_avoid_idx(mog), OURS_MOG, lim=44.0, bw=0.07),
        checker_setting(checker_ckpt),
    ]


def columns(st, a):
    """The 5 figure columns: (label, role, kind, hp) with CLI-set baseline strengths."""
    weak, strong = (a.dng_weak, a.dng_strong) if st["baseline"] == "dng" else (a.safed_weak, a.safed_strong)
    key = "l0" if st["baseline"] == "dng" else "eta"
    return [
        ("Target  p(x|not c)", "good", "target", {}),
        ("Unconditional", "good", "uncond", {}),
        ("Baseline - weak\n(under-avoid -> leak)", "base", st["baseline"], {key: weak}),
        ("Baseline - strong\n(over-avoid -> distort)", "base", st["baseline"], {key: strong}),
        ("Ours  (PU-steering)", "ours", "ours", st["ours_hp"]),
    ]


def run_setting(i, st, a):
    cols = []
    for label, role, kind, hp in columns(st, a):
        X, W, leak, w2 = st["run"](kind, hp, a.seeds, a.M)
        cols.append({"label": label, "role": role, "kind": kind, "X": X.to(torch.float32), "W": W.to(torch.float64), "leak": leak, "w2": w2})
        print(f"[sample] {st['name']:18s} {label.splitlines()[0]:24s} leak={leak:.3f} W2={w2:.2f}", flush=True)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    out = RES_DIR / f"row_{i}.pt"
    torch.save({"name": st["name"], "engine": st["engine"], "lim": st["lim"], "bw": st["bw"], "geom": st["geom"], "cols": cols}, out)
    print(f"[saved] {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Sample the avoidance demo and cache particles/weights.")
    p.add_argument("row", nargs="?", type=int, help="setting index 0/1/2 (default: all)")
    p.add_argument("--seeds", type=int, default=40, help="pooled runs (default 40 reproduces the figure)")
    p.add_argument("--M", type=int, default=64, help="particles per run")
    p.add_argument("--threads", type=int, default=40)
    p.add_argument("--checker-ckpt", default=None, help="override ckpts/checker_score.pt")
    p.add_argument("--dng-weak", type=float, default=1.0)
    p.add_argument("--dng-strong", type=float, default=8.0)
    p.add_argument("--safed-weak", type=float, default=4.0)
    p.add_argument("--safed-strong", type=float, default=20.0)
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    settings = build_settings(a.checker_ckpt)
    for i in [a.row] if a.row is not None else range(len(settings)):
        run_setting(i, settings[i], a)


if __name__ == "__main__":
    main()
