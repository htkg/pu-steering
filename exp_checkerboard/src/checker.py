"""Checkerboard tiles, sampling, metrics, and the learned score (avoided = central tile)."""

import math
from pathlib import Path

import numpy as np
import ot
import torch
from torch import nn

from .targets import sample_checkerboard

_HERE = Path(__file__).resolve().parent
SCALE = 4.0
CELL = SCALE / 2.0
OCCUPIED = [(cx, cy) for cx in (-2, -1, 0, 1) for cy in (-2, -1, 0, 1) if (cx + cy) % 2 == 0]
AVOID_CELLS = [(0, 0)]  # central tile
KEEP_CELLS = [c for c in OCCUPIED if c not in set(AVOID_CELLS)]
CLASS_PRIOR = len(AVOID_CELLS) / len(OCCUPIED)


def _in_cells(x: torch.Tensor, cells) -> torch.Tensor:
    c = torch.floor(x / CELL).to(torch.long)
    cellset = {(int(a), int(b)) for a, b in cells}
    return torch.tensor([(int(c[i, 0]), int(c[i, 1])) in cellset for i in range(c.shape[0])], dtype=torch.bool)


def sample_cells(cells, n: int, seed: int = 0) -> torch.Tensor:
    """Rejection-sample checkerboard points restricted to the given cells."""
    out, got, s = [], 0, seed
    while got < n:
        batch = sample_checkerboard(max(4 * n, 4096), scale=SCALE, seed=s)
        sel = batch[_in_cells(batch, cells)]
        out.append(sel)
        got += sel.shape[0]
        s += 1
    return torch.cat(out, 0)[:n]


def checker_post(x: torch.Tensor) -> torch.Tensor:
    """Oracle p(c|x): 1 if x lands in an avoided tile, else 0."""
    return _in_cells(x, AVOID_CELLS).to(torch.float32)


def checker_w2(x: torch.Tensor, w: torch.Tensor | None = None, n_ref: int = 3000, seed: int = 123, cells=None) -> float:
    """W2 between (optionally weighted) samples x and ground-truth samples on ``cells``
    (default the kept tiles; pass OCCUPIED to score an unconditional generation)."""
    ref = sample_cells(KEEP_CELLS if cells is None else cells, n_ref, seed=seed).numpy().astype(np.float64)
    xn = x.detach().numpy().astype(np.float64)
    a = np.ones(ref.shape[0]) / ref.shape[0]
    b = (w / (w.sum() + 1e-12)).detach().numpy().astype(np.float64) if w is not None else np.ones(xn.shape[0]) / xn.shape[0]
    return float(np.sqrt(ot.emd2(a, b, ot.dist(ref, xn))))


def _time_embed(tf: torch.Tensor, dim: int = 64) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(torch.linspace(0, math.log(1000.0), half))
    ang = tf[:, None] * freqs[None, :]
    return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class EpsNet(nn.Module):
    """MLP eps-predictor with sinusoidal time embedding."""

    def __init__(self, d: int = 2, hidden: int = 256, t_dim: int = 64, depth: int = 4):
        super().__init__()
        self.t_dim = t_dim
        layers = [nn.Linear(d + t_dim, hidden), nn.SiLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, d)]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, tf: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, _time_embed(tf, self.t_dim)], dim=-1))


def load_checker_score(sched, ckpt_path: str | Path | None = None):
    """Return score_fn(x_t, t) -> score = -eps/sqrt(1-abar_t) from the trained eps-net."""
    ckpt_path = ckpt_path or _HERE.parent / "ckpts" / "checker_score.pt"
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    net = EpsNet(d=cfg["d"], hidden=cfg["hidden"], t_dim=cfg["t_dim"], depth=cfg["depth"])
    net.load_state_dict(ck["state_dict"])
    net.eval()
    Tsteps = cfg["T"]

    @torch.no_grad()
    def score_fn(x_t: torch.Tensor, t: int) -> torch.Tensor:
        eps = net(x_t, torch.full((x_t.shape[0],), float(t) / Tsteps))
        return -eps / math.sqrt(1.0 - sched.abar_t(t))

    return score_fn
