"""Train the checkerboard's unconditional score (denoising score matching)."""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.checker import OCCUPIED, EpsNet, _in_cells, checker_w2
from src.diffusion import DDPMSchedule
from src.targets import sample_checkerboard

HERE = Path(__file__).resolve().parent
DEFAULT_CKPT = HERE.parent / "ckpts" / "checker_score.pt"


@torch.no_grad()
def evaluate(net, sched, T, n=4000):
    """Evaluate unconditional samples by on-tile fraction and W2."""
    net.eval()
    g = torch.Generator().manual_seed(0)
    x = torch.randn(n, 2, generator=g)
    for t in range(T, 0, -1):
        ab, al, be = sched.abar_t(t), sched.alpha(t), sched.beta(t)
        eps = net(x, torch.full((n,), t / T))
        mu = (1.0 / math.sqrt(al)) * (x - (be / math.sqrt(1.0 - ab)) * eps)
        x = mu + math.sqrt(max(sched.sigma2(t), 1e-12)) * torch.randn(n, 2, generator=g) if t > 1 else mu
    net.train()
    return float(_in_cells(x, OCCUPIED).float().mean()), checker_w2(x, w=None, n_ref=n, seed=0, cells=OCCUPIED)


def main():
    p = argparse.ArgumentParser(description="Train the checkerboard score net.")
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--bs", type=int, default=4096)
    p.add_argument("--out", default=DEFAULT_CKPT)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=40)
    p.add_argument("--T", type=int, default=100)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--eval-every", type=int, default=5000, help="eval on-tile every N steps; keep the best (0 = only final)")
    a = p.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    sched = DDPMSchedule(T=a.T)
    abar = sched.abar.clone()

    net = EpsNet(hidden=a.hidden)
    opt = torch.optim.Adam(net.parameters(), lr=2e-4)
    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)

    t0, loss_ema = time.time(), None
    best_on_tile, best_state = -1.0, None
    for step in range(a.steps):
        x0 = sample_checkerboard(a.bs, scale=4.0, seed=a.seed * 1_000_000 + step)
        ti = torch.randint(1, a.T + 1, (a.bs,))
        ab = abar[ti - 1][:, None]
        eps = torch.randn_like(x0)
        xt = torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * eps
        loss = ((net(xt, ti.float() / a.T) - eps) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        lr_sched.step()
        loss_ema = loss.item() if loss_ema is None else 0.99 * loss_ema + 0.01 * loss.item()
        if step % a.log_every == 0 or step == a.steps - 1:
            print(f"  step {step:6d}  loss_ema={loss_ema:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if a.eval_every and (step + 1) % a.eval_every == 0:
            ot, w2 = evaluate(net, sched, a.T)
            tag = "  <- best" if ot > best_on_tile else ""
            print(f"  [eval@{step + 1:6d}] on-tile={ot:.3f}  W2={w2:.3f}{tag}", flush=True)
            if ot > best_on_tile:
                best_on_tile, best_state = ot, {k: v.clone() for k, v in net.state_dict().items()}

    if best_state is not None:
        net.load_state_dict(best_state)
    on_tile, w2 = evaluate(net, sched, a.T)
    print(f"[eval] uncond on-tile={on_tile:.3f}  W2={w2:.3f}  (best kept)", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "cfg": {"d": 2, "hidden": a.hidden, "t_dim": 64, "depth": 4, "T": a.T, "scale": 4.0}}, a.out)
    print(f"[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
