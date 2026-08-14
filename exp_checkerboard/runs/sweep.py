"""Sweep a baseline's guidance strength and print the leak / W2 trade-off."""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sample import build_settings

SETTINGS = {"ring": 0, "mog40": 1, "checker": 2}


def main():
    p = argparse.ArgumentParser(description="Sweep baseline guidance strength; print leak/W2.")
    p.add_argument("--setting", choices=list(SETTINGS), default="ring")
    p.add_argument("--strengths", default="1,2,3,5,8", help="comma-separated baseline strengths")
    p.add_argument("--seeds", type=int, default=16, help="pooled runs (smaller than the figure's 40 for speed)")
    p.add_argument("--M", type=int, default=64)
    p.add_argument("--threads", type=int, default=40)
    p.add_argument("--checker-ckpt", default=None)
    a = p.parse_args()

    torch.set_num_threads(a.threads)
    st = build_settings(a.checker_ckpt)[SETTINGS[a.setting]]
    strengths = [float(s) for s in a.strengths.split(",")]
    key = "l0" if st["baseline"] == "dng" else "eta"

    print(f"# {st['name']}  (baseline = {st['baseline']}, sweeping {key}; seeds={a.seeds} M={a.M})", flush=True)
    print(f"{'method':>16s}  {'leak':>7s}  {'W2':>7s}", flush=True)
    _, _, leak, w2 = st["run"]("target", {}, a.seeds, a.M)
    print(f"{'target':>16s}  {leak:7.3f}  {w2:7.2f}", flush=True)
    for s in strengths:
        _, _, leak, w2 = st["run"](st["baseline"], {key: s}, a.seeds, a.M)
        print(f"{st['baseline'] + '=' + format(s, 'g'):>16s}  {leak:7.3f}  {w2:7.2f}", flush=True)
    _, _, leak, w2 = st["run"]("ours", st["ours_hp"], a.seeds, a.M)
    print(f"{'ours':>16s}  {leak:7.3f}  {w2:7.2f}", flush=True)


if __name__ == "__main__":
    main()
