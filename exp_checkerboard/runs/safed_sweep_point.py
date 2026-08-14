"""Run the checker SafeDenoiser baseline at one eta and print 'eta leak W2' (args: eta seeds [threads])."""

import os
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "runs"))
from sample import build_settings

eta = float(sys.argv[1])
seeds = int(sys.argv[2])
threads = int(sys.argv[3]) if len(sys.argv) > 3 else 40
torch.set_num_threads(threads)

st = build_settings()[2]
_, _, leak, w2 = st["run"]("safed", {"eta": eta}, seeds, 64)
print(f"{eta:g}\t{leak:.4f}\t{w2:.4f}", flush=True)
