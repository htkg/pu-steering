"""Merge per-trial JSONL files in m/ and lambda/ into summary tables.

Usage:
    uv run python results/ablation/merge.py
"""

import json
import math
from pathlib import Path

HERE = Path(__file__).parent

M_ROWS = [("OFF", "M"), ("ON", "M"), ("ON_TILT", "M"), ("ON_GRAD40", "M"), ("ON_GRAD48", "M")]
MS = [64, 256, 1024, 4096]
L_ROWS = [
    ("OFF", "tm", [0.5, 1.0, 2.0, 3.0, 5.0]),
    ("ON", "lf", [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]),
    ("ON_TILT", "tm", [0.5, 1.0, 2.0, 3.0, 5.0]),
    ("ON_GRAD40", "le", [0.5, 1.0, 2.0, 3.0, 5.0]),
    ("ON_GRAD48", "le", [0.5, 1.0, 2.0, 3.0, 5.0]),
]
KNOB = {"tm": "tilt_max", "lf": "lam_fixed", "le": "lam_end"}


def load(sub, name):
    p = HERE / sub / f"{name}.jsonl"
    if not p.exists():
        return None
    return [json.loads(ln) for ln in open(p)]


def ms(vals):
    n = len(vals)
    m = sum(vals) / n
    s = math.sqrt(sum((v - m) ** 2 for v in vals) / n) if n > 1 else 0.0
    return m, s


def stat(rows, k):
    m, s = ms([r[k] for r in rows])
    return f"{m:.3f} ± {s:.3f}"


def stat_w(rows, k):
    m, s = ms([r[k] for r in rows])
    return f"{m:.2f} ± {s:.2f}"


def main():
    with open(HERE / "m_merged.txt", "w") as f:
        for var, _ in M_ROWS:
            f.write(f"# {var}: M scaling (Oracle at the same M)\n")
            for M in MS:
                rows = load("m", f"{var}_M{M}")
                if rows is None:
                    continue
                f.write(
                    f"  M={M:<5} n={len(rows):>2}  "
                    f"raw={stat(rows, 'raw_pm')}/{stat_w(rows, 'raw_wass')}  "
                    f"w={stat(rows, 'w_pm')}/{stat_w(rows, 'w_wass')}  "
                    f"oracle={stat(rows, 'orc_pm')}/{stat_w(rows, 'orc_wass')}\n"
                )
            f.write("\n")
    print("m_merged.txt")

    with open(HERE / "lambda_merged.txt", "w") as f:
        for var, knob, vals in L_ROWS:
            f.write(f"# {var}: tilt strength ({KNOB[knob]}), M=1024\n")
            for v in vals:
                rows = load("lambda", f"{var}_{knob}{v}")
                if rows is None:
                    continue
                f.write(
                    f"  {knob}={v:<4} n={len(rows):>2}  "
                    f"raw={stat(rows, 'raw_pm')}/{stat_w(rows, 'raw_wass')}  "
                    f"w={stat(rows, 'w_pm')}/{stat_w(rows, 'w_wass')}\n"
                )
            f.write("\n")
    print("lambda_merged.txt")


if __name__ == "__main__":
    main()
