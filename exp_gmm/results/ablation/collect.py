"""Sort per-trial JSONL dumps from the ablation runs into m/ and lambda/.

Usage:
    uv run python results/ablation/collect.py <dump.txt> [...]
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent


def main():
    recs = {}
    for f in sys.argv[1:]:
        with open(f) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln.startswith("{"):
                    continue
                r = json.loads(ln)
                if not r.get("trial"):
                    continue
                base = r["tag"].rsplit("_s", 1)[0]  # e.g. m_ON_TILT_M64, l_OFF_tm0.5
                sub = "m" if base.startswith("m_") else "lambda"
                recs.setdefault((sub, base[2:]), []).append(r)

    for (sub, name), rows in sorted(recs.items()):
        rows.sort(key=lambda r: (r["ls"], r["seed"]))
        out = HERE / sub / f"{name}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"{out.relative_to(HERE)}: {len(rows)} trials")


if __name__ == "__main__":
    main()
