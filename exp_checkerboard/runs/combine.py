"""Stack the per-row PNGs in figures/ into one figure (pure image op, no sampling)."""

import re
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE.parent / "figures"


def row_index(path: Path) -> int:
    match = re.search(r"row_(\d+)_", path.name)
    if match is None:
        raise ValueError(f"unexpected row figure filename: {path.name}")
    return int(match.group(1))


def main():
    paths = sorted(FIG_DIR.glob("row_*.png"), key=row_index)
    if not paths:
        raise SystemExit("no figures/row_*.png found (run make_panels.py first)")
    imgs = [Image.open(p).convert("RGB") for p in paths]
    w, h = max(im.width for im in imgs), max(im.height for im in imgs)
    canvas = Image.new("RGB", (w, h * len(imgs)), "white")
    for i, im in enumerate(imgs):
        canvas.paste(im, ((w - im.width) // 2, i * h))
    out = FIG_DIR / "demo_avoidance.png"
    canvas.save(out)
    print(f"[saved] combined {len(imgs)} rows -> {out} ({canvas.width}x{canvas.height})", flush=True)


if __name__ == "__main__":
    main()
