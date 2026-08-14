The original PNGs (in the source repo) are stored here as JPEG quality 80 for file size reasons; 4114x4114 kept, 541MB -> 60MB.

```shell
SRC=<source>/exp_stable_diffusion/results/dataset/cafe_terrace/grids
uv run --with pillow python -c '
import sys, os
from PIL import Image
src, dst = sys.argv[1], sys.argv[2]
for r, _, fs in os.walk(src):
    for f in sorted(fs):
        if not f.endswith(".png"):
            continue
        p = os.path.join(r, f)
        o = os.path.join(dst, os.path.relpath(p, src))[:-4] + ".jpg"
        im = Image.open(p).convert("RGB")
        im.save(o, "JPEG", quality=80, subsampling=2, optimize=True, progressive=True, icc_profile=im.info.get("icc_profile"))
' "$SRC" .
```
