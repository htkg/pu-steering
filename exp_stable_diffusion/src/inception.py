"""InceptionV3 pool3 (2048-d) activations via pytorch-fid, cached for FID.

We cache these activations at generation time so FID can be computed later from
the cache without re-running Inception. pytorch-fid's InceptionV3 resizes to 299
and maps [0,1] -> [-1,1] internally, so we feed [0,1] images straight in.
"""

import torch
from pytorch_fid.inception import InceptionV3

_INCEPTION_CACHE: dict = {}


def _get_inception(device: str):
    """Load (and cache per device) the pytorch-fid InceptionV3 2048-d block."""
    block = _INCEPTION_CACHE.get(str(device))
    if block is None:
        idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        block = InceptionV3([idx]).to(device).eval()
        for p in block.parameters():
            p.requires_grad_(False)
        _INCEPTION_CACHE[str(device)] = block
    return block


@torch.no_grad()
def inception_features(images: torch.Tensor, device: str = "cuda", batch_size: int = 64) -> torch.Tensor:
    """(N,3,H,W) float [0,1] -> (N, 2048) pool3 activations, float64 on CPU."""
    block = _get_inception(device)
    outs = []
    for i in range(0, images.shape[0], batch_size):
        chunk = images[i : i + batch_size].to(device).clamp(0.0, 1.0).float()
        feat = block(chunk)[0].squeeze(-1).squeeze(-1)  # (b, 2048)
        outs.append(feat.to(torch.float64).cpu())
    return torch.cat(outs, dim=0)
