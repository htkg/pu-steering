"""CSD ViT-L style embedder (Somepalli et al., 2024).

Produces L2-normalized 768-dimensional style embeddings. The logistic head for
p(c) is fitted later when the reference set is built.

The Hugging Face repository provides a raw state dict with OpenAI CLIP keys, so
the ViT-L/14 visual tower is defined here to match those keys.
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

# CLIP image normalization (CSD uses the CLIP transform).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

CSD_HF_REPO = "tomg-group-umd/CSD-ViT-L"
CSD_HF_WEIGHTS = "pytorch_model.bin"
CSD_STYLE_DIM = 768  # projected style width
CSD_BACKBONE_WIDTH = 1024  # CLIP-L visual width


class _QuickGELU(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class _ResAttnBlock(torch.nn.Module):
    """OpenAI CLIP ResidualAttentionBlock (keys: attn, ln_1, mlp.c_fc/c_proj, ln_2)."""

    def __init__(self, width: int, heads: int):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(width, heads)
        self.ln_1 = torch.nn.LayerNorm(width)
        self.mlp = torch.nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", torch.nn.Linear(width, width * 4)),
                    ("gelu", _QuickGELU()),
                    ("c_proj", torch.nn.Linear(width * 4, width)),
                ]
            )
        )
        self.ln_2 = torch.nn.LayerNorm(width)

    def forward(self, x):
        a = self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0]
        x = x + a
        x = x + self.mlp(self.ln_2(x))
        return x


class Backbone(torch.nn.Module):
    conv1: torch.nn.Conv2d
    class_embedding: torch.nn.Parameter
    positional_embedding: torch.nn.Parameter
    ln_pre: torch.nn.LayerNorm
    transformer: torch.nn.Module
    ln_post: torch.nn.LayerNorm

    def __init__(self):
        super().__init__()


class CSDViTL(torch.nn.Module):
    """OpenAI-CLIP ViT-L/14 visual tower + CSD style projection (768-d, L2-norm)."""

    def __init__(
        self,
        width: int = CSD_BACKBONE_WIDTH,
        layers: int = 24,
        heads: int = 16,
        patch: int = 14,
        image_size: int = 224,
        out_dim: int = CSD_STYLE_DIM,
    ):
        super().__init__()
        scale = width**-0.5
        grid = image_size // patch
        self.backbone = Backbone()
        self.backbone.conv1 = torch.nn.Conv2d(3, width, kernel_size=patch, stride=patch, bias=False)
        self.backbone.class_embedding = torch.nn.Parameter(scale * torch.randn(width))
        self.backbone.positional_embedding = torch.nn.Parameter(scale * torch.randn(grid * grid + 1, width))
        self.backbone.ln_pre = torch.nn.LayerNorm(width)
        self.backbone.transformer = torch.nn.Module()
        self.backbone.transformer.resblocks = torch.nn.ModuleList([_ResAttnBlock(width, heads) for _ in range(layers)])
        self.backbone.ln_post = torch.nn.LayerNorm(width)
        # The CSD projection is stored in last_layer_style.
        self.last_layer_style = torch.nn.Parameter(torch.empty(width, out_dim))
        self.last_layer_content = torch.nn.Parameter(torch.empty(width, out_dim))  # loaded, unused

    def encode_feature(self, x: torch.Tensor) -> torch.Tensor:
        b = self.backbone
        x = b.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = b.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls, x], dim=1) + b.positional_embedding.to(x.dtype)
        x = b.ln_pre(x).permute(1, 0, 2)  # NLD -> LND
        for blk in b.transformer.resblocks:  # type: ignore
            x = blk(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        return b.ln_post(x[:, 0, :])  # CLS token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        style = self.encode_feature(x) @ self.last_layer_style
        return F.normalize(style, dim=1, p=2)


def _load_backbone(device: str) -> CSDViTL:
    """Load the pretrained CSD ViT-L style backbone from the HF checkpoint."""
    ckpt = hf_hub_download(CSD_HF_REPO, CSD_HF_WEIGHTS)
    # This checkpoint contains a pickled NumPy scalar. PyTorch 2.6 and later
    # require weights_only=False to load it.
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)
    sd = raw.get("model_state_dict", raw) if isinstance(raw, dict) else raw
    sd = {k.removeprefix("module."): v for k, v in sd.items()}
    model = CSDViTL()
    missing, _ = model.load_state_dict(sd, strict=False)
    critical = [m for m in missing if m.startswith("backbone.") or m == "last_layer_style"]
    if critical:
        raise RuntimeError(f"CSD checkpoint missing style keys, e.g. {critical[:3]}")
    return model.eval().to(device)


def _preprocess(images: torch.Tensor, size: int = 224) -> torch.Tensor:
    """(N,3,H,W) in [0,1] -> resize 224 + CLIP normalize."""
    x = F.interpolate(images, size=(size, size), mode="bicubic", align_corners=False, antialias=True).clamp(0.0, 1.0)
    m = torch.tensor(CLIP_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    s = torch.tensor(CLIP_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x - m) / s


class CSDStyleEmbedder:
    """Wraps the CSD backbone; `.embed(images)` -> (N, 768) L2-norm style embeddings."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self._model = None

    def load(self):
        self._model = _load_backbone(self.device)
        return self

    @torch.no_grad()
    def embed(self, images: torch.Tensor, batch_size: int = 128) -> torch.Tensor:
        if self._model is None:
            raise RuntimeError("call load() first")
        outs = []
        for i in range(0, images.shape[0], batch_size):
            x = _preprocess(images[i : i + batch_size].to(self.device))
            outs.append(self._model(x))
        return torch.cat(outs, dim=0)
