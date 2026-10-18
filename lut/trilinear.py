"""Trilinear 3D LUT lookup via grid_sample.

LUT layout: (B, 3, D, D, D) indexed as lut[b, c, blue, green, red].
grid_sample maps grid coords (x, y, z) -> (W, H, D) dims of the input volume,
so passing pixel colors as (r, g, b) makes r index the last axis, g the middle
and b the first. align_corners=True puts lattice node i exactly at i / (D - 1).
"""
import torch
import torch.nn.functional as F


def identity_lut(dim: int = 33) -> torch.Tensor:
    """Return a (3, D, D, D) identity LUT in [blue, green, red] index order."""
    v = torch.linspace(0, 1, dim)
    b, g, r = torch.meshgrid(v, v, v, indexing="ij")
    return torch.stack([r, g, b], dim=0)


def apply_lut(img: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Apply per-sample LUTs to images.

    img: (B, 3, H, W) in [0, 1]
    lut: (B, 3, D, D, D) or (3, D, D, D) shared across the batch
    returns (B, 3, H, W)
    """
    if lut.dim() == 4:
        lut = lut.unsqueeze(0).expand(img.shape[0], -1, -1, -1, -1)
    grid = img.permute(0, 2, 3, 1).unsqueeze(1) * 2 - 1  # (B, 1, H, W, 3) as (r, g, b)
    out = F.grid_sample(lut, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return out.squeeze(2)
