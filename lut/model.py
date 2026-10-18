"""Image-adaptive 3D LUT (Zeng et al., TPAMI 2020).

A small CNN sees a 256x256 thumbnail and predicts N weights; the weights fuse N
learnable basis LUTs into one image-specific LUT, which is applied to the
full-resolution image with trilinear interpolation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .trilinear import apply_lut, identity_lut


def _block(cin: int, cout: int, norm: bool = True) -> list[nn.Module]:
    layers = [nn.Conv2d(cin, cout, 3, stride=2, padding=1), nn.LeakyReLU(0.2)]
    if norm:
        layers.append(nn.InstanceNorm2d(cout, affine=True))
    return layers


class WeightPredictor(nn.Module):
    """5 conv blocks (16/32/64/128/128) -> dropout -> 8x8 conv acting as FC. ~269K params for N=3."""

    def __init__(self, n_luts: int = 3, thumb: int = 256, identity_init: bool = True):
        super().__init__()
        self.thumb = thumb
        chans = [3, 16, 32, 64, 128, 128]
        layers = []
        for i, (cin, cout) in enumerate(zip(chans[:-1], chans[1:])):
            layers += _block(cin, cout, norm=i < len(chans) - 2)  # reference code: no norm on the last block
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Conv2d(128, n_luts, thumb // 32))
        if identity_init:
            # start at weights [1, 0, ...] so the fused LUT is the identity. The paper's default init
            # gives random, often negative weights and a much worse-than-input start.
            nn.init.zeros_(self.head[1].weight)
            nn.init.zeros_(self.head[1].bias)
            self.head[1].bias.data[0] = 1.0

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(img, size=(self.thumb, self.thumb), mode="bilinear", align_corners=False)
        return self.head(self.features(x)).flatten(1)


class AdaptiveLUT(nn.Module):
    def __init__(self, n_luts: int = 3, dim: int = 33, identity_init: bool = True):
        super().__init__()
        self.dim = dim
        luts = torch.zeros(n_luts, 3, dim, dim, dim)
        luts[0] = identity_lut(dim)  # paper init: one identity LUT, the rest zero
        if identity_init:
            # With weights [1, 0, ...] a zero LUT and its zero weight both get zero gradient and never
            # move, so the other LUTs need a small nonzero start. Output is still the identity at init.
            luts[1:] = 1e-3 * torch.randn_like(luts[1:])
        self.luts = nn.Parameter(luts)
        self.predictor = WeightPredictor(n_luts, identity_init=identity_init)

    def fuse(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (fused LUT (B, 3, D, D, D), weights (B, N))."""
        w = self.predictor(img)
        return torch.einsum("bn,ncijk->bcijk", w, self.luts), w

    def forward(self, img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lut, w = self.fuse(img)
        return apply_lut(img, lut), w
