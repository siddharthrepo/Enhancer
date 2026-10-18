"""Simple baselines and the per-image oracle LUT.

Every method maps an input image (3, H, W) in [0, 1] to an output image. Methods
fitted on data (global curves, fixed LUT) only see the training split; the
oracle is fitted on each test pair itself and is an upper bound for any single
global colour mapping, not a fair competitor.
"""
import torch

from .losses import LUTRegularizer
from .trilinear import apply_lut, identity_lut


def identity(img: torch.Tensor) -> torch.Tensor:
    return img


def gray_world(img: torch.Tensor) -> torch.Tensor:
    """Scale channels so their means match (classic white-balance assumption)."""
    means = img.mean(dim=(1, 2), keepdim=True)
    return (img * means.mean() / means.clamp_min(1e-6)).clamp(0, 1)


class GlobalCurves:
    """Per-channel 256-entry tone curve: mean target value for each input value, over the training set."""

    def __init__(self, bins: int = 256):
        self.bins = bins
        self.sums = torch.zeros(3, bins, dtype=torch.float64)
        self.counts = torch.zeros(3, bins, dtype=torch.float64)

    def update(self, src: torch.Tensor, tgt: torch.Tensor):
        idx = (src * (self.bins - 1)).round().long().flatten(1)
        self.sums.scatter_add_(1, idx, tgt.flatten(1).double())
        self.counts.scatter_add_(1, idx, torch.ones_like(idx, dtype=torch.float64))

    def finalize(self):
        ramp = torch.linspace(0, 1, self.bins, dtype=torch.float64).expand(3, -1)
        curve = torch.where(self.counts > 0, self.sums / self.counts.clamp_min(1), ramp)
        self.curve = torch.cummax(curve, dim=1).values.float()  # keep curves monotonic

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        idx = (img * (self.bins - 1)).round().long().flatten(1)
        return self.curve.gather(1, idx).view_as(img)


def fit_lut(src_px: torch.Tensor, tgt_px: torch.Tensor, dim: int = 33, steps: int = 500, lr: float = 5e-3,
            lambda_s: float = 1e-4, lambda_m: float = 10.0) -> torch.Tensor:
    """Fit one 3D LUT to paired pixels (3, N) by gradient descent, with the paper's regularizers.

    Lattice cells no pixel touches are held near the identity by the smoothness term.
    """
    device = src_px.device
    lut = identity_lut(dim).to(device).requires_grad_(True)
    reg = LUTRegularizer(dim).to(device)
    opt = torch.optim.Adam([lut], lr=lr)
    src_img, tgt_img = src_px[None, :, None, :], tgt_px[None, :, None, :]  # (1, 3, 1, N) as an image
    for _ in range(steps):
        pred = apply_lut(src_img, lut)
        tv, mn = reg(lut[None])
        loss = torch.mean((pred - tgt_img) ** 2) + lambda_s * tv + lambda_m * mn
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return lut.detach()


def paired_pixels(src: torch.Tensor, tgt: torch.Tensor, n: int, generator: torch.Generator | None = None):
    """Sample the same n pixel locations from two (3, H, W) images -> two (3, n) tensors."""
    s, t = src.flatten(1), tgt.flatten(1)
    idx = torch.randint(0, s.shape[1], (min(n, s.shape[1]),), generator=generator)
    return s[:, idx], t[:, idx]
