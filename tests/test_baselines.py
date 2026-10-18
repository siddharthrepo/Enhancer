import torch

from lut.baselines import GlobalCurves, fit_lut, gray_world, paired_pixels
from lut.trilinear import apply_lut


def test_gray_world_equalizes_channel_means():
    img = torch.rand(3, 32, 32) * torch.tensor([1.0, 0.6, 0.3]).view(3, 1, 1)
    means = gray_world(img).mean(dim=(1, 2))
    assert torch.allclose(means, means.mean().expand(3), atol=1e-3)


def test_global_curves_learn_a_tone_curve():
    curves = GlobalCurves()
    for _ in range(4):
        src = torch.rand(3, 64, 64)
        curves.update(src, src**0.5)
    curves.finalize()
    src = torch.rand(3, 16, 16)
    assert torch.allclose(curves(src), src**0.5, atol=0.02)


def test_fit_lut_recovers_a_colour_transform():
    torch.manual_seed(0)
    src = torch.rand(3, 64, 64)
    tgt = torch.stack([src[1], src[0], src[2] * 0.5])  # swap R/G, halve B
    lut = fit_lut(*paired_pixels(src, tgt, 4096), dim=9, steps=400, lr=2e-2, lambda_m=0.0)
    assert torch.mean((apply_lut(src[None], lut)[0] - tgt) ** 2) < 1e-4
