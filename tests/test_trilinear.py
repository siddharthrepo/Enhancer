import numpy as np
import torch

from lut.trilinear import apply_lut, identity_lut


def naive_trilinear(img: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Reference lookup. img (H, W, 3) in [0, 1], lut (3, D, D, D) indexed [c, b, g, r]."""
    d = lut.shape[-1]
    out = np.zeros_like(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            pos = img[y, x] * (d - 1)  # (r, g, b) in lattice units
            i0 = np.minimum(np.floor(pos).astype(int), d - 2)
            f = pos - i0
            acc = np.zeros(3)
            for dr in (0, 1):
                for dg in (0, 1):
                    for db in (0, 1):
                        wgt = (f[0] if dr else 1 - f[0]) * (f[1] if dg else 1 - f[1]) * (f[2] if db else 1 - f[2])
                        acc += wgt * lut[:, i0[2] + db, i0[1] + dg, i0[0] + dr]
            out[y, x] = acc
    return out


def test_identity_lut_is_identity():
    img = torch.rand(2, 3, 17, 23)
    out = apply_lut(img, identity_lut(33))
    assert torch.allclose(out, img, atol=1e-5)


def test_matches_naive_reference():
    torch.manual_seed(0)
    lut = torch.rand(3, 9, 9, 9)
    img = torch.rand(1, 3, 6, 7)
    img[0, :, 0, 0] = torch.tensor([0.0, 0.0, 0.0])
    img[0, :, 0, 1] = torch.tensor([1.0, 1.0, 1.0])
    img[0, :, 0, 2] = torch.tensor([1.0, 0.0, 0.5])
    out = apply_lut(img, lut)[0].permute(1, 2, 0).numpy()
    ref = naive_trilinear(img[0].permute(1, 2, 0).numpy().astype(np.float64), lut.numpy().astype(np.float64))
    assert np.allclose(out, ref, atol=1e-5)


def test_channel_axis_order():
    # a LUT that outputs only the red input must ignore green and blue
    d = 5
    lut = identity_lut(d)
    lut[1:] = 0
    img = torch.tensor([[[[0.25]], [[0.9]], [[0.6]]]])  # r, g, b
    out = apply_lut(img, lut)
    assert torch.allclose(out.flatten(), torch.tensor([0.25, 0.0, 0.0]), atol=1e-6)


def test_per_sample_luts():
    img = torch.rand(2, 3, 4, 4)
    luts = torch.stack([identity_lut(9), 1 - identity_lut(9)])
    out = apply_lut(img, luts)
    assert torch.allclose(out[0], img[0], atol=1e-5)
    assert torch.allclose(out[1], 1 - img[1], atol=1e-5)
