"""PSNR, SSIM and CIE76 colour error (deltaE in Lab), computed on 8-bit quantized images."""
import numpy as np
import torch
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity


def to_uint8(img: torch.Tensor) -> np.ndarray:
    """(3, H, W) float in [0, 1] -> (H, W, 3) uint8."""
    return (img.clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred.astype(np.float64) - target.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    return float(structural_similarity(pred, target, channel_axis=2, data_range=255))


def delta_e76(pred: np.ndarray, target: np.ndarray) -> float:
    diff = rgb2lab(pred) - rgb2lab(target)
    return float(np.sqrt((diff**2).sum(-1)).mean())


def all_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    p, t = to_uint8(pred), to_uint8(target)
    return {"psnr": psnr(p, t), "ssim": ssim(p, t), "delta_e": delta_e76(p, t)}
