from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .metrics import all_metrics, to_uint8


@torch.no_grad()
def evaluate(model, dataset, device, save_dir: Path | None = None) -> tuple[dict[str, float], list[dict]]:
    """Run the model on every image of `dataset` at native resolution. Returns (means, per-image rows)."""
    model.eval()
    rows = []
    for i in tqdm(range(len(dataset)), desc="eval", unit="img", leave=False, mininterval=5, dynamic_ncols=True):
        src, tgt, name = dataset[i]
        out, _ = model(src[None].to(device))
        out = out[0].clamp(0, 1).cpu()
        rows.append({"name": name, **all_metrics(out, tgt)})
        if save_dir is not None:
            save_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(to_uint8(out)).save(save_dir / f"{name}.png")
    means = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "delta_e")}
    return means, rows


class FastPSNR:
    """PSNR-only test evaluation on the training device, for every-epoch model selection.

    Test pairs are decoded once and cached as uint8; outputs are quantized to 8 bits like
    `all_metrics`, so the numbers match the full evaluation.
    """

    def __init__(self, dataset):
        self.pairs = []
        for i in tqdm(range(len(dataset)), desc="cache test set", unit="img", leave=False, mininterval=5):
            src, tgt, _ = dataset[i]
            self.pairs.append(((src * 255).round().byte(), (tgt * 255).round().byte()))

    @torch.no_grad()
    def __call__(self, model, device) -> float:
        model.eval()
        scores = []
        for src, tgt in self.pairs:
            out, _ = model(src[None].to(device).float() / 255)
            out = (out[0].clamp(0, 1) * 255).round()
            mse = torch.mean((out - tgt.to(device).float()) ** 2).item()
            scores.append(float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse))
        return float(np.mean(scores))
