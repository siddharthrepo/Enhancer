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
