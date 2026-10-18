"""FiveK 480p dataset in the layout released by Zeng et al.

root/
  input/JPG/480p/a0001.jpg        8-bit sRGB inputs
  expertC/JPG/480p/a0001.jpg      expert C targets
  train_input.txt, train_label.txt, test.txt   image ids, one per line
The paired training set is train_input + train_label (4500), test is 500.
"""
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


def read_ids(path: Path) -> list[str]:
    return sorted({Path(line.strip()).stem for line in path.read_text().splitlines() if line.strip()})


def load_rgb(path: Path) -> torch.Tensor:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def adjust_saturation(img: torch.Tensor, factor: float) -> torch.Tensor:
    gray = (0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]).unsqueeze(0)
    return (gray + factor * (img - gray)).clamp(0, 1)


class FiveK(Dataset):
    def __init__(self, root, split: str = "train", crop_size: int | None = None, ids: list[str] | None = None):
        self.root = Path(root)
        self.split = split
        self.crop_size = crop_size  # resize random crops to a fixed size so batches > 1 work
        if ids is None:
            if split == "train":
                ids = read_ids(self.root / "train_input.txt") + read_ids(self.root / "train_label.txt")
            else:
                ids = read_ids(self.root / "test.txt")
        self.ids = sorted(set(ids))

    def __len__(self):
        return len(self.ids)

    def paths(self, i: int) -> tuple[Path, Path]:
        name = self.ids[i] + ".jpg"
        return self.root / "input/JPG/480p" / name, self.root / "expertC/JPG/480p" / name

    def __getitem__(self, i):
        src_path, tgt_path = self.paths(i)
        src, tgt = load_rgb(src_path), load_rgb(tgt_path)
        if self.split == "train":
            src, tgt = self.augment(src, tgt)
        return src, tgt, self.ids[i]

    def augment(self, src, tgt):
        # paper: random crop at 0.6-1.0 scale, horizontal flip, small brightness/saturation jitter on input
        _, h, w = src.shape
        ch, cw = int(h * random.uniform(0.6, 1.0)), int(w * random.uniform(0.6, 1.0))
        y, x = random.randint(0, h - ch), random.randint(0, w - cw)
        src, tgt = src[:, y : y + ch, x : x + cw], tgt[:, y : y + ch, x : x + cw]
        if random.random() < 0.5:
            src, tgt = src.flip(-1), tgt.flip(-1)
        src = (src * random.uniform(0.8, 1.2)).clamp(0, 1)
        src = adjust_saturation(src, random.uniform(0.8, 1.2))
        if self.crop_size:
            size = (self.crop_size, self.crop_size)
            src = F.interpolate(src[None], size=size, mode="bilinear", antialias=True, align_corners=False)[0]
            tgt = F.interpolate(tgt[None], size=size, mode="bilinear", antialias=True, align_corners=False)[0]
        return src, tgt
