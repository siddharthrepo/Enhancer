"""Score simple baselines and the per-image oracle LUT on the FiveK test split.

  identity      input unchanged
  gray_world    channel means equalized
  curves        per-channel tone curves fitted on the training set
  fixed_lut     one 3D LUT fitted on the training set (the model without its CNN)
  oracle_lut    one 3D LUT fitted to each test pair itself: the best any global colour mapping can do

Writes baselines.json and prints a table next to the paper's numbers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from lut.baselines import GlobalCurves, fit_lut, gray_world, identity, paired_pixels
from lut.data import FiveK, load_rgb
from lut.metrics import all_metrics
from lut.trilinear import apply_lut


def score(method, test_set, desc):
    rows = []
    for i in tqdm(range(len(test_set)), desc=desc, unit="img", leave=False, mininterval=5):
        src, tgt, _ = test_set[i]
        rows.append(all_metrics(method(src, tgt), tgt))
    return {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "delta_e")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="runs/baselines")
    p.add_argument("--px-per-image", type=int, default=2000, help="training pixels sampled per image for fixed_lut")
    p.add_argument("--oracle-px", type=int, default=50000, help="pixels sampled per test image for the oracle fit")
    p.add_argument("--oracle-steps", type=int, default=300)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-oracle", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator().manual_seed(0)
    train_set, test_set = FiveK(args.data, "train"), FiveK(args.data, "test")
    if args.limit:
        train_set.ids, test_set.ids = train_set.ids[: args.limit], test_set.ids[: args.limit]

    curves, src_px, tgt_px = GlobalCurves(), [], []
    for i in tqdm(range(len(train_set)), desc="fit on train", unit="img", mininterval=5):
        src_path, tgt_path = train_set.paths(i)  # un-augmented pair
        src, tgt = load_rgb(src_path), load_rgb(tgt_path)
        curves.update(src, tgt)
        s, t = paired_pixels(src, tgt, args.px_per_image, gen)
        src_px.append(s)
        tgt_px.append(t)
    curves.finalize()
    fixed = fit_lut(torch.cat(src_px, 1).to(device), torch.cat(tgt_px, 1).to(device)).cpu()

    def oracle(src, tgt):
        s, t = paired_pixels(src, tgt, args.oracle_px, gen)
        lut = fit_lut(s.to(device), t.to(device), steps=args.oracle_steps).cpu()
        return apply_lut(src[None], lut)[0]

    methods = {
        "identity": lambda s, t: identity(s),
        "gray_world": lambda s, t: gray_world(s),
        "curves": lambda s, t: curves(s),
        "fixed_lut": lambda s, t: apply_lut(s[None], fixed)[0],
    }
    if not args.skip_oracle:
        methods["oracle_lut"] = oracle

    results = {}
    for name, fn in methods.items():
        results[name] = score(fn, test_set, name)
        print(f"{name:12s} psnr {results[name]['psnr']:.2f}  ssim {results[name]['ssim']:.3f}  "
              f"dE {results[name]['delta_e']:.2f}", flush=True)
    print(f"{'paper model':12s} psnr 25.21  ssim 0.922  dE 7.61")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "baselines.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
