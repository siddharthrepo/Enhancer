"""Evaluate a checkpoint on the FiveK test split; writes metrics.json, per_image.csv and optionally outputs."""
import argparse
import csv
import json
from pathlib import Path

import torch

from lut.data import FiveK
from lut.evaluate import evaluate
from lut.model import AdaptiveLUT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", default=None, help="defaults to the checkpoint's folder")
    p.add_argument("--save-images", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    a = ckpt["args"]
    model = AdaptiveLUT(a["n_luts"], a["lut_dim"]).to(device)
    model.load_state_dict(ckpt["model"])
    out = Path(args.out or Path(args.ckpt).parent)

    means, rows = evaluate(model, FiveK(args.data, "test"), device,
                           save_dir=out / "test_outputs" if args.save_images else None)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(means, indent=2))
    with open(out / "per_image.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(means, indent=2))
    print("paper (480p, paired): psnr 25.21  ssim 0.922  delta_e 7.61")


if __name__ == "__main__":
    main()
