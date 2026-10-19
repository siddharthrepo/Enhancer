"""Train the image-adaptive 3D LUT on FiveK 480p (paired, expert C).

Resumable: writes <out>/last.pt every epoch and picks it up automatically, so
training can span several Kaggle sessions.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from lut.data import FiveK
from lut.evaluate import FastPSNR, evaluate
from lut.losses import LUTRegularizer, lut_loss
from lut.model import AdaptiveLUT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="FiveK root containing input/, expertC/, test.txt")
    p.add_argument("--out", default="runs/fivek")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=1, help="paper uses 1; >1 needs --crop-size")
    p.add_argument("--crop-size", type=int, default=None, help="resize random crops to this square size")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n-luts", type=int, default=3)
    p.add_argument("--lut-dim", type=int, default=33)
    p.add_argument("--lambda-s", type=float, default=1e-4)
    p.add_argument("--lambda-m", type=float, default=10.0)
    p.add_argument("--eval-every", type=int, default=10,
                   help="test PSNR every N epochs; full SSIM/deltaE only on a new best PSNR")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="use only N train/test images (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--paper-init", action="store_true", help="random weight-predictor init as in the paper")
    p.add_argument("--max-hours", type=float, default=None,
                   help="stop cleanly after this many hours (e.g. 11 on Kaggle); re-run to resume")
    return p.parse_args()


def _fmt(seconds: float) -> str:
    h, m = divmod(int(seconds) // 60, 60)
    return f"{h}h{m:02d}m"


def main():
    args = parse_args()
    if args.batch_size > 1 and not args.crop_size:
        raise SystemExit("--batch-size > 1 requires --crop-size (480p images have different shapes)")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_set = FiveK(args.data, "train", crop_size=args.crop_size)
    test_set = FiveK(args.data, "test")
    if args.limit:
        train_set.ids, test_set.ids = train_set.ids[: args.limit], test_set.ids[: args.limit]
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=device.type == "cuda", drop_last=True, persistent_workers=args.workers > 0)

    model = AdaptiveLUT(args.n_luts, args.lut_dim, identity_init=not args.paper_init).to(device)
    reg = LUTRegularizer(args.lut_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    start_epoch, best_psnr = 0, -1.0

    if (out / "last.pt").exists():
        ckpt = torch.load(out / "last.pt", map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        start_epoch, best_psnr = ckpt["epoch"] + 1, ckpt["best_psnr"]
        print(f"resumed from epoch {ckpt['epoch']} (best psnr {best_psnr:.2f})")

    print(f"device={device} train={len(train_set)} test={len(test_set)} "
          f"params={sum(p.numel() for p in model.predictor.parameters()) / 1e3:.0f}K (CNN)")
    fast_psnr = FastPSNR(test_set)
    log = open(out / "log.jsonl", "a")
    run_start = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, sums, n = time.time(), {"mse": 0.0, "tv": 0.0, "mn": 0.0}, 0
        bar = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}", unit="it", mininterval=5, dynamic_ncols=True)
        for src, tgt, _ in bar:
            src, tgt = src.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
            pred, w = model(src)
            loss, parts = lut_loss(pred, tgt, w, model.luts, reg, args.lambda_s, args.lambda_m)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            for k in sums:
                sums[k] += parts[k]
            n += 1
            if n % 50 == 0:
                bar.set_postfix(mse=f"{sums['mse'] / n:.5f}")
        bar.close()
        rec = {"epoch": epoch, "time_s": round(time.time() - t0, 1), **{k: v / max(n, 1) for k, v in sums.items()}}

        last_epoch = epoch == args.epochs - 1
        if (epoch + 1) % args.eval_every == 0 or last_epoch:
            rec["psnr"] = fast_psnr(model, device)
            if rec["psnr"] > best_psnr:
                metrics, _ = evaluate(model, test_set, device)  # full metrics only for a new best
                rec.update(metrics)
                best_psnr = metrics["psnr"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": metrics, "args": vars(args)},
                           out / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": epoch,
                    "best_psnr": best_psnr, "args": vars(args)}, out / "last.pt")
        done = epoch + 1 - start_epoch
        eta = (time.time() - run_start) / done * (args.epochs - epoch - 1)
        scores = f" | psnr {rec['psnr']:.2f}" if "psnr" in rec else ""
        if "ssim" in rec:
            scores += f" ssim {rec['ssim']:.3f} dE {rec['delta_e']:.2f} (new best)"
        print(f"epoch {epoch + 1}/{args.epochs} | mse {rec['mse']:.5f} | {rec['time_s']:.0f}s{scores} "
              f"| best psnr {best_psnr:.2f} | ETA {_fmt(eta)}", flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if args.max_hours and time.time() - run_start > args.max_hours * 3600 and not last_epoch:
            print(f"time budget of {args.max_hours}h reached after epoch {epoch + 1}; re-run to resume", flush=True)
            break


if __name__ == "__main__":
    main()
