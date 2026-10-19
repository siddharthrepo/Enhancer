"""Enhance photos (colour and tone only) with a trained image-adaptive 3D LUT.

  python enhance.py photo.jpg                      # -> enhanced/photo.jpg
  python enhance.py ~/Pictures/trip -o out --compare
  python enhance.py photo.heic --save-lut          # also writes the predicted LUT as a .cube file

Output has the same resolution as the input. HEIC needs `pip install pillow-heif`.
"""
import argparse
import time
from pathlib import Path

import torch
from PIL import Image

from lut.inference import enhance, load_model, load_photo, save_photo, to_image, to_tensor

EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".heic", ".heif", ".bmp"}


def collect(inputs: list[str]) -> list[Path]:
    files = []
    for item in map(Path, inputs):
        if item.is_dir():
            files += sorted(p for p in item.rglob("*") if p.suffix.lower() in EXTS)
        else:
            files.append(item)
    return files


def write_cube(lut: torch.Tensor, path: Path, title: str):
    """.cube format: red index varies fastest, which matches our [c, blue, green, red] layout."""
    d = lut.shape[-1]
    rgb = lut[0].permute(1, 2, 3, 0).reshape(-1, 3).clamp(0, 1)
    lines = [f'TITLE "{title}"', f"LUT_3D_SIZE {d}"] + [f"{r:.6f} {g:.6f} {b:.6f}" for r, g, b in rgb.tolist()]
    path.write_text("\n".join(lines) + "\n")


def side_by_side(before: Image.Image, after: Image.Image, max_width: int = 2400) -> Image.Image:
    w, h = before.size
    canvas = Image.new("RGB", (2 * w, h))
    canvas.paste(before, (0, 0))
    canvas.paste(after, (w, 0))
    if canvas.width > max_width:
        canvas = canvas.resize((max_width, round(h * max_width / canvas.width)), Image.LANCZOS)
    return canvas


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("inputs", nargs="+", help="image files or folders")
    p.add_argument("--ckpt", default="results/runs/fivek/best.pt")
    p.add_argument("-o", "--out", default="enhanced")
    p.add_argument("--compare", action="store_true", help="also save a before|after image")
    p.add_argument("--save-lut", action="store_true", help="also save the predicted LUT as .cube")
    p.add_argument("--quality", type=int, default=95)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.ckpt, device)
    out_dir = Path(args.out)
    files = collect(args.inputs)
    if not files:
        raise SystemExit("no images found")

    for path in files:
        t0 = time.perf_counter()
        img, info = load_photo(path)
        out, lut = enhance(model, to_tensor(img))
        result = to_image(out)
        ms = (time.perf_counter() - t0) * 1000
        suffix = ".png" if path.suffix.lower() in (".png", ".tif", ".tiff") else ".jpg"
        dest = out_dir / (path.stem + suffix)
        save_photo(result, dest, info, args.quality)
        if args.compare:
            save_photo(side_by_side(img, result), out_dir / f"{path.stem}_compare.jpg", {}, 90)
        if args.save_lut:
            write_cube(lut.cpu(), out_dir / f"{path.stem}.cube", path.stem)
        note = f" (converted from {info['converted_from']})" if info["converted_from"] else ""
        print(f"{path.name}: {img.width}x{img.height} -> {dest} in {ms:.0f} ms{note}")


if __name__ == "__main__":
    main()
