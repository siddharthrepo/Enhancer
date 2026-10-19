"""Full-resolution inference: load a checkpoint and photos, predict one LUT per photo, apply it.

The CNN only ever sees a thumbnail, made the way training saw it: the photo is first
downscaled with antialiasing to a 480px short side (FiveK 480p), then the model resizes that
to 256x256. The predicted LUT is applied to every pixel of the original image, in row chunks
so 48MP photos stay within a few hundred MB of memory.
"""
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageCms, ImageOps

from .model import AdaptiveLUT
from .trilinear import apply_lut

try:  # optional HEIC/HEIF support (iPhone, recent Samsung)
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

SRGB = ImageCms.createProfile("sRGB")
TRAIN_SHORT_SIDE = 480
EXIF_ORIENTATION = 0x0112


def load_model(ckpt_path, device="cpu") -> AdaptiveLUT:
    ckpt = torch.load(ckpt_path, map_location=device)
    a = ckpt.get("args", {})
    model = AdaptiveLUT(a.get("n_luts", 3), a.get("lut_dim", 33))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def load_photo(path) -> tuple[Image.Image, dict]:
    """Open a photo as 8-bit sRGB with EXIF orientation applied. Returns (image, info for saving)."""
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # phones store rotation as a tag, not in the pixels
    icc = img.info.get("icc_profile")
    converted_from = None
    if icc:
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        name = ImageCms.getProfileDescription(src).strip()
        if "srgb" not in name.lower():  # e.g. Display P3 from iPhones / Samsungs
            img = ImageCms.profileToProfile(img.convert("RGB"), src, SRGB, outputMode="RGB")
            converted_from = name
    exif = img.getexif()
    exif[EXIF_ORIENTATION] = 1  # pixels are now upright
    return img.convert("RGB"), {"exif": exif.tobytes(), "converted_from": converted_from}


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).contiguous()


def to_image(t: torch.Tensor) -> Image.Image:
    arr = (t[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr)


def training_scale_thumbnail(img: torch.Tensor) -> torch.Tensor:
    """Downscale (antialiased) to a 480px short side, like the FiveK 480p training images."""
    h, w = img.shape[-2:]
    scale = TRAIN_SHORT_SIDE / min(h, w)
    if scale >= 1:
        return img
    size = (max(1, round(h * scale)), max(1, round(w * scale)))
    return F.interpolate(img, size=size, mode="bilinear", antialias=True, align_corners=False)


@torch.no_grad()
def enhance(model: AdaptiveLUT, img: torch.Tensor, chunk_rows: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """img (1, 3, H, W) in [0, 1] -> (enhanced (1, 3, H, W), fused LUT (1, 3, D, D, D))."""
    device = next(model.parameters()).device
    lut, _ = model.fuse(training_scale_thumbnail(img).to(device))
    out = torch.empty_like(img)
    for y in range(0, img.shape[-2], chunk_rows):
        rows = img[..., y : y + chunk_rows, :].to(device)
        out[..., y : y + chunk_rows, :] = apply_lut(rows, lut).cpu()
    return out, lut


def save_photo(img: Image.Image, path: Path, info: dict, quality: int = 95):
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs = {"icc_profile": ImageCms.ImageCmsProfile(SRGB).tobytes()}
    if path.suffix.lower() in (".jpg", ".jpeg"):
        kwargs.update(quality=quality, subsampling=0, exif=info.get("exif", b""))
    img.save(path, **kwargs)
