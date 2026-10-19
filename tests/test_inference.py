import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lut.inference import enhance, load_photo, to_tensor
from lut.model import AdaptiveLUT

ROOT = Path(__file__).resolve().parents[1]


def _photo(path: Path, w=300, h=200, orientation=None):
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    exif = img.getexif()
    if orientation:
        exif[0x0112] = orientation
    img.save(path, quality=95, exif=exif.tobytes())
    return img


def test_exif_rotation_is_applied(tmp_path):
    _photo(tmp_path / "rot.jpg", orientation=6)  # 6 = rotate 90 degrees clockwise
    img, _ = load_photo(tmp_path / "rot.jpg")
    assert img.size == (200, 300)


def test_untrained_model_returns_input_and_chunking_is_exact():
    model = AdaptiveLUT().eval()
    img = torch.rand(1, 3, 700, 1100)
    out_full, _ = enhance(model, img, chunk_rows=10_000)
    out_chunked, _ = enhance(model, img, chunk_rows=128)
    assert torch.allclose(out_full, img, atol=1e-5)
    assert torch.equal(out_full, out_chunked)


def test_cli_writes_same_size_output_and_cube(tmp_path):
    _photo(tmp_path / "a.jpg", w=640, h=480)
    ckpt = tmp_path / "m.pt"
    torch.save({"model": AdaptiveLUT().state_dict(), "args": {"n_luts": 3, "lut_dim": 33}}, ckpt)
    out = tmp_path / "out"
    subprocess.run([sys.executable, "enhance.py", str(tmp_path / "a.jpg"), "--ckpt", str(ckpt), "-o", str(out),
                    "--compare", "--save-lut"], cwd=ROOT, check=True)
    assert Image.open(out / "a.jpg").size == (640, 480)
    assert (out / "a_compare.jpg").exists()
    cube = (out / "a.cube").read_text().splitlines()
    assert cube[1] == "LUT_3D_SIZE 33" and len(cube) == 2 + 33**3
    # identity model: first entry is black, second steps only in red (red varies fastest)
    assert cube[2] == "0.000000 0.000000 0.000000"
    assert cube[3].split()[1:] == ["0.000000", "0.000000"] and float(cube[3].split()[0]) > 0
