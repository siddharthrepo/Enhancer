import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from lut.losses import LUTRegularizer
from lut.model import AdaptiveLUT
from lut.trilinear import identity_lut

ROOT = Path(__file__).resolve().parents[1]


def test_param_count_matches_paper():
    model = AdaptiveLUT(n_luts=3, dim=33)
    cnn = sum(p.numel() for p in model.predictor.parameters())
    assert 260_000 < cnn < 280_000  # paper: ~269K
    assert model.luts.numel() == 3 * 3 * 33**3


def test_forward_shapes_any_resolution():
    model = AdaptiveLUT().eval()
    for h, w in ((480, 720), (123, 77)):
        out, weights = model(torch.rand(2, 3, h, w))
        assert out.shape == (2, 3, h, w)
        assert weights.shape == (2, 3)


def test_regularizer_identity_is_monotonic():
    reg = LUTRegularizer(9)
    tv, mn = reg(identity_lut(9)[None])
    assert mn.item() == 0
    tv_flipped, mn_flipped = reg((1 - identity_lut(9))[None])
    assert mn_flipped.item() > 0
    assert torch.isclose(tv, tv_flipped)


def _fake_fivek(root: Path, n_train=4, n_test=2):
    rng = np.random.default_rng(0)
    for sub in ("input/JPG/480p", "expertC/JPG/480p"):
        (root / sub).mkdir(parents=True)
    ids = [f"a{i:04d}" for i in range(1, n_train + n_test + 1)]
    for name in ids:
        img = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        Image.fromarray(img).save(root / "input/JPG/480p" / f"{name}.jpg")
        Image.fromarray(255 - img).save(root / "expertC/JPG/480p" / f"{name}.jpg")
    half = n_train // 2
    (root / "train_input.txt").write_text("\n".join(ids[:half]) + "\n")
    (root / "train_label.txt").write_text("\n".join(ids[half:n_train]) + "\n")
    (root / "test.txt").write_text("\n".join(ids[n_train:]) + "\n")


def test_train_and_eval_smoke(tmp_path):
    data, run = tmp_path / "fivek", tmp_path / "run"
    _fake_fivek(data)
    common = [sys.executable, "train.py", "--data", str(data), "--out", str(run), "--workers", "0", "--eval-every", "1"]
    subprocess.run(common + ["--epochs", "1", "--batch-size", "2", "--crop-size", "32"], cwd=ROOT, check=True)
    # second call must resume from last.pt and continue to epoch 2
    subprocess.run(common + ["--epochs", "2", "--batch-size", "2", "--crop-size", "32"], cwd=ROOT, check=True)
    assert len((run / "log.jsonl").read_text().splitlines()) == 2
    subprocess.run([sys.executable, "eval.py", "--data", str(data), "--ckpt", str(run / "best.pt")],
                   cwd=ROOT, check=True)
    assert (run / "metrics.json").exists()


def test_identity_init_returns_input():
    model = AdaptiveLUT().train()  # dropout active: zero head weights make it irrelevant at init
    img = torch.rand(2, 3, 40, 60)
    out, weights = model(img)
    assert torch.allclose(weights, torch.tensor([[1.0, 0.0, 0.0]] * 2))
    assert torch.allclose(out, img, atol=1e-5)


def test_every_lut_gets_gradient_at_init():
    # regression: with weights [1, 0, 0] and zero LUTs 1..N, those LUTs never received gradient
    model = AdaptiveLUT()
    src = torch.rand(1, 3, 64, 64)
    out, _ = model(src)
    torch.mean((out - src**0.5) ** 2).backward()
    # each LUT's weight must be learnable; once it moves off 0 the LUT itself gets gradient
    head_grad = model.predictor.head[1].weight.grad.flatten(1).norm(dim=1)
    assert (head_grad > 0).all()


def test_fast_psnr_matches_full_eval(tmp_path):
    from lut.data import FiveK
    from lut.evaluate import FastPSNR, evaluate

    _fake_fivek(tmp_path, n_train=2, n_test=4)
    test_set = FiveK(tmp_path, "test")
    torch.manual_seed(0)
    model = AdaptiveLUT(identity_init=False)
    full, _ = evaluate(model, test_set, "cpu")
    assert abs(FastPSNR(test_set)(model, "cpu") - full["psnr"]) < 1e-4
