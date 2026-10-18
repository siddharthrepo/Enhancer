# Image-adaptive 3D LUT photo enhancer

A from-scratch PyTorch implementation of [Zeng et al., "Learning Image-adaptive 3D Lookup Tables"](https://arxiv.org/abs/2009.14468) (TPAMI 2020). A small CNN looks at a 256×256 thumbnail and blends 3 learned 33³ colour LUTs into one image-specific LUT, which is applied to the full-resolution image. It fixes colour and tone only (white balance, contrast, saturation), not noise or detail.

See `feature.txt` for the full project plan.

## Layout

```
lut/trilinear.py   LUT lookup via F.grid_sample (also used for ONNX export)
lut/model.py       weight-predictor CNN + LUT bank
lut/losses.py      MSE + smoothness (3D TV) + monotonicity regularizers
lut/data.py        FiveK 480p loader + paper augmentations
lut/metrics.py     PSNR, SSIM, deltaE (CIE76), on 8-bit images
train.py / eval.py resumable training and test-split evaluation
kaggle/            notebook for training on Kaggle GPUs
tests/             lookup-vs-reference, shapes, regularizers, train/resume smoke test
```

## Local setup (code + tests only, CPU)

```bash
uv venv .venv
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python numpy pillow scikit-image tqdm pytest onnx onnxruntime
.venv/bin/python -m pytest -q
```

## Training on Kaggle

1. Push this repo to GitHub, then set `REPO` in `kaggle/train_fivek.ipynb`.
2. Import the notebook into Kaggle. Turn on GPU (T4) and Internet, and add the dataset
   [`jesucristo/zeng-3lut-fivek-480p`](https://www.kaggle.com/datasets/jesucristo/zeng-3lut-fivek-480p), a mirror of the authors' 480p release.
3. Run all cells. `train.py` writes `last.pt` every epoch and resumes automatically. If a session hits Kaggle's 12h limit, save the output as a dataset version, attach it, and set `RESUME_FROM`.

## Phase 1 target (FiveK 480p, expert C, 4500/500)

| | PSNR | SSIM | ΔE |
|---|---|---|---|
| Paper | 25.21 | 0.922 | 7.61 |
| Ours | – | – | – |
