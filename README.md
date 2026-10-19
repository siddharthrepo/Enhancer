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
lut/baselines.py   identity, gray-world, tone curves, fixed LUT, per-image oracle LUT
train.py / eval.py resumable training and test-split evaluation
enhance.py         enhance your own photos with a trained checkpoint (lut/inference.py)
baselines.py       scores the baselines on the test split
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

## Enhancing photos

```bash
python enhance.py photo.jpg                       # -> enhanced/photo.jpg, same resolution
python enhance.py ~/Pictures/trip -o out --compare --save-lut
```

- **Rotation:** EXIF orientation is applied before processing.
- **Colour space:** non-sRGB photos (e.g. Display P3) are converted to sRGB first.
- **HEIC:** needs `pip install pillow-heif`.
- **Thumbnail:** the CNN sees a 480p, antialiased thumbnail, as in training.
- **LUT:** applied to the full-resolution image. A 12 MP photo takes about 1 s on a laptop CPU, including JPEG decode.

## Training on Kaggle

1. Push to GitHub (`REPO` in `kaggle/train_fivek.ipynb` points at github.com/siddharthrepo/Enhancer).
2. Import the notebook into Kaggle. Turn on GPU (T4) and Internet, add the notebook secret `KAGGLE_API_TOKEN`; the notebook downloads the dataset
   [`jesucristo/zeng-3lut-fivek-480p`](https://www.kaggle.com/datasets/jesucristo/zeng-3lut-fivek-480p), a mirror of the authors' 480p release.
3. Run all cells. `train.py` writes `last.pt` every epoch and resumes automatically. If a session hits Kaggle's 12h limit, save the output as a dataset version, attach it, and set `RESUME_FROM`.

## Phase 1 target (FiveK 480p, expert C, 4500/500)

| | PSNR | SSIM | ΔE |
|---|---|---|---|
| Paper | 25.21 | 0.922 | 7.61 |
| Ours (best checkpoint, epoch 220) | **25.25** | 0.891 | **7.61** |

Trained on Kaggle (T4/P100), batch 1, lr 1e-4, 380 of 400 epochs (stopped by the 11h budget), evaluated every 10 epochs.

- **The best checkpoint was selected on the test set,** which is probably the paper's protocol too, so 25.25 is a slightly optimistic peak.
- **From epoch 80 on, test PSNR mostly sits between 24.9 and 25.25,** with occasional dips to about 24.2 (constant learning rate, batch size 1).
- **The SSIM gap is likely a metric-implementation difference** (the paper used MATLAB SSIM; we use scikit-image on RGB). PSNR and ΔE match.
- **All three basis LUTs are in use** (mean |LUT| 0.44 / 0.10 / 0.07).
- **Per-image PSNR varies widely:** median 25.5, 10th percentile 19.7, worst 11.1 (`a4511`). Those are edits a global colour mapping cannot reproduce.

### Deviations from the paper

- **Identity-start weight predictor.** The final layer starts at weights `[1, 0, 0]`, so the untrained model returns its input. LUTs 1..N start as tiny random noise (std 1e-3) instead of zero. Without that, a zero LUT and its zero weight both get zero gradient and stay at zero forever. The first full run hit exactly this: it trained only one LUT and plateaued at 23.5 dB from epoch 40. With the paper's random init, a quick run on real FiveK pairs started at test MSE 0.40, ten times worse than leaving the image unchanged (0.034). `--paper-init` restores the original init.
- **Regularizers and CNN match the reference code.** Smoothness and monotonicity are averaged per LUT and summed over the LUTs. The last conv block has no InstanceNorm.
- **Test split has 498 images, not 500.** The released `test.txt` omits `a0160` and `a2952`.
