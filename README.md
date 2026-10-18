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

## Training on Kaggle

1. Push to GitHub (`REPO` in `kaggle/train_fivek.ipynb` points at github.com/siddharthrepo/Enhancer).
2. Import the notebook into Kaggle. Turn on GPU (T4) and Internet, add the notebook secret `KAGGLE_API_TOKEN`; the notebook downloads the dataset
   [`jesucristo/zeng-3lut-fivek-480p`](https://www.kaggle.com/datasets/jesucristo/zeng-3lut-fivek-480p), a mirror of the authors' 480p release.
3. Run all cells. `train.py` writes `last.pt` every epoch and resumes automatically. If a session hits Kaggle's 12h limit, save the output as a dataset version, attach it, and set `RESUME_FROM`.

## Phase 1 target (FiveK 480p, expert C, 4500/500)

| | PSNR | SSIM | ΔE |
|---|---|---|---|
| Paper | 25.21 | 0.922 | 7.61 |
| Ours | – | – | – |

### Deviations from the paper

- **Identity-start weight predictor.** The final layer starts at weights `[1, 0, 0]`, so the untrained model returns its input. LUTs 1..N start as tiny random noise (std 1e-3) instead of zero. Without that, a zero LUT and its zero weight both get zero gradient and stay at zero forever. The first full run hit exactly this: it trained only one LUT and plateaued at 23.5 dB from epoch 40. With the paper's random init, a quick run on real FiveK pairs started at test MSE 0.40, ten times worse than leaving the image unchanged (0.034). `--paper-init` restores the original init.
- **Regularizers and CNN match the reference code.** Smoothness and monotonicity are averaged per LUT and summed over the LUTs. The last conv block has no InstanceNorm.
- **Test split has 498 images, not 500.** The released `test.txt` omits `a0160` and `a2952`.
