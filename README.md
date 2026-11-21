# NPSS-Image

Author: [masanli1](https://github.com/masanli1) · Repository: [NPSS-Image](https://github.com/masanli1/NPSS-Image)

[中文说明](README.zh-CN.md)

> **Block-DCT shrinkage with cross-phase structure confidence.**
> Single file, no dependencies beyond numpy + Pillow, no training, no GPU, no model weights.

A training-free image denoiser built on two ideas:

1. **Threshold what is unstable.** Split the image into 8×8 blocks, take the DCT, and
   keep only coefficients above `k0·σ`. DC is always kept. Repeat over 16 (up to 64)
   block phases and average — cycle spinning removes the blocking artefact.
2. **Relax the threshold where structure is consistently recognised.** The per-phase
   keep/kill decisions are re-projected into image space to build a structure-confidence
   map `rho`. The threshold is then scaled by `(1 - λ·rho)`: where a region keeps being
   recognised as structure across all phases, weaker coefficients are allowed through.

`rho` is the interesting part — see [Why the DC must be cleared](#why-the-dc-must-be-cleared).

## Measured

Reproducible with `python3 benchmark.py` (384×384 colour, synthetic + one real texture,
fixed seeds). AWGN, σ on the 0–255 scale.

| λ (enhance) | σ=15 PSNR | σ=15 SSIM | σ=25 PSNR | σ=25 SSIM |
|---|---|---|---|---|
| 0.0 (base) | 29.12 | — | 26.21 | — |
| 0.3 | 29.46 (+0.34) | | 26.54 (+0.32) | |
| **0.5 (default)** | **29.59 (+0.47)** | | **26.66 (+0.45)** | |
| 0.7 | 29.62 (+0.50) | | 26.68 (+0.46) | |
| 0.9 | 29.53 (+0.41) | | 26.55 (+0.34) | |

Per-image at σ=15, λ=0.5:

| texture | base | + enhancement | gain |
|---|---|---|---|
| grass (high-frequency) | 26.63 / 0.771 | 27.28 / 0.832 | +0.65 dB, +0.061 SSIM |
| fabric (woven grid) | 27.36 / 0.939 | 27.61 / 0.956 | +0.25 dB, +0.017 SSIM |
| fractal 1/f | 26.42 / 0.739 | 27.11 / 0.824 | +0.69 dB, +0.085 SSIM |
| brick (periodic) | 36.07 / 0.959 | 36.35 / 0.959 | +0.28 dB, ±0.000 SSIM |

On smooth, low-texture photographs (Kodak `kodim23`) the gain is smaller but still
positive: 34.53 → 34.66 dB at σ=15, and no SSIM loss.

## Usage

```python
import numpy as np
from PIL import Image
import npss_image as M

img = np.asarray(Image.open("photo.jpg").convert("RGB"), np.float32) / 255.0

# best settings: multi-scale + structural enhancement
out, _ = M.denoise_structural(img, sigma=None, k0=3.0, blocks=(4, 8, 16), lam=0.5)

# base algorithm only
out2, _ = M.denoise(img, sigma=None, k0=3.0, blocks=(4, 8, 16))

Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)).save("denoised.png")
```

`sigma=None` estimates the noise level internally. `sigma` is on the 0–1 scale when
passed to the API and 0–255 on the CLI.

CLI:

```bash
python3 demo.py photo.jpg                          # auto sigma, best settings
python3 demo.py photo.jpg --sigma 25 --ref clean.png   # report PSNR / SSIM
python3 demo.py photo.jpg --enhance 0              # disable the enhancement
python3 benchmark.py --sigma 15 25 --lam 0 0.5 0.9
```

## Why the DC must be cleared

The confidence map is built by inverse-transforming the binary keep-mask back into image
space. The DC coefficient is **always** kept, so if it is included in that mask it
dominates the inverse transform and the map collapses to a near-constant image.

Measured separation between structure (top 10% gradient) and flat (bottom 40%) regions:

| mask | structure | flat | separation |
|---|---|---|---|
| DC included | 0.491 | 0.496 | **0.00** (useless) |
| **DC cleared** | **0.415** | **0.072** | **0.34** |

Clearing DC is a one-line change that turns the map from noise into the whole signal.

## What this is not

- **Not a BM3D competitor.** It only looks inside each block; BM3D additionally searches
  the image for similar patches and pools their evidence before deciding. On texture-dense
  images that is worth roughly 2–4 dB, which this algorithm does not attempt to recover.
  On smooth images at moderate noise the gap narrows to under 1 dB.
  Closing it needs non-local redundancy, not smarter thresholding.
- **Not learned.** No training set, no weights, no GPU.
- **Periodic textures are its weak spot.** Brick/checkerboard patterns alias across the
  fixed block grid; the confidence map stays flat and the enhancement adds little
  (SSIM can dip by ~0.002).

## Notes

- Cycle spinning with 64 phases costs 4 full passes over the image; a 600×400 colour
  photo takes ~2 s single-threaded. Use `blocks=(8,)` and `lam=0` for ~4× speed.
- BLAS is pinned to one thread inside the module — small matrix batches are faster that way.

## License

Dual-licensed: free for academic / non-commercial use ([LICENSE-ACADEMIC.md](LICENSE-ACADEMIC.md)),
paid license required for commercial use ([LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md)).
Contact `2459551532@qq.com`.
