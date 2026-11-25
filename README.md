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

# recommended: removes a fixed pattern if one is present, then denoises
out, info = M.denoise_auto(img, sigma=None, blocks=(4, 8, 16), lam=0.5)

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
python3 benchmark.py --pattern          # fixed-pattern removal table
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

## Pipeline (v1.3)

```python
out, info = M.denoise_auto(img)     # everything below, in order
# info = {'pattern_corr': 0.97, 'pattern_used': True, 'flatness': 1.4, 'route': 'white'}
```

1. gated fixed-pattern removal,
2. noise-type diagnostics (per-frequency sigma + flatness),
3. DCT threshold + multi-scale cycle spinning + structure-confidence relaxation.

On white noise and pattern-free images step 1 and the routing are skipped, so the output is
identical to `denoise_structural`.

### Coloured noise (opt-in)

`denoise_auto(img, colored_route=True)` uses per-frequency thresholds when the noise curve
is not flat: **+4.5 / +2.3 / +2.8 dB** on kodim23 / grass / kodim01. It is off by default
because a strongly periodic texture (woven fabric) reads the same flatness and loses
**−3.0 dB**.

## Micro-structure removal (v1.4)

Noise structure is *small*; texture is *large*. `micro_pattern()` high-passes the image
(texture vanishes, contributing negative correlation), pools the residue by pixel position
modulo 2, and gates on two independent halves plus a minimum amplitude (10% of sigma).

| input | two-half corr | amplitude / sigma | fires |
|---|---|---|---|
| 1-pixel column pattern | 0.99 | 0.20 | ✓ |
| white noise | 0.47 | 0.01 | ✗ |
| kodim01 | 0.78 | 0.02 | ✗ |
| woven fabric | −0.85 | 0.00 | ✗ |

`denoise_auto()` runs this **and** the block-level detector; each catches its own scale:

| noise | kodim23 | kodim01 | fabric | grass |
|---|---|---|---|---|
| white | +0.00 | +0.00 | +0.00 | +0.00 |
| 1-pixel pattern | +0.24 | +0.09 | −0.01 | −0.08 |
| block pattern | +1.43 | +0.60 | −0.00 | −0.13 |

## Fixed-pattern removal (v1.1)

Real cameras, scanners and screen captures often carry a *spatial* pattern on top of
random noise: column banding, sensor fixed-pattern noise, moiré grids. A random-noise
threshold cannot touch it — the pattern looks like structure and gets preserved.

`fixed_pattern()` pools every 8×8 block and checks whether two independent halves see the
same pattern. If they do (correlation > 0.5), the common component is subtracted; if they
do not, the image is returned untouched.

| image | base | + pattern removal | gate correlation |
|---|---|---|---|
| fractal | 28.11 | **29.31** (+1.20) | 0.98 |
| brick | 34.77 | **36.36** (+1.60) | 0.99 |
| fabric | 28.94 | **29.92** (+0.99) | 0.98 |
| grass | 28.50 | **29.53** (+1.03) | 0.99 |

On white and coloured noise the gate stays at −0.49…0.12, so the step is skipped and the
output is **bit-identical** to `denoise_structural`.

`denoise_auto()` is the recommended entry point: gated pattern removal, then structural
denoising.

> An iterative version of this loop (subtract residual template, denoise again, repeat)
> was measured and **rejected**: it degrades ordinary images badly (fabric white noise
> 27.62 → 19.90 dB) because the convergence signal cannot distinguish "noise removed"
> from "structure over-smoothed".

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

## 运行档（variants）

| 档位 | 配置 | 256² 用时 | PSNR(σ=25) |
|---|---|---|---|
| `fast` | 8×8 单尺度，不做图案去除 | **0.033 s** | 29.06 |
| `standard` | 4+8+16 多尺度 + 图案去除 | 0.075 s | 29.19 |
| `adaptive` | 逐图 SURE 有限搜索选参 | 约 6 s/图 | 视图像而定 |

```python
from variants.profiles import denoise, denoise_adaptive
out, info = denoise(img, sigma=25/255, mode="fast")     # 高速档
out, info = denoise(img, sigma=25/255, mode="standard")  # 质量档
out, info = denoise_adaptive(img, sigma=25/255)          # 逐图自适应（SURE）
```

### Set12 实测（12 张 × σ=15/25/50）

| σ | 噪声原图 | fast | standard | BM3D |
|---|---|---|---|---|
| 15 | 24.67 | 31.71 | 31.76 | 32.40 |
| 25 | 20.33 | 29.04 | 29.10 | 29.98 |
| 50 | 14.76 | 25.20 | 25.33 | 26.35 |

`standard` 与 BM3D 差 0.85 dB，但快约 22 倍（0.075 s vs 11.8 s，2 vCPU）。

## 参数有限搜索

`tools/finite_search.py` 用「三节点法」把无界参数空间变成有限计算：

- 固定位置：`blocks`（7 种）、`remove_pattern`（2 种）→ 直接枚举
- 固定数量：连续参数只有 `k0` / `lam` / `flat` 三个
- 不固定：离散成有限槽位后，**粗格 → 细化**

实测：生物进化 360 次评估 / 368 s / 28.502 → **分层有限 50 次 / 67 s / 28.504**（快 5.5 倍）。

```bash
python3 tools/finite_search.py --images data/set12
python3 benchmark_set12.py --set12 data/set12
```
