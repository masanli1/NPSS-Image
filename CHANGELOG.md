## v2.0.0 — 2026-11-25

**运行档 + 参数有限搜索 + Set12 基准**

- 新增 `variants/profiles.py`：`fast`（8×8 单尺度，0.033 s/256²）、`standard`（4+8+16，0.075 s）、
  `adaptive`（逐图 SURE 有限搜索选参）
- 新增 `tools/finite_search.py`：三节点法把无界参数空间变成有限计算
  （生物进化 360 次/368 s → 分层有限 **50 次/67 s**，快 5.5 倍，结果持平）
- 新增 `benchmark_set12.py`：Set12 × σ=15/25/50，对比 fast / standard / BM3D
- 参数默认改为离线搜索结果：`k0=3.1`、`lam=0.7`（原 3.0 / 0.3）
- Set12 实测：standard **31.76 / 29.10 / 25.33 dB**（BM3D 32.40 / 29.98 / 26.35）

# Changelog

## 1.4.0 — 2026-11-21

Micro-structure detection — noise structure is *small*.

Noise has a fine, pixel-scale structure; texture is a large-scale structure. Pooling whole
8×8 blocks mixes the two, which is why earlier "template" attempts were fooled by woven
fabric. `micro_pattern()` separates them:

1. high-pass the image (removes large-scale structure — texture disappears, and in fact
   reads as *negative* correlation),
2. pool the residue by pixel position modulo `P` (default 2),
3. compare two independent halves, and require the pattern amplitude to be at least 10% of
   the noise sigma.

Measured two-half correlation:

| input | corr | fires |
|---|---|---|
| 1-pixel column pattern | **0.99** | ✓ |
| white noise (parrot) | 0.47 | ✗ (amplitude 0.01σ) |
| kodim01 | 0.78 | ✗ (amplitude 0.02σ) |
| woven fabric | **−0.85** | ✗ |
| grass | −0.26 | ✗ |

The amplitude gate is what makes it safe: a real pattern is ~20% of sigma, content-induced
spurious correlation is 1–3%.

`denoise_auto()` now runs micro detection (1–2 px) **and** block detection (8 px), each with
its own gate — a 1-pixel pattern is caught by the micro detector, an 8-pixel grid by the
block detector, neither catches the other:

| noise | kodim23 | kodim01 | fabric | grass |
|---|---|---|---|---|
| white | +0.00 | +0.00 | +0.00 | +0.00 |
| 1-pixel pattern | +0.24 | +0.09 | −0.01 | −0.08 |
| block pattern | +1.43 | +0.60 | −0.00 | −0.13 |

## 1.3.0 — 2026-11-21

> Covers what was drafted as 1.2.0 (noise-profile routing); both landed in this release.

`denoise_auto()` is now the single entry point and runs the whole validated pipeline:

1. gated fixed-pattern removal,
2. noise-type diagnostics (`noise_profile` → per-frequency sigma + flatness),
3. DCT thresholding with multi-scale cycle spinning and structure-confidence relaxation.

Measured with auto sigma on both sides:

| noise | before | `denoise_auto` | change |
|---|---|---|---|
| white (4 images) | — | identical | ±0.00 |
| fixed pattern (kodim23 / kodim01) | 32.61 / 27.01 | **33.36 / 27.36** | +0.75 / +0.35 |
| fixed pattern (grass) | 23.08 | 22.92 | −0.16 |

### Coloured-noise routing: implemented, **off by default**

`colored_route=True` switches to per-frequency thresholds when the curve is not flat:

| image | default | coloured_route | change |
|---|---|---|---|
| kodim23 | 22.17 | **26.66** | +4.49 |
| grass | 21.47 | **23.79** | +2.32 |
| kodim01 | 22.16 | **24.96** | +2.80 |
| fabric | 22.49 | 19.52 | **−2.97** |

The fabric failure is the reason it stays opt-in: that image's *pure white noise* already
reads a flatness of 3.90, overlapping the 4.14 of coloured noise on grass. Flatness alone
cannot separate "coloured noise" from "strongly periodic texture", so auto-routing is unsafe.

## 1.1.0 — 2026-11-21

Gated fixed-pattern removal.

- `fixed_pattern()` / `fixed_pattern_rgb()`: split every block into two halves; if both
  halves see the same spatial pattern (correlation > 0.5) it is subtracted. Otherwise
  nothing changes.
- `denoise_auto()`: recommended entry — gated pattern removal, then `denoise_structural`.
- Measured on 4 images: **+1.0 to +3.3 dB** on synthetic periodic patterns
  (gate correlation 0.94–0.99); **exactly 0 change** on white and coloured noise
  (gate correlation −0.49 to 0.12).

### Rejected: iterative closed loop

A loop of "denoise → pool patches → subtract residual template → denoise again" was
implemented and tested (`denoise_iterative`, not shipped). It looked like +4.8 dB on a
single image but degraded most others badly: fabric white noise 27.62 → 19.90 dB,
grass 26.33 → 21.25 dB. The convergence signal (pooled template energy) decreases both
when noise is removed *and* when structure is over-smoothed, so it cannot tell the two
apart. Only the one-shot, gated version is safe.

## 1.0.0 — 2026-11-21

First public release, extracted from the site's demo module.

- Base denoiser: 8×8 block DCT, 3σ hard threshold with DC protection,
  16/64-phase cycle spinning, optional multi-scale (4 / 8 / 16).
- **Structural enhancement** (`denoise_structural`, λ default 0.5): cross-phase
  keep/kill agreement becomes a structure-confidence map that relaxes the threshold
  where structure is consistently recognised. +0.47 dB / +0.06 SSIM on texture-dense
  images at σ=15, +0.45 dB at σ=25.
- `structure_confidence()` exposes the confidence map on its own.
- `benchmark.py`: reproducible texture benchmark with fixed seeds.
- `demo.py`: single-image CLI with auto noise estimation.

### Negative results kept for the record

- Threshold modulation driven by `quant_stability()` (quantisation-based perturbation)
  was measured in three directions — all worse (down to 24.3 dB at σ=25).
- Per-pixel median / consistency-weighted fusion across phases: +0.03 dB, i.e. noise.
- Confidence map without clearing DC: separation 0.00, unusable.
- Multi-scale confidence (4+8+16 combined): PSNR flat, ~+0.005 SSIM, 30% slower — not adopted.
