# Changelog

## 1.3.0 — 2026-11-21

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

## 1.2.0 — 2026-11-21

Unified entry point with noise-type routing.

- `noise_profile()` — estimates the noise sigma per DCT frequency from the flattest
  blocks, and returns how curved the curve is (`flatness`).
- `denoise_auto()` now routes:
  - flatness < 1.6 → white noise → scalar sigma (`_estimate_sigma`) + structural
    enhancement — bit-identical to before;
  - flatness ≥ 1.6 → coloured noise → **per-frequency thresholds** + structural
    enhancement.
- Measured (auto sigma on both sides, so the comparison is fair):

| noise | before | routed | gain | flatness |
|---|---|---|---|---|
| white (kodim23) | 33.40 | 33.40 | +0.00 | 1.4 |
| white (grass) | 22.91 | 22.91 | +0.00 | 1.4 |
| white (kodim01) | 27.37 | 27.37 | +0.00 | 1.3 |
| **coloured (kodim23)** | 22.17 | **26.66** | **+4.49** | 48.5 |
| **coloured (grass)** | 21.47 | **23.79** | **+2.32** | 4.1 |
| **coloured (kodim01)** | 22.16 | **24.96** | **+2.80** | 30.6 |

- Flatness separates cleanly: white 1.3–1.5, coloured 4.1–48.5.
- Pattern removal stays gated; grass shows −0.16 dB, i.e. within noise.

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
