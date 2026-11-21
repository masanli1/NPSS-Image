# Changelog

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
