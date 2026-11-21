# Changelog

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
