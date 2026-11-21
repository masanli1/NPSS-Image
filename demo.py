#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Denoise a single image.

    python3 demo.py photo.jpg                      # auto noise estimate, best settings
    python3 demo.py photo.jpg --sigma 25           # known noise level (0-255 scale)
    python3 demo.py photo.jpg --enhance 0          # base algorithm, no structural boost
    python3 demo.py photo.jpg --ref clean.png      # also report PSNR / SSIM
"""
import argparse
import sys

import numpy as np
from PIL import Image

import npss_image as M


def to_uint8(a):
    return (np.clip(a, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--sigma", type=float, default=None,
                    help="noise sigma on the 0-255 scale; omit to auto-estimate")
    ap.add_argument("--k0", type=float, default=3.0, help="threshold multiplier (default 3.0)")
    ap.add_argument("--blocks", default="4,8,16", help="block sizes, e.g. 8 or 4,8,16")
    ap.add_argument("--enhance", type=float, default=0.5,
                    help="structural enhancement lambda; 0 disables (default 0.5)")
    ap.add_argument("--ref", default=None, help="clean reference for PSNR / SSIM")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    im = Image.open(args.image).convert("RGB")
    a = np.asarray(im, dtype=np.float32) / 255.0
    blocks = tuple(int(x) for x in args.blocks.split(",") if x.strip().isdigit()) or (8,)
    sigma = None if args.sigma is None else args.sigma / 255.0

    if args.enhance and args.enhance > 0:
        out, _ = M.denoise_structural(a, sigma=sigma, k0=args.k0, blocks=blocks,
                                      lam=args.enhance)
    else:
        out, _ = M.denoise(a, sigma=sigma, k0=args.k0, blocks=blocks)
    out = np.clip(out, 0, 1)

    path = args.out or (args.image.rsplit(".", 1)[0] + "_denoised.png")
    Image.fromarray(to_uint8(out)).save(path)
    print(f"saved {path}  ({im.size[0]}x{im.size[1]}, blocks={blocks}, "
          f"enhance={args.enhance}, sigma={'auto' if sigma is None else args.sigma})")

    if args.ref:
        ref = np.asarray(Image.open(args.ref).convert("RGB"), dtype=np.float32) / 255.0
        noisy = a if a.shape == ref.shape else None
        if noisy is not None:
            print(f"PSNR  noisy {M.psnr(noisy, ref):.2f} -> denoised {M.psnr(out, ref):.2f} dB")
            print(f"SSIM  noisy {M.ssim(noisy, ref):.4f} -> denoised {M.ssim(out, ref):.4f}")


if __name__ == "__main__":
    sys.exit(main())
