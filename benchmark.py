#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reproducible benchmark: AWGN on texture-dense images.

    python3 benchmark.py                 # built-in synthetic textures, sigma 15 / 25
    python3 benchmark.py --kodak DIR     # also include kodimXX.png from DIR
    python3 benchmark.py --sigma 10 25 50 --lam 0 0.3 0.5

Every image is generated with a fixed seed (crc32 of its name), so numbers are stable.
"""
import argparse
import zlib

import numpy as np
from PIL import Image

import npss_image as M

SIZE = 384


# ---------------------------------------------------------------- test images
def _save(a):
    return np.clip(a, 0, 1).astype(np.float32)


def tex_fractal():
    rng = np.random.default_rng(11)
    f = np.fft.fftfreq(SIZE)
    F = np.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    F[0, 0] = 1e-6
    spec = rng.normal(size=(SIZE, SIZE)) + 1j * rng.normal(size=(SIZE, SIZE))
    b = np.real(np.fft.ifft2(spec / F ** 0.9))
    b = (b - b.min()) / (b.max() - b.min() + 1e-9)
    return _save(np.stack([b, b * 0.9 + 0.05, b * 0.8 + 0.1], -1))


def tex_brick():
    rng = np.random.default_rng(12)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    brick = np.zeros((SIZE, SIZE))
    bh, bw = 40, 90
    for r in range(0, SIZE, bh):
        off = (bw // 2) if (r // bh) % 2 else 0
        for c in range(-bw, SIZE + bw, bw):
            m = ((yy >= r) & (yy < r + bh - 4) & (xx >= c + off + 4) & (xx < c + off + bw - 4))
            brick[m] = 0.72 + 0.06 * rng.random()
    brick[~((yy % bh) < bh - 4)] = 0.92
    return _save(np.stack([brick * 0.85, brick * 0.45, brick * 0.35], -1))


def tex_fabric():
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    w = 0.5 + 0.22 * np.sin(xx * 1.7) * np.sin(yy * 1.7) + 0.12 * np.sin((xx + yy) * 3.1)
    w = w + 0.05 * np.random.default_rng(13).normal(size=(SIZE, SIZE))
    return _save(np.stack([w * 0.9, w * 0.8, w * 0.75], -1))


def tex_grass():
    rng = np.random.default_rng(14)
    g = np.zeros((SIZE, SIZE))
    for _ in range(2600):
        x, y = rng.integers(0, SIZE, 2)
        L = rng.integers(6, 20)
        ang = rng.normal(1.2, 0.5)
        for t in range(L):
            px, py = int(x + t * np.cos(ang)) % SIZE, int(y + t * np.sin(ang)) % SIZE
            g[py, px] = rng.uniform(0.25, 0.75)
    g = 0.25 + 0.5 * g + 0.05 * rng.normal(size=(SIZE, SIZE))
    return _save(np.stack([g * 0.5, g, g * 0.45], -1))


BUILTIN = {"fractal": tex_fractal, "brick": tex_brick, "fabric": tex_fabric,
           "grass": tex_grass}


# ---------------------------------------------------------------- benchmark
def run_one(name, clean, sigma, lams, blocks=(4, 8, 16)):
    seed = zlib.crc32(f"{name}-{sigma}".encode()) % 100000
    rng = np.random.default_rng(seed)
    noisy = np.clip(clean + rng.normal(0, sigma / 255.0, clean.shape), 0, 1)
    res = {}
    for lam in lams:
        if lam == 0:
            out, _ = M.denoise(noisy, sigma=sigma / 255.0, k0=3.0, blocks=blocks)
        else:
            out, _ = M.denoise_structural(noisy, sigma=sigma / 255.0, k0=3.0,
                                          blocks=blocks, lam=lam)
        out = np.clip(out, 0, 1)
        res[lam] = (M.psnr(out, clean), M.ssim(out, clean))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, nargs="+", default=[15, 25])
    ap.add_argument("--lam", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--kodak", default=None, help="directory containing kodim01.png ...")
    ap.add_argument("--pattern", action="store_true",
                    help="also test a periodic fixed-pattern noise")
    args = ap.parse_args()

    images = {k: f() for k, f in BUILTIN.items()}
    if args.kodak:
        import os
        for f in sorted(os.listdir(args.kodak)):
            if f.lower().startswith("kodim") and f.lower().endswith(".png"):
                im = Image.open(os.path.join(args.kodak, f)).convert("RGB")
                im = im.crop((0, 0, min(SIZE, im.width), min(SIZE, im.height)))
                images[f[:-4]] = np.asarray(im, np.float32) / 255.0

    for sigma in args.sigma:
        print(f"\n=== sigma = {sigma} ===")
        hdr = "image".ljust(10) + "".join(f"  lam={l:<4}" for l in args.lam)
        print(hdr)
        acc = {l: [] for l in args.lam}
        for name, clean in images.items():
            res = run_one(name, clean, sigma, args.lam)
            line = name.ljust(10)
            for l in args.lam:
                p, s = res[l]
                acc[l].append(p)
                line += f"  {p:6.2f}/ {s:.3f}"
            print(line)
        line = "mean".ljust(10)
        for l in args.lam:
            line += f"  {np.mean(acc[l]):6.2f}/"
        print(line)
        if 0.0 in acc:
            for l in args.lam:
                if l == 0:
                    continue
                d = np.mean(acc[l]) - np.mean(acc[0])
                print(f"  gain at lam={l}: {d:+.2f} dB")

    if args.pattern:
        print("\n=== periodic fixed pattern (+ AWGN sigma=10) ===")
        print("image".ljust(10) + "  base    with pattern removal")
        for name, clean in images.items():
            h, w = clean.shape[:2]
            yy, xx = np.mgrid[0:h, 0:w]
            pat = 0.025 * np.sin(2 * np.pi * xx / 8) + 0.015 * np.cos(2 * np.pi * yy / 8)
            rng = np.random.default_rng(zlib.crc32(f"{name}-pat".encode()) % 100000)
            noisy = np.clip(clean + rng.normal(0, 10 / 255.0, clean.shape) + pat[..., None], 0, 1)
            base, _ = M.denoise_structural(noisy, sigma=10 / 255.0, k0=3.0, blocks=(8,), lam=0.5)
            auto, info = M.denoise_auto(noisy, sigma=10 / 255.0, k0=3.0, blocks=(8,), lam=0.5)
            pb = M.psnr(np.clip(base, 0, 1), clean)
            pa = M.psnr(np.clip(auto, 0, 1), clean)
            print(f"{name:<10}  {pb:6.2f}  {pa:6.2f}  ({pa-pb:+.2f})")


if __name__ == "__main__":
    main()
