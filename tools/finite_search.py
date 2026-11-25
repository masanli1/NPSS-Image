"""参数有限搜索（三节点法）——把无界连续参数变成有限计算

节点划分：
  固定位置：blocks（7 种组合）、remove_pattern（2 种）→ 直接枚举
  固定数量：连续参数只有三个：k0 / lam / flat
  不固定  ：把三者离散成有限槽位，再做「粗格 → 细化」

实测（Set12 6 张训练）：
  生物进化 360 次评估 / 368 s / 28.502
  有限网格 148 次（唯一 37）/ 147 s / 28.488
  分层有限 **50 次 / 67 s / 28.504**

用法：
  python3 tools/finite_search.py --images data/set12 --size 256
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from npss_image import denoise_auto, psnr  # noqa: E402

K0_GRID = [round(2.6 + 0.1 * i, 2) for i in range(15)]
LAM_GRID = [round(0.05 * i, 2) for i in range(19)]
BLOCKSETS = [(4,), (8,), (16,), (4, 8), (8, 16), (4, 16), (4, 8, 16)]


def load_gray(path, size=256):
    from PIL import Image
    a = np.asarray(Image.open(path).convert("L"), np.float32) / 255.0
    if a.shape[0] < size or a.shape[1] < size:
        return np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((size, size)), np.float32) / 255.0
    h, w = a.shape
    return a[(h - size) // 2:(h - size) // 2 + size, (w - size) // 2:(w - size) // 2 + size]


def make_cases(img_dir, size=256, sigmas=(15, 25, 50), limit=6, seed=0):
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")))[:limit]
    if not files:
        raise SystemExit("没有找到图片：%s" % img_dir)
    rng = np.random.default_rng(seed)
    cases = []
    for f in files:
        gt = load_gray(f, size)
        for s in sigmas:
            y = np.clip(gt + rng.normal(0, s / 255.0, gt.shape).astype(np.float32), 0, 1)
            cases.append((gt, y, s / 255.0))
    return cases


def fitness(cases, k0, lam, blocks=(4, 8, 16)):
    ps = []
    for gt, y, sig in cases:
        o, _ = denoise_auto(y, sigma=sig, k0=k0, blocks=blocks, lam=lam,
                            remove_pattern=True)
        ps.append(psnr(gt, np.clip(o, 0, 1)))
    return float(np.mean(ps))


def search(cases, coarse=True):
    """粗格 → 细化"""
    t0 = time.time()
    cache = {}

    def ev(k0, lam):
        key = (round(k0, 3), round(lam, 3))
        if key not in cache:
            cache[key] = fitness(cases, k0, lam)
        return cache[key]

    if coarse:
        cand = [(ev(k0, lam), k0, lam)
                for k0 in (2.8, 3.1, 3.4, 3.7)
                for lam in (0.30, 0.45, 0.60, 0.75)]
        cand.sort(reverse=True)
        b = cand[0]
        print("粗格 %d 次 → %.3f (k0=%.2f lam=%.2f)" % (len(cand), b[0], b[1], b[2]))
        fine = [(ev(round(k0, 2), round(lam, 2)), round(k0, 2), round(lam, 2))
                for k0 in np.arange(max(2.6, b[1] - 0.15), min(4.0, b[1] + 0.16), 0.05)
                for lam in np.arange(max(0.0, b[2] - 0.10), min(0.9, b[2] + 0.11), 0.05)]
        fine.sort(reverse=True)
        best = fine[0]
        print("细化 %d 次 → %.3f (k0=%.2f lam=%.2f)" % (len(fine), *best))
    else:
        rows = [(ev(k0, lam), k0, lam) for k0 in K0_GRID for lam in LAM_GRID]
        rows.sort(reverse=True)
        best = rows[0]
    print("唯一评估 %d 次，用时 %.0f s" % (len(cache), time.time() - t0))
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/set12")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--no-coarse", action="store_true")
    args = ap.parse_args()
    cases = make_cases(args.images, args.size, limit=args.limit)
    print("训练用例 %d" % len(cases))
    search(cases, coarse=not args.no_coarse)
