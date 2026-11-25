"""Set12 基准：NPSS / NPSS-fast / BM3D 三方对比

用法：
  python3 benchmark_set12.py --set12 data/set12            # 12 张 × σ=15/25/50
  python3 benchmark_set12.py --set12 data/set12 --holdout  # 只用后 6 张（留出集）
"""
from __future__ import annotations

import argparse
import glob
import os
import statistics as st

import numpy as np
from PIL import Image

import npss_image as N
from variants.profiles import denoise


def load(p):
    return np.asarray(Image.open(p).convert("L"), np.float32) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set12", default="data/set12")
    ap.add_argument("--holdout", action="store_true", help="只用后 6 张")
    ap.add_argument("--no-bm3d", action="store_true")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.set12, "*.png")))
    if not files:
        raise SystemExit("没有找到图片：%s（见 data/set12/README.md）" % args.set12)
    if args.holdout:
        files = files[6:]

    bm3d = None
    if not args.no_bm3d:
        try:
            import bm3d as bm3dmod
            bm3d = bm3dmod
        except ImportError:
            print("（未安装 bm3d，跳过 BM3D 对比：pip install bm3d）")

    rng = np.random.default_rng(0)
    rows = []
    for f in files:
        gt = load(f)
        for s255 in (15, 25, 50):
            sig = s255 / 255.0
            y = np.clip(gt + rng.normal(0, sig, gt.shape).astype(np.float32), 0, 1)
            fast, _ = denoise(y, sig, "fast")
            std, _ = denoise(y, sig, "standard")
            rec = {"name": f.split("/")[-1][:-4], "sigma": s255,
                   "noisy": N.psnr(gt, y),
                   "fast": N.psnr(gt, np.clip(fast, 0, 1)),
                   "standard": N.psnr(gt, np.clip(std, 0, 1))}
            if bm3d is not None:
                rec["bm3d"] = N.psnr(gt, np.clip(bm3d.bm3d(y.astype(np.float32), sig), 0, 1))
            rows.append(rec)

    print("%-8s %5s %8s %8s %9s %8s" % ("image", "σ", "noisy", "fast", "standard", "BM3D"))
    for r in rows:
        print("%-8s %5d %8.2f %8.2f %9.2f %8s" %
              (r["name"], r["sigma"], r["noisy"], r["fast"], r["standard"],
               ("%.2f" % r["bm3d"]) if "bm3d" in r else "—"))

    print("\n平均（%d 张）" % len(files))
    print("%5s %8s %8s %9s %8s" % ("σ", "noisy", "fast", "standard", "BM3D"))
    for s in (15, 25, 50):
        sel = [r for r in rows if r["sigma"] == s]
        def m(k):
            return st.mean(r[k] for r in sel) if all(k in r for r in sel) else float("nan")
        print("%5d %8.2f %8.2f %9.2f %8.2f" % (s, m("noisy"), m("fast"), m("standard"), m("bm3d")))
    allm = {k: st.mean(r[k] for r in rows) for k in ("noisy", "fast", "standard") if k in rows[0]}
    if "bm3d" in rows[0]:
        allm["bm3d"] = st.mean(r["bm3d"] for r in rows)
        print("%5s %8.2f %8.2f %9.2f %8.2f" % ("平均", allm["noisy"], allm["fast"],
                                               allm["standard"], allm["bm3d"]))
        print("\nfast 比 BM3D 低 %.2f dB；standard 低 %.2f dB" %
              (allm["bm3d"] - allm["fast"], allm["bm3d"] - allm["standard"]))


if __name__ == "__main__":
    main()
