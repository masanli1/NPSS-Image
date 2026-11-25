"""三种运行档：速度 / 质量 / 自适应

| 档位 | 配置 | 256² 用时 | PSNR(σ=25) |
|---|---|---|---|
| fast     | blocks=(8,), 无图案去除 | 0.033 s | 29.06 |
| standard | blocks=(4,8,16), 图案去除 | 0.075 s | 29.19 |
| adaptive | 逐图 SURE 选参 | ~6 s/图 | 视图像而定 |
"""
from __future__ import annotations

import numpy as np

try:
    from npss_image import denoise_auto
except ImportError:                      # 作为包使用时
    from ..npss_image import denoise_auto  # type: ignore

FAST = {"k0": 3.1, "lam": 0.7, "blocks": (8,), "remove_pattern": False}
STANDARD = {"k0": 3.1, "lam": 0.7, "blocks": (4, 8, 16), "remove_pattern": True}

# 自适应档的有限网格（离线搜索得到，5 个候选）
GRID = [(3.1, 0.70), (2.9, 0.50), (3.0, 0.20), (3.2, 0.50), (3.3, 0.85)]


def denoise(img, sigma=None, mode="standard"):
    """返回 (降噪图, 信息 dict)"""
    cfg = FAST if mode == "fast" else STANDARD
    out, info = denoise_auto(img, sigma=sigma, k0=cfg["k0"], blocks=cfg["blocks"],
                             lam=cfg["lam"], remove_pattern=cfg["remove_pattern"])
    info["mode"] = mode
    return out, info


def _sure(y, sigma, k0, lam, eps=1e-3, seed=0):
    """SURE 无参考风险估计（蒙特卡洛散度）"""
    n = y.size
    f0, _ = denoise_auto(y, sigma=sigma, k0=k0, blocks=(4, 8, 16), lam=lam,
                         remove_pattern=True)
    b = np.random.default_rng(seed).standard_normal(y.shape).astype(np.float32)
    f1, _ = denoise_auto(np.clip(y + eps * b, 0, 1), sigma=sigma, k0=k0,
                         blocks=(4, 8, 16), lam=lam, remove_pattern=True)
    div = float(np.sum(b * (f1 - f0))) / eps
    return float(np.mean((f0 - y) ** 2)) - sigma ** 2 + 2 * sigma ** 2 * div / n


def denoise_adaptive(img, sigma, grid=None, verbose=False):
    """逐图有限搜索：用 SURE 选 k0/lam，无需真值"""
    import numpy as np
    y = np.asarray(img, np.float32)
    grid = grid or GRID
    scores = [(float(_sure(y, sigma, k0, lam)), k0, lam) for k0, lam in grid]
    best = min(scores)
    out, info = denoise_auto(y, sigma=sigma, k0=best[1], blocks=(4, 8, 16),
                             lam=best[2], remove_pattern=True)
    info.update({"mode": "adaptive", "chosen": {"k0": best[1], "lam": best[2]},
                 "evals": len(grid) * 2,
                 "log": [{"k0": k0, "lam": lam, "sure": s} for s, k0, lam in scores]})
    if verbose:
        for s, k0, lam in scores:
            print("  k0=%.1f lam=%.2f  SURE=%.6e" % (k0, lam, s))
    return out, info
