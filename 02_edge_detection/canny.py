"""
Canny 边缘检测（Canny Edge Detection）
=======================================

四步流程：
    1. 高斯平滑：       I_s = G_σ * I
    2. Sobel 梯度：     (Gx, Gy) → 幅值 M、方向 θ
    3. 非极大值抑制：   沿梯度方向仅保留局部最大值（细化边缘）
    4. 双阈值 + 边缘连通：
           强边缘  ← M ≥ T_high
           弱边缘  ← T_low ≤ M < T_high（若连通到强边缘则保留）
           噪声    ← M < T_low

参数建议：
    T_high ≈ 2 × T_low   （经验比例 2:1 到 3:1）
"""

import numpy as np
from collections import deque
import sys
import os
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "01_image_basics"))
from filters import gaussian_filter  # noqa: E402
from convolution import cross_correlate2d  # noqa: E402

from sobel import SOBEL_X, SOBEL_Y  # noqa: E402


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _angle_to_direction(angle_deg: np.ndarray) -> np.ndarray:
    """将梯度角度（度）量化为 4 个离散方向之一：0/45/90/135°。

    梯度方向 θ 用于确定"沿梯度方向"的两个相邻像素。
    量化规则：将 [0°, 180°) 分为 4 等份，每份 45° 宽：
        [0°,  22.5°)  和 [157.5°, 180°) → 0°   (水平)
        [22.5°, 67.5°)                   → 45°  (右上/左下)
        [67.5°, 112.5°)                  → 90°  (垂直)
        [112.5°, 157.5°)                 → 135° (左上/右下)
    """
    # 将角度映射到 [0°, 180°)
    angle_deg = angle_deg % 180.0
    direction = np.zeros_like(angle_deg, dtype=np.int32)
    direction[(angle_deg >= 22.5)  & (angle_deg < 67.5)]  = 45
    direction[(angle_deg >= 67.5)  & (angle_deg < 112.5)] = 90
    direction[(angle_deg >= 112.5) & (angle_deg < 157.5)] = 135
    return direction


def _non_maximum_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """非极大值抑制（Non-Maximum Suppression, NMS）。

    沿梯度方向比较当前像素与其两侧相邻像素的幅值：
        若当前像素幅值 ≥ 两侧像素幅值，则保留；否则置零。

    四个方向对应的相邻偏移：
        0°  → (0, ±1)   水平方向
        45° → (±1, ∓1)  右上 / 左下
        90° → (±1, 0)   垂直方向
        135°→ (±1, ±1)  左上 / 右下

    Parameters
    ----------
    magnitude : ndarray (H, W)  梯度幅值
    direction : ndarray (H, W)  量化方向（0/45/90/135）

    Returns
    -------
    ndarray (H, W)  细化后的幅值图
    """
    H, W = magnitude.shape
    suppressed = np.zeros_like(magnitude)

    # 方向 → 相邻偏移量 (dr1, dc1), (dr2, dc2)
    offsets = {
        0:   ((0,  1), (0, -1)),
        45:  ((-1, 1), (1, -1)),
        90:  ((-1, 0), (1,  0)),
        135: ((-1,-1), (1,  1)),
    }

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            d = direction[i, j]
            (dr1, dc1), (dr2, dc2) = offsets[d]
            n1 = magnitude[i + dr1, j + dc1]
            n2 = magnitude[i + dr2, j + dc2]
            if magnitude[i, j] >= n1 and magnitude[i, j] >= n2:
                suppressed[i, j] = magnitude[i, j]
    return suppressed


def _double_threshold(
    image: np.ndarray, low: float, high: float
) -> Tuple[np.ndarray, np.ndarray]:
    """双阈值分割，返回强边缘掩膜和弱边缘掩膜。

    Returns
    -------
    strong : ndarray bool  M ≥ high
    weak   : ndarray bool  low ≤ M < high
    """
    strong = image >= high
    weak   = (image >= low) & ~strong
    return strong, weak


def _edge_tracking_bfs(
    strong: np.ndarray, weak: np.ndarray
) -> np.ndarray:
    """BFS 边缘追踪：弱边缘若与强边缘 8-邻域相连则升级为强边缘，否则丢弃。

    Parameters
    ----------
    strong : ndarray bool (H, W)
    weak   : ndarray bool (H, W)

    Returns
    -------
    ndarray bool (H, W)  最终边缘图
    """
    H, W = strong.shape
    result = strong.copy()
    queue = deque(zip(*np.where(strong)))  # 初始化队列为所有强边缘像素

    while queue:
        r, c = queue.popleft()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and weak[nr, nc] and not result[nr, nc]:
                    result[nr, nc] = True
                    queue.append((nr, nc))
    return result


# ---------------------------------------------------------------------------
# 主函数：Canny 边缘检测
# ---------------------------------------------------------------------------

def canny(
    image: np.ndarray,
    sigma: float = 1.0,
    k: int = 5,
    low_threshold: float = 20.0,
    high_threshold: float = 50.0,
) -> np.ndarray:
    """Canny 边缘检测完整实现。

    Parameters
    ----------
    image          : ndarray (H, W)  灰度图像 [0, 255]
    sigma          : 高斯平滑标准差
    k              : 高斯核大小
    low_threshold  : 双阈值下限
    high_threshold : 双阈值上限（建议 2~3 倍 low_threshold）

    Returns
    -------
    ndarray bool (H, W)  边缘掩膜（True 为边缘）
    """
    # 步骤 1：高斯平滑
    smoothed = gaussian_filter(image.astype(np.float64), k=k, sigma=sigma)

    # 步骤 2：Sobel 梯度
    Gx = cross_correlate2d(smoothed, SOBEL_X, padding="same")
    Gy = cross_correlate2d(smoothed, SOBEL_Y, padding="same")
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    angle_deg = np.degrees(np.arctan2(Gy, Gx))

    # 步骤 3：非极大值抑制
    direction = _angle_to_direction(angle_deg)
    suppressed = _non_maximum_suppression(magnitude, direction)

    # 步骤 4：双阈值 + 边缘追踪
    strong, weak = _double_threshold(suppressed, low_threshold, high_threshold)
    edges = _edge_tracking_bfs(strong, weak)

    return edges


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(1)
    # 构造一张中央有方形的合成图像
    img = np.zeros((15, 15), dtype=np.float64)
    img[4:11, 4:11] = 200.0          # 白色方块
    img += np.random.randn(15, 15)   # 添加微量噪声

    print("输入图像（含噪）：")
    print(np.round(img).astype(int))

    edges = canny(img, sigma=1.0, k=5, low_threshold=10.0, high_threshold=30.0)
    print("\nCanny 边缘（True = 边缘）：")
    print(edges.astype(int))
