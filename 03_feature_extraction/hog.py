"""
HOG 特征描述子（Histogram of Oriented Gradients）
==================================================

参考：Dalal & Triggs, CVPR 2005

参数（默认值与原始论文一致）：
    cell_size  : 每个 cell 的像素大小，默认 8×8
    block_size : 每个 block 包含的 cell 数（行×列），默认 2×2
    n_bins     : 方向 bin 数，默认 9（每 bin 覆盖 20°，范围 [0°, 180°)）
    clip_val   : L2-Hys 截断值，默认 0.2

特征向量长度计算：
    n_blocks_row = (n_cells_row - block_rows + 1)
    n_blocks_col = (n_cells_col - block_cols + 1)
    descriptor_length = n_blocks_row × n_blocks_col × block_rows × block_cols × n_bins
"""

import numpy as np
import sys
import os
from typing import Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "01_image_basics"))
from convolution import cross_correlate2d  # noqa: E402


# 简单的 1-D 差分核（中心差分）
_DIFF_X = np.array([[-1, 0, 1]], dtype=np.float64)
_DIFF_Y = np.array([[-1], [0], [1]], dtype=np.float64)


def _compute_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算图像的梯度幅值和（无符号）方向。

    使用 [-1, 0, 1] 中心差分核。

    Returns
    -------
    magnitude : ndarray (H, W)  梯度幅值
    direction : ndarray (H, W)  无符号方向，单位°，范围 [0, 180)
    """
    img = image.astype(np.float64)
    Gx = cross_correlate2d(img, _DIFF_X, padding="same")
    Gy = cross_correlate2d(img, _DIFF_Y, padding="same")
    magnitude = np.sqrt(Gx ** 2 + Gy ** 2)
    # 无符号方向：arctan2 结果映射到 [0°, 180°)
    direction = np.degrees(np.arctan2(Gy, Gx)) % 180.0
    return magnitude, direction


def _cell_histogram(
    mag_cell: np.ndarray,
    dir_cell: np.ndarray,
    n_bins: int = 9,
) -> np.ndarray:
    """对单个 cell 构建梯度方向直方图（幅值加权软投票）。

    软投票（线性插值）：
        bin_width = 180 / n_bins
        k = floor(θ / bin_width) % n_bins
        ratio = (θ - k * bin_width) / bin_width
        hist[k]   += magnitude * (1 - ratio)
        hist[k+1] += magnitude * ratio

    Parameters
    ----------
    mag_cell : ndarray (cell_h, cell_w)  幅值
    dir_cell : ndarray (cell_h, cell_w)  无符号方向°

    Returns
    -------
    ndarray (n_bins,)
    """
    bin_width = 180.0 / n_bins
    hist = np.zeros(n_bins, dtype=np.float64)

    for mag, angle in zip(mag_cell.ravel(), dir_cell.ravel()):
        # 确定左侧 bin 及插值比例
        bin_idx = int(angle / bin_width) % n_bins
        ratio = (angle - bin_idx * bin_width) / bin_width
        hist[bin_idx] += mag * (1.0 - ratio)
        hist[(bin_idx + 1) % n_bins] += mag * ratio
    return hist


def _l2_hys_normalize(v: np.ndarray, eps: float = 1e-5, clip: float = 0.2) -> np.ndarray:
    """L2-Hys 归一化：先 L2 归一化 → 截断 → 再 L2 归一化。

    公式：
        v̂  = v / sqrt(||v||² + ε²)
        v̂  = clip(v̂, 0, clip_val)
        v̂  = v̂ / sqrt(||v̂||² + ε²)
    """
    v = v / np.sqrt(np.dot(v, v) + eps ** 2)
    v = np.minimum(v, clip)
    v = v / np.sqrt(np.dot(v, v) + eps ** 2)
    return v


def hog_descriptor(
    image: np.ndarray,
    cell_size: int = 8,
    block_size: int = 2,
    n_bins: int = 9,
    clip_val: float = 0.2,
) -> np.ndarray:
    """计算图像的 HOG 特征描述子。

    Parameters
    ----------
    image      : ndarray (H, W)  灰度图像
    cell_size  : int  cell 边长（像素），通常 8
    block_size : int  block 由 block_size×block_size 个 cell 组成，通常 2
    n_bins     : int  方向直方图 bin 数，通常 9
    clip_val   : float  L2-Hys 截断值，通常 0.2

    Returns
    -------
    ndarray (descriptor_length,)

    特征向量长度：
        n_cells_row = H // cell_size
        n_cells_col = W // cell_size
        n_blocks_row = n_cells_row - block_size + 1
        n_blocks_col = n_cells_col - block_size + 1
        length = n_blocks_row * n_blocks_col * block_size * block_size * n_bins
    """
    H, W = image.shape
    magnitude, direction = _compute_gradients(image)

    n_cells_row = H // cell_size
    n_cells_col = W // cell_size

    # 第一步：计算每个 cell 的方向直方图
    cell_hists = np.zeros((n_cells_row, n_cells_col, n_bins), dtype=np.float64)
    for r in range(n_cells_row):
        for c in range(n_cells_col):
            r0, r1 = r * cell_size, (r + 1) * cell_size
            c0, c1 = c * cell_size, (c + 1) * cell_size
            cell_hists[r, c] = _cell_histogram(
                magnitude[r0:r1, c0:c1], direction[r0:r1, c0:c1], n_bins
            )

    # 第二步：block 归一化并串联成描述子
    descriptor_parts = []
    n_blocks_row = n_cells_row - block_size + 1
    n_blocks_col = n_cells_col - block_size + 1

    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            block = cell_hists[r : r + block_size, c : c + block_size]  # (bs, bs, n_bins)
            block_vec = block.ravel()
            block_vec = _l2_hys_normalize(block_vec, clip=clip_val)
            descriptor_parts.append(block_vec)

    return np.concatenate(descriptor_parts)


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(7)
    # 构造一张 64×128 的合成图像（行人检测标准窗口大小）
    img = np.random.randint(0, 256, (64, 128), dtype=np.uint8).astype(np.float64)

    descriptor = hog_descriptor(img, cell_size=8, block_size=2, n_bins=9)

    expected_cells_r = 64 // 8   # 8
    expected_cells_c = 128 // 8  # 16
    expected_blocks_r = expected_cells_r - 2 + 1  # 7
    expected_blocks_c = expected_cells_c - 2 + 1  # 15
    expected_len = expected_blocks_r * expected_blocks_c * 2 * 2 * 9  # 7*15*4*9 = 3780

    print(f"图像大小：{img.shape}")
    print(f"HOG 描述子长度：{len(descriptor)}  （期望 {expected_len}）")
    print(f"描述子前 20 个值：{np.round(descriptor[:20], 4)}")
    assert len(descriptor) == expected_len, "描述子长度不匹配！"
    print("✓ 长度验证通过")
