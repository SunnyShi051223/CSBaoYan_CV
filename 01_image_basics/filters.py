"""
空间域图像滤波器（Spatial Image Filters）
=========================================

均值滤波（Mean Filter）
------------------------
    K_mean = (1/k²) · [1]_{k×k}

高斯滤波（Gaussian Filter）
----------------------------
    G(x,y) = 1/(2πσ²) · exp(-(x²+y²) / (2σ²))

    离散化步骤：
    1. 以核中心为原点建立整数坐标网格
    2. 代入高斯公式计算每点权重
    3. 归一化（除以权重之和），确保总增益为 1

中值滤波（Median Filter）
--------------------------
    output[i,j] = median({ I[i+m, j+n] | (m,n) ∈ Ω })

    非线性；对椒盐噪声效果最好。
"""

import numpy as np
from convolution import cross_correlate2d, pad2d


# ---------------------------------------------------------------------------
# 1. 均值滤波
# ---------------------------------------------------------------------------

def mean_kernel(k: int = 3) -> np.ndarray:
    """生成 k×k 均值卷积核。

    >>> mean_kernel(3).sum()
    1.0
    """
    return np.ones((k, k), dtype=np.float64) / (k * k)


def mean_filter(image: np.ndarray, k: int = 3) -> np.ndarray:
    """均值滤波。

    Parameters
    ----------
    image : ndarray (H, W)
    k     : 核大小（奇数）

    Returns
    -------
    ndarray (H, W)
    """
    return cross_correlate2d(image, mean_kernel(k), padding="same")


# ---------------------------------------------------------------------------
# 2. 高斯滤波
# ---------------------------------------------------------------------------

def gaussian_kernel(k: int = 5, sigma: float = 1.0) -> np.ndarray:
    """生成 k×k 高斯卷积核。

    公式：
        G(x,y) = exp(-(x²+y²) / (2σ²))   然后归一化

    Parameters
    ----------
    k     : 核大小（奇数）
    sigma : 标准差

    Returns
    -------
    ndarray (k, k)，所有元素之和为 1

    >>> gaussian_kernel(3, 1.0).sum()  # 应约为 1.0
    np.float64(1.0)
    """
    half = k // 2
    xs = np.arange(-half, half + 1, dtype=np.float64)
    # 建立二维坐标网格
    xx, yy = np.meshgrid(xs, xs)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return kernel / kernel.sum()


def gaussian_filter(image: np.ndarray, k: int = 5, sigma: float = 1.0) -> np.ndarray:
    """高斯滤波（使用可分离性可拆成两次 1-D 滤波，此处直接 2-D 实现）。

    可分离性公式：
        G(x,y) = G(x) · G(y)
        → 先按行做 1-D 高斯，再按列做 1-D 高斯（减少计算量 O(k²)→O(2k)）

    Parameters
    ----------
    image : ndarray (H, W)
    k     : 核大小
    sigma : 标准差

    Returns
    -------
    ndarray (H, W)
    """
    kernel = gaussian_kernel(k, sigma)
    return cross_correlate2d(image, kernel, padding="same")


def gaussian_filter_separable(
    image: np.ndarray, k: int = 5, sigma: float = 1.0
) -> np.ndarray:
    """利用高斯核的可分离性，先行后列做两次 1-D 卷积，复杂度更低。

    1-D 高斯核：g(x) = exp(-x²/(2σ²))，归一化后用于行/列方向。
    """
    half = k // 2
    xs = np.arange(-half, half + 1, dtype=np.float64)
    g1d = np.exp(-(xs ** 2) / (2.0 * sigma ** 2))
    g1d /= g1d.sum()

    # 行方向（横向 1-D 滤波）
    row_kernel = g1d.reshape(1, k)
    tmp = cross_correlate2d(image, row_kernel, padding="same")

    # 列方向（纵向 1-D 滤波）
    col_kernel = g1d.reshape(k, 1)
    return cross_correlate2d(tmp, col_kernel, padding="same")


# ---------------------------------------------------------------------------
# 3. 中值滤波
# ---------------------------------------------------------------------------

def median_filter(image: np.ndarray, k: int = 3) -> np.ndarray:
    """中值滤波（非线性）。

    对每个 k×k 窗口内的像素值取中位数，可有效去除椒盐噪声。

    Parameters
    ----------
    image : ndarray (H, W)
    k     : 窗口大小（奇数）

    Returns
    -------
    ndarray (H, W)
    """
    H, W = image.shape
    half = k // 2
    padded = pad2d(image, half, half)
    output = np.zeros_like(image, dtype=np.float64)
    for i in range(H):
        for j in range(W):
            window = padded[i : i + k, j : j + k]
            output[i, j] = np.median(window)
    return output


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(42)
    # 创建一张含高斯噪声的 7×7 "图像"
    img = np.arange(49, dtype=float).reshape(7, 7)
    noise = np.random.randn(7, 7) * 5
    noisy = img + noise

    print("原始图像：")
    print(np.round(img, 1))

    print("\n含噪图像：")
    print(np.round(noisy, 1))

    print("\n均值滤波（3×3）：")
    print(np.round(mean_filter(noisy, k=3), 1))

    print("\n高斯滤波（5×5, σ=1.0）：")
    print(np.round(gaussian_filter(noisy, k=5, sigma=1.0), 1))

    print("\n高斯滤波（可分离版本）：")
    print(np.round(gaussian_filter_separable(noisy, k=5, sigma=1.0), 1))

    # 添加椒盐噪声
    salt_pepper = img.copy()
    idx = np.random.choice(49, 10, replace=False)
    salt_pepper.flat[idx[:5]] = 255
    salt_pepper.flat[idx[5:]] = 0
    print("\n含椒盐噪声的图像：")
    print(salt_pepper)
    print("\n中值滤波（3×3）：")
    print(median_filter(salt_pepper, k=3))
