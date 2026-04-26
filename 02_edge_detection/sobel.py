"""
Sobel 算子（Sobel Operator）
============================

公式：
    Gx = [[-1, 0, 1],        Gy = [[-1, -2, -1],
           [-2, 0, 2],               [ 0,  0,  0],
           [-1, 0, 1]]               [ 1,  2,  1]]

    梯度幅值：|∇I| = sqrt(Gx² + Gy²)
    梯度方向：θ    = arctan(Gy / Gx)   (rad)

Sobel = 平滑（高斯方向）× 差分（垂直方向）可分离分解：
    Gx = [1, 2, 1]^T · [-1, 0, 1]
    Gy = [-1, -2, -1]^T · [1, 0, -1]（或等价写法）
"""

import numpy as np
import sys
import os
from typing import Tuple

# 允许直接 import 上一目录中的 convolution 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "01_image_basics"))
from convolution import cross_correlate2d  # noqa: E402


SOBEL_X = np.array([[-1, 0, 1],
                     [-2, 0, 2],
                     [-1, 0, 1]], dtype=np.float64)

SOBEL_Y = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float64)


def sobel_gradients(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """计算图像的 Sobel x/y 梯度。

    Parameters
    ----------
    image : ndarray (H, W)  灰度图像，像素值 [0, 255]

    Returns
    -------
    Gx, Gy : ndarray (H, W)  x 方向与 y 方向梯度
    """
    Gx = cross_correlate2d(image.astype(np.float64), SOBEL_X, padding="same")
    Gy = cross_correlate2d(image.astype(np.float64), SOBEL_Y, padding="same")
    return Gx, Gy


def sobel_magnitude(image: np.ndarray) -> np.ndarray:
    """计算 Sobel 梯度幅值 |∇I| = sqrt(Gx² + Gy²)。

    Parameters
    ----------
    image : ndarray (H, W)

    Returns
    -------
    ndarray (H, W)  梯度幅值
    """
    Gx, Gy = sobel_gradients(image)
    return np.sqrt(Gx ** 2 + Gy ** 2)


def sobel_direction(image: np.ndarray) -> np.ndarray:
    """计算 Sobel 梯度方向 θ = arctan(Gy / Gx)（弧度）。

    Parameters
    ----------
    image : ndarray (H, W)

    Returns
    -------
    ndarray (H, W)  方向（弧度），范围 (-π, π)
    """
    Gx, Gy = sobel_gradients(image)
    return np.arctan2(Gy, Gx)


def sobel_edge(image: np.ndarray, threshold: float = 50.0) -> np.ndarray:
    """简单阈值化的 Sobel 边缘检测。

    Parameters
    ----------
    image     : ndarray (H, W)
    threshold : 幅值阈值，超过则判定为边缘

    Returns
    -------
    ndarray (H, W)  bool 数组，True 表示边缘
    """
    mag = sobel_magnitude(image)
    return mag >= threshold


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 构造一张中间有竖直边缘的测试图像
    img = np.zeros((7, 7), dtype=np.float64)
    img[:, 3:] = 200.0        # 右半部分为亮区
    img += np.random.RandomState(0).randn(7, 7) * 2  # 少量噪声

    Gx, Gy = sobel_gradients(img)
    mag = sobel_magnitude(img)
    angle = sobel_direction(img)

    print("输入图像（含细微噪声）：")
    print(np.round(img, 1))

    print("\nSobel Gx：")
    print(np.round(Gx, 1))

    print("\nSobel Gy：")
    print(np.round(Gy, 1))

    print("\n梯度幅值：")
    print(np.round(mag, 1))

    print("\n边缘掩膜（阈值=50）：")
    print(sobel_edge(img, threshold=50).astype(int))
