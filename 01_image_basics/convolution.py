"""
二维卷积 / 互相关（2-D Convolution / Cross-correlation）
=======================================================

数学公式（卷积）：
    (I * K)[i, j] = Σ_m Σ_n  I[i+m, j+n] · K[m, n]
                    （严格卷积：K 翻转 180°）

数学公式（互相关，深度学习框架实际使用）：
    (I ⊛ K)[i, j] = Σ_m Σ_n  I[i+m, j+n] · K[m, n]
                    （核不翻转）

输出尺寸（无填充，步长为 1）：
    H_out = H - k_h + 1
    W_out = W - k_w + 1

输出尺寸（same 填充，步长为 s）：
    H_out = ceil(H / s)
    W_out = ceil(W / s)
    padding = max((H_out-1)*s + k_h - H, 0)  （分配给上下两侧）
"""

import numpy as np


def pad2d(image: np.ndarray, pad_h: int, pad_w: int, value: float = 0.0) -> np.ndarray:
    """对二维图像进行零填充（或常数填充）。

    Parameters
    ----------
    image : ndarray, shape (H, W)
    pad_h : int  上下各填充的行数
    pad_w : int  左右各填充的列数
    value : float  填充值，默认 0

    Returns
    -------
    ndarray, shape (H + 2*pad_h, W + 2*pad_w)
    """
    return np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), constant_values=value)


def conv2d(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
    padding: str = "valid",
) -> np.ndarray:
    """二维卷积（严格卷积：核先翻转 180° 再做互相关）。

    Parameters
    ----------
    image   : ndarray, shape (H, W)  — 输入灰度图像
    kernel  : ndarray, shape (kH, kW) — 卷积核
    stride  : int  — 步长，默认 1
    padding : str  — 'valid'（无填充）或 'same'（输出与输入尺寸相同，步长 1 时）

    Returns
    -------
    ndarray  卷积结果

    示例（均值模糊）
    ---------------
    >>> img = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=float)
    >>> k = np.ones((3,3)) / 9
    >>> out = conv2d(img, k, padding='same')
    >>> out.shape
    (3, 3)
    """
    # 严格卷积 = 互相关 + 翻转核
    flipped_kernel = kernel[::-1, ::-1]
    return cross_correlate2d(image, flipped_kernel, stride=stride, padding=padding)


def cross_correlate2d(
    image: np.ndarray,
    kernel: np.ndarray,
    stride: int = 1,
    padding: str = "valid",
) -> np.ndarray:
    """二维互相关（深度学习框架中习惯称为"卷积"）。

    Parameters
    ----------
    image   : ndarray, shape (H, W)
    kernel  : ndarray, shape (kH, kW)
    stride  : int
    padding : 'valid' | 'same'

    Returns
    -------
    ndarray
    """
    H, W = image.shape
    kH, kW = kernel.shape

    if padding == "same":
        # 计算使输出尺寸 = ceil(H/s) 所需的填充
        pad_h = max((kH - 1) // 2, 0)
        pad_w = max((kW - 1) // 2, 0)
        image = pad2d(image, pad_h, pad_w)
        H, W = image.shape

    # 输出尺寸
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1

    output = np.zeros((H_out, W_out), dtype=np.float64)
    for i in range(H_out):
        for j in range(W_out):
            patch = image[i * stride : i * stride + kH, j * stride : j * stride + kW]
            output[i, j] = np.sum(patch * kernel)
    return output


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)
    img = np.arange(25, dtype=float).reshape(5, 5)
    print("输入图像：")
    print(img)

    # 边缘检测水平核
    kernel_h = np.array([[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]], dtype=float)

    out_valid = cross_correlate2d(img, kernel_h, padding="valid")
    out_same  = cross_correlate2d(img, kernel_h, padding="same")

    print("\n互相关（valid）：")
    print(out_valid)
    print("\n互相关（same）：")
    print(out_same)

    # 验证：卷积 vs 互相关
    out_conv = conv2d(img, kernel_h, padding="same")
    print("\n卷积（same，核翻转后互相关）：")
    print(out_conv)
