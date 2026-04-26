"""
单元测试 — CV 基础模块
======================
运行方式：
    python tests/test_cv_basics.py
"""

import sys
import os
import numpy as np

# 将各模块目录添加到 path
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "01_image_basics"))
sys.path.insert(0, os.path.join(ROOT, "02_edge_detection"))
sys.path.insert(0, os.path.join(ROOT, "03_feature_extraction"))
sys.path.insert(0, os.path.join(ROOT, "04_cnn_basics"))


def assert_close(a, b, atol=1e-6, msg=""):
    assert np.allclose(a, b, atol=atol), f"{msg}\n期望：{b}\n得到：{a}"


# ---------------------------------------------------------------------------
# 01 卷积
# ---------------------------------------------------------------------------
from convolution import conv2d, cross_correlate2d  # noqa: E402


def test_cross_correlate_identity():
    """单位卷积核（中心为 1 的 delta）应返回原图（same 填充）。"""
    img = np.arange(9, dtype=float).reshape(3, 3)
    delta = np.array([[0, 0, 0],
                      [0, 1, 0],
                      [0, 0, 0]], dtype=float)
    out = cross_correlate2d(img, delta, padding="same")
    assert_close(out, img, msg="单位核互相关应返回原图")
    print("✓ test_cross_correlate_identity")


def test_cross_correlate_valid_size():
    """valid 模式输出尺寸：H_out = H - kH + 1"""
    img = np.ones((7, 9))
    kernel = np.ones((3, 5))
    out = cross_correlate2d(img, kernel, padding="valid")
    assert out.shape == (5, 5), f"期望 (5,5) 得到 {out.shape}"
    print("✓ test_cross_correlate_valid_size")


def test_conv2d_symmetric_kernel():
    """对称核：卷积 == 互相关（因为翻转后不变）。"""
    img = np.random.RandomState(0).randn(5, 5)
    k = np.array([[1, 2, 1],
                  [2, 4, 2],
                  [1, 2, 1]], dtype=float)
    out_cc = cross_correlate2d(img, k, padding="same")
    out_cv = conv2d(img, k, padding="same")
    assert_close(out_cc, out_cv, msg="对称核：卷积 vs 互相关应相等")
    print("✓ test_conv2d_symmetric_kernel")


# ---------------------------------------------------------------------------
# 01 滤波器
# ---------------------------------------------------------------------------
from filters import gaussian_kernel, gaussian_filter, gaussian_filter_separable  # noqa: E402


def test_gaussian_kernel_sum():
    """高斯核归一化后所有元素之和为 1。"""
    for k in (3, 5, 7):
        for sigma in (0.5, 1.0, 2.0):
            g = gaussian_kernel(k, sigma)
            assert_close(g.sum(), 1.0, atol=1e-10,
                         msg=f"高斯核求和 k={k} σ={sigma}")
    print("✓ test_gaussian_kernel_sum")


def test_gaussian_separable_equals_2d():
    """可分离高斯滤波 ≈ 二维高斯滤波（误差 < 1e-10）。"""
    img = np.random.RandomState(1).randn(10, 10)
    out_2d  = gaussian_filter(img, k=5, sigma=1.0)
    out_sep = gaussian_filter_separable(img, k=5, sigma=1.0)
    assert_close(out_2d, out_sep, atol=1e-10,
                 msg="可分离高斯 vs 二维高斯应几乎相等")
    print("✓ test_gaussian_separable_equals_2d")


# ---------------------------------------------------------------------------
# 02 Sobel
# ---------------------------------------------------------------------------
from sobel import sobel_gradients, sobel_magnitude  # noqa: E402


def test_sobel_flat_image():
    """均匀图像（所有像素相同）的梯度幅值应全为 0。"""
    img = np.full((5, 5), 100.0)
    mag = sobel_magnitude(img)
    # 边界处因 same 填充不为 0，仅检查内部
    assert_close(mag[1:-1, 1:-1], 0.0,
                 msg="均匀图像内部梯度应为 0")
    print("✓ test_sobel_flat_image")


def test_sobel_vertical_edge():
    """垂直边缘：Gx 应显著，Gy 应接近 0（内部）。"""
    img = np.zeros((7, 7))
    img[:, 3:] = 255.0
    Gx, Gy = sobel_gradients(img)
    # 在竖直边缘中部行，Gx 幅值较大
    assert Gx[3, 3] > 100, f"竖直边缘 Gx 应较大，得到 {Gx[3,3]}"
    print("✓ test_sobel_vertical_edge")


# ---------------------------------------------------------------------------
# 04 激活函数
# ---------------------------------------------------------------------------
from activations import relu, sigmoid, tanh, softmax, relu_grad, sigmoid_grad  # noqa: E402


def test_relu():
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    assert_close(relu(x), expected, msg="ReLU")
    print("✓ test_relu")


def test_sigmoid_range():
    """Sigmoid 输出应在 (0, 1)。"""
    x = np.linspace(-10, 10, 100)
    s = sigmoid(x)
    assert s.min() > 0 and s.max() < 1, "Sigmoid 应在 (0,1)"
    print("✓ test_sigmoid_range")


def test_sigmoid_at_zero():
    assert_close(sigmoid(np.array([0.0])), np.array([0.5]),
                 msg="σ(0) 应为 0.5")
    print("✓ test_sigmoid_at_zero")


def test_softmax_sums_to_one():
    z = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    probs = softmax(z, axis=-1)
    assert_close(probs.sum(axis=-1), np.ones(2),
                 msg="softmax 每行应归一化为 1")
    print("✓ test_softmax_sums_to_one")


# ---------------------------------------------------------------------------
# 04 CNN 层
# ---------------------------------------------------------------------------
from layers import Conv2d, MaxPool2d, AvgPool2d, BatchNorm2d, Linear  # noqa: E402


def test_conv2d_output_shape():
    """验证 Conv2d 前向传播输出形状。"""
    x = np.random.randn(2, 3, 8, 8)
    conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
    out = conv.forward(x)
    assert out.shape == (2, 16, 8, 8), f"期望 (2,16,8,8) 得到 {out.shape}"
    print("✓ test_conv2d_output_shape")


def test_maxpool_output_shape():
    x = np.random.randn(2, 8, 8, 8)
    pool = MaxPool2d(kernel_size=2, stride=2)
    out = pool.forward(x)
    assert out.shape == (2, 8, 4, 4), f"期望 (2,8,4,4) 得到 {out.shape}"
    print("✓ test_maxpool_output_shape")


def test_batchnorm_mean_std():
    """BatchNorm2d 输出（γ=1,β=0）内部均值≈0，方差≈1（沿 N,H,W）。"""
    np.random.seed(42)
    x = np.random.randn(16, 4, 8, 8) * 5 + 10  # 偏移均值和方差
    bn = BatchNorm2d(num_features=4)
    out = bn.forward(x)
    mu  = out.mean(axis=(0, 2, 3))
    var = out.var(axis=(0, 2, 3))
    assert_close(mu, np.zeros(4), atol=1e-6, msg="BN 后均值应为 0")
    assert_close(var, np.ones(4), atol=1e-4, msg="BN 后方差应为 1")
    print("✓ test_batchnorm_mean_std")


def test_linear_output_shape():
    x = np.random.randn(4, 32)
    fc = Linear(in_features=32, out_features=10)
    out = fc.forward(x)
    assert out.shape == (4, 10), f"期望 (4,10) 得到 {out.shape}"
    print("✓ test_linear_output_shape")


# ---------------------------------------------------------------------------
# HOG
# ---------------------------------------------------------------------------
from hog import hog_descriptor  # noqa: E402


def test_hog_descriptor_length():
    """HOG 描述子长度应与理论值一致。"""
    img = np.random.RandomState(0).randint(0, 256, (64, 128)).astype(float)
    desc = hog_descriptor(img, cell_size=8, block_size=2, n_bins=9)
    expected = 7 * 15 * 4 * 9  # 3780
    assert len(desc) == expected, f"期望 {expected} 得到 {len(desc)}"
    print("✓ test_hog_descriptor_length")


def test_hog_normalized():
    """HOG 描述子每个 block 向量的 L2 范数应接近 1（±ε）。"""
    img = np.random.RandomState(1).randint(0, 256, (16, 16)).astype(float)
    desc = hog_descriptor(img, cell_size=8, block_size=2, n_bins=9)
    # 每个 block 有 2*2*9 = 36 个值
    block_size = 2 * 2 * 9
    n_blocks = len(desc) // block_size
    for i in range(n_blocks):
        v = desc[i * block_size : (i + 1) * block_size]
        norm = np.linalg.norm(v)
        assert norm <= 1.0 + 1e-6, f"Block {i} 范数 {norm:.4f} 超过 1"
    print("✓ test_hog_normalized")


# ---------------------------------------------------------------------------
# 运行所有测试
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        test_cross_correlate_identity,
        test_cross_correlate_valid_size,
        test_conv2d_symmetric_kernel,
        test_gaussian_kernel_sum,
        test_gaussian_separable_equals_2d,
        test_sobel_flat_image,
        test_sobel_vertical_edge,
        test_relu,
        test_sigmoid_range,
        test_sigmoid_at_zero,
        test_softmax_sums_to_one,
        test_conv2d_output_shape,
        test_maxpool_output_shape,
        test_batchnorm_mean_std,
        test_linear_output_shape,
        test_hog_descriptor_length,
        test_hog_normalized,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"✗ {t.__name__}: {e}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"通过：{passed} / {len(tests)}   失败：{failed}")
    if failed:
        sys.exit(1)
