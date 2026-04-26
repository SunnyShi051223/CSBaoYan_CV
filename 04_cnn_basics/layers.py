"""
CNN 核心层前向传播（CNN Layers — Forward Pass）
================================================

实现（仅含前向传播，用于学习理解，不含反向传播）：
    1. Conv2d        — 二维卷积层
    2. MaxPool2d     — 最大池化层
    3. AvgPool2d     — 平均池化层
    4. BatchNorm2d   — 对卷积层特征图做批归一化
    5. Linear        — 全连接层（矩阵乘法 + 偏置）

所有实现均以 NumPy 完成，批量维度 N 在第 0 轴。

张量形状约定（NCHW）：
    (N, C, H, W) — 批大小、通道数、高度、宽度
"""

import numpy as np


# ---------------------------------------------------------------------------
# 1. 二维卷积层（Conv2d）
# ---------------------------------------------------------------------------

class Conv2d:
    """二维卷积层（前向传播）。

    公式：
        Z[n, c_out, h, w] = b[c_out]
                           + Σ_{c_in} Σ_i Σ_j  W[c_out, c_in, i, j]
                                               · X[n, c_in, h*s+i, w*s+j]

    参数
    ----
    in_channels  : 输入通道数 C_in
    out_channels : 输出通道数 C_out
    kernel_size  : 卷积核大小 k（方形）
    stride       : 步长 s
    padding      : 填充大小 p（零填充）

    输出尺寸：
        H_out = floor((H_in + 2p - k) / s) + 1
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        self.stride = stride
        self.padding = padding
        k = kernel_size
        # He 初始化（适合 ReLU）
        fan_in = in_channels * k * k
        std = np.sqrt(2.0 / fan_in)
        self.weight = np.random.randn(out_channels, in_channels, k, k) * std
        self.bias = np.zeros(out_channels)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播。

        Parameters
        ----------
        x : ndarray (N, C_in, H_in, W_in)

        Returns
        -------
        ndarray (N, C_out, H_out, W_out)
        """
        N, C_in, H_in, W_in = x.shape
        C_out, _, k, _ = self.weight.shape
        p, s = self.padding, self.stride

        # 零填充
        if p > 0:
            x_pad = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)))
        else:
            x_pad = x

        H_out = (H_in + 2 * p - k) // s + 1
        W_out = (W_in + 2 * p - k) // s + 1
        out = np.zeros((N, C_out, H_out, W_out), dtype=np.float64)

        for h in range(H_out):
            for w in range(W_out):
                # 取 patch：(N, C_in, k, k)
                patch = x_pad[:, :, h * s : h * s + k, w * s : w * s + k]
                # W: (C_out, C_in, k, k) → (C_out, C_in*k*k)
                # patch: (N, C_in*k*k)
                out[:, :, h, w] = (
                    patch.reshape(N, -1) @ self.weight.reshape(C_out, -1).T
                ) + self.bias
        return out

    @staticmethod
    def output_size(H_in: int, k: int, s: int = 1, p: int = 0) -> int:
        """计算单个维度的输出大小。"""
        return (H_in + 2 * p - k) // s + 1


# ---------------------------------------------------------------------------
# 2. 最大池化层（MaxPool2d）
# ---------------------------------------------------------------------------

class MaxPool2d:
    """最大池化层。

    公式：
        out[n, c, h, w] = max_{i,j ∈ [0,k)} X[n, c, h*s+i, w*s+j]
    """

    def __init__(self, kernel_size: int, stride: int = None):
        self.k = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Parameters: x (N, C, H, W) → (N, C, H_out, W_out)"""
        N, C, H, W = x.shape
        k, s = self.k, self.stride
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1
        out = np.zeros((N, C, H_out, W_out), dtype=np.float64)
        for h in range(H_out):
            for w in range(W_out):
                patch = x[:, :, h * s : h * s + k, w * s : w * s + k]
                out[:, :, h, w] = patch.reshape(N, C, -1).max(axis=-1)
        return out


# ---------------------------------------------------------------------------
# 3. 平均池化层（AvgPool2d）
# ---------------------------------------------------------------------------

class AvgPool2d:
    """平均池化层。

    公式：
        out[n, c, h, w] = (1/k²) · Σ_{i,j} X[n, c, h*s+i, w*s+j]
    """

    def __init__(self, kernel_size: int, stride: int = None):
        self.k = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Parameters: x (N, C, H, W) → (N, C, H_out, W_out)"""
        N, C, H, W = x.shape
        k, s = self.k, self.stride
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1
        out = np.zeros((N, C, H_out, W_out), dtype=np.float64)
        for h in range(H_out):
            for w in range(W_out):
                patch = x[:, :, h * s : h * s + k, w * s : w * s + k]
                out[:, :, h, w] = patch.reshape(N, C, -1).mean(axis=-1)
        return out


# ---------------------------------------------------------------------------
# 4. 批归一化（BatchNorm2d）
# ---------------------------------------------------------------------------

class BatchNorm2d:
    """对卷积特征图（N, C, H, W）做批归一化（训练模式）。

    公式（沿 N, H, W 维度统计，每个通道独立）：
        μ_c   = mean(X[:, c, :, :])
        σ²_c  = var (X[:, c, :, :])
        X̂[:,c] = (X[:,c] - μ_c) / sqrt(σ²_c + ε)
        Y[:,c] = γ_c · X̂[:,c] + β_c

    参数
    ----
    num_features : 通道数 C
    eps          : 数值稳定项，默认 1e-5
    """

    def __init__(self, num_features: int, eps: float = 1e-5):
        self.gamma = np.ones(num_features, dtype=np.float64)   # 初始化 γ=1
        self.beta  = np.zeros(num_features, dtype=np.float64)  # 初始化 β=0
        self.eps = eps

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Parameters: x (N, C, H, W) → (N, C, H, W)"""
        N, C, H, W = x.shape
        # 沿 (N, H, W) 维度计算均值和方差，形状 (C,)
        mu    = x.mean(axis=(0, 2, 3))          # (C,)
        var   = x.var (axis=(0, 2, 3))          # (C,)

        # 广播：(C,) → (1, C, 1, 1)
        mu4d  = mu  .reshape(1, C, 1, 1)
        var4d = var .reshape(1, C, 1, 1)
        g4d   = self.gamma.reshape(1, C, 1, 1)
        b4d   = self.beta .reshape(1, C, 1, 1)

        x_hat = (x - mu4d) / np.sqrt(var4d + self.eps)
        return g4d * x_hat + b4d


# ---------------------------------------------------------------------------
# 5. 全连接层（Linear）
# ---------------------------------------------------------------------------

class Linear:
    """全连接层（仿射变换）。

    公式：
        Y = X · W^T + b
        X: (N, in_features)
        W: (out_features, in_features)
        b: (out_features,)
        Y: (N, out_features)
    """

    def __init__(self, in_features: int, out_features: int):
        std = np.sqrt(2.0 / in_features)
        self.weight = np.random.randn(out_features, in_features) * std
        self.bias   = np.zeros(out_features)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (N, in_features) → (N, out_features)"""
        return x @ self.weight.T + self.bias


# ---------------------------------------------------------------------------
# 演示：小型前向传播示例
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    # 输入：2 张 3 通道 8×8 图像
    x = np.random.randn(2, 3, 8, 8)

    # 卷积 (3→8 通道, 核 3×3, 步长 1, 填充 1 → 输出 8×8)
    conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
    z_conv = conv.forward(x)
    print(f"Conv2d 输出：{z_conv.shape}")  # (2, 8, 8, 8)

    # 批归一化
    bn = BatchNorm2d(num_features=8)
    z_bn = bn.forward(z_conv)
    print(f"BatchNorm2d 输出：{z_bn.shape}")  # (2, 8, 8, 8)

    # 最大池化 (2×2, stride=2 → 输出 4×4)
    pool = MaxPool2d(kernel_size=2, stride=2)
    z_pool = pool.forward(z_bn)
    print(f"MaxPool2d 输出：{z_pool.shape}")  # (2, 8, 4, 4)

    # 展平
    z_flat = z_pool.reshape(2, -1)
    print(f"展平后：{z_flat.shape}")          # (2, 128)

    # 全连接 (128 → 10)
    fc = Linear(in_features=128, out_features=10)
    logits = fc.forward(z_flat)
    print(f"全连接输出（logits）：{logits.shape}")  # (2, 10)

    print("\n✓ 前向传播完整流程演示成功")
