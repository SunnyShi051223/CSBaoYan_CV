"""
激活函数（Activation Functions）
=================================

常用激活函数及其导数的 NumPy 实现。

公式汇总：
    ReLU(x)       = max(0, x)
    Sigmoid(x)    = 1 / (1 + exp(-x))
    Tanh(x)       = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    Leaky ReLU(x) = max(α·x, x)
    Softmax(z)_i  = exp(z_i) / Σ_j exp(z_j)   （数值稳定版本：减去 max）
    ELU(x)        = x  if x > 0 else α·(exp(x) - 1)
    GELU(x)       ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
"""

import numpy as np


# ---------------------------------------------------------------------------
# ReLU
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    """ReLU(x) = max(0, x)"""
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """ReLU 导数：1 if x > 0 else 0"""
    return (x > 0).astype(np.float64)


# ---------------------------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """σ(x) = 1 / (1 + exp(-x))

    数值稳定处理：
        x >= 0 → 1 / (1 + exp(-x))
        x <  0 → exp(x) / (1 + exp(x))
    """
    out = np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )
    return out


def sigmoid_grad(x: np.ndarray) -> np.ndarray:
    """σ'(x) = σ(x) · (1 - σ(x))"""
    s = sigmoid(x)
    return s * (1.0 - s)


# ---------------------------------------------------------------------------
# Tanh
# ---------------------------------------------------------------------------

def tanh(x: np.ndarray) -> np.ndarray:
    """tanh(x) — 直接调用 np.tanh（内部已数值稳定）"""
    return np.tanh(x)


def tanh_grad(x: np.ndarray) -> np.ndarray:
    """tanh'(x) = 1 - tanh²(x)"""
    return 1.0 - np.tanh(x) ** 2


# ---------------------------------------------------------------------------
# Leaky ReLU
# ---------------------------------------------------------------------------

def leaky_relu(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU(x) = max(α·x, x)"""
    return np.where(x > 0, x, alpha * x)


def leaky_relu_grad(x: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Leaky ReLU'(x) = 1 if x > 0 else α"""
    return np.where(x > 0, 1.0, alpha)


# ---------------------------------------------------------------------------
# Softmax
# ---------------------------------------------------------------------------

def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax(z)_i = exp(z_i) / Σ_j exp(z_j)

    数值稳定：先减去沿 axis 的最大值（log-sum-exp 技巧）。

    Parameters
    ----------
    z    : ndarray  logits（任意形状）
    axis : int      沿哪个轴做 softmax，默认 -1（最后一维）

    Returns
    -------
    ndarray  与 z 形状相同，每行之和为 1
    """
    z_shifted = z - z.max(axis=axis, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / exp_z.sum(axis=axis, keepdims=True)


# ---------------------------------------------------------------------------
# ELU
# ---------------------------------------------------------------------------

def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ELU(x) = x  if x > 0  else  α·(exp(x) - 1)"""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1.0))


def elu_grad(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ELU'(x) = 1  if x > 0  else  α·exp(x)"""
    return np.where(x > 0, 1.0, alpha * np.exp(x))


# ---------------------------------------------------------------------------
# GELU（近似版本）
# ---------------------------------------------------------------------------

def gelu(x: np.ndarray) -> np.ndarray:
    """GELU(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))

    Hendrycks & Gimpel (2016) 的近似公式，GPT/BERT 中广泛使用。
    """
    c = np.sqrt(2.0 / np.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x ** 3)))


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    x = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
    print("输入 x：", x)
    print()

    funcs = [
        ("ReLU",       relu,       relu_grad),
        ("Sigmoid",    sigmoid,    sigmoid_grad),
        ("Tanh",       tanh,       tanh_grad),
        ("Leaky ReLU", leaky_relu, leaky_relu_grad),
        ("ELU",        elu,        elu_grad),
        ("GELU",       gelu,       None),
    ]

    for name, fn, grad_fn in funcs:
        out = fn(x)
        grad = grad_fn(x) if grad_fn else "N/A"
        print(f"{name:12s}  f(x)={np.round(out, 4)}  f'(x)={np.round(grad, 4) if not isinstance(grad, str) else grad}")

    print()
    print("Softmax 示例：")
    logits = np.array([[2.0, 1.0, 0.1], [-1.0, 3.0, 0.5]])
    probs = softmax(logits, axis=-1)
    print("logits：\n", logits)
    print("softmax：\n", np.round(probs, 4))
    print("每行之和：", probs.sum(axis=-1))
