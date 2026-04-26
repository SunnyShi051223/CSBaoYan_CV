# 04 CNN 基础构件（CNN Building Blocks）

## 1. 卷积层（Convolutional Layer）

### 前向传播

$$
Z^{(l)}[n, c_{out}, h, w] = b^{(l)}[c_{out}] +
\sum_{c_{in}=0}^{C_{in}-1}
\sum_{i=0}^{k_H-1}
\sum_{j=0}^{k_W-1}
W^{(l)}[c_{out}, c_{in}, i, j] \cdot
X^{(l-1)}[n, c_{in},\, h \cdot s + i,\, w \cdot s + j]
$$

其中 $s$ 为步长（stride）。

### 输出尺寸

$$
H_{out} = \left\lfloor \frac{H_{in} + 2p - k_H}{s} \right\rfloor + 1
$$

---

## 2. 池化层（Pooling Layer）

### 最大池化

$$
\text{MaxPool}[n, c, h, w] = \max_{i \in [0,k_H),\, j \in [0,k_W)}
X[n, c,\, h \cdot s + i,\, w \cdot s + j]
$$

### 平均池化

$$
\text{AvgPool}[n, c, h, w] = \frac{1}{k_H k_W}
\sum_{i=0}^{k_H-1} \sum_{j=0}^{k_W-1}
X[n, c,\, h \cdot s + i,\, w \cdot s + j]
$$

---

## 3. 激活函数（Activation Functions）

| 函数 | 公式 | 导数 |
|------|------|------|
| ReLU | $f(x) = \max(0, x)$ | $f'(x) = \mathbf{1}[x > 0]$ |
| Sigmoid | $\sigma(x) = \dfrac{1}{1+e^{-x}}$ | $\sigma'(x) = \sigma(x)(1-\sigma(x))$ |
| Tanh | $\tanh(x) = \dfrac{e^x - e^{-x}}{e^x + e^{-x}}$ | $\tanh'(x) = 1 - \tanh^2(x)$ |
| Leaky ReLU | $f(x) = \max(\alpha x, x),\ \alpha \ll 1$ | $f'(x) = \mathbf{1}[x>0] + \alpha\,\mathbf{1}[x \leq 0]$ |
| Softmax | $\sigma(\mathbf{z})_i = \dfrac{e^{z_i}}{\sum_j e^{z_j}}$ | — |

---

## 4. 批归一化（Batch Normalization）

$$
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \varepsilon}},
\qquad
y_i = \gamma \hat{x}_i + \beta
$$

- $\mu_B = \dfrac{1}{m}\sum_{i=1}^m x_i$ — mini-batch 均值  
- $\sigma_B^2 = \dfrac{1}{m}\sum_{i=1}^m (x_i - \mu_B)^2$ — mini-batch 方差  
- $\gamma, \beta$ — 可学习的缩放与偏移参数  
- $\varepsilon$ — 数值稳定项，通常取 $10^{-5}$

---

## 代码文件

- [`layers.py`](layers.py) — 卷积层、池化层、批归一化的 NumPy 前向传播实现
- [`activations.py`](activations.py) — 常用激活函数及其导数
