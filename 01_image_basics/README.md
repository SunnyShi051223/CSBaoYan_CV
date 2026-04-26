# 01 图像基础 — 卷积与空间滤波

## 1. 二维离散卷积

### 公式

$$
(I * K)[i, j] = \sum_{m=-\lfloor h/2 \rfloor}^{\lfloor h/2 \rfloor}
                \sum_{n=-\lfloor w/2 \rfloor}^{\lfloor w/2 \rfloor}
                I[i+m,\, j+n] \cdot K[m, n]
$$

其中：
- $I$ — 输入图像（$H \times W$）
- $K$ — 卷积核（$h \times w$，通常为奇数尺寸）
- `*` — 卷积运算（注意：严格意义上卷积需要翻转核，相关/互相关不翻转）

### 与互相关的区别

| 运算 | 核翻转 | 用途 |
|------|--------|------|
| 卷积 (Convolution) | 是（180°） | 信号处理理论 |
| 互相关 (Cross-correlation) | 否 | 深度学习框架中的实际实现 |

---

## 2. 均值滤波（Mean Filter）

$$
K_{\text{mean}} = \frac{1}{k^2}
\begin{bmatrix}1 & \cdots & 1\\ \vdots & \ddots & \vdots\\ 1 & \cdots & 1\end{bmatrix}_{k \times k}
$$

平均卷积核，具有平滑作用，但会模糊边缘。

---

## 3. 高斯滤波（Gaussian Filter）

$$
G(x, y) = \frac{1}{2\pi\sigma^2}
\exp\!\left(-\frac{x^2 + y^2}{2\sigma^2}\right)
$$

- $\sigma$ — 标准差，控制平滑程度
- 高斯核在空间域与频率域均为高斯形状，具有最优时频局部化特性
- 实际使用时将连续公式离散化并归一化

---

## 4. 中值滤波（Median Filter）

非线性滤波，将窗口内所有像素值排序后取**中位数**：

$$
\text{output}[i,j] = \text{median}\bigl(\{I[i+m, j+n] \mid m,n \in \Omega\}\bigr)
$$

优势：对椒盐噪声（Salt-and-Pepper noise）抑制效果优于线性滤波器。

---

## 代码文件

- [`convolution.py`](convolution.py) — 二维卷积 / 互相关的 NumPy 实现
- [`filters.py`](filters.py) — 均值、高斯、中值滤波器实现
