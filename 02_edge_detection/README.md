# 02 边缘检测（Edge Detection）

## 概述

边缘 = 图像亮度发生**显著变化**的位置，对应一阶导数（梯度）的极大值，或二阶导数的**零交叉点**。

---

## 1. 梯度（Gradient）

连续域：

$$
\nabla I = \left(\frac{\partial I}{\partial x},\, \frac{\partial I}{\partial y}\right)
$$

梯度幅值与方向：

$$
|\nabla I| = \sqrt{\left(\frac{\partial I}{\partial x}\right)^2 + \left(\frac{\partial I}{\partial y}\right)^2},
\qquad
\theta = \arctan\!\left(\frac{\partial I / \partial y}{\partial I / \partial x}\right)
$$

离散近似（有限差分）：

$$
\frac{\partial I}{\partial x} \approx I[i, j+1] - I[i, j-1]
$$

---

## 2. Sobel 算子

Sobel 在差分的基础上加入了高斯平滑，对噪声更鲁棒：

$$
G_x = \begin{bmatrix}-1 & 0 & 1\\-2 & 0 & 2\\-1 & 0 & 1\end{bmatrix} * I,
\qquad
G_y = \begin{bmatrix}-1 & -2 & -1\\ 0 & 0 & 0\\ 1 & 2 & 1\end{bmatrix} * I
$$

$$
|\nabla I| = \sqrt{G_x^2 + G_y^2}, \qquad \theta = \arctan(G_y / G_x)
$$

> 可分解为：$G_x = \begin{bmatrix}1\\2\\1\end{bmatrix}\begin{bmatrix}-1 & 0 & 1\end{bmatrix}$

---

## 3. Canny 边缘检测（四步法）

| 步骤 | 操作 | 目的 |
|------|------|------|
| 1 | 高斯平滑 | 抑制噪声 |
| 2 | Sobel 梯度 | 计算幅值与方向 |
| 3 | 非极大值抑制（NMS） | 细化边缘为单像素宽 |
| 4 | 双阈值 + 边缘追踪 | 连接弱边缘，去除噪声响应 |

### 3.1 非极大值抑制

沿梯度方向对幅值进行插值，若当前点不是局部极大值则置零：

$$
\text{NMS}[i,j] =
\begin{cases}
|\nabla I|[i,j] & \text{若 } |\nabla I|[i,j] > \max(\text{neighbor}_{+}, \text{neighbor}_{-}) \\
0 & \text{否则}
\end{cases}
$$

### 3.2 双阈值

$$
\text{edge}[i,j] =
\begin{cases}
\text{强边缘} & |\nabla I|[i,j] \geq T_{\text{high}} \\
\text{弱边缘} & T_{\text{low}} \leq |\nabla I|[i,j] < T_{\text{high}} \\
\text{非边缘} & |\nabla I|[i,j] < T_{\text{low}}
\end{cases}
$$

弱边缘**连接到强边缘**则保留，否则丢弃（BFS / DFS 追踪）。

---

## 代码文件

- [`sobel.py`](sobel.py) — Sobel 算子实现
- [`canny.py`](canny.py) — Canny 边缘检测完整实现
