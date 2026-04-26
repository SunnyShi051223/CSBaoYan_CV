# 03 特征提取 — HOG（方向梯度直方图）

## 1. 什么是 HOG？

HOG（Histogram of Oriented Gradients）是一种描述局部图像纹理与形状的特征描述子，
由 Dalal & Triggs（CVPR 2005）提出，广泛用于行人检测等任务。

---

## 2. 算法流程

```
原始图像
   │
   ▼  (可选) Gamma 校正 / 颜色归一化
   │
   ▼  计算梯度
   │  Gx = I * [-1, 0, 1]
   │  Gy = I * [-1, 0, 1]^T
   │  M[i,j]  = sqrt(Gx² + Gy²)
   │  θ[i,j]  = arctan(Gy / Gx)   ∈ [0°, 180°)（无符号）
   │
   ▼  在 cell 内构建梯度方向直方图
   │  将方向量化为 B 个 bin（通常 B=9，每 bin 20°）
   │  软投票（线性插值）
   │
   ▼  Block 归一化（L2-Hys）
   │  block = 2×2 cells
   │  v_norm = v / sqrt(||v||² + ε²)，然后截断到 0.2，再归一化
   │
   ▼  串联所有 block 特征向量 → HOG 描述子
```

---

## 3. 关键公式

### 梯度

$$
M[i,j] = \sqrt{G_x[i,j]^2 + G_y[i,j]^2}, \qquad
\theta[i,j] = \arctan\!\left(\frac{G_y[i,j]}{G_x[i,j]}\right) \cdot \frac{180°}{\pi} \bmod 180°
$$

### Cell 直方图（软投票）

对每个像素 $(i,j)$，设其属于 bin $k = \lfloor\theta / \Delta\theta\rfloor$：

$$
h_k \mathrel{+}= M[i,j] \cdot \left(1 - \frac{\theta - k\cdot\Delta\theta}{\Delta\theta}\right),
\quad
h_{k+1} \mathrel{+}= M[i,j] \cdot \frac{\theta - k\cdot\Delta\theta}{\Delta\theta}
$$

### Block 归一化（L2-Hys）

$$
\hat{v} = \frac{v}{\sqrt{\|v\|_2^2 + \varepsilon^2}}, \qquad
\hat{v} = \min(\hat{v},\, 0.2), \qquad
\hat{v} = \frac{\hat{v}}{\sqrt{\|\hat{v}\|_2^2 + \varepsilon^2}}
$$

---

## 代码文件

- [`hog.py`](hog.py) — HOG 特征描述子 NumPy 实现
