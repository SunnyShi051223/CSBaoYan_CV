# CSBaoYan_CV — 计算机视觉基础知识学习

> 面向 **CS 保研 / 考研** 及入门深度学习 CV 方向的同学，整理了核心数学公式 + NumPy 代码实现，便于快速上手与复习。

---

## 目录

| 模块 | 内容 |
|------|------|
| [01\_image\_basics](01_image_basics/) | 图像卷积、空间滤波（均值、高斯、中值） |
| [02\_edge\_detection](02_edge_detection/) | Sobel 算子、Canny 边缘检测 |
| [03\_feature\_extraction](03_feature_extraction/) | HOG 特征描述子 |
| [04\_cnn\_basics](04_cnn_basics/) | 卷积层、池化层、激活函数、批归一化 |

---

## 快速开始

```bash
# 安装依赖（仅需 numpy）
pip install numpy

# 运行任意模块的示例
python 01_image_basics/convolution.py
python 02_edge_detection/sobel.py
python 03_feature_extraction/hog.py
python 04_cnn_basics/layers.py
```

---

## 学习路线

```
图像基础（卷积 & 滤波）
       ↓
边缘检测（梯度计算）
       ↓
特征提取（HOG / 方向梯度直方图）
       ↓
CNN 基础构件（卷积层 → 池化 → 激活 → BN）
```
