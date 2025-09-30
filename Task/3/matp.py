import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 3D 投影

# === 1. 读取数据 ===
df = pd.read_csv("2d_scalar_field.csv")  # 替换为你的文件路径
x = df['x'].values
y = df['y'].values
S = df['S'].values

# === 2. 转换为网格 ===
# 数据是按行存储的 (100x100)
n = int(np.sqrt(len(S)))  # 这里应该是 100
X = x.reshape((n, n))
Y = y.reshape((n, n))
Z = S.reshape((n, n))

# === 3. 可视化：高度表示函数值 ===
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')

ax.set_title("2D Scalar Field Visualization (Height = S)")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("S")

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()
