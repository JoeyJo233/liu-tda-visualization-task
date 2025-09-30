import pandas as pd
import numpy as np
import pyvista as pv
import sys

# === 1. 读取数据 ===
df = pd.read_csv("2d_scalar_field.csv")  # 替换为你的文件路径
x = df['x'].values
y = df['y'].values
S = df['S'].values

# === 2. 转换为网格 (100x100) ===
n = int(np.sqrt(len(S)))  # 应该是 100
n = int(np.sqrt(len(S)))  # 应该是 100
X = x.reshape((n, n))
Y = y.reshape((n, n))
Z = S.reshape((n, n))

# === 3. 构造 PyVista StructuredGrid ===
# 使用 Z 值作为高度，创建3D表面
z_scale = 1.5  # Z轴缩放因子，增大高度差异
grid = pv.StructuredGrid(X, Y, Z * z_scale)  # 将 Z 缩放1.5倍
grid["values"] = Z.flatten(order="C")  # 把函数值 S 作为 scalar（用于颜色映射）

# === 4. 处理命令行参数和设置等高线值 ===
# 检查是否有命令行参数传入
if len(sys.argv) > 1:
    try:
        contour_value = float(sys.argv[1])  # 从命令行获取等高线值
        print(f"使用命令行参数指定的等高线值: S = {contour_value}")
    except ValueError:
        print(f"警告: 无效的参数 '{sys.argv[1]}'，使用默认值 S = 0")
        contour_value = 0  # 参数无效时使用默认值
else:
    contour_value = 0  # 默认值: S = 0
    print(f"使用默认等高线值: S = {contour_value}")

# === 6. 可视化 ===
plotter = pv.Plotter()

# 添加主表面
plotter.add_mesh(grid, scalars="values", cmap="viridis", show_edges=False, 
                 smooth_shading=True)  # 平滑着色效果更好

# 添加等高线
contours = grid.contour(isosurfaces=[contour_value])  # 在指定值处绘制等高线
plotter.add_mesh(contours, color="red", line_width=3, 
                 label=f"Contour at S = {contour_value}")

plotter.add_title("2D Scalar Field: Height + Color Map + Contour Lines")

# 添加坐标轴 - 多种方式确保显示
plotter.show_axes()  # 基本坐标轴
plotter.add_axes()   # 添加轴标识器
plotter.show_grid()  # 显示网格线

# 自定义坐标轴标签
plotter.add_text("X", position='lower_left')
plotter.add_text("Y", position='lower_right') 
# plotter.add_text("Z (Scalar Value)", position='upper_left')

# 添加图例
plotter.add_legend()

plotter.show()
