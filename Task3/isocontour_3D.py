import pandas as pd
import numpy as np
import pyvista as pv
import sys

# === 1. read data ===
df = pd.read_csv("2d_scalar_field.csv")
x = df['x'].values
y = df['y'].values
S = df['S'].values

# === 2. convert to grid (100x100) ===
n = int(np.sqrt(len(S)))  # should be 100
X = x.reshape((n, n))
Y = y.reshape((n, n))
Z = S.reshape((n, n))

# === 3. construct PyVista StructuredGrid ===
z_scale = 1.5  # Z axis scaling factor to enhance height variation
grid = pv.StructuredGrid(X, Y, Z * z_scale)  # scale Z by 1.5
grid["values"] = Z.flatten(order="C")  # use function values S as scalar for color mapping

# === 4. process command line arguments and set contour values ===
if len(sys.argv) > 1:
    try:
        contour_value = float(sys.argv[1])  # get contour value from command line
        print(f"use command line argument specified contour value: S = {contour_value}")
    except ValueError:
        print(f"Warning: invalid input '{sys.argv[1]}', using default value S = 0")
        contour_value = 0 
else:
    contour_value = 0  # default value: S = 0
    print(f"Using default contour value: S = {contour_value}")

# === 6. visualization ===
plotter = pv.Plotter()

# ---- 优化后的 colorbar 参数 ----
scalar_bar_args = dict(
    title="Scalar field S(x,y)\n",   # 更清晰的标题
    vertical=False,                # 横向放置
    fmt="%.2f",                    # 刻度格式
    n_labels=5,                    # 刻度数量
    label_font_size=10,
    title_font_size=11,
    position_x=0.15,               # 下方左侧位置
    position_y=0.05,               # 底部位置
    width=0.70,                    # 横向宽度
    height=0.04,                   # 横向高度
)

# add main surface
plotter.add_mesh(
    grid, scalars="values", cmap="viridis", show_edges=False,
    smooth_shading=True, scalar_bar_args=scalar_bar_args
)

# add contour
contours = grid.contour(isosurfaces=[contour_value])
plotter.add_mesh(contours, color="red", line_width=3, label=f"Contour at S = {contour_value}")

plotter.add_title("2D Scalar Field: Height + Color Map + Contour Lines")

# add axes and grid
plotter.show_axes()   # orientation widget (小三维坐标指示器)
plotter.add_axes()    # 外部坐标轴
plotter.show_grid()   # 网格线与刻度

# 删除了手动添加的 X/Y 文本，使图更简洁

# add legend (用于 contour 曲线说明)
# plotter.add_legend()

plotter.show()
