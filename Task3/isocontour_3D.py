import pandas as pd
import numpy as np
import pyvista as pv
import sys

df = pd.read_csv("2d_scalar_field.csv")
x = df['x'].values
y = df['y'].values
S = df['S'].values

# convert to grid (100x100)
n = int(np.sqrt(len(S))) 
X = x.reshape((n, n))
Y = y.reshape((n, n))
Z = S.reshape((n, n))

# construct PyVista StructuredGrid
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

plotter = pv.Plotter()

scalar_bar_args = dict(
    title="Scalar field S(x,y)\n",   
    vertical=False,                
    fmt="%.2f",                    
    n_labels=5,                    
    label_font_size=10,
    title_font_size=11,
    position_x=0.15,               
    position_y=0.05,               
    width=0.70,                    
    height=0.04,                   
)

plotter.add_mesh(
    grid, scalars="values", cmap="viridis", show_edges=False,
    smooth_shading=True, scalar_bar_args=scalar_bar_args
)

# add contour
contours = grid.contour(isosurfaces=[contour_value])
plotter.add_mesh(contours, color="red", line_width=3, label=f"Contour at S = {contour_value}")

plotter.add_title("2D Scalar Field: Height + Color Map + Contour Lines")
plotter.show_axes()   
plotter.add_axes()    
plotter.show_grid()  

plotter.show()
