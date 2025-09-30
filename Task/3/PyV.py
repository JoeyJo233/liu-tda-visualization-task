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
n = int(np.sqrt(len(S)))  # should be 100
X = x.reshape((n, n))
Y = y.reshape((n, n))
Z = S.reshape((n, n))

# === 3. construct PyVista StructuredGrid ===
# use Z as height to create 3D surface
z_scale = 1.5  # Z axis scaling factor to enhance height variation
grid = pv.StructuredGrid(X, Y, Z * z_scale)  # scale Z by 1.5
grid["values"] = Z.flatten(order="C")  # use function values S as scalar for color mapping

# === 4. process command line arguments and set contour values ===
# check if there are command line arguments
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

# add main surface
plotter.add_mesh(grid, scalars="values", cmap="viridis", show_edges=False, 
                 smooth_shading=True)  # smooth shading looks better

# add contour
contours = grid.contour(isosurfaces=[contour_value])  # draw contour at specified value
plotter.add_mesh(contours, color="red", line_width=3, 
                 label=f"Contour at S = {contour_value}")

plotter.add_title("2D Scalar Field: Height + Color Map + Contour Lines")

# add axes - multiple ways to ensure display
plotter.show_axes()  # basic axes
plotter.add_axes()   # add axis labels
plotter.show_grid()  # show grid lines

# custom axis labels
plotter.add_text("X", position='lower_left')
plotter.add_text("Y", position='lower_right') 
# plotter.add_text("Z (Scalar Value)", position='upper_left')

# add legend
plotter.add_legend()

plotter.show()