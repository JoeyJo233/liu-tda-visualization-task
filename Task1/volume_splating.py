import numpy as np
import pyvista as pv
from pyvista import examples
import vtk
from vtk.util.numpy_support import numpy_to_vtk

img = examples.download_head_2()     
spacing = img.spacing
origin  = img.origin
nx, ny, nz = img.dimensions

arr = img.active_scalars.reshape((nx, ny, nz), order="F")
p80 = np.percentile(arr, 80.0)
step_xyz = (1, 1, 1)  

ix = np.arange(0, nx, step_xyz[0])
iy = np.arange(0, ny, step_xyz[1])
iz = np.arange(0, nz, step_xyz[2])

X, Y, Z = np.meshgrid(ix, iy, iz, indexing='ij')
vals = arr[X, Y, Z]
mask = vals >= p80

X, Y, Z, V = X[mask], Y[mask], Z[mask], vals[mask]

PX = origin[0] + (X + 0.5) * spacing[0]
PY = origin[1] + (Y + 0.5) * spacing[1]
PZ = origin[2] + (Z + 0.5) * spacing[2]
pts_np = np.c_[PX, PY, PZ].astype(np.float32)

pts = vtk.vtkPoints()
pts.SetData(numpy_to_vtk(pts_np, deep=True))

poly = vtk.vtkPolyData()
poly.SetPoints(pts)

npts = pts.GetNumberOfPoints()
verts = vtk.vtkCellArray()
verts.Allocate(npts)
for i in range(npts):
    verts.InsertNextCell(1)
    verts.InsertCellPoint(i)
poly.SetVerts(verts)

radius = (0.9 * min(spacing)) * np.ones(npts, dtype=np.float32)
poly.GetPointData().AddArray(numpy_to_vtk(radius, deep=True, array_type=vtk.VTK_FLOAT))
poly.GetPointData().GetArray(0).SetName("Radius")

smin, smax = float(arr.min()), float(arr.max())
gray = np.clip((V - smin) / (smax - smin + 1e-6), 0, 1)
alpha = np.clip((V - p80) / (smax - p80 + 1e-6), 0, 1) ** 1.2

rgba = np.c_[gray*255, gray*255, gray*255, alpha*255].astype(np.uint8)
rgba_vtk = numpy_to_vtk(rgba, deep=True, array_type=vtk.VTK_UNSIGNED_CHAR)
rgba_vtk.SetNumberOfComponents(4)
rgba_vtk.SetName("RGBA")
poly.GetPointData().SetScalars(rgba_vtk)  


mapper = vtk.vtkOpenGLPointGaussianMapper()
mapper.SetInputData(poly)
mapper.EmissiveOff()
mapper.SetScaleArray("Radius")               #
mapper.SetScalarVisibility(True)
mapper.SetColorModeToDirectScalars()         

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetOpacity(1.0)


pl = pv.Plotter(window_size=(900, 900))
pl.add_actor(actor)
pl.add_axes()
pl.add_text(f"Splatting (object-space)\npoints={npts:,}\nstep={step_xyz}, thr={p80:.1f}",
            font_size=10)


pl.camera_position = 'iso'        


pl.show()
