import numpy as np
import pyvista as pv
from pyvista import examples
import vtk
from vtk.util.numpy_support import vtk_to_numpy

dataset = examples.download_head_2()

arr = dataset.active_scalars
p60, p85, p95 = np.percentile(arr, [60, 85, 95])
smin, smax = float(arr.min()), float(arr.max())

def to_idx(v):
    t = 0.0 if smax == smin else (v - smin) / (smax - smin)
    return int(np.clip(round(t * 255), 0, 255))

i60, i85, i95 = map(to_idx, [p60, p85, p95])
opacity_lut = np.zeros(256, dtype=float)
if i85 > i60:
    opacity_lut[i60:i85+1] = np.linspace(0.0, 0.6, i85 - i60 + 1)
if i95 > i85:
    opacity_lut[i85:i95+1] = np.linspace(0.6, 0.95, i95 - i85 + 1)
opacity_lut[i95:] = 1.0

scalar_opacity = vtk.vtkPiecewiseFunction()
scalar_opacity.AddPoint(smin, 0.0)
scalar_opacity.AddPoint(p60, 0.0)
scalar_opacity.AddPoint(p85, 0.6)
scalar_opacity.AddPoint(p95, 0.85)
scalar_opacity.AddPoint(smax, 1.0)

# Compute gradient magnitude |∇S| statistics (for designing gradient opacity) ----------
# Use VTK's image gradient and magnitude filters to get gradient distribution for automatic thresholding
grad = vtk.vtkImageGradient()
grad.SetInputData(dataset)
grad.SetDimensionality(3)
grad.SetHandleBoundaries(True)
grad.Update()

mag = vtk.vtkImageMagnitude()
mag.SetInputConnection(grad.GetOutputPort())
mag.Update()

mag_img = mag.GetOutput()
garr = vtk_to_numpy(mag_img.GetPointData().GetScalars())

g70, g90, g99 = np.percentile(garr, [70, 90, 99])
gmin, gmax = float(garr.min()), float(garr.max())

# Gradient Opacity: low gradient transparent, high gradient gradually opaque
grad_opacity = vtk.vtkPiecewiseFunction()
grad_opacity.AddPoint(gmin, 0.0)
grad_opacity.AddPoint(g70, 0.0)
grad_opacity.AddPoint(g90, 0.5)
grad_opacity.AddPoint(g99, 1.0)
grad_opacity.AddPoint(gmax, 1.0)

# Rendering (GPU volume rendering + sampling step + 2D TF = Scalar TF × Gradient TF) ----------
pl = pv.Plotter(window_size=(900, 900))
actor = pl.add_volume(
    dataset,
    cmap="gray",
    opacity=opacity_lut,   
    shade=True,
    mapper="gpu",
    opacity_unit_distance=3,  
    blending='composite'  
)

mapper = actor.GetMapper()
if hasattr(mapper, "SetAutoAdjustSampleDistances"):
    mapper.SetAutoAdjustSampleDistances(False)

vx, vy, vz = dataset.spacing
step = 0.25 * min(vx, vy, vz)
if hasattr(mapper, "SetSampleDistance"):
    mapper.SetSampleDistance(step)
if hasattr(mapper, "SetInteractiveSampleDistance"):
    mapper.SetInteractiveSampleDistance(step * 2)

prop = actor.GetProperty()
prop.SetScalarOpacity(scalar_opacity)
# Set gradient opacity (= second dimension)
prop.SetGradientOpacity(grad_opacity)

prop.SetShade(True)
if hasattr(prop, "SetInterpolationTypeToLinear"):
    prop.SetInterpolationTypeToLinear()

mapper.SetSampleDistance(step * 0.8)

pl.view_isometric()
pl.show()
