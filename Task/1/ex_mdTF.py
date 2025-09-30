import numpy as np
import pyvista as pv
from pyvista import examples
import vtk
from vtk.util.numpy_support import vtk_to_numpy

# ---------- 1) 加载体数据 ----------
dataset = examples.download_head_2()

# ---------- 2) 标量不透明度（与你原本的逻辑一致，稍作函数化） ----------
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

# 为了给 VTK 的 PiecewiseFunction，也构一份“连续标量-不透明度”版本
scalar_opacity = vtk.vtkPiecewiseFunction()
scalar_opacity.AddPoint(smin, 0.0)
scalar_opacity.AddPoint(p60, 0.0)
scalar_opacity.AddPoint(p85, 0.6)
scalar_opacity.AddPoint(p95, 0.85)
scalar_opacity.AddPoint(smax, 1.0)

# ---------- 3) 计算梯度大小 |∇S| 的统计量（用于设计梯度不透明度） ----------
# 用 VTK 的图像梯度与模长滤波，得到梯度大小分布，便于自动定阈
grad = vtk.vtkImageGradient()
grad.SetInputData(dataset)
grad.SetDimensionality(3)
grad.SetHandleBoundaries(True)
grad.Update()

mag = vtk.vtkImageMagnitude()
mag.SetInputConnection(grad.GetOutputPort())
mag.Update()

# 拿到梯度大小数组（point data）
mag_img = mag.GetOutput()
garr = vtk_to_numpy(mag_img.GetPointData().GetScalars())

# 经验上“高梯度”可用较高分位点来选（你可以按需调整）
g70, g90, g99 = np.percentile(garr, [70, 90, 99])
gmin, gmax = float(garr.min()), float(garr.max())

# 梯度不透明度（Gradient Opacity）：低梯度透明，高梯度逐渐不透明
grad_opacity = vtk.vtkPiecewiseFunction()
grad_opacity.AddPoint(gmin, 0.0)
grad_opacity.AddPoint(g70, 0.0)
grad_opacity.AddPoint(g90, 0.5)
grad_opacity.AddPoint(g99, 1.0)
grad_opacity.AddPoint(gmax, 1.0)

# ---------- 4) 绘制（GPU 体绘制 + 采样步长 + 二维TF = 标量TF × 梯度TF） ----------
pl = pv.Plotter(window_size=(900, 900))
actor = pl.add_volume(
    dataset,
    cmap="gray",
    opacity=opacity_lut,   # 这会先给一个标量不透明度 LUT
    shade=True,
    mapper="gpu",
    opacity_unit_distance=2.0  # 适当偏大一些，更平滑（可调 1~3
)

# 取到底层 mapper / property，精准控制
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
# 用我们构造的“连续”标量不透明度，覆盖 LUT（更平滑、可控）
prop.SetScalarOpacity(scalar_opacity)
# 关键：设置梯度不透明度（= 第二维）
prop.SetGradientOpacity(grad_opacity)

# 其余观感相关
prop.SetShade(True)
if hasattr(prop, "SetInterpolationTypeToLinear"):
    prop.SetInterpolationTypeToLinear()

# 可选：略降采样距离以突出边界（代价是更耗性能）
# mapper.SetSampleDistance(step * 0.8)

pl.add_text("2D Transfer Function: Scalar × Gradient\n(梯度越高越不透明 → 边界更清楚)",
            font_size=10)
pl.view_isometric()
pl.show()
