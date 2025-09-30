import numpy as np
import pyvista as pv
from pyvista import examples

# 1) 加载体数据（你也可以换成自己的 .vti / .mha / DICOM 文件夹）
dataset = examples.download_head_2()

# 2) 自适应构造一个更“锐利”的不透明度传递函数（关键提升清晰度）
arr = dataset.active_scalars
p70, p85, p95 = np.percentile(arr, [70, 85, 95])
smin, smax = float(arr.min()), float(arr.max())

def to_idx(v):
    t = 0.0 if smax == smin else (v - smin) / (smax - smin)
    return int(np.clip(round(t * 255), 0, 255))

i70, i85, i95 = map(to_idx, [p70, p85, p95])
opacity = np.zeros(256, dtype=float)
if i85 > i70:
    opacity[i70:i85+1] = np.linspace(0.0, 0.6, i85 - i70 + 1)
if i95 > i85:
    opacity[i85:i95+1] = np.linspace(0.6, 0.95, i95 - i85 + 1)
opacity[i95:] = 1.0

# 3) 绘制
pl = pv.Plotter(window_size=(900, 900))
actor = pl.add_volume(
    dataset,
    cmap="bone",
    opacity=opacity,
    shade=True,                 # 开启着色，增强轮廓
    mapper="gpu",               # GPU 体绘制
    opacity_unit_distance=0.25  # 适当偏小更锐利（可调 0.1~0.5）
)

# 4) 关键：直接设置底层 VTK 的采样步长（高清的核心）
mapper = actor.GetMapper()

# （可选）关闭自动步长
if hasattr(mapper, "SetAutoAdjustSampleDistances"):
    mapper.SetAutoAdjustSampleDistances(False)

# 用体素最小间距的 0.25 倍作为步长起点；想更清晰就再减小（更耗性能）
vx, vy, vz = dataset.spacing
step = 0.25 * min(vx, vy, vz)

if hasattr(mapper, "SetSampleDistance"):
    mapper.SetSampleDistance(step)
if hasattr(mapper, "SetInteractiveSampleDistance"):
    mapper.SetInteractiveSampleDistance(step * 2)  # 交互时稍微大一点更流畅

# （可选）如果你想更锐利，还可调体属性
prop = actor.GetProperty()
if hasattr(prop, "SetShade"):
    prop.SetShade(True)
if hasattr(prop, "SetInterpolationTypeToLinear"):
    prop.SetInterpolationTypeToLinear()

pl.view_isometric()
pl.show()
