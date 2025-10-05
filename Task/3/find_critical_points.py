import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def load_and_prepare_data(filename):
    """Load CSV data and organize into regular grid."""
    df = pd.read_csv(filename)
    
    # Get unique x and y coordinates (sorted)
    x_unique = np.sort(df['x'].unique())
    y_unique = np.sort(df['y'].unique())
    
    # Create meshgrid
    nx, ny = len(x_unique), len(y_unique)
    
    # Reshape S values into 2D grid
    # Assuming data is row-major ordered
    S_grid = df['S'].values.reshape(ny, nx)
    
    return x_unique, y_unique, S_grid

def compute_derivatives(x, y, S):
    """Compute first and second derivatives using finite differences."""
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    
    # First derivatives (central differences for interior points)
    dS_dx = np.zeros_like(S)
    dS_dy = np.zeros_like(S)
    
    # Interior points
    dS_dx[:, 1:-1] = (S[:, 2:] - S[:, :-2]) / (2 * dx)
    dS_dy[1:-1, :] = (S[2:, :] - S[:-2, :]) / (2 * dy)
    
    # Boundary points (forward/backward differences)
    dS_dx[:, 0] = (S[:, 1] - S[:, 0]) / dx
    dS_dx[:, -1] = (S[:, -1] - S[:, -2]) / dx
    dS_dy[0, :] = (S[1, :] - S[0, :]) / dy
    dS_dy[-1, :] = (S[-1, :] - S[-2, :]) / dy
    
    # Second derivatives
    d2S_dx2 = np.zeros_like(S)
    d2S_dy2 = np.zeros_like(S)
    d2S_dxdy = np.zeros_like(S)
    
    # Second derivatives (interior points)
    d2S_dx2[:, 1:-1] = (S[:, 2:] - 2*S[:, 1:-1] + S[:, :-2]) / (dx**2)
    d2S_dy2[1:-1, :] = (S[2:, :] - 2*S[1:-1, :] + S[:-2, :]) / (dy**2)
    
    # Mixed derivative (interior points)
    d2S_dxdy[1:-1, 1:-1] = (S[2:, 2:] - S[2:, :-2] - S[:-2, 2:] + S[:-2, :-2]) / (4 * dx * dy)
    
    return dS_dx, dS_dy, d2S_dx2, d2S_dy2, d2S_dxdy

from scipy import ndimage as ndi
import numpy as np

def find_critical_points(x, y, S, gradient_threshold=1e-2, det_threshold=1e-6, smooth_sigma=None):
    """
    在规则网格上查找临界点（critical points），并用连通域聚合去重。
    - gradient_threshold: 判为“梯度近零”的阈值
    - det_threshold: Hessian 判型的容差
    - smooth_sigma: 若给定，对 S 先做高斯平滑（pixels 为单位）
    """
    if smooth_sigma is not None and smooth_sigma > 0:
        from scipy.ndimage import gaussian_filter
        S_use = gaussian_filter(S, sigma=smooth_sigma)
    else:
        S_use = S

    dS_dx, dS_dy, d2S_dx2, d2S_dy2, d2S_dxdy = compute_derivatives(x, y, S_use)

    # 梯度模长（gradient magnitude）
    grad_mag = np.sqrt(dS_dx**2 + dS_dy**2)

    # 候选掩膜：梯度近零，且排除边界
    critical_mask = (grad_mag < gradient_threshold)
    critical_mask[0, :] = critical_mask[-1, :] = False
    critical_mask[:, 0] = critical_mask[:, -1] = False

    # --- 关键：连通域聚合（8邻域） ---
    structure = np.ones((3, 3), dtype=bool)
    labels, n_comp = ndi.label(critical_mask, structure=structure)

    minima, maxima, saddles = [], [], []

    for lab in range(1, n_comp + 1):
        ys, xs = np.where(labels == lab)
        # 在该连通域内挑一个代表点：梯度模长最小的像素
        k = np.argmin(grad_mag[ys, xs])
        i, j = ys[k], xs[k]

        # Hessian
        H = np.array([[d2S_dx2[i, j], d2S_dxdy[i, j]],
                      [d2S_dxdy[i, j], d2S_dy2[i, j]]], dtype=float)
        det_H = np.linalg.det(H)
        tr_H  = np.trace(H)

        x_coord, y_coord, s_value = x[j], y[i], S[i, j]

        if det_H > det_threshold:
            # 正定 / 负定 用迹（或最稳妥用特征值）
            if tr_H > 0:
                minima.append((x_coord, y_coord, s_value, i, j))
            elif tr_H < 0:
                maxima.append((x_coord, y_coord, s_value, i, j))
            else:
                eigs = np.linalg.eigvalsh(H)
                if np.all(eigs > 0):
                    minima.append((x_coord, y_coord, s_value, i, j))
                elif np.all(eigs < 0):
                    maxima.append((x_coord, y_coord, s_value, i, j))
                else:
                    saddles.append((x_coord, y_coord, s_value, i, j))
        elif det_H < -det_threshold:
            saddles.append((x_coord, y_coord, s_value, i, j))
        else:
            # det ~ 0 的不稳定点：可忽略或另行统计
            pass

    return minima, maxima, saddles

def visualize_results(x, y, S, minima, maxima, saddles):
    """Visualize scalar field with critical points marked."""
    X, Y = np.meshgrid(x, y)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot contour map
    levels = 20
    contour = ax.contourf(X, Y, S, levels=levels, cmap='viridis', alpha=0.7)
    ax.contour(X, Y, S, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    
    # Plot critical points
    if minima:
        min_x, min_y = zip(*[(m[0], m[1]) for m in minima])
        ax.scatter(min_x, min_y, c='blue', s=100, marker='v', 
                  edgecolors='black', linewidths=2, label='Local Minima', zorder=5)
    
    if maxima:
        max_x, max_y = zip(*[(m[0], m[1]) for m in maxima])
        ax.scatter(max_x, max_y, c='red', s=100, marker='^', 
                  edgecolors='black', linewidths=2, label='Local Maxima', zorder=5)
    
    if saddles:
        sad_x, sad_y = zip(*[(s[0], s[1]) for s in saddles])
        ax.scatter(sad_x, sad_y, c='yellow', s=100, marker='X', 
                  edgecolors='black', linewidths=2, label='Saddle Points', zorder=5)
    
    plt.colorbar(contour, ax=ax, label='S value')
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_title('Critical Points in 2D Scalar Field', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.show()

def print_results(minima, maxima, saddles):
    """Print critical points information."""
    print(f"\nCritical Points Found:")
    print(f"  Minima: {len(minima)}, Maxima: {len(maxima)}, Saddles: {len(saddles)}")
    print(f"  Total: {len(minima) + len(maxima) + len(saddles)}")
    
    if minima:
        print("\nLOCAL MINIMA:")
        for idx, (x, y, s, i, j) in enumerate(minima, 1):
            print(f"  {idx}. ({x:.4f}, {y:.4f}), S = {s:.6f}")
    
    if maxima:
        print("\nLOCAL MAXIMA:")
        for idx, (x, y, s, i, j) in enumerate(maxima, 1):
            print(f"  {idx}. ({x:.4f}, {y:.4f}), S = {s:.6f}")
    
    if saddles:
        print("\nSADDLE POINTS:")
        for idx, (x, y, s, i, j) in enumerate(saddles, 1):
            print(f"  {idx}. ({x:.4f}, {y:.4f}), S = {s:.6f}")

# Main execution
if __name__ == "__main__":
    # Load data
    x, y, S = load_and_prepare_data('2d_scalar_field.csv')
    print(f"Grid: {len(x)} x {len(y)}, S range: [{S.min():.4f}, {S.max():.4f}]")
    
    # Find critical points
    minima, maxima, saddles = find_critical_points(x, y, S, gradient_threshold=0.08, det_threshold=1e-7)
    
    # Print results
    print_results(minima, maxima, saddles)
    
    # Visualize
    visualize_results(x, y, S, minima, maxima, saddles)