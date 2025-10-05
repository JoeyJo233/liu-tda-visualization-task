import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

def load_and_prepare_data(filename):
    """Load CSV data and organize into regular grid."""
    df = pd.read_csv(filename)
    
    # Get unique x and y coordinates (sorted)
    x_unique = np.sort(df['x'].unique())
    y_unique = np.sort(df['y'].unique())
    
    # Create meshgrid
    nx, ny = len(x_unique), len(y_unique)
    
    # Reshape S values into 2D grid
    S_grid = df['S'].values.reshape(ny, nx)
    
    return x_unique, y_unique, S_grid

def visualize_isocontour(x, y, S, target_isovalue=0.0):
    """Visualize scalar field with highlighted target isocontour."""
    X, Y = np.meshgrid(x, y)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Plot contour map
    levels = 20
    contour = ax.contourf(X, Y, S, levels=levels, cmap='viridis', alpha=0.7)
    ax.contour(X, Y, S, levels=levels, colors='black', alpha=0.3, linewidths=0.5)
    
    # Highlight target isocontour in red
    target_contour = ax.contour(X, Y, S, levels=[target_isovalue], 
                                colors='red', linewidths=3, zorder=5)
    
    # Colorbar
    plt.colorbar(contour, ax=ax, label='S value')
    
    # Add red line to legend for target isocontour
    from matplotlib.lines import Line2D
    red_line = Line2D([0], [0], color='red', linewidth=3, 
                     label=f'Isocontour (S={target_isovalue:.3f})')
    
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    
    ax.legend(handles=[red_line], fontsize=10, loc='upper left')
    
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.show()

def print_data_info(x, y, S, target_isovalue):
    """Print information about the data and target isovalue."""
    print(f"\nGrid: {len(x)} × {len(y)}, S range: [{S.min():.6f}, {S.max():.6f}]")
    print(f"Target isovalue: S = {target_isovalue:.6f}")
    
    if target_isovalue < S.min():
        print(f"Warning: Target value below S_min, isocontour will NOT appear")
    elif target_isovalue > S.max():
        print(f"Warning: Target value above S_max, isocontour will NOT appear")

# Main execution
if __name__ == "__main__":

    target_isovalue = 0.0  
    if len(sys.argv) > 1:
        try:
            target_isovalue = float(sys.argv[1])
        except ValueError:
            print(f"Invalid isovalue '{sys.argv[1]}'. Using default: 0.0")
            target_isovalue = 0.0

    x, y, S = load_and_prepare_data('2d_scalar_field.csv')
    print_data_info(x, y, S, target_isovalue)
    visualize_isocontour(x, y, S, target_isovalue=target_isovalue)