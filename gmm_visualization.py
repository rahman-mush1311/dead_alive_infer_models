"""
Overlay Moving and NotMoving GMMs per grid cell for comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


def plot_covariance_ellipse(mean, cov, ax, n_std=1.0, color='blue', alpha=0.3, 
                            linestyle='-', linewidth=2, label=None):
    """
    Plot covariance ellipse for a 2D Gaussian.
    
    Parameters:
    - mean: [μx, μy]
    - cov: 2x2 covariance matrix
    - ax: matplotlib axis
    - n_std: number of standard deviations for ellipse
    - color: ellipse color
    - alpha: transparency
    - linestyle: line style
    - linewidth: line width
    - label: legend label
    """
    # Eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Angle of ellipse
    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
    
    # Width and height of ellipse
    width, height = 2 * n_std * np.sqrt(eigenvalues)
    
    # Create ellipse
    ellipse = Ellipse(mean, width, height, angle=angle,
                     facecolor=color, alpha=alpha, edgecolor=color, 
                     linewidth=linewidth, linestyle=linestyle, label=label)
    
    ax.add_patch(ellipse)


def plot_gmm_overlay_single_cell(gmm_moving_cell, gmm_notmoving_cell, 
                                 row, col, ax=None,
                                 n_moving=0, n_notmoving=0):
    """
    Overlay Moving and NotMoving GMMs for a single grid cell.
    
    Parameters:
    - gmm_moving_cell: fitted GMM for moving class (this cell)
    - gmm_notmoving_cell: fitted GMM for notmoving class (this cell)
    - row: grid row index
    - col: grid column index
    - ax: matplotlib axis (if None, creates new figure)
    - n_moving: number of moving samples
    - n_notmoving: number of notmoving samples
    
    Returns:
    - fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure
    
    has_moving = gmm_moving_cell is not None
    has_notmoving = gmm_notmoving_cell is not None
    
    # Colors
    color_moving = '#FF4444'      # Red
    color_notmoving = '#4444FF'   # Blue
    
    # Plot Moving GMM
    if has_moving:
        means = gmm_moving_cell.means_
        covariances = gmm_moving_cell.covariances_
        weights = gmm_moving_cell.weights_
        n_components = gmm_moving_cell.n_components
        
        for k in range(n_components):
            mean = means[k]
            cov = covariances[k]
            weight = weights[k]
            
            # Plot mean (star)
            #label = f'Moving K={k+1} (π={weight:.2f})' if k == 0 else f'Moving K={k+1} (π={weight:.2f})'
            ax.scatter(mean[0], mean[1], s=400, c=color_moving, marker='*', 
                      edgecolors='black', linewidths=2)
            
            # Plot covariance ellipses
            plot_covariance_ellipse(mean, cov, ax, n_std=2.0, color=color_moving, 
                                   alpha=0.2, linestyle='-', linewidth=2)
            plot_covariance_ellipse(mean, cov, ax, n_std=1.0, color=color_moving, 
                                   alpha=0.3, linestyle='--', linewidth=1.5)
            
            # Annotate mean
            text = f'M{k+1}: [{mean[0]:.2f}, {mean[1]:.2f}]'
            ax.annotate(text, xy=mean, xytext=(15, 15), textcoords='offset points',
                       fontsize=9, bbox=dict(boxstyle='round,pad=0.4', 
                       facecolor=color_moving, alpha=0.7, edgecolor='black'),
                       color='white', weight='bold')
    
    # Plot NotMoving GMM
    if has_notmoving:
        means = gmm_notmoving_cell.means_
        covariances = gmm_notmoving_cell.covariances_
        weights = gmm_notmoving_cell.weights_
        n_components = gmm_notmoving_cell.n_components
        
        for k in range(n_components):
            mean = means[k]
            cov = covariances[k]
            weight = weights[k]
            
            # Plot mean (star)
            #label = f'NotMoving K={k+1} (π={weight:.2f})'
            ax.scatter(mean[0], mean[1], s=400, c=color_notmoving, marker='*', 
                      edgecolors='black', linewidths=2)
            
            # Plot covariance ellipses
            plot_covariance_ellipse(mean, cov, ax, n_std=2.0, color=color_notmoving, 
                                   alpha=0.2, linestyle='-', linewidth=2)
            plot_covariance_ellipse(mean, cov, ax, n_std=1.0, color=color_notmoving, 
                                   alpha=0.3, linestyle='--', linewidth=1.5)
            
            # Annotate mean
            text = f'N{k+1}: [{mean[0]:.2f}, {mean[1]:.2f}]'
            ax.annotate(text, xy=mean, xytext=(-15, -15), textcoords='offset points',
                       fontsize=9, bbox=dict(boxstyle='round,pad=0.4', 
                       facecolor=color_notmoving, alpha=0.7, edgecolor='black'),
                       color='white', weight='bold')
    
    # Handle empty cells
    if not has_moving and not has_notmoving:
        ax.text(0, 0, f'Cell [{row}][{col}]\n\nNo GMMs\n(insufficient data)', 
               ha='center', va='center', fontsize=14, color='gray',
               bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    # Styling
    #ax.set_xlabel('Normalized dx', fontsize=13, weight='bold')
    #ax.set_ylabel('Normalized dy', fontsize=13, weight='bold')
    
    title = f'Cell [{row}][{col}]'
    if has_moving or has_notmoving:
        title += f'Moving: n={n_moving} | NotMoving: n={n_notmoving}'
    #ax.set_title(title, fontsize=14, weight='bold', pad=15)
    
    ax.legend(loc='best', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    
    # Set axis limits based on what's present
    if has_moving or has_notmoving:
        ax.axis('equal')
    else:
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
    
    return fig, ax


def plot_gmm_overlay_grid(gmm_moving, gmm_notmoving, figsize_per_cell=6, save_path=None):
    """
    Create grid of all cells with Moving and NotMoving GMMs overlaid.
    
    Parameters:
    - gmm_moving: GMMDisplacementModel for moving class
    - gmm_notmoving: GMMDisplacementModel for notmoving class
    - figsize_per_cell: size of each subplot
    - save_path: optional path to save figure
    
    Returns:
    - fig
    """
    n_rows = gmm_moving.num_rows()
    n_cols = gmm_moving.num_cols()
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(figsize_per_cell * n_cols, figsize_per_cell * n_rows))
    
    # Handle single cell case
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    print(f"\n{'='*70}")
    print(f"Creating GMM Overlay Grid ({n_rows}x{n_cols})")
    print(f"{'='*70}\n")
    
    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            
            gmm_mov = gmm_moving.gmm[row][col]
            gmm_notmov = gmm_notmoving.gmm[row][col]
            n_mov = gmm_moving.n[row][col]
            n_notmov = gmm_notmoving.n[row][col]
            
            print(f"Cell [{row}][{col}]")
            
            plot_gmm_overlay_single_cell(
                gmm_mov, gmm_notmov, row, col, ax,
                n_moving=n_mov, n_notmoving=n_notmov
            )
    
    fig.suptitle('GMM Overlay: Moving (Red) vs NotMoving (Blue)', 
                fontsize=8, weight='bold', y=0.998)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to: {save_path}")
    
    print(f"\n{'='*70}")
    print("Grid complete!")
    print(f"{'='*70}\n")
    
    return fig


def plot_gmm_overlay_selected_cells(gmm_moving, gmm_notmoving, cells_to_plot, 
                                    cols=3, figsize_per_cell=6, save_path=None):
    """
    Plot overlay for selected cells only.
    
    Parameters:
    - gmm_moving: GMMDisplacementModel for moving class
    - gmm_notmoving: GMMDisplacementModel for notmoving class
    - cells_to_plot: list of (row, col) tuples to plot
    - cols: number of columns in subplot grid
    - figsize_per_cell: size of each subplot
    - save_path: optional path to save figure
    
    Returns:
    - fig
    
    Example:
    cells = [(0, 0), (0, 1), (1, 1), (2, 2)]
    fig = plot_gmm_overlay_selected_cells(gmm_moving, gmm_notmoving, cells)
    """
    n_cells = len(cells_to_plot)
    rows = int(np.ceil(n_cells / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(figsize_per_cell * cols, figsize_per_cell * rows))
    axes = np.array(axes).flatten()  # Flatten for easy iteration
    
    print(f"\n{'='*70}")
    print(f"Creating GMM Overlay for {n_cells} Selected Cells")
    print(f"{'='*70}\n")
    
    for idx, (row, col) in enumerate(cells_to_plot):
        ax = axes[idx]
        
        gmm_mov = gmm_moving.gmm[row][col]
        gmm_notmov = gmm_notmoving.gmm[row][col]
        n_mov = gmm_moving.n[row][col]
        n_notmov = gmm_notmoving.n[row][col]
        
        print(f"Cell [{row}][{col}]: Moving n={n_mov}, NotMoving n={n_notmov}")
        
        plot_gmm_overlay_single_cell(
            gmm_mov, gmm_notmov, row, col, ax,
            n_moving=n_mov, n_notmoving=n_notmov
        )
    
    # Hide extra subplots
    for idx in range(n_cells, len(axes)):
        axes[idx].axis('off')
    
    fig.suptitle('GMM Overlay: Moving (Red) vs NotMoving (Blue)', 
                fontsize=18, weight='bold', y=0.998)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved to: {save_path}")
    
    print(f"\n{'='*70}")
    print("Selected cells complete!")
    print(f"{'='*70}\n")
    
    return fig


def create_comparison_summary(gmm_moving, gmm_notmoving):
    """
    Print summary comparison of Moving vs NotMoving GMMs.
    
    Parameters:
    - gmm_moving: GMMDisplacementModel for moving class
    - gmm_notmoving: GMMDisplacementModel for notmoving class
    """
    n_rows = gmm_moving.num_rows()
    n_cols = gmm_moving.num_cols()
    
    print(f"\n{'='*70}")
    print("GMM Comparison Summary")
    print(f"{'='*70}\n")
    
    print(f"{'Cell':<12} {'Moving K':<12} {'Moving n':<12} {'NotMoving K':<15} {'NotMoving n':<12}")
    print("-" * 70)
    
    for row in range(n_rows):
        for col in range(n_cols):
            cell_label = f"[{row}][{col}]"
            
            mov_k = gmm_moving.gmm[row][col].n_components if gmm_moving.gmm[row][col] else 0
            notmov_k = gmm_notmoving.gmm[row][col].n_components if gmm_notmoving.gmm[row][col] else 0
            
            mov_n = gmm_moving.n[row][col]
            notmov_n = gmm_notmoving.n[row][col]
            
            mov_k_str = str(mov_k) if mov_k > 0 else "-"
            notmov_k_str = str(notmov_k) if notmov_k > 0 else "-"
            
            print(f"{cell_label:<12} {mov_k_str:<12} {mov_n:<12} {notmov_k_str:<15} {notmov_n:<12}")
    
    print(f"\n{'='*70}\n")


