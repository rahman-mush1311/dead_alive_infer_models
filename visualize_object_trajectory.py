import subprocess
import os
import platform
import shlex
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score,roc_curve,ConfusionMatrixDisplay,auc
from PIL import Image


TRUE_LABEL = "true_label"
PREDICTED_LABEL= "predicted_label"
LOG_PDFS="log_pdfs"
DEAD_PDFS="dead_log_sum_pdfs"
ALIVE_PDFS="alive_log_sum_pdfs"
TRACKING_DATA = "tracking_data"
SCORES = "scores"

MOTILE = 1
NOTMOTILE = 0


def plot_confusion_matrix(curr_obs, obs_type,color, model_type):
    
    true_label = [curr_obs[obj_id][TRUE_LABEL] for obj_id in curr_obs]
    predicted_label= [curr_obs[obj_id][PREDICTED_LABEL] for obj_id in curr_obs]
        
    # Create the confusion matrix
    cm = confusion_matrix(true_label, predicted_label, labels=[NOTMOTILE, MOTILE])
    accuracy = accuracy_score(true_label, predicted_label)
    f1 = f1_score(true_label, predicted_label, pos_label=1, average='binary')
    recall = recall_score(true_label, predicted_label, pos_label=1, average='binary')
    precision = precision_score(true_label, predicted_label, pos_label=1, average='binary')
    print(f"{accuracy:<10.3f}{f1:<10.3f}{recall:<10.3f}{precision:<10.3f}")
    return accuracy,f1,recall,precision
    '''    
    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-moving(0)", "Moving(1)"])
    disp.plot(cmap=color)
    disp.ax_.set_title(f"{model_type} Confusion Matrix Using {obs_type}")
    disp.ax_.set_xlabel("Predicted Labels")
    disp.ax_.set_ylabel("True Labels")
    
    metrics_text = (
        f"Accuracy: {accuracy:.3f}\n"
        f"F1-Score: {f1:.3f}\n"
        f"Recall: {recall:.3f}\n"
        f"Precision: {precision:.3f}\n"
            
    )
    
    disp.ax_.legend(
        handles=[plt.Line2D([], [], color='white', label=metrics_text)],
            loc='lower right',
            fontsize=10,
            frameon=False
        )
    
    plt.show()
    '''
def extract_correlations_from_model(model):
    """
    Converts each trained cell's covariance matrix into a correlation matrix.
    
    Parameters:
    - model : trained GridFeatureModel or GridDisplacementModel
              (must have .cov_matrix and .n attributes)
    
    Returns:
    - corr_matrices : dict {(row, col): correlation_matrix}
    """
    corr_matrices = {}
    rows, cols = model.num_rows(), model.num_cols()

    for r in range(rows):
        for c in range(cols):
            cov = model.cov_matrix[r][c]
            n = model.n[r][c]

            # skip cells with too few observations
            if n < 2:
                continue
            if np.allclose(cov, 0):
                continue

            # compute per-feature std deviations
            std = np.sqrt(np.diag(cov))
            # avoid divide-by-zero
            std[std == 0] = np.inf
            corr = cov / np.outer(std, std)
            corr_matrices[(r, c)] = corr

    return corr_matrices
    
def visualize_model_features_correlations(curr_model):
    
    features = ['dx', 'dy', 'heading', 'turning', 'ax', 'ay']
    for r in range(curr_model.num_rows()):
        for c in range(curr_model.num_cols()):
            if curr_model.n[r][c] > 1:
                cov = curr_model.cov_matrix[r][c]
                corr = np.corrcoef(cov)
                sns.heatmap(corr, xticklabels=features, yticklabels=features,
                        annot=True, vmin=-1, vmax=1, cmap='coolwarm')
                plt.title(f'Correlation matrix cell [{r},{c}]')
                plt.show()

def plot_corr(corr, title):
    features = ['dx', 'dy', 'heading', 'turning', 'ax', 'ay']
    sns.heatmap(corr, vmin=-1, vmax=1, cmap='coolwarm',
                xticklabels=features, yticklabels=features, annot=True)
    plt.title(title)
    plt.show()

def visualize_auc_score(auc_pairs):
    feature_names = ['dx', 'dy', 'heading', 'turning', 'ax', 'ay']
    n_feat = len(feature_names)

    # Initialize AUC matrix with NaNs
    auc_mat = np.full((n_feat, n_feat), np.nan)

    # Fill symmetric matrix: AUC for pair (i,j) goes in [i,j] and [j,i]
    for fi, fj, auc in auc_pairs:
        i = feature_names.index(fi)
        j = feature_names.index(fj)
        auc_mat[i, j] = auc
        auc_mat[j, i] = auc

    # Diagonal: set to 0.5 or NaN (no pair)
    np.fill_diagonal(auc_mat, 0.5)

    plt.figure(figsize=(6, 5))
    sns.heatmap(auc_mat,vmin=0.5, vmax=1.0,annot=True, fmt=".2f",xticklabels=feature_names,yticklabels=feature_names)
    plt.title("Feature-Pair AUC Matrix")
    plt.tight_layout()
    plt.show()

def _plot_top_ranked_pair(X_all,y_all):
    feature_names = ['dx', 'dy', 'heading', 'turning', 'ax', 'ay']

    # Choose which two features to plot
    i = feature_names.index('dy')   # X-axis
    j = feature_names.index('ay')   # Y-axis
    plt.figure(figsize=(6,5))
    plt.scatter(X_all[y_all==0, i], X_all[y_all==0, j],alpha=0.3, color='red', label='Non-motile')
    plt.scatter(X_all[y_all==1, i], X_all[y_all==1, j],alpha=0.3, color='green', label='Motile')

    plt.xlabel(feature_names[i])
    plt.ylabel(feature_names[j])
    plt.legend()
    plt.title(f'{feature_names[i]} vs {feature_names[j]}')
    plt.tight_layout()
    plt.show()

def _svm_feature_coorelation (X, feature_names):
    
    # Compute correlation matrix
    correlation_matrix = np.corrcoef(X.T)
    
    # Create heatmap
    plt.figure(figsize=(16, 14))
    sns.heatmap(correlation_matrix, 
                xticklabels=feature_names, 
                yticklabels=feature_names,
                cmap='coolwarm', 
                center=0,
                vmin=-1, 
                vmax=1,
                annot=True,  # Show correlation values
                fmt='.2f',
                square=True,
                cbar_kws={'label': 'Correlation'})
    
    plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    plt.show()


def visualize_grid_displacements(grid_dis, save_path=None):
    """
    Visualize dy displacements for each grid cell with overlaid histograms
    for motile vs non-motile observations.
    
    Args:
        grid_dis: 3D list [row][col] -> list of tuples (dx, dy, label)
        save_path: Optional path to save the figure
    """

    
    num_rows = 3  # 3
    num_cols = 3  # 3
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 15))
    fig.suptitle('Vertical Displacement (dy) Distribution by Grid Cell', 
                 fontsize=16, y=0.995)
    
    # Process each grid cell
    for row in range(num_rows):
        for col in range(num_cols):
            ax = axes[row, col]
            cell_data = grid_dis[row][col]
            
            if len(cell_data) == 0:
                # Empty cell - leave blank
                ax.set_title(f'Cell ({row},{col})\nNo data', fontsize=10)
                ax.set_xlabel('dy (pixels/frame)')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
            else:
                # Separate dy values by label
                motile_dy = [dy for dx, dy, label in cell_data if label == MOTILE]
                nonmotile_dy = [dy for dx, dy, label in cell_data if label == NOTMOTILE]
                
                # Plot overlaid histograms
                if len(motile_dy) > 0:
                    ax.hist(motile_dy, bins=30, alpha=0.6, color='red', 
                           label=f'Motile (n={len(motile_dy)})', edgecolor='black')
                
                if len(nonmotile_dy) > 0:
                    ax.hist(nonmotile_dy, bins=30, alpha=0.6, color='blue', 
                           label=f'Non-motile (n={len(nonmotile_dy)})', edgecolor='black')
                
                # Formatting
                ax.set_title(f'Cell ({row},{col}) dy ', fontsize=6)
                #ax.set_xlabel(f'Cell ({row},{col}) dy')
                ax.set_ylabel('Frequency')
                ax.legend(fontsize=5)
                ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

def visualize_grid_displacement_points(grid_dis, save_path=None):
    """
    Visualize displacement points (dx, dy) for each grid cell with scatter plots
    for motile vs non-motile observations.
    
    Args:
        grid_dis: 3D list [row][col] -> list of tuples (dx, dy, label)
        save_path: Optional path to save the figure
    """
    
    num_rows = 3  # 3
    num_cols = 2  # 3
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 18))
    fig.suptitle('Points (x, y) by Grid Cell', 
                 fontsize=16, y=0.998)
    
    # Process each grid cell
    for row in range(num_rows):
        for col in range(num_cols):
            ax = axes[row, col]
            cell_data = grid_dis[row][col]
            
            if len(cell_data) == 0:
                # Empty cell - leave blank
                ax.set_title(f'Cell ({row},{col}) - No data', fontsize=10, pad=12)
                ax.set_xlabel('x (pixels)', fontsize=9)
                ax.set_ylabel('y (pixels)', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
                ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5)
                '''
                # Add directional arrow (left to right)
                ax.annotate('', xy=(0.8, 0.9), xytext=(0.2, 0.9),
                           xycoords='axes fraction',
                           arrowprops=dict(arrowstyle='->', color='green', lw=2.5),
                           annotation_clip=False)
                '''
            else:
                # Separate points by label
                motile_points = [(dx, dy) for dx, dy, label in cell_data if label == MOTILE]
                nonmotile_points = [(dx, dy) for dx, dy, label in cell_data if label == NOTMOTILE]
                
                # Plot non-motile points first (so motile appears on top)
                if len(nonmotile_points) > 0:
                    nonmotile_dx = [dx for dx, dy in nonmotile_points]
                    nonmotile_dy = [dy for dx, dy in nonmotile_points]
                    ax.scatter(nonmotile_dx, nonmotile_dy, c='blue', alpha=0.4, s=10,
                             label=f'Non-motile (n={len(nonmotile_points)})', 
                             edgecolors='none', marker='o')
                
                # Plot motile points
                if len(motile_points) > 0:
                    motile_dx = [dx for dx, dy in motile_points]
                    motile_dy = [dy for dx, dy in motile_points]
                    ax.scatter(motile_dx, motile_dy, c='red', alpha=0.5, s=10,
                             label=f'Motile (n={len(motile_points)})', 
                             edgecolors='none', marker='o')
                
                # Formatting
                ax.set_title(f'Cell ({row},{col}) - Total: {len(cell_data)}', 
                           fontsize=10, pad=12)
                ax.set_xlabel('x (pixel)', fontsize=9)
                ax.set_ylabel('y (pixel)', fontsize=9)
                ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
                ax.grid(True, alpha=0.3)
                ax.axhline(y=0, color='k', linestyle='-', linewidth=0.8)
                ax.axvline(x=0, color='k', linestyle='-', linewidth=0.8)
                
                # Add directional arrow at the top
                ylim = ax.get_ylim()
                xlim = ax.get_xlim()
                arrow_y = ylim[1] * 0.9  # Position arrow near top
                arrow_x_start = xlim[0] + (xlim[1] - xlim[0]) * 0.05
                arrow_x_end = xlim[0] + (xlim[1] - xlim[0]) * 0.35
                '''
                ax.annotate('', xy=(arrow_x_end, arrow_y), xytext=(arrow_x_start, arrow_y),
                           arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
                '''
                # Adjust tick label size
                ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Adjust spacing to prevent overlap
    plt.tight_layout(rect=[0, 0, 1, 0.995], h_pad=3.0, w_pad=2.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()

def visualize_grid_positions(grid_dis, save_path=None):
    """
    Visualize raw position points (x, y) for each grid cell
    for motile vs non-motile observations. Only shows cells with data.
    
    Args:
        grid_dis: 3D list [row][col] -> list of tuples (x, y, label)
        save_path: Optional path to save the figure
    """
    
    num_rows = 3
    num_cols = 3
    
    # First, identify which cells have data
    cells_with_data = []
    for row in range(num_rows):
        for col in range(num_cols):
            if len(grid_dis[row][col]) > 0:
                cells_with_data.append((row, col, grid_dis[row][col]))
    
    if len(cells_with_data) == 0:
        print("No data to plot!")
        return
    
    # Calculate subplot layout (try to keep it squarish)
    n_plots = len(cells_with_data)
    n_cols_plot = min(3, n_plots)  # Max 3 columns
    n_rows_plot = (n_plots + n_cols_plot - 1) // n_cols_plot  # Ceiling division
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows_plot, n_cols_plot, 
                             figsize=(6*n_cols_plot, 6*n_rows_plot))
    fig.suptitle('Object Positions (x, y) by Grid Cell', 
                 fontsize=16, y=0.995)
    
    # Handle case where there's only one subplot
    if n_plots == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    # Plot each cell with data
    for idx, (row, col, cell_data) in enumerate(cells_with_data):
        ax = axes[idx]
        
        # Separate points by label
        motile_points = [(x, y) for x, y, label in cell_data if label == MOTILE]
        nonmotile_points = [(x, y) for x, y, label in cell_data if label == NOTMOTILE]
        
        # Plot non-motile points first (so motile appears on top)
        if len(nonmotile_points) > 0:
            nonmotile_x = [x for x, y in nonmotile_points]
            nonmotile_y = [y for x, y in nonmotile_points]
            ax.scatter(nonmotile_x, nonmotile_y, c='blue', alpha=0.4, s=10,
                     label=f'Non-motile (n={len(nonmotile_points)})', 
                     edgecolors='none', marker='o')
        
        # Plot motile points
        if len(motile_points) > 0:
            motile_x = [x for x, y in motile_points]
            motile_y = [y for x, y in motile_points]
            ax.scatter(motile_x, motile_y, c='red', alpha=0.5, s=10,
                     label=f'Motile (n={len(motile_points)})', 
                     edgecolors='none', marker='o')
        
        # Formatting
        ax.set_title(f'Cell ({row},{col}) - Total: {len(cell_data)}', 
                   fontsize=11, pad=15)
        ax.set_xlabel('x (pixels)', fontsize=9, labelpad=8)
        ax.set_ylabel('y (pixels)', fontsize=9, labelpad=8)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Hide any unused subplots
    for idx in range(len(cells_with_data), len(axes)):
        axes[idx].axis('off')
    
    # Adjust spacing with more room
    plt.tight_layout(rect=[0, 0.02, 1, 0.98], h_pad=3.5, w_pad=2.5)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


    