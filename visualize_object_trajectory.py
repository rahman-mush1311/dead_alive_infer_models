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

MOVING=1
NOTMOVING=0


def plot_confusion_matrix(curr_obs, obs_type,color, model_type):
    
    true_label = [curr_obs[obj_id][TRUE_LABEL] for obj_id in curr_obs]
    predicted_label= [curr_obs[obj_id][PREDICTED_LABEL] for obj_id in curr_obs]
        
    # Create the confusion matrix
    cm = confusion_matrix(true_label, predicted_label, labels=[NOTMOVING, MOVING])
    accuracy = accuracy_score(true_label, predicted_label)
    f1 = f1_score(true_label, predicted_label, pos_label=1, average='binary')
    recall = recall_score(true_label, predicted_label, pos_label=1, average='binary')
    precision = precision_score(true_label, predicted_label, pos_label=1, average='binary')
    print(f"{accuracy:<10.3f}{f1:<10.3f}{recall:<10.3f}{precision:<10.3f}")
        
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


    