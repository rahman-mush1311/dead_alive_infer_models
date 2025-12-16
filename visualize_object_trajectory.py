import subprocess
import os
import platform
import shlex
import numpy

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score,roc_curve,ConfusionMatrixDisplay,auc

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

    return accuracy,f1,precision,recall
    
