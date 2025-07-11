import subprocess
import os
import platform
import shlex
import numpy

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score,roc_curve,ConfusionMatrixDisplay,auc

TRUE_LABELS = "true_labels"
PREDICTED_LABELS= "predicted_labels"
LOG_PDFS="log_pdfs"
DEAD_PDFS="dead_log_sum_pdfs"
ALIVE_PDFS="alive_log_sum_pdfs"
TRACKING_DATA = "tracking_data"

MOVING=1
NOTMOVING=0
       
def run_ffplay(video_path, width, height,start_frame=None, end_frame=None, fps=30, slow_factor=1.0):
    if not os.path.exists(video_path):
        print(f"File not found: {video_path}")
        return
    
    start_sec = start_frame / fps if start_frame is not None else None
    duration = ((end_frame - start_frame) / fps) if start_frame is not None and end_frame is not None else None
    # Build ffplay command
    cmd = ['ffplay', video_path]
    if width and height:
        cmd += ['-x', str(width), '-y', str(height)]
   
    # Build ffplay command
    cmd = ['ffplay', '-autoexit']
    
    vf_filters = []
    
    # Slow motion
    if slow_factor > 1.0:
        cmd += ['-vf', f'setpts={slow_factor}*PTS']

    # Frame-based timing
    if start_sec is not None:
        cmd += ['-ss', str(start_sec)]
    if duration is not None:
        cmd += ['-t', str(duration)]

    # Window size
    if width and height:
        cmd += ['-x', str(width), '-y', str(height)]

    cmd.append(video_path)

    try:
        subprocess.run(cmd)
    except FileNotFoundError:
        print("ffplay not found. Make sure FFmpeg is installed and in your system PATH.")
    except Exception as e:
        print(f"Error: {e}")

def get_video_fps(video_path):
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    try:
        output = subprocess.check_output(cmd).decode().strip()
        num, denom = map(int, output.split('/'))
        return num / denom
    except Exception as e:
        print(f"Error extracting FPS: {e}")
        return None
        
def plot_object_trajectories(curr_obs,extracted_ids,model_type):

    """
    Plots one object trajectory using x/y limits based on all objects combined.
    
    Params:
        - observations: dict of {object_id: TRACKING DATA: [(frame, x, y), ...], TRUE_LABELS: 0/1, PREDICTED_LABELS: 0,1}
        - all_ids: list of all relevant object IDs (subset of keys in observations)
    """
    
    i=0
    for obj_id in extracted_ids:        
        if obj_id in curr_obs:
            label_true = curr_obs[obj_id][TRUE_LABELS]
            label_predicted = curr_obs[obj_id][PREDICTED_LABELS]
            points = curr_obs[obj_id][TRACKING_DATA]
            
            x = [p[1] for p in points]  # Extract x-coordinates
            y = [p[2] for p in points]  # Extract y-coordinates
    
            plt.plot(x, y, marker="o", linestyle="-", color="brown")  # Plot trajectory

            # Labels & Formatting
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            plt.title(f"{model_type} True Label {'MOVING' if label_true == MOVING else 'NOT_MOVING'} | Predicted: {'MOVING' if label_predicted == MOVING else 'NOT_MOVING'}")
            plt.legend(title=f"Object id {obj_id}")
            plt.grid(True, linestyle="--", alpha=0.6)  
            #plt.savefig(f"mislabled true label is: {label_true} and predicted_label is {label_predicted}.png")
            #i+=1
            plt.show()

def get_axis_limits(curr_obs):

    all_x=[]
    all_y=[]
    
    for obj_id in curr_obs:
        points = curr_obs[obj_id][TRACKING_DATA]
        all_x.extend(p[1] for p in points)
        all_y.extend(p[2] for p in points)

           
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    
    return x_min,x_max,y_min,y_max
    
def mean_covariance_overlay_plot(grid_mu_alive, grid_cov_alive, grid_mu_dead, grid_cov_dead):
    # Step 1: Compute global min/max across both alive and dead models
    global_min_x, global_max_x = float('inf'), float('-inf')
    global_min_y, global_max_y = float('inf'), float('-inf')

    all_models = [(grid_mu_alive, grid_cov_alive), (grid_mu_dead, grid_cov_dead)]

    for grid_mu, grid_cov in all_models:
        for i, (mu_row_item, cov_row_item) in enumerate(zip(grid_mu, grid_cov)):
            for j, (mu_col_item, cov_col_item) in enumerate(zip(mu_row_item, cov_row_item)):
                mu = mu_col_item
                cov_matrix = cov_col_item

                eigenvalues, _ = numpy.linalg.eigh(cov_matrix)
                width, height = 2 * numpy.sqrt(eigenvalues)
                max_range = max(width, height) * 1.5

                global_min_x = min(global_min_x, mu[0] - max_range)
                global_max_x = max(global_max_x, mu[0] + max_range)
                global_min_y = min(global_min_y, mu[1] - max_range)
                global_max_y = max(global_max_y, mu[1] + max_range)

    # Step 2: Plot overlay for each grid cell
    for i in range(len(grid_mu_alive)):
        for j in range(len(grid_mu_alive[0])):
            mu_alive = grid_mu_alive[i][j]
            cov_alive = grid_cov_alive[i][j]
            mu_dead = grid_mu_dead[i][j]
            cov_dead = grid_cov_dead[i][j]

            fig, ax = plt.subplots(figsize=(8, 8))

            # Plot alive mean
            ax.plot(mu_alive[0], mu_alive[1], 'go', label="Moving Mean", markersize=10)
            eigenvalues, eigenvectors = numpy.linalg.eigh(cov_alive)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            angle = numpy.degrees(numpy.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * numpy.sqrt(eigenvalues)
            ellipse_alive = Ellipse(xy=mu_alive, width=width, height=height, angle=angle,
                                    edgecolor='green', linestyle='--', linewidth=4, facecolor='none', label="Moving 1 Std Dev")
            ax.add_patch(ellipse_alive)

            # Plot dead mean
            ax.plot(mu_dead[0], mu_dead[1], 'ro', label="Non-Moving Mean",  markersize=10)
            eigenvalues, eigenvectors = numpy.linalg.eigh(cov_dead)
            order = eigenvalues.argsort()[::-1]
            eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
            angle = numpy.degrees(numpy.arctan2(*eigenvectors[:, 0][::-1]))
            width, height = 2 * numpy.sqrt(eigenvalues)
            ellipse_dead = Ellipse(xy=mu_dead, width=width, height=height, angle=angle,
                                   edgecolor='red', linestyle='-', linewidth=4, facecolor='none', label="Non-Moving 1 Std Dev")
            ax.add_patch(ellipse_dead)

            # Set plot limits and labels
            ax.set_xlim(global_min_x, global_max_x)
            ax.set_ylim(global_min_y, global_max_y)
           
            
            ax.set_aspect('equal', adjustable='datalim')
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.grid(True)
            ax.set_title(f"Moving vs Non-Moving Covariance Ellipses Grid [{i}][{j}]")
            ax.legend()
            plt.tight_layout()
            plt.show()

def plot_confusion_matrix(curr_obs, obs_type,color, model_type):
    
    true_label = [curr_obs[obj_id][TRUE_LABELS] for obj_id in curr_obs]
    predicted_label= [curr_obs[obj_id][PREDICTED_LABELS] for obj_id in curr_obs]
        
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

    
