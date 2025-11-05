import subprocess
import os
import platform
import shlex
import numpy

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
            label_true = curr_obs[obj_id][TRUE_LABEL]
            label_predicted = curr_obs[obj_id][PREDICTED_LABEL]
            points = curr_obs[obj_id][TRACKING_DATA]
            
            x = [p[1] for p in points]  # Extract x-coordinates
            y = [p[2] for p in points]  # Extract y-coordinates
    
            plt.plot(x, y, marker="o", linestyle="-", color="green")  # Plot trajectory

            # Labels & Formatting
            plt.xlabel("X Coordinate")
            plt.ylabel("Y Coordinate")
            #plt.title(f" Assigned True Label {'MOVING' if label_true == MOVING else 'NOT_MOVING'}")
            plt.title(f"{model_type} True Label {'MOVING' if label_true == MOVING else 'NOT_MOVING'} | Predicted: {'MOVING' if label_predicted == MOVING else 'NOT_MOVING'}")
            plt.legend(title=f"Object id {obj_id}")
            plt.grid(True, linestyle="--", alpha=0.6)  
            #plt.savefig(f"mislabled true label is: {label_true} and predicted_label is {label_predicted}.png")
            #i+=1
            plt.show()

def plot_displacements_across_frames(obj_displacements):
    """
    Plot displacement magnitudes for selected objects.
    """
    frames = []
    mags = []
    
    for obj_id, (frame_list,magnitudes_list) in obj_displacements.items():
        print(f"working in here! {obj_id}: {frame_list}, {magnitudes_list}")
        frames.extend(frame_list)
        mags.extend(magnitudes_list)

    plt.figure(figsize=(10, 6))
    plt.hist2d(frames, mags, bins=[100, 20], cmap='viridis')
    plt.xlabel("Frame Number")
    plt.ylabel("Displacement Magnitude")
    plt.title("2D Histogram of Displacement Magnitude Over Time")
    plt.colorbar(label='Count')
    plt.tight_layout()
    plt.show()
    
    return 

def plot_score_components(labeled_obs,all_zdx_zdy):


    # Separate scores by label
    moving_scores = [v[SCORES] for v in labeled_obs.values() if v[TRUE_LABELS] == MOVING]
    notmoving_scores = [v[SCORES] for v in labeled_obs.values() if v[TRUE_LABELS] == NOTMOVING]

    # Define bins based on combined data
    all_scores = moving_scores + notmoving_scores
    bins = numpy.histogram_bin_edges(all_scores, bins='auto')

    # Plot both histograms
    plt.figure(figsize=(8, 5))

    plt.hist(notmoving_scores, bins=bins, color='lightcoral', edgecolor='black', alpha=0.7, label='NOTMOVING')
    plt.hist(moving_scores, bins=bins, color='mediumseagreen', edgecolor='black', alpha=0.7, label='MOVING')

    # Optional mean line
    global_mean = numpy.mean(all_zdx_zdy)
    plt.axvline(global_mean, color='blue', linestyle='--', linewidth=2, label=f'Mean of Z-scoce Distances= {global_mean:.2f}')

    # Formatting
    plt.title("Histogram of Object Scores by Label")
    plt.xlabel("Mean Z-Score Distance")
    plt.ylabel("Number of Objects")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_labeled_mean_displacements_by_lines(curr_obs,global_mean,global_cov):
    moving_dx = []
    moving_dy=[]
    notmoving_dx = []
    notmoving_dy=[]
    print(f"global mean: {global_mean}, {global_cov}")
    for obj_id, data in curr_obs.items():
        stats=data[SCORES]
        #print(f"for {obj_id}:{stats} mean is: {stats[0]},{stats[1]}, {data[TRUE_LABELS]}")
    
        if data[TRUE_LABELS] == MOVING:
            moving_dx.append(stats[0])
            moving_dy.append(stats[1])
        else:
            notmoving_dx.append(stats[0])
            notmoving_dy.append(stats[1])
    print(f"{len(moving_dx)},{len(moving_dy)},{len(notmoving_dx)},{len(notmoving_dy)}")
    plt.figure(figsize=(8, 6))

    # Global mean vector (from origin)
    plt.axvline(global_mean[0], color='blue', linestyle='--', linewidth=2, label='Global Mean dx')
    plt.axhline(global_mean[1], color='orange', linestyle='--', linewidth=2, label='Global Mean dy')

    # 1σ spread lines (optional)
    #plt.axvline(global_mean[0] + numpy.sqrt(global_cov[0][0]), color='black', linestyle='--', linewidth=1, label='dx ± 1σ')
    #plt.axvline(global_mean[0] - numpy.sqrt(global_cov[0][0]), color='black', linestyle='--', linewidth=1)

    # Horizontal dashed gray lines for dy ± std
    #plt.axhline(global_mean[1] + numpy.sqrt(global_cov[1][1]), color='gray', linestyle='--', linewidth=1, label='dy ± 1σ')
    #plt.axhline(global_mean[1] - numpy.sqrt(global_cov[1][1]), color='gray', linestyle='--', linewidth=1)

    # Scatter of object displacements by label
    plt.scatter(moving_dx, moving_dy, color='green', label='MOVING', alpha=0.7, edgecolors='k')
    plt.scatter(notmoving_dx, notmoving_dy, color='red', label='NOTMOVING', alpha=0.7, edgecolors='k')

   
    plt.xlabel("Mean X Displacement")
    plt.ylabel("Mean Y Displacement")
    plt.title("Labeled Object Displacement Means with Global Vector")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
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
    global_min_x, global_max_x = -2.00, 2.00
    global_min_y, global_max_y = -3.50, 3.50

    all_models = [(grid_mu_alive, grid_cov_alive), (grid_mu_dead, grid_cov_dead)]

    for grid_mu, grid_cov in all_models:
        for i, (mu_row_item, cov_row_item) in enumerate(zip(grid_mu, grid_cov)):
            for j, (mu_col_item, cov_col_item) in enumerate(zip(mu_row_item, cov_row_item)):
                mu = mu_col_item
                cov_matrix = cov_col_item

                eigenvalues, _ = numpy.linalg.eigh(cov_matrix)
                width, height = 2 * numpy.sqrt(eigenvalues)
                max_range = max(width, height) * 1.5
                '''
                global_min_x = min(global_min_x, mu[0] - max_range)
                global_max_x = max(global_max_x, mu[0] + max_range)
                global_min_y = min(global_min_y, mu[1] - max_range)
                global_max_y = max(global_max_y, mu[1] + max_range)
                '''

    # Step 2: Plot overlay for each grid cell
    for i in range(len(grid_mu_alive)):
        for j in range(len(grid_mu_alive[0])):
            mu_alive = grid_mu_alive[i][j]
            cov_alive = grid_cov_alive[i][j]
            mu_dead = grid_mu_dead[i][j]
            cov_dead = grid_cov_dead[i][j]

            fig, ax = plt.subplots(figsize=(5, 5))

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

def plot_hourly_prediction(hour_list,total_list,alive_list,dose_rate):
    
    # Calculate alive percentages
    alive_percentages = [(alive / total) * 100 for alive, total in zip(alive_list, total_list)]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(hour_list, alive_percentages, marker='o', linestyle='-', color='purple')
    plt.xticks(hour_list)
    # Labels and formatting
    plt.xlabel("Hour of Imaging")
    plt.ylabel("Percentage of Motile Organisms Predicted (%)")
    plt.title(f"{dose_rate}ppb Motile Object Prediction Rate Over Time With MGD")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def grouped_bar_chart():

    '''
    # Example data (replace with your actual values)
    time_points = ['0 hr', '4 hr', '8 hr']
    alive_60 = [71, 68, 86]
    alive_240 = [81, 37, 25]
    alive_480 = [69, 7, 18]
    alive_960 = [50, 22, 18]

    # Position settings
    x = numpy.arange(len(time_points))
    bar_width = 0.2

    # Create grouped bars
    plt.bar(x - 1.5*bar_width, alive_60, bar_width, label='60 ppb')
    plt.bar(x - 0.5*bar_width, alive_240, bar_width, label='240 ppb')
    plt.bar(x + 0.5*bar_width, alive_480, bar_width, label='480 ppb')
    plt.bar(x + 1.5*bar_width, alive_960, bar_width, label='960 ppb')

    # Labels and formatting
    plt.xlabel('Hour of Imaging')
    plt.ylabel('Percentage of Alive Samples (%)')
    plt.title('Alive Prediction Rate Over Time for Different Concentrations')
    plt.xticks(x, time_points)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()
    '''
    #"480 ppb (Days-old)": {0: 127, 4: 12, 8: 378},
    # Data
    data = {
    "60 ppb (Days-old)": {0: 107, 4: 180, 8: 46, 12: 194, 16: 68, 28: 124, 32: 138, 33: 50, 34: 81, 35: 86, 36: 43},
    "240 ppb (Days-old)": {0: 32, 4: 37, 8: 386, 12: 215},
    "480 ppb (Days-old)": {0: 52, 1: 32, 1.5: 36, 2: 33, 2.5: 30, 3: 31, 3.5: 40},
    "960 ppb (Week-old)": {0: 32, 4: 156, 8: 96, 12: 59}
    }

    plt.figure(figsize=(8, 5))
    for i, (label, t_counts) in enumerate(data.items()):
        times = list(t_counts.keys())
        counts = list(t_counts.values())
        plt.scatter(times, [i]*len(times), s=[c*5 for c in counts], label=label, alpha=0.6)

    plt.yticks(range(len(data)), list(data.keys()))
    plt.xlabel("Imaging Time (hours)")
    plt.ylabel("Concentration and Age")
    plt.title("Toxic Sample Collection with Object Counts")
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_grid_coordinates(labeled_observations):

    max_x=4128
    max_y=2196
    color_map={0: "red", 1: "blue"}   # 0 = non-motile, 1 = motile
    show_legend=True
    point_size=2
    alpha=0.6
    
    fig, ax = plt.subplots(figsize=(9, 6))

    # collect and plot points label-wise (for clean legend)
    plotted_labels = set()
    for obj_id, content in labeled_observations.items():
        label = content[TRUE_LABEL]
        clr = color_map.get(label, "gray")
        # your tuples are (frame, x, y)
        xs = [p[1] for p in content[TRACKING_DATA]]
        ys = [p[2] for p in content[TRACKING_DATA]]

        # Only add a label once to legend
        leg_lbl = "Motile (1)" if label == 1 else "Non-motile (0)"
        ax.scatter(ys, xs, s=point_size, alpha=alpha, c=clr,
                   label=(leg_lbl if label not in plotted_labels else None))
        plotted_labels.add(label)

    # draw 3×3 grid lines (thirds of the field)
    x_third = max_x / 3
    y_third = max_y / 3
    for xv in (x_third, 2 * x_third):
        ax.axvline(x=xv, linestyle="--", linewidth=1, alpha=0.6)
    for yv in (y_third, 2 * y_third):
        ax.axhline(y=yv, linestyle="--", linewidth=1, alpha=0.6)

    ax.set_xlim(0, max_x)
    ax.set_ylim(0, max_y)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("One sample file (x, y) Points with 3×3 Grid Overlay")
    ax.grid(False)
    ax.set_aspect("equal", adjustable="box")

    if show_legend:
        ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    plt.show()

    return 

def plot_accuracy_window():
    
    window_sizes = list(range(1, 11))  # Window sizes from 1 to 10
    accuracy_values = [0.716, 0.731, 0.80, 0.735, 0.734, 0.730, 0.730, 0.730, 0.730, 0.730] 
    threshold = [-28.91, -28.91, -5.507, -28.91, -28.91, -28.91, -28.91, -28.91, -28.91, -28.91] 
    precision = [0.429, 0.556,0.715, 0.557, 1.00, 1.00, 1.00, 1.00, 1.00, 1.00] 
    recall = [0.144,0.044,0.419, 0.026,0.017,0.004,0.004,0.004,0.004,0.004]
    f1 =[0.216,0.081,0.529, 0.051,0.034,0.009,0.009, 0.009,0.009,0.009]
    
    # Plotting
    plt.figure(figsize=(8, 5))
    
    plt.plot(window_sizes, accuracy_values, label='accuracy',linestyle='-', color='orange')
    plt.plot(window_sizes, precision, label='precision',linestyle='-', color='green')
    plt.plot(window_sizes, recall, label='recall',linestyle='-', color='red')
    plt.plot(window_sizes, f1, label='f1-score',linestyle='-', color='blue')
    '''
    plt.plot(window_sizes,threshold,linestyle='-', color='purple')
    '''
   
    plt.xticks(window_sizes)
    plt.xlabel('Window Size')
    plt.ylabel('Evaluation Metrics')
    plt.title('Accuracy,Precision,Recall,F1-Score vs Window')
    #plt.ylim(0.1, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.show()

def create_montage(image_directory, output_name,grid_size=(5, 5), image_size=(100, 100)):
    """
    Create a montage from grid stat images
    
    Parameters:
    - image_directory: directory containing the images
    - output_name: output filename
    - grid_size: tuple (rows, cols) for the grid
    - image_size: tuple (width, height) to resize each image
    """
    
    rows, cols = grid_size
    width, height = image_size
    
    # Create the montage canvas
    montage_width = cols * width
    montage_height = rows * height
    montage = Image.new('RGB', (montage_width, montage_height), 'white')
    
    # Collect all images
    images = []
    missing_files = []
    loaded_files = []
    
    for row in range(rows):
        for col in range(cols):
            filename = f"{row}_{col}.png"
            filepath = os.path.join(image_directory, filename)
            # Debug: Print the full path being checked
            print(f"Checking: {filepath}")
            
            if os.path.exists(filepath):
                try:
                    with Image.open(filepath) as img:
                        # Resize image to standardize
                        img = img.resize((width, height), Image.Resampling.LANCZOS)
                        images.append(img)
                        loaded_files.append(filename)
                        print(f" Successfully loaded: {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    print(f"Full traceback: {traceback.format_exc()}")
                    missing_files.append(filename)
            else:
                print(f"File not found: {filename}")
                # Create a placeholder image
                placeholder = Image.new('RGB', (width, height), 'lightgray')
                images.append(placeholder)
                missing_files.append(filename)
    
    # Arrange images in the montage
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        x = col * width
        y = row * height
        montage.paste(img, (x, y))
    
    # Save the montage
    montage.save(output_name)
    print(f"Montage saved as: {output_name}")
    print(f"Montage size: {montage_width}x{montage_height}")
    
    if missing_files:
        print(f"Missing files replaced with placeholders: {missing_files}")
    
    return montage

def plot_motile_fraction_heatmap(data_dict, figsize=(10, 6)):
    """
    Create heatmap: rows=doses, cols=time bins, cells=motile fraction
    
    Parameters:
    - data_dict: {dose: {'times': [0,4,8,12], 'motile_fractions': [1.0,0.8,0.4,0.1]}}
    """
    
    # Get all unique time points
    all_times = sorted(set(time for data in data_dict.values() for time in data['times']))
    doses = sorted(data_dict.keys())
    
    # Create matrix
    matrix = numpy.full((len(doses), len(all_times)), numpy.nan)
    
    for i, dose in enumerate(doses):
        for j, time in enumerate(all_times):
            if time in data_dict[dose]['times']:
                time_idx = data_dict[dose]['times'].index(time)
                matrix[i, j] = data_dict[dose]['motile_fractions'][time_idx]
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(matrix, 
                xticklabels=[f'{t}h' for t in all_times],
                yticklabels=[f'{d} ppb' for d in doses],
                annot=True, fmt='.2f', cmap='RdYlGn', 
                cbar_kws={'label': 'Motile Fraction'},
                ax=ax)
    
    ax.set_title('Motile Fraction Over Time by Dose', fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Dose')
    
    plt.tight_layout()
    plt.show()

def plot_two_treatment_curves(figsize=(10, 6)):
    """
    Plot two treatment curves on the same plot
    """
    
    # Data
    time_480 = [0, 1, 1.5, 2, 2.5, 3, 3.5]
    motile_480 = [0.59, 0.56, 0.47, 0.33, 0.10, 0.05, 0.06]
    
    time_960 = [0, 4, 8, 12]
    motile_960 = [0.28, 0.04, 0.06, 0.01]
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.plot(time_480, motile_480, 'o-', linewidth=2, markersize=8, 
            color='blue', label='480 ppb (young ostracods)')
    
    ax.plot(time_960, motile_960, 's-', linewidth=2, markersize=8, 
            color='red', label='960 ppb (week-old ostracods)')
    
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Motile Fraction', fontsize=12)
    ax.set_title('Motile Response Under Different Treatments', fontweight='bold', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
def plot_three_confidence_ellipses(means, covariances, labels=['File 1', 'File 2', 'File 3'], 
                                  figsize=(8, 8), n_std=2.0):
    """
    Plot three confidence ellipses with different means and covariances
    
    Parameters:
    - means: list of 3 mean vectors [[mu_x1, mu_y1], [mu_x2, mu_y2], [mu_x3, mu_y3]]
    - covariances: list of 3 covariance matrices [cov1, cov2, cov3]
    - labels: list of 3 labels for the ellipses
    - n_std: number of standard deviations for ellipse size
    """
    
    fig, ax = matplotlib.pyplot.subplots(figsize=figsize)
    colors = ['red', 'green', 'blue']
    
    for i in range(3):
        mean = numpy.array(means[i])
        cov = covariances[i]
        
        # Plot mean point
        ax.scatter(mean[0], mean[1], color=colors[i], s=200, 
                  marker='o', label=labels[i], edgecolor='black', 
                  linewidth=2, zorder=5)
        
        # Calculate ellipse parameters
        eigenvals, eigenvecs = numpy.linalg.eigh(cov)
        eigenvals = numpy.maximum(eigenvals, 1e-8)  # Avoid negative eigenvalues
        
        angle = numpy.degrees(numpy.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width = 2 * n_std * numpy.sqrt(eigenvals[0])
        height = 2 * n_std * numpy.sqrt(eigenvals[1])
        
        # Create and add ellipse
        ellipse = Ellipse(mean, width, height, angle=angle,
                         facecolor=colors[i], alpha=0.3, 
                         edgecolor=colors[i], linewidth=2)
        ax.add_patch(ellipse)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'mean & covariance of 3 different sample files')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.show()

def plot_different_charts():
    '''
    hour_list=[0,4,8,12]
    total_list=[28,167,94,61]
    alive_list=[8,7,6,1]
    plot_hourly_prediction(hour_list,total_list,alive_list,960)
    '''
    #grouped_bar_chart()
    data = {
    60: {'times': [0, 4, 8, 12], 'motile_fractions': [.61, .55, .82, .51]},
    240: {'times': [0, 4, 8, 12], 'motile_fractions': [.72, .13, 0.038, 0.083]},
    960: {'times': [0, 4, 8, 12], 'motile_fractions': [.28, 0.04, 0.06, 0.01]}
}
    
    #plot_motile_fraction_heatmap(data)
    #plot_two_treatment_curves()
    #mean_covariance_overlay_plot(alive_model.mu,alive_model.cov_matrix,dead_model.mu,dead_model.cov_matrix)

    
