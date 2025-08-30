from driver_data_preprocessing import PreProcessingObservations
from driver_data_preprocessing_utils import run_hourly_graph,get_visualization_ids,run_tracked_videos_by_filename,infer_with_trained_model,run_trajectory_plot,run_outlier_model,collect_infer_data,prepare_train_infer_data,calculate_class_probability
from visualize_object_trajectory import plot_object_trajectories,plot_confusion_matrix

from driver_GridDisplacementModel import GridDisplacementModel
from GridOutlierModel import OutlierModelEvaluation
from GridBayesianModel import BayesianModel


import numpy
import os
import math
from collections import Counter
from PIL import Image
import matplotlib.pyplot

TRUE_LABELS = "true_labels"
PREDICTED_LABELS = "predicted_labels"
LOG_PDFS="log_pdfs"
TRACKING_DATA = "tracking_data"


MOTILE=1
NOTMOTILE=0   


def collect_files(fileForTrain,typeOffile):
    """
    takes the folder path from user and returns the filelists contained inside that folder. 
    Parameters:
    fileForTrain: string containing either train/ infer
    typeOffile: string either .txt or .xlsx
    Returns:
    file_list- a list containing the filename along with the folder location
    """
    
    user_base_dir = input(f"Enter the base directory where your {fileForTrain} data folder is located: ")
    if not os.path.isdir(user_base_dir):
        raise ValueError(f"{user_base_dir} is not a valid directory")
        
    else:
        if typeOffile==".txt":
            file_list = [
                os.path.join(user_base_dir, f)
                for f in os.listdir(user_base_dir)
                if os.path.isfile(os.path.join(user_base_dir, f)) and f.lower().endswith(".txt")
            ]
        elif typeOffile==".xlsx":
            file_list = [
                os.path.join(user_base_dir, f)
                for f in os.listdir(user_base_dir)
                if os.path.isfile(os.path.join(user_base_dir, f)) and f.lower().endswith(".xlsx")
            ]
        else:
            raise ValueError("Unsupported file type. Please use '.txt' or '.xlsx'")

    return file_list

def stats(x):
    n = len(x)
    s = sum(x)
    mu = s / n if n else 0
    std = math.sqrt(sum([(xi - mu) ** 2 for xi in x]))
    max_x = max(x) if n else 0
    min_x = min(x) if n else 0
    print(f'{n=} {s=} {mu=} {std=} {max_x=} {min_x=}')

def analyze_object(o):
    occurrences = [r[0] for r in o]
    assert occurrences == list(range(1, len(o) + 1)) # occurrences are sequential
    dx = []
    dy = []
    for i in range(1, len(o)):
        dx.append(o[i][1] - o[i-1][1])
        dy.append(o[i][2] - o[i-1][2])
    print('stats dx')
    stats(dx)
    print('stats dy')
    stats(dy)
    frames = [r[3] for r in o]
    if frames != list(range(o[0][3], o[0][3] + len(o))):
        print(f'**** FRAMES ARE NOT SEQUENTIAL: {frames}') # frame numbers are sequential

def analyze(objects):
    for objectid in objects:
        print()
        print(f'analyzing {objectid} with {len(objects[objectid])} objects')
        analyze_object(objects[objectid])
    
if __name__ == "__main__":
    collected_train_txt_file_lists=collect_files("train",".txt")
    #print(collected_train_txt_file_lists)
    collected_train_excel_file_lists=collect_files("train",".xlsx")
    #print(collected_train_excel_file_lists)
    file_processor=PreProcessingObservations()
    for text_file, excel_file in zip(collected_train_txt_file_lists,collected_train_excel_file_lists):
        #print(f" txt file is: {text_file},{excel_file}")
        labeles_loaded=file_processor.load_labels(excel_file)
        tracking_observations=file_processor.load_observations(text_file)
        print(f"{text_file} has {len(tracking_observations)}")
        print(f"{excel_file} has {len(labeles_loaded)}")
        #analyze(observations)
    
    