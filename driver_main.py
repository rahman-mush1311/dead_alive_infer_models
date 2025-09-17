from driver_data_preprocessing import PreProcessingObservations
from driver_training_processer import run_outlier_model,run_bayesian_model,infer_with_trained_outlier_model,infer_with_trained_bayesian_model
#from driver_data_preprocessing_utils import run_hourly_graph,get_visualization_ids,run_tracked_videos_by_filename,infer_with_trained_model,run_trajectory_plot,run_outlier_model,collect_infer_data,prepare_train_infer_data,calculate_class_probability
from visualize_object_trajectory import plot_object_trajectories,plot_confusion_matrix,plot_hourly_prediction,plot_grid_coordinates,plot_accuracy_window

from driver_GridDisplacementModel import GridDisplacementModel
from GridOutlierModel import OutlierModelEvaluation
from GridBayesianModel import BayesianModel


import numpy
import os
import math
from collections import Counter
from PIL import Image
import matplotlib.pyplot

TRUE_LABEL = "true_label"
PREDICTED_LABEL = "predicted_label"
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

def collect_tox_file_from_user_input():
    """
    takes folder location and filename as user input returns the full file path in a list
    Params:
    -N/A
    Returns:
    folder_path or None
    """

    full_file_path = input("Enter folder location and file name (e.g., C:/data/tracks/obj_track_01.txt): ").strip()
    file_list=[]

    try:
        file_list.append(full_file_path)
        return file_list
    except FileNotFoundError:
        print(f"File not found: {full_file_path}")
        return None

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

def prepare_train_data(collected_text_file_lists,collected_excel_file_lists):
    
    observation_stats ={}
    
    all_train_observations={}
    all_test_observations={}
    
    
    for text_file, excel_file in zip(collected_train_txt_file_lists,collected_train_excel_file_lists):
        #print(f" txt file is: {text_file},{excel_file}")
        file_processor=PreProcessingObservations()
        labeles_loaded=file_processor.load_labels(excel_file)
        tracking_observations=file_processor.load_observations(text_file)
        labeled_observations=file_processor.label_observations_by_expert_labels(text_file,excel_file,tracking_observations,labeles_loaded)
        print(f"{text_file} has {len(tracking_observations)}")
        print(f"{excel_file} has {len(labeles_loaded)}")
        print(f"final labeled obs size is: {len(labeled_observations)}")
        first_key, first_value = next(iter(labeled_observations.items()))
        #print(f"{first_key},{first_value}")
        train_observations,test_observations=file_processor.prepare_train_test(labeled_observations,train_ratio=0.8)
        if len(train_observations)>0:
            file_processor.compute_global_stats(train_observations)
            all_train_observations[text_file]=train_observations
            observation_stats[text_file]={'mu': file_processor.total_mu, 'cov': file_processor.total_cov_matrix}
        if len(test_observations)>0:
                all_test_observations[text_file]=test_observations
        
    return observation_stats,all_train_observations,all_test_observations
    
if __name__ == "__main__":
    #collected_train_txt_file_lists=collect_files("train text files",".txt")
    #collected_train_excel_file_lists=collect_files("train excel files",".xlsx")

    
    hour_list=[0,4,8,12]
    total_list=[33,36,360,205]
    alive_list=[24,5,14,17]
    plot_hourly_prediction(hour_list,total_list,alive_list,240)
    
    
    '''
    file_processor=PreProcessingObservations()
    loaded_observations=file_processor.load_observations(collected_train_txt_file_lists[2])
    labeles_loaded=file_processor.load_labels(collected_train_excel_file_lists[3])
    name_matching=file_processor.match_txt_excel_file_prefix(collected_train_txt_file_lists[2],collected_train_excel_file_lists[3])
    print(f"{collected_train_excel_file_lists[3]}: {collected_train_txt_file_lists[2]}, {name_matching}")
    
    observation_stats,all_train_observations,all_test_observations=prepare_train_data(collected_train_txt_file_lists,collected_train_excel_file_lists)
    #dead_model,outlier_model_eval=run_outlier_model(collected_train_txt_file_lists,observation_stats,all_train_observations,all_test_observations,True)
    #dead_model,alive_model,bayesian_model_without_threshold=run_bayesian_model(collected_train_txt_file_lists,observation_stats,all_train_observations,all_test_observations,False,True)
    dead_model,alive_model,bayesian_model_with_threshold=run_bayesian_model(collected_train_txt_file_lists,observation_stats,all_train_observations,all_test_observations,True,True)
    
    
    infer_observation_stats={}
    all_infer_observations={}
    collected_tox_text_file_lists=collect_tox_file_from_user_input()
    file_processor=PreProcessingObservations()
    loaded_infer_observations=file_processor.load_observations(collected_tox_text_file_lists[0])
    labeled_infer_observations=file_processor.label_tox_observations(loaded_infer_observations)
    #labeled_infer_observations=file_processor.label_tox_observations_by_ranking(loaded_infer_observations,.4)
    print(f"{collected_tox_text_file_lists[0]} has {len(labeled_infer_observations)}")
    
    
    file_processor.compute_global_stats(labeled_infer_observations)
    infer_observation_stats[collected_tox_text_file_lists[0]]={'mu': file_processor.total_mu, 'cov': file_processor.total_cov_matrix}
    all_infer_observations[collected_tox_text_file_lists[0]]=labeled_infer_observations
    #infer_obs_predicted=infer_with_trained_outlier_model(collected_tox_text_file_lists,infer_observation_stats,all_infer_observations,dead_model,outlier_model_eval)
    infer_observations_pred=infer_with_trained_bayesian_model(collected_tox_text_file_lists,infer_observation_stats,all_infer_observations,dead_model,alive_model,bayesian_model_with_threshold)
    '''
    
    '''
    file_processor=PreProcessingObservations()
    for text_file, excel_file in zip(collected_train_txt_file_lists,collected_train_excel_file_lists):
        #print(f" txt file is: {text_file},{excel_file}")
        labeles_loaded=file_processor.load_labels(excel_file)
        tracking_observations=file_processor.load_observations(text_file)
        labeled_observations=file_processor.label_observations_by_expert_labels(text_file,tracking_observations,labeles_loaded)
        print(f"{text_file} has {len(tracking_observations)}")
        print(f"{excel_file} has {len(labeles_loaded)}")
        print(f"final labeled obs size is: {len(labeled_observations)}")
        first_key, first_value = next(iter(labeled_observations.items()))
        print(f"{first_key},{first_value}")
    '''
    
    