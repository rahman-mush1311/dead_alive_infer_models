from driver_data_preprocessing import PreProcessingObservations
from driver_training_gmm_processer import prepare_train_test_setup
from feature_model_combined_processor import train_test_setup_for_combined_model
#from training_processor_combined_model import prepare_train_test_setup_for_combined

from driver_frame_stat_collector import TrackStatisticsCollector
from GridBayesianModel import BayesianModel

from driver_GridDisplacementGMM import GMMDisplacementModel
from visualize_object_trajectory import plot_confusion_matrix

import numpy
import os
import glob
import math
from collections import Counter
from PIL import Image
import matplotlib.pyplot
from collections import Counter
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import pprint
import matplotlib.pyplot as plt

TRUE_LABEL = "true_label"
PREDICTED_LABEL = "predicted_label"
LOG_PDFS="log_pdfs"
TRACKING_DATA = "tracking_data"
DEAD_PDFS="dead_log_sum_pdfs"
ALIVE_PDFS="alive_log_sum_pdfs"


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

def prepare_train_data(collected_text_file_lists,collected_excel_file_lists):
    
    observation_stats ={}
    
    all_train_observations={}
    all_test_observations={}
    
    
    for text_file, excel_file in zip(collected_train_txt_file_lists,collected_train_excel_file_lists):
        print(f" txt file is: {text_file},{excel_file}")
        file_processor=PreProcessingObservations()
        tracking_observations=file_processor.load_observations(text_file)
        labeles_loaded=file_processor.load_labels(excel_file)
        labeled_observations=file_processor.label_observations_by_expert_labels(text_file,excel_file,tracking_observations,labeles_loaded)      
      
        train_observations,test_observations=file_processor.prepare_train_test(labeled_observations,train_ratio=0.8)
        if len(train_observations)>0:
            #file_processor.compute_global_stats(train_observations)
            #all_train_observations[text_file]=train_observations
            #observation_stats[text_file]={'mu': file_processor.total_mu, 'cov': file_processor.total_cov_matrix}
            for obj_id, obj_data in train_observations.items():
                all_train_observations[obj_id] = obj_data
        if len(test_observations)>0:
            #all_test_observations[text_file]=test_observations
            for obj_id, obj_data in test_observations.items():
                all_test_observations[obj_id] = obj_data
    
    global_processor = PreProcessingObservations()

    global_processor.compute_global_stats(all_train_observations)

    observation_stats = {
        'mu': global_processor.total_mu,
        'cov': global_processor.total_cov_matrix,
        'n_obs': global_processor.total_obs
    }
    return observation_stats,all_train_observations,all_test_observations

def get_only_tracking_data(curr_obs, type_of_filtering):   
    #tracking_only_obs = {obj_id: obj_data[TRACKING_DATA]for obj_id, obj_data in curr_obs.items()}
    filtered_tracking_obs={obj_id: obj_data[TRACKING_DATA] for obj_id, obj_data in curr_obs.items() if obj_data[TRUE_LABEL] == type_of_filtering}
    return filtered_tracking_obs


def combine_dictionary_probs_labels(curr_obs_probs_motile_gmm,curr_obs_probs_non_motile_gmm,curr_obs):
    combined_log_probs = {}
    for obj_id in curr_obs:
        motile_entry = curr_obs_probs_motile_gmm[obj_id][LOG_PDFS]
        non_motile_entry = curr_obs_probs_non_motile_gmm[obj_id][LOG_PDFS]
        label=curr_obs[obj_id][TRUE_LABEL]
        combined_log_probs[obj_id] = {
                ALIVE_PDFS: motile_entry,
                DEAD_PDFS: non_motile_entry,
                TRUE_LABEL: label
            }
        #print(combined_log_probs[obj_id])
    return combined_log_probs
    
if __name__ == "__main__":
    #prepare_train_test_setup()
    #prepare_train_test_setup_for_combined()
    train_test_setup_for_combined_model()
    '''
    for i in range(5):
        collected_train_txt_file_lists=collect_files("train text files",".txt")
        collected_train_excel_file_lists=collect_files("train excel files",".xlsx")
        observation_stats,all_train_observations,all_test_observations=prepare_train_data(collected_train_txt_file_lists,collected_train_excel_file_lists)
    
        training_motile_obs=get_only_tracking_data(all_train_observations,MOTILE)
        training_nonmotile_obs=get_only_tracking_data(all_train_observations,NOTMOTILE)
    
        motile_GMM=GMMDisplacementModel()
        motile_GMM.set_normalization_params(observation_stats['mu'], observation_stats['cov'])
        curr_motile_displacements=motile_GMM.collect_displacements(training_motile_obs)
        curr_motile_normalized_displacements=motile_GMM.apply_normalization(curr_motile_displacements)
        motile_GMM.calculate_GMM_parameters(curr_motile_normalized_displacements,1)
    
        non_motile_GMM=GMMDisplacementModel()
        non_motile_GMM.set_normalization_params(observation_stats['mu'], observation_stats['cov'])
        curr_nonmotile_displacements=non_motile_GMM.collect_displacements(training_nonmotile_obs)
        curr_nonmotile_normalized_displacements=non_motile_GMM.apply_normalization(curr_nonmotile_displacements)
        non_motile_GMM.calculate_GMM_parameters(curr_nonmotile_normalized_displacements,0)
    
        tracking_train_obs = {obj_id: obj_data[TRACKING_DATA]for obj_id, obj_data in all_train_observations.items()}
        train_motile_probs=motile_GMM.compute_probabilities(tracking_train_obs)  
        train_non_motile_probs=non_motile_GMM.compute_probabilities(tracking_train_obs)
    
        combined_probs_train=combine_dictionary_probs_labels(train_motile_probs, train_non_motile_probs,all_train_observations)
    
        bayesian_model_without_threshold=BayesianModel()  
        bayesian_model_without_threshold.calculate_prior(training_nonmotile_obs,training_motile_obs)
        train_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_probs_train)
        train_acc, train_F1, train_precision, train_recall=plot_confusion_matrix(train_probs_bayesin_model_without_threshold, "Train","Greens", "Bayesian")
    
        tracking_test_obs = {obj_id: obj_data[TRACKING_DATA]for obj_id, obj_data in all_test_observations.items()}
        test_motile_probs=motile_GMM.compute_probabilities(tracking_test_obs)  
        test_non_motile_probs=non_motile_GMM.compute_probabilities(tracking_test_obs)
        #test_motile_probs, test_non_motile_probs=probability_estimation(motile_GMM, non_motile_GMM, labeled_test_observations)
        combined_probs_test=combine_dictionary_probs_labels(test_motile_probs, test_non_motile_probs,all_test_observations)
        test_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_probs_test)
        test_acc, test_F1, test_precision, test_recall=plot_confusion_matrix(test_probs_bayesin_model_without_threshold, "Test","Greens", "Bayesian")
    #dead_model,alive_model,bayesian_model_without_threshold=run_bayesian_model(collected_train_txt_file_lists,observation_stats,all_train_observations,all_test_observations,False,True)
    
    #stat_collector=TrackStatisticsCollector()
    #collector, all_data=stat_collector.collector_frame_stat_data() 
    #stat_collector.call_frame_stat_visualizor()
    
    '''