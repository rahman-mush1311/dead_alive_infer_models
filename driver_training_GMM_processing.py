from driver_data_preprocessing import PreProcessingObservations
from driver_GridDisplacement_GMM import GMMGridDisplacementModel
from GridOutlierModel import OutlierModelEvaluation 
from GridBayesianModel import BayesianModel

from visualize_object_trajectory import plot_confusion_matrix

import os
import numpy
import matplotlib.pyplot

TRACKING_DATA = "tracking_data"
TRUE_LABEL = "true_label"
PREDICTED_LABEL= "predicted_label"
LOG_PDFS="log_pdfs"
DEAD_PDFS="dead_log_sum_pdfs"
ALIVE_PDFS="alive_log_sum_pdfs"

MOVING=1
NOTMOVING=0

def get_sample_file_stats(curr_obs_stats):
    
    contains_valid_stats=False
    
    dx_norm=0.0
    dy_norm=0.0
    sx_norm=0.0
    sy_norm=0.0
    
    if len(curr_obs_stats)!=0:
        contains_valid_stats=True
        dx_norm, dy_norm = curr_obs_stats['mu']
        sx_norm, sy_norm = numpy.sqrt(numpy.diag(curr_obs_stats['cov']))
        
    return contains_valid_stats,dx_norm,dy_norm,sx_norm,sy_norm
    
def gmm_dead_model_training(collected_file_lists,observation_stats,train_observations):
    
    all_dead_grid_displacements = [[[] for _ in range(3)] for _ in range(3)]
    
    for file in collected_file_lists:
         
        if file in train_observations and observation_stats:
            curr_obs_stats= observation_stats[file]
            curr_train_obs=train_observations[file]
            
            contains_valid_stats, dx_norm, dy_norm, sx_norm, sy_norm = get_sample_file_stats(curr_obs_stats)
            filtered_curr_nonmoving_obs={obj_id: obj_data[TRACKING_DATA] for obj_id, obj_data in curr_train_obs.items() if obj_data[TRUE_LABEL] == NOTMOVING}
            if len(curr_obs_stats)!=0 and len(filtered_curr_nonmoving_obs)!=0:
                gmm_grid_displacement_model=GMMGridDisplacementModel() 
                curr_grid_norm_displacements=gmm_grid_displacement_model.calculate_displacements_grid_cell(filtered_curr_nonmoving_obs,curr_obs_stats['mu'],curr_obs_stats['cov'])
                #print(f"$$$$ dead model training for single file {file}")
                for row in range(3):
                    for col in range(3):
                        all_dead_grid_displacements[row][col].extend(curr_grid_norm_displacements[row][col])
                
            elif len(filtered_curr_nonmoving_obs)==0:
                print(f"!!!!!!!Warning!!!!!!!!:  after filtering {file} doesn't contain any dead examples {len(filtered_curr_nonmoving_obs)}.")
            else:
                print(f"!!!!!!!Warning!!!!!!!!: normalization content empty for {file} {len(curr_obs_stats)}.")
                
        elif file not in observation_stats:
            print(f"!!!!!!!Warning!!!!!!!!: normalization content empty for {file} {len(observation_stats[file])}.")
            continue
        else:
            print(f"!!!!!!!Warning!!!!!!!!:  {file} doesn't contain any dead examples {len(train_observations[file])}.")
            continue
    
    combined_dead_GMM_model=GMMGridDisplacementModel()
    combined_dead_GMM_model.calculate_parameters(all_dead_grid_displacements)
    '''
    for row in range(3):
        for col in range(3):
            cell_gmm=combined_dead_GMM_model.gmms[row][col]
            print(f"for {row}{col}:")
            if cell_gmm is not None:
                for i in range(cell_gmm.n_components):
                    mu = cell_gmm.means_[i]
                    cov = cell_gmm.covariances_[i]
                    weight = cell_gmm.weights_[i]
                    print(f"component {i} has weight of {weight}, mean of {mu} and cov of {cov}")
    '''
                    
    return combined_dead_GMM_model


def gmm_alive_model_training(collected_file_lists,observation_stats,train_observations):
    
    all_alive_grid_displacements = [[[] for _ in range(3)] for _ in range(3)]
    
    for file in collected_file_lists:
         
        if file in train_observations and observation_stats:
            curr_obs_stats= observation_stats[file]
            curr_train_obs=train_observations[file]
            
            contains_valid_stats, dx_norm, dy_norm, sx_norm, sy_norm = get_sample_file_stats(curr_obs_stats)
            filtered_curr_moving_obs={obj_id: obj_data[TRACKING_DATA] for obj_id, obj_data in curr_train_obs.items() if obj_data[TRUE_LABEL] == MOVING}
            if len(curr_obs_stats)!=0 and len(filtered_curr_moving_obs)!=0:
                gmm_grid_displacement_model=GMMGridDisplacementModel() 
                curr_grid_norm_displacements=gmm_grid_displacement_model.calculate_displacements_grid_cell(filtered_curr_moving_obs,curr_obs_stats['mu'],curr_obs_stats['cov'])
                #print(f"$$$$ dead model training for single file {file}")
                for row in range(3):
                    for col in range(3):
                        all_alive_grid_displacements[row][col].extend(curr_grid_norm_displacements[row][col])
                
            elif len(filtered_curr_nmoving_obs)==0:
                print(f"!!!!!!!Warning!!!!!!!!:  after filtering {file} doesn't contain any dead examples {len(filtered_curr_moving_obs)}.")
            else:
                print(f"!!!!!!!Warning!!!!!!!!: normalization content empty for {file} {len(curr_obs_stats)}.")
                
        elif file not in observation_stats:
            print(f"!!!!!!!Warning!!!!!!!!: normalization content empty for {file} {len(observation_stats[file])}.")
            continue
        else:
            print(f"!!!!!!!Warning!!!!!!!!:  {file} doesn't contain any dead examples {len(train_observations[file])}.")
            continue
    
    combined_alive_GMM_model=GMMGridDisplacementModel()
    combined_alive_GMM_model.calculate_parameters(all_alive_grid_displacements)
    '''
    for row in range(3):
        for col in range(3):
            cell_gmm=combined_dead_GMM_model.gmms[row][col]
            print(f"for {row}{col}:")
            if cell_gmm is not None:
                for i in range(cell_gmm.n_components):
                    mu = cell_gmm.means_[i]
                    cov = cell_gmm.covariances_[i]
                    weight = cell_gmm.weights_[i]
                    print(f"component {i} has weight of {weight}, mean of {mu} and cov of {cov}")
    '''
                    
    return combined_alive_GMM_model

def get_dictionary_of_tracking_data(curr_obs_for_probability_calculation):
    
    curr_tracking_obs={obj_id: obj_data[TRACKING_DATA] for obj_id, obj_data in curr_obs_for_probability_calculation.items()}
    
    return curr_tracking_obs
    
def computed_probability_with_labels(curr_log_pdf_dict,dis_prob_with_label,obs_dict_with_labels):
    """
    Combines log-probability values from a current dictionary into a master dictionary with true labels.    
    Parameters:
    - curr_log_pdf_dict: {obj_id: {LOG_PDFS: [...]}}, from one file
    - dis_prob_with_label: master dict accumulating all log PDFs and labels
    - obs_dict_with_labels: {obj_id: {obs: [...], true_labels}}, from one file
    
    Returns:
    - dis_prob_with_label: updated with new entries or extended values
    """
    
    for obj_id, values in curr_log_pdf_dict.items():
        if obj_id not in dis_prob_with_label:
            dis_prob_with_label[obj_id] = {} 
        
            dis_prob_with_label[obj_id][LOG_PDFS] = values[LOG_PDFS]
            dis_prob_with_label[obj_id][TRUE_LABEL] = obs_dict_with_labels[obj_id][TRUE_LABEL]

    return dis_prob_with_label
    
def calculate_class_probability(combined_model,collected_file_lists,observation_stats,curr_observations):
    
    displacement_probabilities_labeled={}
    
    for file in collected_file_lists:
         
        if file in curr_observations and observation_stats:
            curr_obs_stats= observation_stats[file]
            curr_obs_for_probability_calculation=curr_observations[file]
            
            curr_tracking_obs=get_dictionary_of_tracking_data(curr_obs_for_probability_calculation)            
            contains_valid_stats, dx_norm, dy_norm, sx_norm, sy_norm = get_sample_file_stats(curr_obs_stats)

            if contains_valid_stats and curr_tracking_obs:
                
                curr_log_pdf_dict= combined_model.compute_probabilities(curr_tracking_obs, dx_norm, dy_norm, sx_norm, sy_norm)      
                displacement_probabilities_labeled=computed_probability_with_labels(curr_log_pdf_dict,displacement_probabilities_labeled,curr_obs_for_probability_calculation)
                
            elif not curr_tracking_obs:
                print(f"!!!!!!!Warning!!!!!!!! for {file} doesn't contain any examples and obs size before filter is {len(curr_obs_for_probability_calculation)}.")
            else:
                print(f"!!!!!!!Warning!!!!!!!!: normalization content empty for {file} and contents are: {curr_obs_stats['mu']}, {curr_obs_stats['cov']}.")
               
        else:
            print(f"{file} contains {len(curr_obs_for_probability_calculation)} examples and normalization len is: {len(curr_obs_stats)}")
            continue
    
    return displacement_probabilities_labeled

def outlier_model_threshold_selection(train_probs_labeled):

    outlier_model_eval=OutlierModelEvaluation()
    
    window_sizes=[1,2,3,4,5,6,7,8,9,10]
    outlier_model_eval.evaluate_thresholds_window_sizes(train_probs_labeled, window_sizes)   
    train_probs_predicted_labeled=outlier_model_eval.predict_probabilities_dictionary_update(train_probs_labeled)
    
    
    return train_probs_predicted_labeled,outlier_model_eval

def combine_dictionary_dead_alive_probs(train_obs_probs_dead_model,train_obs_probs_alive_model):
    
    alive_train_obs_probs = {}
    dead_train_obs_probs = {}
    combined_log_probs = {}

    for obj_id in train_obs_probs_alive_model:
        moving_entry = train_obs_probs_alive_model[obj_id]
        non_moving_entry = train_obs_probs_dead_model[obj_id]
    
        if moving_entry[TRUE_LABEL] == non_moving_entry[TRUE_LABEL]:
            label=moving_entry[TRUE_LABEL]
            # Separate based on true label
            if label == MOVING:
                alive_train_obs_probs[obj_id] = moving_entry
            else:
                dead_train_obs_probs[obj_id] = non_moving_entry

            # Combine alive and dead log_pdfs for later classification
            combined_log_probs[obj_id] = {
                ALIVE_PDFS: moving_entry[LOG_PDFS],
                DEAD_PDFS: non_moving_entry[LOG_PDFS],
                TRUE_LABEL: label
            }
        else:
            print(f"!!!WARNING!!! {obj_id} have mismatching true labels, the alive model calculated dictionary has {moving_entry[TRUE_LABEL]} dead {non_moving_entry[TRUE_LABEL]}")
    return dead_train_obs_probs,alive_train_obs_probs,combined_log_probs

def get_obs_labeled_by_prediction(all_infer_obs,infer_probs_predicted_labeled):
    
    infer_obs_labeled={}
    for file_name,objects in all_infer_obs.items():     
        for obj_id, obj_data in objects.items():
            infer_obs_labeled[obj_id] = {
            TRACKING_DATA: obj_data[TRACKING_DATA],
            TRUE_LABEL: obj_data[TRUE_LABEL],
            PREDICTED_LABEL: infer_probs_predicted_labeled[obj_id][PREDICTED_LABEL]   
        }
    return infer_obs_labeled
    
def infer_with_GMM_bayesian_model(collect_infer_file_lists,infer_obs_stats,all_infer_obs,dead_model,alive_model,bayesian_model_without_threshold):

    infer_obs_probs_dead_model=calculate_class_probability(dead_model,collect_infer_file_lists,infer_obs_stats,all_infer_obs)
    infer_obs_probs_alive_model=calculate_class_probability(alive_model,collect_infer_file_lists,infer_obs_stats,all_infer_obs)
    
    dead_infer_obs_probs,alive_infer_obs_probs,combined_infer_obs_probs=combine_dictionary_dead_alive_probs(infer_obs_probs_dead_model,infer_obs_probs_alive_model)
     
    #bayesian_model_without_threshold.calculate_prior(dead_infer_obs_probs,alive_infer_obs_probs)
    infer_probs_labeled_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_infer_obs_probs)
    plot_confusion_matrix(infer_probs_labeled_bayesin_model_without_threshold, "Infer","Oranges", "Bayesian")
    #all_infer_obs_labeled=get_obs_labeled_by_prediction(all_infer_obs,infer_probs_labeled_bayesin_model_without_threshold)
    return 

def infer_with_GMM_outlier_model(collect_infer_file_lists,infer_obs_stats,all_infer_obs,dead_model,outlier_model_eval):
    
    infer_obs_probs_dead_model=calculate_class_probability(dead_model,collect_infer_file_lists,infer_obs_stats,all_infer_obs)
    infer_probs_predicted_for_dead_model=outlier_model_eval.predict_probabilities_dictionary_update(infer_obs_probs_dead_model)
    plot_confusion_matrix(infer_probs_predicted_for_dead_model,  "Infer","Oranges", "Bayesian")