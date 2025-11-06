
from driver_GridFeatureModel import GridFeatureModel
from GridBayesianModel import BayesianModel

from visualize_object_trajectory import plot_confusion_matrix

import os
import numpy
import matplotlib.pyplot

TRUE_LABEL = "true_label"
LOG_PDFS="log_pdfs"
PREDICTED_LABEL= "predicted_label"
DEAD_PDFS="dead_log_sum_pdfs"
ALIVE_PDFS="alive_log_sum_pdfs"
TRACKING_DATA = "tracking_data"

TRAIN="train"
INFER="infer"

MOTILE=1
NOTMOTILE=0   

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
    
def dead_model_training(collected_file_lists,observation_stats,train_observations):

    dead_models_params = {}
    
    for file in collected_file_lists:
         
        if file in train_observations and observation_stats:
            curr_obs_stats= observation_stats[file]
            curr_train_obs=train_observations[file]
            
            contains_valid_stats, dx_norm, dy_norm, sx_norm, sy_norm = get_sample_file_stats(curr_obs_stats)
            filtered_curr_nonmoving_obs={obj_id: obj_data[TRACKING_DATA] for obj_id, obj_data in curr_train_obs.items() if obj_data[TRUE_LABEL] == NOTMOTILE}
            if len(curr_obs_stats)!=0 and len(filtered_curr_nonmoving_obs)!=0:
                grid_feature_model=GridFeatureModel() 
                grid_feature_model.total_mu=curr_obs_stats['mu']
                grid_feature_model.total_cov_matrix=curr_obs_stats['cov']
                print(f"$$$$ dead model training for single file {file}")
                curr_grid_features= grid_feature_model.calculate_displacements(filtered_curr_nonmoving_obs)
                curr_grid_model_parameters= grid_feature_model.calculate_parameters(curr_grid_features)
        
                dead_models_params[file] =  grid_feature_model
                
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
                    
    return dead_models_params

def alive_model_training(collected_file_lists,observation_stats,train_observations):

    alive_models_params = {}
    
    for file in collected_file_lists:
        if file not in train_observations:
            print(f"!!!!!!!Warning!!!!!!!!: {file} not found in train_observations.")
            continue  
        else:
            curr_obs_stats= observation_stats[file]
            curr_train_obs=train_observations[file]
           
            filtered_curr_moving_obs={obj_id: obj_data[TRACKING_DATA] for obj_id, obj_data in curr_train_obs.items() if obj_data[TRUE_LABEL] == MOTILE}
            if len(curr_obs_stats)!=0 and len(filtered_curr_moving_obs)!=0:
                grid_feature_model=GridFeatureModel() 
                grid_feature_model.total_mu=curr_obs_stats['mu']
                grid_feature_model.total_cov_matrix=curr_obs_stats['cov']
                print(f"$$$$ alive model training for single file {file}")
                curr_grid_features=grid_feature_model.calculate_displacements(filtered_curr_moving_obs)
                grid_feature_model.calculate_parameters(curr_grid_features)
        
                alive_models_params[file] = grid_feature_model
            else:
                if len(curr_obs_stats)==0:
                    print(f"!!!!!!!Warning!!!!!!!!: normalization content empty for {file} {len(curr_obs_stats)}.")
                else:
                    print(f"!!!!!!!Warning!!!!!!!!:  {file} doesn't contain any moving examples {len(filtered_curr_moving_obs)}.")
                    
    return alive_models_params

def combine_trained_models(collected_file_lists, curr_models_params):
    
    combined_model = GridFeatureModel()
    
    # Track the models
    calculated_models = []
    valid_file_size=0
    
    for file in collected_file_lists:    
        if file not in curr_models_params:
            print(f"!!!!!Warning: {file} doesn't have parameters to combine in current_models_params!!!!!!")
            continue  
        else:
            calculated_models.append(curr_models_params[file])
            valid_file_size+=1

    if valid_file_size>0:
        combined_model = combined_model.add_models(*calculated_models)
        #print(f"models to combine {valid_file_size} and models stored {len(calculated_models)}\n"
              #f"combined models stats are: {combined_model.mu}, {combined_model.cov_matrix}")             
    else:
        print(f"!!!!WARNING!!!! no valid models found {valid_file_size} and model size is {len(calculated_models)}")
    return combined_model

def get_dictionary_of_tracking_data(curr_obs_for_probability_calculation):
    
    curr_tracking_obs={obj_id: obj_data[TRACKING_DATA] for obj_id, obj_data in curr_obs_for_probability_calculation.items()}
    
    return curr_tracking_obs
    
def calculate_class_probability(combined_model,collected_file_lists,observation_stats,curr_observations):
    
    feature_probabilities_labeled={}
    
    for file in collected_file_lists:
         
        if file in curr_observations and observation_stats:
            curr_obs_stats= observation_stats[file]
            curr_obs_for_probability_calculation=curr_observations[file]
            
            curr_tracking_obs=get_dictionary_of_tracking_data(curr_obs_for_probability_calculation)            
            contains_valid_stats, dx_norm, dy_norm, sx_norm, sy_norm = get_sample_file_stats(curr_obs_stats)

            if contains_valid_stats and curr_tracking_obs:
                      
                calculator = GridFeatureModel()
                calculator.mu = combined_model.mu
                calculator.cov_matrix = combined_model.cov_matrix
                calculator.n = combined_model.n
                
                curr_log_pdf_dict= calculator.compute_probabilities(curr_tracking_obs, dx_norm, dy_norm, sx_norm, sy_norm)      
                feature_probabilities_labeled=computed_probability_with_labels(curr_log_pdf_dict,feature_probabilities_labeled,curr_obs_for_probability_calculation)
                
            elif not curr_tracking_obs:
                print(f"!!!!!!!Warning!!!!!!!! for {file} doesn't contain any examples and obs size before filter is {len(curr_obs_for_probability_calculation)}.")
            else:
                print(f"!!!!!!!Warning!!!!!!!!: normalization content empty for {file} and contents are: {curr_obs_stats['mu']}, {curr_obs_stats['cov']}.")
               
        else:
            print(f"{file} contains {len(curr_obs_for_probability_calculation)} examples and normalization len is: {len(curr_obs_stats)}")
            continue
    
    return feature_probabilities_labeled 

def computed_probability_with_labels(curr_log_pdf_dict,feature_prob_with_label,obs_dict_with_labels):
    """
    Combines log-probability values from a current dictionary into a master dictionary with true labels.    
    Parameters:
    - curr_log_pdf_dict: {obj_id: {LOG_PDFS: [...]}}, from one file
    - feature_prob_with_label: master dict accumulating all log PDFs and labels
    - obs_dict_with_labels: {obj_id: {obs: [...], true_labels}}, from one file
    
    Returns:
    - feature_prob_with_label: updated with new entries or extended values
    """
    
    for obj_id, values in curr_log_pdf_dict.items():
        if obj_id not in feature_prob_with_label:
            feature_prob_with_label[obj_id] = {} 
        
        feature_prob_with_label[obj_id][LOG_PDFS] = values[LOG_PDFS]
        feature_prob_with_label[obj_id][TRUE_LABEL] = obs_dict_with_labels[obj_id][TRUE_LABEL]

    return feature_prob_with_label
    
def combine_dictionary_nonmotile_motile_probs(train_obs_probs_dead_model,train_obs_probs_alive_model):
    
    alive_train_obs_probs = {}
    dead_train_obs_probs = {}
    combined_log_probs = {}

    for obj_id in train_obs_probs_alive_model:
        moving_entry = train_obs_probs_alive_model[obj_id]
        non_moving_entry = train_obs_probs_dead_model[obj_id]
    
        if moving_entry[TRUE_LABEL] == non_moving_entry[TRUE_LABEL]:
            label=moving_entry[TRUE_LABEL]
            # Separate based on true label
            if label == MOTILE:
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
    
def run_bayesian_model(collected_file_lists,obs_stats,all_train_obs,all_test_obs,test_performance):
    
    
    dead_model_params=dead_model_training(collected_file_lists,obs_stats,all_train_obs)
    print(f"$$$$Bayesian combined dead model training $$$$$$")
    dead_model=combine_trained_models(collected_file_lists, dead_model_params)
    
    alive_model_params=alive_model_training(collected_file_lists,obs_stats,all_train_obs)
    print(f"$$$$Bayesian combined alive model training $$$$$$")
    alive_model=combine_trained_models(collected_file_lists, alive_model_params)
    
    train_obs_probs_dead_model=calculate_class_probability(dead_model,collected_file_lists,obs_stats,all_train_obs)
    train_obs_probs_alive_model=calculate_class_probability(alive_model,collected_file_lists,obs_stats,all_train_obs)
    
    dead_train_obs_probs,alive_train_obs_probs,combined_obs_probs=combine_dictionary_nonmotile_motile_probs(train_obs_probs_dead_model,train_obs_probs_alive_model)
    
    bayesian_model_without_threshold=BayesianModel()  
    bayesian_model_without_threshold.calculate_prior(dead_train_obs_probs,alive_train_obs_probs)
    train_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_obs_probs)
    plot_confusion_matrix(train_probs_bayesin_model_without_threshold, "Train","Greens", "Bayesian")
    
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$TESTING WITHOUT MARGIN###############################
    if test_performance==True:
        test_obs_probs_dead_model=calculate_class_probability(dead_model,collected_file_lists,obs_stats,all_test_obs)
        test_obs_probs_alive_model=calculate_class_probability(alive_model,collected_file_lists,obs_stats,all_test_obs)
        dead_train_obs_probs,alive_train_obs_probs,combined_test_obs_probs=combine_dictionary_nonmotile_motile_probs(test_obs_probs_dead_model,test_obs_probs_alive_model)
        test_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_test_obs_probs)
        plot_confusion_matrix(test_probs_bayesin_model_without_threshold, "Test","Greens", "Bayesian")
    else:
        print(f"user doesn't want the model to see the test set performance")
       
    return dead_model,alive_model,bayesian_model_without_threshold
    '''
    return dead_model,alive_model
    '''