from driver_data_preprocessing import PreProcessingObservations
from driver_GridDisplacementGMM import GMMDisplacementModel
from GridBayesianModel import BayesianModel
from visualize_object_trajectory import plot_hourly_prediction,plot_confusion_matrix
from gmm_visualization import plot_gmm_overlay_grid


import numpy
import os
import math
from collections import Counter
from PIL import Image
from collections import Counter
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.mixture import GaussianMixture
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
    
def count_lables(curr_obs_dict):
    curr_obs_label_counter = Counter(data[TRUE_LABEL] for data in curr_obs_dict.values())
    return curr_obs_label_counter[MOTILE],curr_obs_label_counter[NOTMOTILE],

def prepare_train_data(collected_text_file_lists,collected_excel_file_lists):
    
    observation_stats ={}
    
    all_train_observations={}
    all_test_observations={}
    
    total_obs_size=0
    total_train, total_test = 0, 0
    motile_train, nonmotile_train = 0, 0
    motile_test, nonmotile_test = 0, 0
    
    
    for text_file, excel_file in zip(collected_train_txt_file_lists,collected_train_excel_file_lists):
        print(f" txt file is: {text_file},{excel_file}")
        file_processor=PreProcessingObservations()
        tracking_observations=file_processor.load_observations(text_file)
        labeles_loaded=file_processor.load_labels(excel_file)
        labeled_observations=file_processor.label_observations_by_expert_labels(text_file,excel_file,tracking_observations,labeles_loaded)
        
        #---File wise summary---
        curr_motile_obs,curr_non_motile_obs=count_lables(labeled_observations)
        print(f"{text_file} has {len(tracking_observations)}")
        print(f"{excel_file} has {len(labeles_loaded)}")
        print(f"final labeled obs size is: {len(labeled_observations)}") 
        print(f"it has {curr_motile_obs} motile and {curr_non_motile_obs} non-motile")      
        
        train_observations,test_observations=file_processor.prepare_train_test(labeled_observations,train_ratio=0.8)
        if len(train_observations) > 0:
        # Add prefix to avoid obj_id collisions across files
            for obj_id, obj_data in train_observations.items():
                all_train_observations[obj_id] = obj_data
    
        if len(test_observations) > 0:
            for obj_id, obj_data in test_observations.items():
                all_test_observations[obj_id] = obj_data
    
    # Summary
    print("\n=== Overall Summary ===")
    print(f"Total train observations: {len(all_train_observations)}")
    print(f"Total test observations: {len(all_test_observations)}")

    # Count labels
    train_motile = sum(1 for obj_data in all_train_observations.values() 
                   if obj_data[TRUE_LABEL] == MOTILE)
    train_non_motile = len(all_train_observations) - train_motile

    test_motile = sum(1 for obj_data in all_test_observations.values() 
                  if obj_data[TRUE_LABEL] == NOTMOTILE)
    test_non_motile = len(all_test_observations) - test_motile

    print(f"Train: {train_motile} motile, {train_non_motile} non-motile")
    print(f"Test: {test_motile} motile, {test_non_motile} non-motile")

    # Compute GLOBAL statistics on ALL training data
    print("\n=== Computing Global Statistics ===")
    global_processor = PreProcessingObservations()

    global_processor.compute_global_stats(all_train_observations)

    observation_stats = {
        'mu': global_processor.total_mu,
        'cov': global_processor.total_cov_matrix,
        'n_obs': global_processor.total_obs
    }

    print(f"Global mu: {observation_stats['mu']}")
    print(f"Global cov:\n{observation_stats['cov']}")
    print(f"Total train observations: {observation_stats['n_obs']}")
    
    return observation_stats,all_train_observations,all_test_observations

def normalizing_displacements(observations):

    tracking_only_obs = {obj_id: obj_data[TRACKING_DATA]for obj_id, obj_data in observations.items()}
    #gets the displacement sequence
    file_processor=PreProcessingObservations()
    curr_obs_displacements=file_processor.get_displacement_sequence(tracking_only_obs)
    file_processor.compute_global_stats(observations)
    
    global_mu = file_processor.total_mu  # [mean_dx, mean_dy]
    global_cov = file_processor.total_cov_matrix  # [[var_dx, cov], [cov, var_dy]]
    global_std = numpy.sqrt(numpy.diag(global_cov))
    
    normalized_displacements = {}
    for obj_id, displacements in curr_obs_displacements.items():
        if len(displacements) > 0:
            # Convert to numpy array
            disp_array = numpy.array(displacements)  # Shape: (n, 2)
            
            # Apply Z-score normalization
            normalized = (disp_array - global_mu) / global_std
            
            # Convert back to list
            normalized_displacements[obj_id] = normalized.tolist()
        else:
            print(f"Warning: {obj_id} has no displacements to normalize")
    
    return normalized_displacements
    
def flatten_displacements_dict(displacement_dict):
    """
    Convert displacement dictionary to flat numpy array.
    
    Parameters:
    - displacement_dict: {obj_id: [[dx1, dy1], [dx2, dy2], ...]}
    
    Returns:
    - flat_array: numpy array of shape (N, 2) with all displacements
    """
    all_displacements = []
    
    for obj_id, displacements in displacement_dict.items():
        all_displacements.extend(displacements)
    
    return numpy.array(all_displacements) 
    
def select_optimal_components(normalized_displacements, max_components=10, verbose=True):
    """
    Use AIC to select optimal number of GMM components.
        
    Parameters:
    - normalized_displacements: numpy array of shape (N, 2)
        
    Returns:
    - best_n_components: optimal number of components
    - aic_scores: list of AIC scores for each n_components tried
    """
        
    
    aic_scores = []
    n_samples = len(normalized_displacements)
    data = flatten_displacements_dict(normalized_displacements)
    if verbose:
        print(f"\n{'='*60}")
        print(f"AIC-based Component Selection")
        print(f"{'='*60}")
        print(f"Total samples: {n_samples}")
    
    # Adjust max_components based on available samples
    # Rule of thumb: at least 10 samples per component
    max_k = min(max_components, n_samples // 10)
    
    if max_k < 1:
        if verbose:
            print(f"Warning: Not enough samples ({n_samples}), using 1 component")
        return 1, [0], [0]
    
    if verbose:
        print(f"Testing 1 to {max_k} components...\n")
    
    aic_scores = []
    bic_scores = []
    
    # Try different numbers of components
    for n_components in range(1, max_k + 1):
        try:
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='full',
                max_iter=100,
                n_init=10,  # More initializations for stability
                random_state=42
            )
            gmm.fit(data)
            
            aic = gmm.aic(data)
            bic = gmm.bic(data)
            
            aic_scores.append(aic)
            bic_scores.append(bic)
            
            if verbose:
                print(f"  K={n_components:2d}: AIC={aic:10.2f}, BIC={bic:10.2f}")
        
        except Exception as e:
            if verbose:
                print(f"  K={n_components:2d}: Failed - {e}")
            aic_scores.append(float('inf'))
            bic_scores.append(float('inf'))
    
    # Select components with minimum AIC
    best_n_components = numpy.argmin(aic_scores) + 1
    best_aic = aic_scores[best_n_components - 1]
    
    # Also show BIC choice
    best_n_bic = numpy.argmin(bic_scores) + 1
    best_bic = bic_scores[best_n_bic - 1]
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Results:")
        print(f"  AIC selects: K={best_n_components} (AIC={best_aic:.2f})")
        print(f"  BIC selects: K={best_n_bic} (BIC={best_bic:.2f})")
        print(f"{'='*60}\n")
    
    return best_n_components, aic_scores, bic_scores

def get_only_tracking_data(curr_obs, type_of_filtering):   
    #tracking_only_obs = {obj_id: obj_data[TRACKING_DATA]for obj_id, obj_data in curr_obs.items()}
    filtered_tracking_obs={obj_id: obj_data[TRACKING_DATA] for obj_id, obj_data in curr_obs.items() if obj_data[TRUE_LABEL] == type_of_filtering}
    return filtered_tracking_obs

def training_parameters(training_motile_obs,training_nonmotile_obs,observation_stats):

    motile_GMM=GMMDisplacementModel()
    motile_GMM.set_normalization_params(observation_stats['mu'], observation_stats['cov'])
    curr_motile_displacements=motile_GMM.collect_displacements(training_motile_obs)
    curr_motile_normalized_displacements=motile_GMM.apply_normalization(curr_motile_displacements)
    motile_GMM.calculate_GMM_parameters(curr_motile_normalized_displacements)
    #train_motile_obs_probs_motile_gmm=motile_GMM.compute_probabilities(training_motile_obs)
    #train_nonmotile_obs_probs_motile_gmm=motile_GMM.compute_probabilities(training_nonmotile_obs)
    
    non_motile_GMM=GMMDisplacementModel()
    non_motile_GMM.set_normalization_params(observation_stats['mu'], observation_stats['cov'])
    curr_nonmotile_displacements=motile_GMM.collect_displacements(training_nonmotile_obs)
    curr_nonmotile_normalized_displacements=motile_GMM.apply_normalization(curr_nonmotile_displacements)
    non_motile_GMM.calculate_GMM_parameters(curr_nonmotile_normalized_displacements)
    #train_motile_obs_probs_nonmotile_gmm=non_motile_GMM.compute_probabilities(training_motile_obs)
    #train_nonmotile_obs_probs_nonmotile_gmm=non_motile_GMM.compute_probabilities(training_nonmotile_obs)
    
    
    return motile_GMM,non_motile_GMM

def probability_estimation(motile_gmm, non_motile_gmm, curr_obs):
    tracking_only_obs = {obj_id: obj_data[TRACKING_DATA]for obj_id, obj_data in curr_obs.items()}
    curr_obs_probs_motile_gmm=motile_GMM.compute_probabilities(tracking_only_obs)
    '''
    first_key = next(iter(curr_obs_probs_motile_gmm))
    print(f"First key: {first_key}")  # 'obj1'

    first_item_motile = next(iter(curr_obs_probs_motile_gmm.items()))
    print(f"First item: {first_item_motile}") 
    '''
    curr_obs_probs_non_motile_gmm=non_motile_GMM.compute_probabilities(tracking_only_obs)
    '''
    first_item_non_motile = next(iter(curr_obs_probs_non_motile_gmm.items()))
    print(f"First item: {first_item_non_motile}") 
    '''
    return curr_obs_probs_motile_gmm,curr_obs_probs_non_motile_gmm

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
    collected_train_txt_file_lists=collect_files("train text files",".txt")
    collected_train_excel_file_lists=collect_files("train excel files",".xlsx")
    observation_stats,all_train_observations,all_test_observations=prepare_train_data(collected_train_txt_file_lists,collected_train_excel_file_lists)
    '''
    normalized_displacements=normalizing_displacements(all_train_observations)
    best_k, aic_scores, bic_scores = select_optimal_components(normalized_displacements,max_components=10,verbose=True)
    print(best_k)
    '''
    training_motile_obs=get_only_tracking_data(all_train_observations,MOTILE)
    training_nonmotile_obs=get_only_tracking_data(all_train_observations,NOTMOTILE)
    #print(len(training_motile_obs),len(training_nonmotile_obs))
    motile_GMM,non_motile_GMM=training_parameters(training_motile_obs,training_nonmotile_obs,observation_stats)
    
    fig = plot_gmm_overlay_grid(motile_GMM,non_motile_GMM)
    plt.show()
    '''
    train_motile_probs, train_non_motile_probs=probability_estimation(motile_GMM, non_motile_GMM, all_train_observations)
    combined_probs_train=combine_dictionary_probs_labels(train_motile_probs, train_non_motile_probs,all_train_observations)
    
    bayesian_model_without_threshold=BayesianModel()  
    bayesian_model_without_threshold.calculate_prior(training_nonmotile_obs,training_motile_obs)
    train_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_probs_train)
    plot_confusion_matrix(train_probs_bayesin_model_without_threshold, "Train","Greens", "Bayesian")
    
    test_motile_obs=get_only_tracking_data(all_test_observations,MOTILE)
    test_nonmotile_obs=get_only_tracking_data(all_test_observations,NOTMOTILE)
    test_motile_probs, test_non_motile_probs=probability_estimation(motile_GMM, non_motile_GMM, all_test_observations)
    combined_probs_test=combine_dictionary_probs_labels(test_motile_probs, test_non_motile_probs,all_test_observations)
    test_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_probs_test)
    plot_confusion_matrix(test_probs_bayesin_model_without_threshold, "Test","Greens", "Bayesian")
    '''