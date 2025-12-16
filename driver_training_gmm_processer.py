from driver_data_preprocessing import PreProcessingObservations
from GridBayesianModel import BayesianModel
from driver_GridDisplacementGMM import GMMDisplacementModel
from driver_GridDisplacementModel import GridDisplacementModel

from gmm_visualization import plot_gmm_overlay_grid
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



def collect_files_from_nested_structure(base_dir):
    """
    Collect text and excel files from nested folder structure.
    Files are paired by position (like your existing zip() approach).
    
    Parameters:
    -----------
    base_dir : str
        Path to main folder containing population subfolders
        
    Returns:
    --------
    dict : {
        'day-old': {'text': [...], 'excel': [...]},
        'week-old': {'text': [...], 'excel': [...]},
        'mixed-organics': {'text': [...], 'excel': [...]}
    }
    """
    
    if not os.path.exists(base_dir):
        raise ValueError(f"Base directory does not exist: {base_dir}")
    
    populations = ['day-old', 'week-old', 'mixed-organics']
    
    all_text_files = []
    all_excel_files = []
    population_map = {}
    
    print("="*70)
    print("COLLECTING FILES FROM ALL POPULATIONS")
    print("="*70)
    print(f"Base directory: {base_dir}\n")
    
    for pop in populations:
        pop_dir = os.path.join(base_dir, pop)
        
        if not os.path.exists(pop_dir):
            print(f"WARNING: Population folder not found: {pop_dir}")
            continue
        
        text_dir = os.path.join(pop_dir, 'text')
        excel_dir = os.path.join(pop_dir, 'excel')
        
        text_files = []
        excel_files = []
        
        if os.path.exists(text_dir):
            text_files = sorted(glob.glob(os.path.join(text_dir, '*.txt')))
        
        if os.path.exists(excel_dir):
            excel_files = sorted(glob.glob(os.path.join(excel_dir, '*.xlsx')))
        
        # Verify same number of files
        if len(text_files) != len(excel_files):
            print(f"WARNING: {pop} has {len(text_files)} text files but {len(excel_files)} excel files!")
            min_len = min(len(text_files), len(excel_files))
            text_files = text_files[:min_len]
            excel_files = excel_files[:min_len]
        
        print(f"{pop}:")
        print(f"  Found {len(text_files)} file pairs")
        
        # Add to unified lists
        for txt, xls in zip(text_files, excel_files):
            all_text_files.append(txt)
            all_excel_files.append(xls)
            population_map[txt] = pop
            population_map[xls] = pop
            print(f"    - {os.path.basename(txt)} + {os.path.basename(xls)}")
        print()
    
    print(f"\nTOTAL FILES ACROSS ALL POPULATIONS:")
    print(f"  {len(all_text_files)} file pairs")
    print("="*70)
    
    return {
        'all_text': all_text_files,
        'all_excel': all_excel_files,
        'population_map': population_map
    }


def create_unified_lovo_cv_splits(collected_data):
    """
    Create LOVO-CV splits across ALL populations.
    
    Each fold:
    - Training: ALL files except one (from all populations)
    - Testing: ONE file (from any population)
    
    Parameters:
    -----------
    collected_data : dict
        Output from collect_all_files_unified()
        
    Returns:
    --------
    list of dicts : [
        {
            'fold': 1,
            'train_text': [...],      # Files from ALL populations
            'train_excel': [...],
            'test_text': 'path.txt',
            'test_excel': 'path.xlsx',
            'test_population': 'day-old'  # Which population test file is from
        },
        ...
    ]
    """
    
    all_text = collected_data['all_text']
    all_excel = collected_data['all_excel']
    pop_map = collected_data['population_map']
    
    n_total_files = len(all_text)
    
    if n_total_files < 2:
        raise ValueError(f"Need at least 2 files total, got {n_total_files}")
    
    print("\n" + "="*70)
    print("CREATING UNIFIED LOVO-CV SPLITS")
    print("="*70)
    print(f"Total files: {n_total_files}")
    print(f"Number of folds: {n_total_files} (leave-one-out)")
    print()
    
    folds = []
    
    # Count files per population for display
    pop_counts = {}
    for txt in all_text:
        pop = pop_map[txt]
        pop_counts[pop] = pop_counts.get(pop, 0) + 1
    
    print("Files per population:")
    for pop, count in sorted(pop_counts.items()):
        print(f"  {pop}: {count} files")
    print()
    
    # Create folds
    for i in range(n_total_files):
        # Test files are the i-th files
        test_text = all_text[i]
        test_excel = all_excel[i]
        test_population = pop_map[test_text]
        
        # Train files are ALL except the i-th files
        train_text = [all_text[j] for j in range(n_total_files) if j != i]
        train_excel = [all_excel[j] for j in range(n_total_files) if j != i]
        
        # Count populations in training set
        train_pop_counts = {}
        for txt in train_text:
            pop = pop_map[txt]
            train_pop_counts[pop] = train_pop_counts.get(pop, 0) + 1
        
        fold = {
            'fold': i + 1,
            'train_text': train_text,
            'train_excel': train_excel,
            'test_text': test_text,
            'test_excel': test_excel,
            'test_population': test_population
        }
        
        folds.append(fold)
        
        # Display
        train_pop_str = ", ".join([f"{p}:{c}" for p, c in sorted(train_pop_counts.items())])
        print(f"Fold {i+1:2d}: Train[{len(train_text)} files: {train_pop_str}] | "
              f"Test[{os.path.basename(test_text)} from {test_population}]")
    
    print("\n" + "="*70)
    
    return folds

def count_lables(curr_obs_dict):
    curr_obs_label_counter = Counter(data[TRUE_LABEL] for data in curr_obs_dict.values())
    return curr_obs_label_counter[MOTILE],curr_obs_label_counter[NOTMOTILE]
    
def prepare_train_data(fold_train_text_file_lists,fold_train_excel_file_lists):
    
    all_train_observations={}
    train_observation_stats ={}
    motile_train, nonmotile_train = 0, 0

    for text_file, excel_file in zip(fold_train_text_file_lists,fold_train_excel_file_lists):
        #print(f" txt file is: {text_file},{excel_file}")
        file_processor=PreProcessingObservations()
        tracking_observations=file_processor.load_observations(text_file)
        labeles_loaded=file_processor.load_labels(excel_file)
        labeled_observations=file_processor.label_observations_by_expert_labels(text_file,excel_file,tracking_observations,labeles_loaded)
        
        curr_motile_obs,curr_non_motile_obs=count_lables(labeled_observations)
        for obj_id, obj_data in labeled_observations.items():
                all_train_observations[obj_id] = obj_data
      
    train_motile = sum(1 for obj_data in all_train_observations.values() 
                   if obj_data[TRUE_LABEL] == MOTILE)
    train_non_motile = len(all_train_observations) - train_motile
    
    print("\n=== Overall Summary ===")
    print(f"Total train observations: {len(all_train_observations)}")
    print(f"Train: {train_motile} motile, {train_non_motile} non-motile")
    
    # Compute GLOBAL statistics on ALL training data
    print("\n=== Computing Global Statistics ===")
    global_processor = PreProcessingObservations()

    global_processor.compute_global_stats(all_train_observations)

    train_observation_stats = {
        'mu': global_processor.total_mu,
        'cov': global_processor.total_cov_matrix,
        'n_obs': global_processor.total_obs
    }

    print(f"Global mu: {train_observation_stats['mu']}")
    print(f"Global cov:\n{train_observation_stats['cov']}")
    print(f"Total train observations: {train_observation_stats['n_obs']}")
    
    return train_observation_stats,all_train_observations, len(all_train_observations), train_motile, train_non_motile
    

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
    
    non_motile_GMM=GMMDisplacementModel()
    non_motile_GMM.set_normalization_params(observation_stats['mu'], observation_stats['cov'])
    curr_nonmotile_displacements=non_motile_GMM.collect_displacements(training_nonmotile_obs)
    curr_nonmotile_normalized_displacements=non_motile_GMM.apply_normalization(curr_nonmotile_displacements)
    non_motile_GMM.calculate_GMM_parameters(curr_nonmotile_normalized_displacements)
    
    
    return motile_GMM,non_motile_GMM

def mgd_training_parameters(training_motile_obs,training_nonmotile_obs,observation_stats):
    motile_mgd=GridDisplacementModel()
    motile_mgd.set_normalization_params(observation_stats['mu'], observation_stats['cov'])
    curr_motile_displacements=motile_mgd.calculate_displacements(training_motile_obs)
    curr_motile_normalized_displacements=motile_mgd.apply_normalization(curr_motile_displacements)
    motile_mgd.calculate_parameters(curr_motile_normalized_displacements)
   
    
    non_motile_mgd=GridDisplacementModel()
    non_motile_mgd.set_normalization_params(observation_stats['mu'], observation_stats['cov'])
    curr_nonmotile_displacements=non_motile_mgd.calculate_displacements(training_nonmotile_obs)
    curr_nonmotile_normalized_displacements=non_motile_mgd.apply_normalization(curr_nonmotile_displacements)
    non_motile_mgd.calculate_parameters(curr_nonmotile_normalized_displacements)
    
    return motile_mgd, non_motile_mgd

def probability_estimation(motile_gmm, non_motile_gmm, curr_obs):
    tracking_only_obs = {obj_id: obj_data[TRACKING_DATA]for obj_id, obj_data in curr_obs.items()}
    curr_obs_probs_motile_gmm=motile_gmm.compute_probabilities(tracking_only_obs)
    '''
    first_key = next(iter(curr_obs_probs_motile_gmm))
    print(f"First key: {first_key}")  # 'obj1'

    first_item_motile = next(iter(curr_obs_probs_motile_gmm.items()))
    print(f"First item: {first_item_motile}") 
    '''
    curr_obs_probs_non_motile_gmm=non_motile_gmm.compute_probabilities(tracking_only_obs)
    '''
    first_item_non_motile = next(iter(curr_obs_probs_non_motile_gmm.items()))
    print(f"First item: {first_item_non_motile}") 
    '''
    return curr_obs_probs_motile_gmm,curr_obs_probs_non_motile_gmm

def probability_estimation_with_mgd(motile_mgd, non_motile_mgd, curr_obs):
    tracking_only_obs = {obj_id: obj_data[TRACKING_DATA]for obj_id, obj_data in curr_obs.items()}
    curr_obs_probs_motile_mgd=motile_mgd.compute_probabilities(tracking_only_obs)
    curr_obs_probs_non_motile_mgd=non_motile_mgd.compute_probabilities(tracking_only_obs)
   
    return curr_obs_probs_motile_mgd,curr_obs_probs_non_motile_mgd

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

def evaluate_gmm_models_with_test_data(fold_test_text_file,fold_test_excel_file,motile_GMM, non_motile_GMM,bayesian_model_without_threshold):
    
    file_processor=PreProcessingObservations()
    tracking_test_observations=file_processor.load_observations(fold_test_text_file)
    labeles_test_loaded=file_processor.load_labels(fold_test_excel_file)
    labeled_test_observations=file_processor.label_observations_by_expert_labels(fold_test_text_file,fold_test_excel_file,tracking_test_observations,labeles_test_loaded)
    
    test_motile = sum(1 for obj_data in labeled_test_observations.values() 
                   if obj_data[TRUE_LABEL] == MOTILE)
    test_non_motile = len(labeled_test_observations) - test_motile
    
    file_processor.compute_global_stats(labeled_test_observations)
    motile_GMM.set_normalization_params(file_processor.total_mu, file_processor.total_cov_matrix)
    non_motile_GMM.set_normalization_params(file_processor.total_mu, file_processor.total_cov_matrix)
    
    test_motile_probs, test_non_motile_probs=probability_estimation(motile_GMM, non_motile_GMM, labeled_test_observations)
    combined_probs_test=combine_dictionary_probs_labels(test_motile_probs, test_non_motile_probs,labeled_test_observations)
    test_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_probs_test)
    test_acc, test_F1, test_precision, test_recall=plot_confusion_matrix(test_probs_bayesin_model_without_threshold, "Test","Greens", "Bayesian")
    
    return test_acc, test_F1, test_precision, test_recall, len(labeled_test_observations), test_motile, test_non_motile
    
def estimate_evaluate_gmm_models_with_train_data(train_text_files,train_excel_files):
    train_observation_stats,all_train_observations,train_obs, train_motile, train_non_motile=prepare_train_data(train_text_files,train_excel_files)
    training_motile_obs=get_only_tracking_data(all_train_observations,MOTILE)
    training_nonmotile_obs=get_only_tracking_data(all_train_observations,NOTMOTILE)
            
    motile_GMM,non_motile_GMM=training_parameters(training_motile_obs,training_nonmotile_obs,train_observation_stats)
    train_motile_probs, train_non_motile_probs=probability_estimation(motile_GMM, non_motile_GMM, all_train_observations)
    combined_probs_train=combine_dictionary_probs_labels(train_motile_probs, train_non_motile_probs,all_train_observations)
    
    bayesian_model_without_threshold=BayesianModel()  
    bayesian_model_without_threshold.calculate_prior(training_nonmotile_obs,training_motile_obs)
    train_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_probs_train)
    train_acc, train_F1, train_precision, train_recall=plot_confusion_matrix(train_probs_bayesin_model_without_threshold, "Train","Greens", "Bayesian")
    
    return motile_GMM,non_motile_GMM,bayesian_model_without_threshold, train_acc, train_F1, train_precision, train_recall, train_obs, train_motile, train_non_motile

def estimate_evaluate_mgd_models_with_train_data(train_text_files,train_excel_files):
    train_observation_stats,all_train_observations,train_obs, train_motile, train_non_motile=prepare_train_data(train_text_files,train_excel_files)
    training_motile_obs=get_only_tracking_data(all_train_observations,MOTILE)
    training_nonmotile_obs=get_only_tracking_data(all_train_observations,NOTMOTILE)
            
    motile_mgd,non_motile_mgd=mgd_training_parameters(training_motile_obs,training_nonmotile_obs,train_observation_stats)
    train_motile_probs, train_non_motile_probs=probability_estimation_with_mgd(motile_mgd,non_motile_mgd, all_train_observations)
    combined_probs_train=combine_dictionary_probs_labels(train_motile_probs, train_non_motile_probs,all_train_observations)
    
    bayesian_model_without_threshold=BayesianModel()  
    bayesian_model_without_threshold.calculate_prior(training_nonmotile_obs,training_motile_obs)
    train_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_probs_train)
    train_acc, train_F1, train_precision, train_recall=plot_confusion_matrix(train_probs_bayesin_model_without_threshold, "Train","Greens", "Bayesian")
    
    return motile_mgd,non_motile_mgd,bayesian_model_without_threshold, train_acc, train_F1, train_precision, train_recall, train_obs, train_motile, train_non_motile

def evaluate_mgd_models_with_test_data(fold_test_text_file,fold_test_excel_file,motile_mgd, non_motile_mgd,bayesian_model_without_threshold):
    
    file_processor=PreProcessingObservations()
    tracking_test_observations=file_processor.load_observations(fold_test_text_file)
    labeles_test_loaded=file_processor.load_labels(fold_test_excel_file)
    labeled_test_observations=file_processor.label_observations_by_expert_labels(fold_test_text_file,fold_test_excel_file,tracking_test_observations,labeles_test_loaded)
    
    test_motile = sum(1 for obj_data in labeled_test_observations.values() 
                   if obj_data[TRUE_LABEL] == MOTILE)
    test_non_motile = len(labeled_test_observations) - test_motile
    
    file_processor.compute_global_stats(labeled_test_observations)
    motile_mgd.set_normalization_params(file_processor.total_mu, file_processor.total_cov_matrix)
    non_motile_mgd.set_normalization_params(file_processor.total_mu, file_processor.total_cov_matrix)
    
    test_motile_probs, test_non_motile_probs=probability_estimation(motile_mgd, non_motile_mgd, labeled_test_observations)
    combined_probs_test=combine_dictionary_probs_labels(test_motile_probs, test_non_motile_probs,labeled_test_observations)
    test_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_probs_test)
    test_acc, test_F1, test_precision, test_recall=plot_confusion_matrix(test_probs_bayesin_model_without_threshold, "Test","Greens", "Bayesian")
    
    return test_acc, test_F1, test_precision, test_recall, len(labeled_test_observations), test_motile, test_non_motile
    
def prepare_train_test_setup():
    
    user_base_dir = input(f"Enter the base directory where your data folder is located: ")
    if not os.path.isdir(user_base_dir):
        raise ValueError(f"{user_base_dir} is not a valid directory")
    else:
        collected_file_lists=collect_files_from_nested_structure(user_base_dir)
        # Example 2: Create LOVO-CV splits
        print("\n### EXAMPLE 2: Create LOVO-CV Splits ###\n")
    
        splits = create_unified_lovo_cv_splits(collected_file_lists)
        #pprint.pprint(splits)
        all_results = []
        
        for fold_data in splits:
            fold_num = fold_data['fold']
            test_pop = fold_data['test_population']
            #print(test_pop)
            train_text_files = fold_data['train_text']
            train_excel_files= fold_data['train_excel']
            
            # Get the filename only for cleaner output
            test_text_file = fold_data['test_text']
            test_excel_file = fold_data['test_excel']
        
            print(f"\nFold {fold_num}:")
            #motile_GMM,non_motile_GMM,bayesian_model_without_threshold, train_acc, train_F1, train_precision, train_recall, train_obs, train_motile, train_non_motile=estimate_evaluate_gmm_models_with_train_data(train_text_files,train_excel_files)
            motile_mgd,non_motile_mgd,bayesian_model_without_threshold, train_acc, train_F1, train_precision, train_recall, train_obs, train_motile, train_non_motile=estimate_evaluate_mgd_models_with_train_data(train_text_files,train_excel_files)
            print(f"Train size: {train_obs}, Train_motile: {train_motile}, Train non-motile: {train_non_motile}")
            print(f"Train Acc: {train_acc}, Train F1: {train_F1}, Train Recall: {train_recall}, Train Precision: {train_precision}")
           
            #test_acc, test_F1, test_precision, test_recall,test_obs, test_motile, test_non_motile=evaluate_gmm_models_with_test_data(test_text_file,test_excel_file,motile_GMM, non_motile_GMM,bayesian_model_without_threshold)
            test_acc, test_F1, test_precision, test_recall,test_obs, test_motile, test_non_motile=evaluate_mgd_models_with_test_data(test_text_file,test_excel_file,motile_mgd, non_motile_mgd,bayesian_model_without_threshold)
            print(f"Test size: {test_obs}, Test_motile: {test_motile}, Test non-motile: {test_non_motile}")
            print(f"Test Acc: {test_acc}, Test F1: {test_F1}, Test Recall: {test_recall}, Test Precision: {test_precision}")
            fold_results = {
            'fold_number': fold_num,
            
            # Observation Counts
            'total_train_obs': train_obs,
            'total_test_obs': test_obs,
            'motile_train_obs_size': train_motile,
            'non_motile_train_obs_size': train_non_motile,
            'motile_test_obs_size': test_motile,
            'non_motile_test_obs_size': test_non_motile,
            
            # Training Metrics
            'train_accuracy': train_acc,
            'train_f1_score': train_F1,
            'train_recall': train_recall,
            'train_precision': train_precision,
            
            # Testing Metrics
            'test_accuracy': test_acc,
            'test_f1_score': test_F1,
            'test_recall': test_recall,
            'test_precision': test_precision,
            'test_population': test_pop
            }
        
            all_results.append(fold_results)
        
        print("\n--- Calculating Final Statistics ---")
    
        # 3.1. Extract all metric values into separate lists
        test_accs = [r['test_accuracy'] for r in all_results]
        test_f1s = [r['test_f1_score'] for r in all_results]
        test_recalls = [r['test_recall'] for r in all_results]
        test_precisions = [r['test_precision'] for r in all_results]

        train_accs = [r['train_accuracy'] for r in all_results]
        train_f1s = [r['train_f1_score'] for r in all_results]
        train_recalls = [r['train_recall'] for r in all_results]
        train_precisions = [r['train_precision'] for r in all_results]

        # 3.2. Calculate Mean and Standard Deviation (using numpy)
        final_stats = {
            'Test_Accuracy_Mean': numpy.mean(test_accs),
            'Test_Accuracy_Std': numpy.std(test_accs),
            'Test_F1_Mean': numpy.mean(test_f1s),
            'Test_F1_Std': numpy.std(test_f1s),
            'Test_Recall_Mean': numpy.mean(test_recalls),
            'Test_Recall_Std': numpy.std(test_recalls),
            'Test_Precision_Mean': numpy.mean(test_precisions),
            'Test_Precision_Std': numpy.std(test_precisions),
            
            'Train_Accuracy_Mean': numpy.mean(train_accs),
            'Train_Accuracy_Std': numpy.std(train_accs),
            'Train_F1_Mean': numpy.mean(train_f1s),
            'Train_F1_Std': numpy.std(train_f1s),
            'Train_Recall_Mean': numpy.mean(train_recalls),
            'Train_Recall_Std': numpy.std(train_recalls),
            'Train_Precision_Mean': numpy.mean(train_precisions),
            'Train_Precision_Std': numpy.std(train_precisions),
        }
        pprint.pprint( final_stats)
        