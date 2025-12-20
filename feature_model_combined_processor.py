from driver_data_preprocessing import PreProcessingObservations
from driver_GridFeatureModel import GridFeatureModel
from GridBayesianModel import BayesianModel
from visualize_object_trajectory import plot_confusion_matrix

import numpy
import os
import random
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

def prepare_train_set(fold_train_text_files, fold_train_excel_files):

    all_train_observations={}
    observation_stats ={}

    for text_file, excel_file in zip(fold_train_text_files,fold_train_excel_files):
        file_processor=PreProcessingObservations()
        tracking_observations=file_processor.load_observations(text_file)
        labeles_loaded=file_processor.load_labels(excel_file)
        labeled_observations=file_processor.label_observations_by_expert_labels(text_file,excel_file,tracking_observations,labeles_loaded)
        
        file_processor.compute_global_stats(labeled_observations)
        all_train_observations[text_file]=labeled_observations
        observation_stats[text_file]={'mu': file_processor.total_mu, 'cov': file_processor.total_cov_matrix}
   
    return observation_stats,all_train_observations

def motile_model_training(collected_file_lists,fold_observation_stats,fold_train_observations):

    motile_models_params = {}
    
    for file in collected_file_lists:
        if file not in fold_train_observations:
            print(f"!!!!!!!Warning!!!!!!!!: {file} not found in train_observations.")
            continue  
        else:
            curr_obs_stats= fold_observation_stats[file]
            curr_train_obs=fold_train_observations[file]
           
            filtered_curr_moving_obs={obj_id: obj_data[TRACKING_DATA] for obj_id, obj_data in curr_train_obs.items() if obj_data[TRUE_LABEL] == MOTILE}
            
            if len(curr_obs_stats)!=0 and len(filtered_curr_moving_obs)!=0:
                grid_mgd_motile_model=GridFeatureModel()
                grid_mgd_motile_model.total_mu=curr_obs_stats['mu']
                grid_mgd_motile_model.total_cov_matrix=curr_obs_stats['cov']
                print(f"$$$$ motile model training for single file {os.path.basename(file)}")
                curr_grid_displacements=grid_mgd_motile_model.calculate_displacements(filtered_curr_moving_obs)
                grid_mgd_motile_model.calculate_parameters(curr_grid_displacements)
        
                motile_models_params[file] = grid_mgd_motile_model
            else:
                if len(curr_obs_stats)==0:
                    print(f"!!!!!!!Warning!!!!!!!!: normalization content empty for {file} {len(curr_obs_stats)}.")
                else:
                    print(f"!!!!!!!Warning!!!!!!!!:  {os.path.basename(file)} doesn't contain any moving examples {len(filtered_curr_moving_obs)}.")
                    
    return motile_models_params
    
    
def non_motile_model_training(collected_file_lists,fold_observation_stats,fold_train_observations):

    non_motile_models_params = {}
    
    for file in collected_file_lists:
         
        if file in fold_train_observations and fold_observation_stats:
            curr_obs_stats= fold_observation_stats[file]
            curr_train_obs=fold_train_observations[file]
            
            filtered_curr_nonmoving_obs={obj_id: obj_data[TRACKING_DATA] for obj_id, obj_data in curr_train_obs.items() if obj_data[TRUE_LABEL] == NOTMOTILE}
            if len(curr_obs_stats)!=0 and len(filtered_curr_nonmoving_obs)!=0:
                grid_non_motile_model=GridFeatureModel()
                grid_non_motile_model.total_mu=curr_obs_stats['mu']
                grid_non_motile_model.total_cov_matrix=curr_obs_stats['cov']
                print(f"$$$$ non-motile model training for single file {os.path.basename(file)}")
                curr_grid_displacements=grid_non_motile_model.calculate_displacements(filtered_curr_nonmoving_obs)
                curr_grid_model_parameters=grid_non_motile_model.calculate_parameters(curr_grid_displacements)
        
                non_motile_models_params[file] = grid_non_motile_model
                
            elif len(filtered_curr_nonmoving_obs)==0:
                print(f"!!!!!!!Warning!!!!!!!!:  after filtering {os.path.basename(file)} doesn't contain any dead examples {len(filtered_curr_nonmoving_obs)}.")
            else:
                print(f"!!!!!!!Warning!!!!!!!!: normalization content empty for {os.path.basename(file)} {len(curr_obs_stats)}.")
                
        elif file not in observation_stats:
            print(f"!!!!!!!Warning!!!!!!!!: normalization content empty for {os.path.basename(file)} {len(observation_stats[file])}.")
            continue
        else:
            print(f"!!!!!!!Warning!!!!!!!!:  {os.path.basename(file)} doesn't contain any dead examples {len(train_observations[file])}.")
            continue
                    
    return non_motile_models_params
    
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
        print(f"models to combine {valid_file_size} and models stored {len(calculated_models)}\n")
              #f"combined models stats are: {combined_model.mu}, {combined_model.cov_matrix}")             
    else:
        print(f"!!!!WARNING!!!! no valid models found {valid_file_size} and model size is {len(calculated_models)}")
    return combined_model

def get_dictionary_of_tracking_data(curr_obs_for_probability_calculation):
    
    curr_tracking_obs={obj_id: obj_data[TRACKING_DATA] for obj_id, obj_data in curr_obs_for_probability_calculation.items()}   
    return curr_tracking_obs

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
    
def calculate_class_probability(combined_model,collected_file_lists,fold_observation_stats,fold_curr_observations):
    
    displacement_probabilities_labeled={}
    
    for file in collected_file_lists:
         
        if file in fold_curr_observations and fold_observation_stats:
            curr_obs_stats= fold_observation_stats[file]
            curr_obs_for_probability_calculation=fold_curr_observations[file]
            
            curr_tracking_obs=get_dictionary_of_tracking_data(curr_obs_for_probability_calculation)            
            contains_valid_stats, dx_norm, dy_norm, sx_norm, sy_norm = get_sample_file_stats(curr_obs_stats)

            if contains_valid_stats and curr_tracking_obs:
                      
                calculator = combined_model
                calculator.total_mu=curr_obs_stats['mu']
                calculator.total_cov_matrix=curr_obs_stats['cov']
                curr_log_pdf_dict= calculator.compute_probabilities(curr_tracking_obs)      
                displacement_probabilities_labeled=computed_probability_with_labels(curr_log_pdf_dict,displacement_probabilities_labeled,curr_obs_for_probability_calculation)
                
            elif not curr_tracking_obs:
                print(f"!!!!!!!Warning!!!!!!!! for {file} doesn't contain any examples and obs size before filter is {len(curr_obs_for_probability_calculation)}.")
            else:
                print(f"!!!!!!!Warning!!!!!!!!: normalization content empty for {file} and contents are: {curr_obs_stats['mu']}, {curr_obs_stats['cov']}.")
               
        else:
            print(f"{file} contains {len(curr_obs_for_probability_calculation)} examples and normalization len is: {len(curr_obs_stats)}")
            continue
    
    return displacement_probabilities_labeled

def combine_dictionary_non_motile_motile_probs(train_obs_probs_non_motile_model,train_obs_probs_motile_model):
    
    motile_train_obs_probs = {}
    non_motile_train_obs_probs = {}
    combined_log_probs = {}

    for obj_id in train_obs_probs_motile_model:
        moving_entry = train_obs_probs_motile_model[obj_id]
        non_moving_entry = train_obs_probs_non_motile_model[obj_id]
    
        if moving_entry[TRUE_LABEL] == non_moving_entry[TRUE_LABEL]:
            label=moving_entry[TRUE_LABEL]
            # Separate based on true label
            if label == MOTILE:
                motile_train_obs_probs[obj_id] = moving_entry
            else:
                non_motile_train_obs_probs[obj_id] = non_moving_entry

            # Combine alive and dead log_pdfs for later classification
            combined_log_probs[obj_id] = {
                ALIVE_PDFS: moving_entry[LOG_PDFS],
                DEAD_PDFS: non_moving_entry[LOG_PDFS],
                TRUE_LABEL: label
            }
        else:
            print(f"!!!WARNING!!! {obj_id} have mismatching true labels, the motile model calculated dictionary has {moving_entry[TRUE_LABEL]} non-motile {non_moving_entry[TRUE_LABEL]}")
    return combined_log_probs,motile_train_obs_probs,non_motile_train_obs_probs

def per_fold_test_evaluate(fold_test_text_file, fold_test_excel_file, combined_motile_model, combined_non_motile_model, bayesian_model_without_threshold):
    
    motile_probs_labeled={}
    non_motile_probs_labeled={}
    
    file_processor=PreProcessingObservations()
    tracking_observations=file_processor.load_observations(fold_test_text_file)
    labeles_loaded=file_processor.load_labels(fold_test_excel_file)
    test_labeled_observations=file_processor.label_observations_by_expert_labels(fold_test_text_file,fold_test_excel_file,tracking_observations,labeles_loaded)
    '''
    all_test_observations[fold_test_text_file]=test_labeled_observations
    if len(test_labeled_observations)>0:
        file_processor.compute_global_stats(test_labeled_observations)
    
    test_observation_stats[fold_test_text_file]={'mu': file_processor.total_mu, 'cov': file_processor.total_cov_matrix}
    test_probs_motile_model=calculate_class_probability(combined_motile_model,fold_test_text_file,test_observation_stats,all_test_observations)
    test_probs_non_motile_model=calculate_class_probability(combined_non_motile_model,fold_test_text_file,test_observation_stats,all_test_observations)
    
    
    fold_test_probs,_,_=combine_dictionary_non_motile_motile_probs(test_probs_motile_model,test_probs_non_motile_model)
    test_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(fold_test_probs)
    plot_confusion_matrix(test_probs_bayesin_model_without_threshold, "Test","Greens", "Bayesian")
    '''
    file_processor.compute_global_stats(test_labeled_observations)
    print(f"{file_processor.total_mu}, { file_processor.total_cov_matrix}")
    
    motile_calculator=combined_motile_model
    motile_calculator.total_mu=file_processor.total_mu
    motile_calculator.total_cov_matrix=file_processor.total_cov_matrix
    
    test_tracking_obs=get_dictionary_of_tracking_data(test_labeled_observations)
    test_motile_log_pdf_dict= motile_calculator.compute_probabilities(test_tracking_obs)
    motile_probs_labeled=computed_probability_with_labels(test_motile_log_pdf_dict,motile_probs_labeled,test_labeled_observations)
    
    non_motile_calculator=combined_non_motile_model
    non_motile_calculator.total_mu=file_processor.total_mu
    non_motile_calculator.total_cov_matrix=file_processor.total_cov_matrix
    test_non_motile_log_pdf_dict= motile_calculator.compute_probabilities(test_tracking_obs)
    non_motile_probs_labeled=computed_probability_with_labels(test_non_motile_log_pdf_dict,non_motile_probs_labeled,test_labeled_observations)
    
    combined_test_probs,_,_=combine_dictionary_non_motile_motile_probs(non_motile_probs_labeled,motile_probs_labeled)
    test_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_test_probs)
    #test_probs_bayesin_model_with_threshold=bayesian_model_without_threshold.predict_with_bayesian_threshold(test_probs_bayesin_model_without_threshold)
    plot_confusion_matrix(test_probs_bayesin_model_without_threshold, "Test","Greens", "Bayesian")
    #print(f"{len(combined_test_probs)},{len(test_motile_log_pdf_dict)}, {len(motile_probs_labeled)}, {len(test_non_motile_log_pdf_dict)}, {len(non_motile_probs_labeled)}")
    
def train_test_setup_for_combined_model():
    
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
        
        for fold_data in splits[:9]:
            fold_num = fold_data['fold']
            test_pop = fold_data['test_population']
            #print(test_pop)
            train_text_files = fold_data['train_text']
            train_excel_files= fold_data['train_excel']
            
            # Get the filename only for cleaner output
            test_text_file = fold_data['test_text']
            test_excel_file = fold_data['test_excel']
        
            print(f"\nFold {fold_num}:")
            
            fold_observation_stats,fold_train_observations=prepare_train_set(train_text_files, train_excel_files)
            fold_motile_model_params=motile_model_training(train_text_files,fold_observation_stats,fold_train_observations)
            fold_non_motile_model_params=non_motile_model_training(train_text_files,fold_observation_stats,fold_train_observations)
            
            combined_motile_model=combine_trained_models(train_text_files,fold_motile_model_params)
            combined_non_motile_model=combine_trained_models(train_text_files,fold_non_motile_model_params)
            
            train_probs_motile_model=calculate_class_probability(combined_motile_model,train_text_files,fold_observation_stats,fold_train_observations)
            train_probs_non_motile_model=calculate_class_probability(combined_non_motile_model,train_text_files,fold_observation_stats,fold_train_observations)
            
            train_combined_log_probs,motile_train_obs_probs,non_motile_train_obs_probs=combine_dictionary_non_motile_motile_probs(train_probs_non_motile_model,train_probs_motile_model)
            
            bayesian_model_with_threshold=BayesianModel()
            bayesian_model_with_threshold.calculate_prior(non_motile_train_obs_probs,motile_train_obs_probs)
            train_probs_bayesin_model_without_threshold=bayesian_model_with_threshold.sum_log_probabilities(train_combined_log_probs)
            #bayesian_model_with_threshold.find_optimal_threshold(train_probs_bayesin_model_without_threshold)
            #train_probs_bayesin_model_with_threshold=bayesian_model_with_threshold.predict_with_bayesian_threshold(train_probs_bayesin_model_without_threshold)
            plot_confusion_matrix(train_probs_bayesin_model_without_threshold, "Train","Greens", "Bayesian")
            
            per_fold_test_evaluate(test_text_file, test_excel_file,combined_motile_model,combined_non_motile_model,bayesian_model_with_threshold)
            
            '''
            fold_results = {
            'fold_number': fold_num,
   
            
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
        '''