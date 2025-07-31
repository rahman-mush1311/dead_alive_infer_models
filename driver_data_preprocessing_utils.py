from driver_data_preprocessing import PreProcessingObservations
from driver_GridDisplacementModel import GridDisplacementModel
from GridOutlierModel import OutlierModelEvaluation 
from GridBayesianModel import BayesianModel

from visualize_object_trajectory import plot_hourly_prediction,plot_confusion_matrix,run_ffplay,get_video_fps,get_axis_limits,plot_object_trajectories,plot_displacements_across_frames,plot_score_components,plot_labeled_mean_displacements_by_lines

import os
import numpy

MOVING=1
NOTMOVING=0

TRAIN="train"
INFER="infer"
SEARCH="search"

TRACKING_DATA = "tracking_data"
TRUE_LABELS = "true_labels"
PREDICTED_LABELS= "predicted_labels"
LOG_PDFS="log_pdfs"
DEAD_PDFS="dead_log_sum_pdfs"
ALIVE_PDFS="alive_log_sum_pdfs"

def collect_files_by_mode(base_dir, mode=TRAIN):
    """
    Collects .txt files for either training or inference.
    
    Parameters:
    - base_dir: root directory to start from (default = current dir)
    - mode: "train" for taking the entire training  folder
            "infer" for taking the entire testing folder
            "search" for taking only one file from testing folder
    
    Returns:
    - List of file paths
    """
    file_list = []

    if mode == TRAIN:
        target_dirs = ["ostracod_files", "non_ostracod_files", "mixed_files"]
    elif mode == INFER:
        target_dirs = ["tox_files"]
    elif mode == SEARCH:
        target_dirs = ["."]
    else:
        raise ValueError("Mode must be 'train' or 'infer'")
    
    if mode == SEARCH and os.path.isfile(base_dir) and base_dir.endswith(".txt"):
        return [base_dir]
        
    for subfolder in target_dirs:
        folder_path = os.path.join(base_dir, subfolder)
        if not os.path.exists(folder_path):
            print(f"!!! Warning !!! {folder_path} does not exist.")
            continue

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".txt"):
                    file_list.append(os.path.join(root, file))

    return file_list


def collect_train_data():

    user_base_dir = input("Enter the base directory where your train data folders are located: ")
    user_mode = TRAIN
    
    try:
        collected_files = collect_files_by_mode(base_dir=user_base_dir, mode=user_mode)
        #$$$$$$$$$$$$SANITY CHECKING$$$$$$$$$$$$$$$$$$$$$
        print(f"Found {len(collected_files)} files for mode '{user_mode}'")
        return collected_files
        
    except ValueError as e:
        print(f"Error: {e}")
    
def collect_infer_data(user_file_selected_mode):

    user_base_dir = input("Enter the base directory where your infer data folders are located: ")
    user_mode = user_file_selected_mode
    
    try:
        collected_files = collect_files_by_mode(base_dir=user_base_dir, mode=user_mode)
        #$$$$$$$$$$$$SANITY CHECKING$$$$$$$$$$$$$$$$$$$$$
        print(f"Found {len(collected_files)} files for mode '{user_mode}'")
        return collected_files
        
    except ValueError as e:
        print(f"Error: {e}")
        
def prepare_train_infer_data(collected_file_lists,mode):
    
    observation_stats ={}
    
    all_train_observations={}
    all_test_observations={}
    
    for files in collected_file_lists:
    
        file_pre_processor = PreProcessingObservations()        
        observations,file_type=file_pre_processor.load_observations(files)
        truncated_observations=file_pre_processor.trajectory_quality_analysis(observations)
        if mode==TRAIN:
            train_observations,test_observations=file_pre_processor.prepare_train_test(truncated_observations,train_ratio=0.8)
     
            labeled_train_obs=file_pre_processor.observations_labeling_by_average_variance(train_observations, file_type, True)
            #labeled_train_obs=file_pre_processor.label_obs_by_z_score_ranking(train_observations, file_type, True)
            if len(labeled_train_obs)>0:               
                all_train_observations[files]=labeled_train_obs
                observation_stats[files]={'mu': file_pre_processor.total_mu, 'cov': file_pre_processor.total_cov_matrix}
        
            labeled_test_obs=file_pre_processor.observations_labeling_by_average_variance(test_observations, file_type, False)
            #labeled_test_obs=file_pre_processor.label_obs_by_z_score_ranking(test_observations, file_type, False)
            if len(labeled_test_obs)>0:               
                all_test_observations[files]=labeled_test_obs
        else:
            labeled_infer_obs=file_pre_processor.observations_labeling_by_average_variance(observations, file_type, True)
            #labeled_infer_obs=file_pre_processor.label_obs_by_z_score_ranking(observations, file_type, True)
            if len(labeled_infer_obs)>0:               
                all_train_observations[files]=labeled_infer_obs
                observation_stats[files]={'mu': file_pre_processor.total_mu, 'cov': file_pre_processor.total_cov_matrix}
            
            #$$$$$$$$$$$$$$$$$$$$$$$$$$SANITY CHECKING$$$$$$$$$$$$$$$$$$$$$
            print(f"{files} stats are: {file_pre_processor.total_obs}{file_pre_processor.total_mu}, \n{file_pre_processor.total_cov_matrix}")
        
    return observation_stats,all_train_observations,all_test_observations
    
def alive_model_training(collected_file_lists,observation_stats,train_observations):

    alive_models_params = {}
    
    for file in collected_file_lists:
        if file not in train_observations:
            print(f"!!!!!!!Warning!!!!!!!!: {file} not found in train_observations.")
            continue  
        else:
            curr_obs_stats= observation_stats[file]
            curr_train_obs=train_observations[file]
           
            filtered_curr_moving_obs={obj_id: obj_data[TRACKING_DATA] for obj_id, obj_data in curr_train_obs.items() if obj_data[TRUE_LABELS] == MOVING}
            if len(curr_obs_stats)!=0 and len(filtered_curr_moving_obs)!=0:
                grid_displacement_model=GridDisplacementModel() 
                grid_displacement_model.total_mu=curr_obs_stats['mu']
                grid_displacement_model.total_cov_matrix=curr_obs_stats['cov']
                print(f"$$$$ alive model training for single file {file}")
                curr_grid_displacements=grid_displacement_model.calculate_displacements(filtered_curr_moving_obs)
                grid_displacement_model.calculate_parameters(curr_grid_displacements)
        
                alive_models_params[file] = grid_displacement_model
            else:
                if len(curr_obs_stats)==0:
                    print(f"!!!!!!!Warning!!!!!!!!: normalization content empty for {file} {len(curr_obs_stats)}.")
                else:
                    print(f"!!!!!!!Warning!!!!!!!!:  {file} doesn't contain any moving examples {len(filtered_curr_moving_obs)}.")
                    
    return alive_models_params
    
    
def dead_model_training(collected_file_lists,observation_stats,train_observations):

    dead_models_params = {}
    
    for file in collected_file_lists:
         
        if file in train_observations and observation_stats:
            curr_obs_stats= observation_stats[file]
            curr_train_obs=train_observations[file]
            
            contains_valid_stats, dx_norm, dy_norm, sx_norm, sy_norm = get_sample_file_stats(curr_obs_stats)
            filtered_curr_nonmoving_obs={obj_id: obj_data[TRACKING_DATA] for obj_id, obj_data in curr_train_obs.items() if obj_data[TRUE_LABELS] == NOTMOVING}
            if len(curr_obs_stats)!=0 and len(filtered_curr_nonmoving_obs)!=0:
                grid_displacement_model=GridDisplacementModel() 
                grid_displacement_model.total_mu=curr_obs_stats['mu']
                grid_displacement_model.total_cov_matrix=curr_obs_stats['cov']
                print(f"$$$$ dead model training for single file {file}")
                curr_grid_displacements=grid_displacement_model.calculate_displacements(filtered_curr_nonmoving_obs)
                curr_grid_model_parameters=grid_displacement_model.calculate_parameters(curr_grid_displacements)
        
                dead_models_params[file] = grid_displacement_model
                
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
    
def combine_trained_models(collected_file_lists, curr_models_params):
    
    combined_model = GridDisplacementModel()
    
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
        dis_prob_with_label[obj_id][TRUE_LABELS] = obs_dict_with_labels[obj_id][TRUE_LABELS]

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
                      
                calculator = GridDisplacementModel()
                calculator.mu = combined_model.mu
                calculator.cov_matrix = combined_model.cov_matrix
                calculator.n = combined_model.n
                
                curr_log_pdf_dict= calculator.compute_probabilities(curr_tracking_obs, dx_norm, dy_norm, sx_norm, sy_norm)      
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
    
        if moving_entry[TRUE_LABELS] == non_moving_entry[TRUE_LABELS]:
            label=moving_entry[TRUE_LABELS]
            # Separate based on true label
            if label == MOVING:
                alive_train_obs_probs[obj_id] = moving_entry
            else:
                dead_train_obs_probs[obj_id] = non_moving_entry

            # Combine alive and dead log_pdfs for later classification
            combined_log_probs[obj_id] = {
                ALIVE_PDFS: moving_entry[LOG_PDFS],
                DEAD_PDFS: non_moving_entry[LOG_PDFS],
                TRUE_LABELS: label
            }
        else:
            print(f"!!!WARNING!!! {obj_id} have mismatching true labels, the alive model calculated dictionary has {moving_entry[TRUE_LABELS]} dead {non_moving_entry[TRUE_LABELS]}")
    return dead_train_obs_probs,alive_train_obs_probs,combined_log_probs
        
    
def run_outlier_model(test_performance):
    
    collected_file_lists=collect_train_data()
    obs_stats,all_train_obs,all_test_obs=prepare_train_infer_data(collected_file_lists,TRAIN)
    
    dead_model_params=dead_model_training(collected_file_lists,obs_stats,all_train_obs)
    print(f"$$$$ Outlier combined model training $$$$$$")
    dead_model=combine_trained_models(collected_file_lists, dead_model_params)
    
    train_obs_probs_dead_model=calculate_class_probability(dead_model,collected_file_lists,obs_stats,all_train_obs)
    train_probs_predicted_for_dead_model,outlier_model_eval=outlier_model_threshold_selection(train_obs_probs_dead_model)
    plot_confusion_matrix(train_probs_predicted_for_dead_model, "Train","Blues", "Outlier")
    
    if test_performance==True:
        test_obs_probs_dead_model=calculate_class_probability(dead_model,collected_file_lists,obs_stats,all_test_obs)
        test_probs_predicted_for_dead_model=outlier_model_eval.predict_probabilities_dictionary_update(test_obs_probs_dead_model)
        plot_confusion_matrix(test_probs_predicted_for_dead_model, "Test","Blues", "Outlier")
    else:
        print("user doesn't want to see the test set performance")
    return dead_model,outlier_model_eval

def run_bayesian_model(margin_adjustment,test_performance):
    
    collected_file_lists=collect_train_data()
    obs_stats,all_train_obs,all_test_obs=prepare_train_infer_data(collected_file_lists,TRAIN)
    
    dead_model_params=dead_model_training(collected_file_lists,obs_stats,all_train_obs)
    print(f"$$$$Bayesian combined dead model training $$$$$$")
    dead_model=combine_trained_models(collected_file_lists, dead_model_params)
    
    alive_model_params=alive_model_training(collected_file_lists,obs_stats,all_train_obs)
    print(f"$$$$Bayesian combined alive model training $$$$$$")
    alive_model=combine_trained_models(collected_file_lists, alive_model_params)
   
    train_obs_probs_dead_model=calculate_class_probability(dead_model,collected_file_lists,obs_stats,all_train_obs)
    train_obs_probs_alive_model=calculate_class_probability(alive_model,collected_file_lists,obs_stats,all_train_obs)
    
    dead_train_obs_probs,alive_train_obs_probs,combined_obs_probs=combine_dictionary_dead_alive_probs(train_obs_probs_dead_model,train_obs_probs_alive_model)
    
    bayesian_model_without_threshold=BayesianModel()  
    bayesian_model_without_threshold.calculate_prior(dead_train_obs_probs,alive_train_obs_probs)
    train_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_obs_probs)
    if margin_adjustment==False:
        plot_confusion_matrix(train_probs_bayesin_model_without_threshold, "Train","Greens", "Bayesian")
    
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$TESTING WITHOUT MARGIN###############################
    if test_performance==True:
        test_obs_probs_dead_model=calculate_class_probability(dead_model,collected_file_lists,obs_stats,all_test_obs)
        test_obs_probs_alive_model=calculate_class_probability(alive_model,collected_file_lists,obs_stats,all_test_obs)
        dead_train_obs_probs,alive_train_obs_probs,combined_test_obs_probs=combine_dictionary_dead_alive_probs(test_obs_probs_dead_model,test_obs_probs_alive_model)
        test_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_test_obs_probs)
        if margin_adjustment==False:
            plot_confusion_matrix(test_probs_bayesin_model_without_threshold, "Test","Greens", "Bayesian")
    else:
        print(f"user doesn't want to see the test set performance on bayesian model")
    
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$THRESHOLDING$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    if margin_adjustment==True:
        bayesian_model_without_threshold.find_optimal_threshold(train_probs_bayesin_model_without_threshold)
        train_probs_with_threshold=bayesian_model_without_threshold.predict_with_bayesian_threshold(train_probs_bayesin_model_without_threshold)
        plot_confusion_matrix(train_probs_with_threshold, "Train","Purples", "Bayesian With Threshold")
        
        #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$THRESHOLDING TESTING DATA$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        if test_performance==True:
            test_obs_probs_dead_model=calculate_class_probability(dead_model,collected_file_lists,obs_stats,all_test_obs)
            test_obs_probs_alive_model=calculate_class_probability(alive_model,collected_file_lists,obs_stats,all_test_obs)
            dead_train_obs_probs,alive_train_obs_probs,combined_test_obs_probs=combine_dictionary_dead_alive_probs(test_obs_probs_dead_model,test_obs_probs_alive_model)
            test_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_test_obs_probs)
            
            test_probs_bayesin_model_with_threshold=bayesian_model_without_threshold.predict_with_bayesian_threshold(test_probs_bayesin_model_without_threshold)
            plot_confusion_matrix(test_probs_bayesin_model_with_threshold, "Test","Purples", "Bayesian With Threshold")
        else:
            print(f"user doesn't want to see the test set performance on bayesian margin adjustment")
    else:
        print(f"user doesn't want the model to adjust margin of the bayesian model")
        
    return dead_model,alive_model,bayesian_model_without_threshold

def run_tracked_videos_by_filename():

    trajectory_video_file_name = input("Enter the path of the video file the desired object is located: ").strip()
    trajectory_file_location = input("Enter the path of the .txt the desired object is located: ").strip()
    object_id = int(input("Enter the id of the desired object is located: ").strip())
    
    try:
        collected_files = collect_files_by_mode(base_dir=trajectory_file_location, mode=SEARCH)
        #$$$$$$$$$$$$SANITY CHECKING$$$$$$$$$$$$$$$$$$$$$
        print(f"Found {collected_files}")
        
        file_pre_processor_for_visualization = PreProcessingObservations()
        obs,obs_type=file_pre_processor_for_visualization.load_observations(collected_files[0],INFER)       
        prefix,extension,obs_type=file_pre_processor_for_visualization.get_file_prefix(collected_files[0])
        trajectory_object_id=f"{prefix}_{object_id}_{extension}"
        
        if trajectory_object_id in obs:
            print("=== FFplay Video Player ===")
            fps=get_video_fps(trajectory_video_file_name)
            run_ffplay(trajectory_video_file_name, 1920, 1080,obs[trajectory_object_id][0][0],obs[trajectory_object_id][-1][0], fps,0.5)
        else:
            print(f"{len(obs)}")
            
    except ValueError as e:
        print(f"Error: {e}") 

def get_obs_labeled_by_prediction(all_infer_obs,infer_probs_predicted_labeled):
    
    infer_obs_labeled={}
    for file_name,objects in all_infer_obs.items():     
        for obj_id, obj_data in objects.items():
            infer_obs_labeled[obj_id] = {
            TRACKING_DATA: obj_data[TRACKING_DATA],
            TRUE_LABELS: obj_data[TRUE_LABELS],
            PREDICTED_LABELS: infer_probs_predicted_labeled[obj_id][PREDICTED_LABELS]   
        }
    return infer_obs_labeled
    
def get_visualization_ids(all_infer_obs_labeled, label_true, label_predicted):

    ids = [ obj_id for obj_id, details in all_infer_obs_labeled.items()
        if details[TRUE_LABELS] == label_true and details[PREDICTED_LABELS] == label_predicted
        ]
    print(len(ids))        
    return ids

def infer_with_trained_model(user_model_mode, user_test_performance, user_file_selected_mode):

    collect_infer_file_lists=collect_infer_data(user_file_selected_mode)
    infer_obs_stats,all_infer_obs,_=prepare_train_infer_data(collect_infer_file_lists,INFER)
        
    if user_model_mode ==1:        
        dead_model,outlier_model_eval=run_outlier_model(user_test_performance)
        infer_obs_probs_dead_model=calculate_class_probability(dead_model,collect_infer_file_lists,infer_obs_stats,all_infer_obs)
        infer_probs_predicted_labeled=outlier_model_eval.predict_probabilities_dictionary_update(infer_obs_probs_dead_model)
        plot_confusion_matrix(infer_probs_predicted_labeled, "Infer","Oranges", "Outlier")
        all_infer_obs_labeled=get_obs_labeled_by_prediction(all_infer_obs,infer_probs_predicted_labeled)
        
        return all_infer_obs_labeled
            
    elif user_model_mode ==2:
        dead_model,alive_model,bayesian_model_without_threshold=run_bayesian_model(False,user_test_performance)
        infer_obs_probs_dead_model=calculate_class_probability(dead_model,collect_infer_file_lists,infer_obs_stats,all_infer_obs)
        infer_obs_probs_alive_model=calculate_class_probability(alive_model,collect_infer_file_lists,infer_obs_stats,all_infer_obs)
    
        dead_infer_obs_probs,alive_infer_obs_probs,combined_infer_obs_probs=combine_dictionary_dead_alive_probs(infer_obs_probs_dead_model,infer_obs_probs_alive_model)
     
        #bayesian_model_without_threshold.calculate_prior(dead_infer_obs_probs,alive_infer_obs_probs)
        infer_probs_labeled_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_infer_obs_probs)
        plot_confusion_matrix(infer_probs_labeled_bayesin_model_without_threshold, "Infer","Oranges", "Bayesian")
        all_infer_obs_labeled=get_obs_labeled_by_prediction(all_infer_obs,infer_probs_labeled_bayesin_model_without_threshold)
        
        return all_infer_obs_labeled
        
    else:
        dead_model,alive_model,bayesian_model_without_threshold=run_bayesian_model(True,user_test_performance)
        infer_obs_probs_dead_model=calculate_class_probability(dead_model,collect_infer_file_lists,infer_obs_stats,all_infer_obs)
        infer_obs_probs_alive_model=calculate_class_probability(alive_model,collect_infer_file_lists,infer_obs_stats,all_infer_obs)
    
        dead_infer_obs_probs,alive_infer_obs_probs,combined_infer_obs_probs=combine_dictionary_dead_alive_probs(infer_obs_probs_dead_model,infer_obs_probs_alive_model)
     
        #bayesian_model_without_threshold.calculate_prior(dead_infer_obs_probs,alive_infer_obs_probs)
        infer_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_infer_obs_probs)
        infer_probs_labeled_bayesin_model_with_threshold=bayesian_model_without_threshold.predict_with_bayesian_threshold(infer_probs_bayesin_model_without_threshold)
        plot_confusion_matrix(infer_probs_labeled_bayesin_model_with_threshold, "Infer","Oranges", "Bayesian With Threshold")
        all_infer_obs_labeled=get_obs_labeled_by_prediction(all_infer_obs,infer_probs_labeled_bayesin_model_with_threshold)
        
        return all_infer_obs_labeled

def run_trajectory_plot():
    
    user_base_dir = input("Enter the base directory where your desired data folder is located to show trajectory: ")
    user_mode = SEARCH
    
    try:
        collected_files = collect_files_by_mode(base_dir=user_base_dir, mode=user_mode)
        #$$$$$$$$$$$$SANITY CHECKING$$$$$$$$$$$$$$$$$$$$$
        print(f"Found {len(collected_files)} files for mode '{user_mode}'")
        file_pre_processor = PreProcessingObservations()        
        observations,file_type=file_pre_processor.load_observations(collected_files[0])
        ranked_obs=file_pre_processor.observations_labeling_by_average_variance(observations,file_type,True)
       
        obs_mu = file_pre_processor.total_mu
        obs_cov = file_pre_processor.total_cov_matrix
        plot_labeled_mean_displacements_by_lines(ranked_obs,obs_mu,obs_cov)
        
        #plot_score_components(ranked_obs,all_zdx_zdy)
        #labeled_obs=file_pre_processor.observations_labeling_by_average_variance(observations, file_type, True)
        '''
        moving_objects = sorted([obj_id for obj_id, obs in ranked_obs.items() if obs[TRUE_LABELS] == MOVING])
        
        print(f"Objects labeled as NOT MOVING:{len(moving_objects)}, {len(ranked_obs)}")
        
        for obj_id in moving_objects:
            print(obj_id)
        '''
        #plot_object_trajectories(ranked_obs,moving_objects,0)
        
    except ValueError as e:
        print(f"Error: {e}")
    
   
    
def run_hourly_graph():
    #hour_list=[0,4,8,12,16,24,28,32,33,34,35,36]
    #total_list=[119,190,58,204,78,40,127,146,55,91,94,62]
    #alive_list=[32,44,10,56,14,9,22,37,8,16,13,9]
    hour_list=[0,4,8]
    total_list=[122,16,387]
    alive_list=[34,2,54]
    dose_rate="480 pbb days old ostracod"
    plot_hourly_prediction(hour_list,total_list,alive_list,dose_rate)