from driver_observation_parser import ParsingObservations
from driver_training_processer import run_outlier_model,run_bayesian_model,infer_with_trained_model
#from driver_training_GMM_processing import gmm_dead_model_training,gmm_alive_model_training,calculate_class_probability,outlier_model_threshold_selection,combine_dictionary_dead_alive_probs,infer_with_GMM_bayesian_model,infer_with_GMM_outlier_model
#from driver_data_preprocessing_utils import run_hourly_graph,get_visualization_ids,run_tracked_videos_by_filename,infer_with_trained_model,run_trajectory_plot,run_outlier_model,collect_infer_data,prepare_train_infer_data,calculate_class_probability
from visualize_object_trajectory import plot_object_trajectories,plot_confusion_matrix,plot_hourly_prediction,mean_covariance_overlay_plot,grouped_bar_chart

from driver_GridDisplacementModel import GridDisplacementModel
from GridOutlierModel import OutlierModelEvaluation
from GridBayesianModel import BayesianModel


import numpy
import os
from collections import Counter
from PIL import Image
import matplotlib.pyplot

TRUE_LABEL = "true_label"
PREDICTED_LABEL = "predicted_label"
LOG_PDFS="log_pdfs"
TRACKING_DATA = "tracking_data"

TRAIN="train"
INFER="infer"

MOVING=1
NOTMOVING=0   


def collect_tox_file_from_user_input():
    """
    takes folder location and filename as user input returns the full path to do the parsing
    Params:
    -N/A
    Returns:
    folder_path or None
    """

    folder = input("Enter folder toxic path (e.g., C:/data/tracks): ").strip()
    filename = input("Enter file name (e.g., obj_track_01.txt): ").strip()

    full_path = os.path.join(folder, filename)
    file_list=[]

    try:
        with open(full_path, 'r') as f:
            contents = f.readlines()
            
        print(f"File loaded: {filename} ({len(contents)} lines)")
        file_list.append(full_path)
        return file_list
    except FileNotFoundError:
        print(f"File not found: {full_path}")
        return None
    except Exception as e:
        print(f"Error opening file: {e}")
        return None

def collect_train_file_from_user_input():
    file_list = []
    
    user_base_dir = input("Enter the base directory where your train data folders are located: ")
    target_dirs = ["mixed_files"]
        
    for subfolder in target_dirs:
        folder_path = os.path.join(user_base_dir, subfolder)
        if not os.path.exists(folder_path):
            print(f"!!! Warning !!! {folder_path} does not exist.")
            continue

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".txt"):
                    file_list.append(os.path.join(root, file))

    return file_list

def prepare_train_infer_data(collected_file_lists,model_setup):
    
    observation_stats ={}
    
    all_train_observations={}
    all_test_observations={}
    
    for file in collected_file_lists:
    
        file_pre_processor = ParsingObservations()        
        observations=file_pre_processor.load_observations(file)
        filetred_observations=file_pre_processor.filtering_tracks(observations)
        print(f"before after filter length is: {len(observations)}->{len(filetred_observations)}")
        labeled_observations=file_pre_processor.rank_observations_by_distance(filetred_observations)
        if model_setup==TRAIN:
            train_observations,test_observations=file_pre_processor.prepare_train_test(labeled_observations,train_ratio=0.8)
            if len(train_observations)>0:
                #train_labeled_obs=file_pre_processor.observations_labeling_by_angle(train_observations,True)
                file_pre_processor.compute_global_stats(train_observations)
                all_train_observations[file]=train_observations
                observation_stats[file]={'mu': file_pre_processor.total_mu, 'cov': file_pre_processor.total_cov_matrix}
            
                label_counter_train = Counter(data[TRUE_LABEL] for data in train_observations.values())
                
                first_key, first_value = next(iter(train_observations.items()))
                print(f"{file} has train obs of size: {len(train_observations)}")
                print(f"✅ MOVING: {label_counter_train[MOVING]}")
                print(f"🛑 NON_MOVING: {label_counter_train[NOTMOVING]}")
                plot_trajectories_interactive(train_observations)
                
            all_test_observations[file]=test_observations
            label_counter_test = Counter(data[TRUE_LABEL] for data in test_observations.values())
                
            first_key, first_value = next(iter(test_observations.items()))
            print(f"{file} has test obs of size: {len(test_observations)}")
            print(f"✅ MOVING: {label_counter_test[MOVING]}")
            print(f"🛑 NON_MOVING: {label_counter_test[NOTMOVING]}")
            plot_trajectories_interactive(test_observations)
            '''
            if len(test_observations)>0:
                test_labeled_obs=file_pre_processor.observations_labeling_by_angle(test_observations,False)
                all_test_observations[file]=test_labeled_obs
                
                label_counter = Counter(data[TRUE_LABEL] for data in test_labeled_obs.values())
                
                first_key, first_value = next(iter(test_labeled_obs.items()))
                print(f"{file} has test obs of size: {len(test_labeled_obs)}")
                print(f"✅ MOVING: {label_counter[MOVING]}")
                print(f"🛑 NON_MOVING: {label_counter[NOTMOVING]}")
                
                #plot_trajectories_interactive(test_labeled_obs)
            '''
        else:
            if len(filetred_observations):
                #labeled_observations=file_pre_processor.rank_observations_by_distance(filetred_observations)
                #infer_labeled_obs=file_pre_processor.observations_labeling_by_angle(filetred_observations,True)
                file_pre_processor.compute_global_stats(labeled_observations)
                all_test_observations[file]=labeled_observations
                observation_stats[file]={'mu': file_pre_processor.total_mu, 'cov': file_pre_processor.total_cov_matrix}
                label_counter_infer = Counter(data[TRUE_LABEL] for data in labeled_observations.values())
                
                first_key, first_value = next(iter(labeled_observations.items()))
                print(f"{file} has test obs of size: {len(labeled_observations)}")
                print(f"✅ MOVING: {label_counter_infer[MOVING]}")
                print(f"🛑 NON_MOVING: {label_counter_infer[NOTMOVING]}")
                #plot_trajectories_interactive(infer_labeled_obs)
        
    return observation_stats,all_train_observations,all_test_observations

def plot_trajectories_interactive(curr_obs):
    """
    Plots interested objects' trajectory only x and y coordinates one by one. With the key left it goes left, with right goes right
    
    Params:
        - curr_obs: dict of {object_id:[(frame, x, y), ...]}
        - extracted_ids: list of all interested object IDs (subset of keys in observations)
        - track_type: 0/1 indicating good/bad track
    Returns:
        N/A
    """
    ids_to_plot = [obj_id for obj_id in curr_obs if len(curr_obs[obj_id][TRACKING_DATA]) >= 2]
    if not ids_to_plot:
        print("!!!!No tracks to display!!!")
        return

    total = len(ids_to_plot)
    current_index = [0]
    
    global_xmin = min(p[1] for obj_id in ids_to_plot for p in curr_obs[obj_id][TRACKING_DATA])
    global_xmax = max(p[1] for obj_id in ids_to_plot for p in curr_obs[obj_id][TRACKING_DATA])
    global_ymin = min(p[2] for obj_id in ids_to_plot for p in curr_obs[obj_id][TRACKING_DATA])
    global_ymax = max(p[2] for obj_id in ids_to_plot for p in curr_obs[obj_id][TRACKING_DATA])
    
    def plot_one(index):
        obj_id = ids_to_plot[index]
        points = curr_obs[obj_id][TRACKING_DATA]
        x = [p[1] for p in points]
        y = [p[2] for p in points]
        color = "blue" if curr_obs[obj_id][TRUE_LABEL] == MOVING else "salmon"
        
        true_label_str = "MOVING" if curr_obs[obj_id][TRUE_LABEL] == 1 else "NOTMOVING"
        #pred_label_str = "MOVING" if curr_obs[obj_id][PREDICTED_LABEL] == 1 else "NOTMOVING"

        matplotlib.pyplot.clf()
        #matplotlib.pyplot.plot(x, y, marker="o", linestyle="-", color=color, label=f"Object ID: {obj_id}\n TRUE_LABEL: {curr_obs[obj_id][TRUE_LABEL]} PREDICTED_LABEL: {curr_obs[obj_id][PREDICTED_LABEL]}\n")
        matplotlib.pyplot.plot(x, y, marker="o", linestyle="-", color=color,label=f"Object ID: {obj_id}\nTRUE_LABEL: {true_label_str}")
        matplotlib.pyplot.xlim(global_xmin - 10, global_xmax + 10)
        matplotlib.pyplot.ylim(global_ymin - 10, global_ymax + 10)
        
        matplotlib.pyplot.xlabel("X Coordinate")
        matplotlib.pyplot.ylabel("Y Coordinate")
        matplotlib.pyplot.title(f"{curr_obs[obj_id][TRUE_LABEL]}\nTrack {index+1} of {total}")
        matplotlib.pyplot.legend()
        matplotlib.pyplot.grid(True, linestyle="--", alpha=0.6)
        matplotlib.pyplot.draw()

    def on_key(event):
        if event.key == 'right':
            if current_index[0] < total - 1:
                current_index[0] += 1
                plot_one(current_index[0])
        elif event.key == 'left':
            if current_index[0] > 0:
                current_index[0] -= 1
                plot_one(current_index[0])
        elif event.key == 'escape':
            matplotlib.pyplot.close()

    # Launch plot
    fig = matplotlib.pyplot.figure()
    fig.canvas.mpl_connect('key_press_event', on_key)
    plot_one(current_index[0])
    matplotlib.pyplot.show()


if __name__ == "__main__":
    '''
    hour_list=[0,4,8]
    total_list=[127,12,378]
    alive_list=[88,1,69]
    plot_hourly_prediction(hour_list,total_list,alive_list,480)
    '''
    #grouped_bar_chart()
    collected_train_file_list=collect_train_file_from_user_input()
    observation_stats,all_train_observations,all_test_observations=prepare_train_infer_data(collected_train_file_list,TRAIN)
    #dead_model,outlier_model_eval=run_outlier_model(collected_train_file_list,observation_stats,all_train_observations,all_test_observations,True)
    
    #dead_model,alive_model,bayesian_model_without_threshold=run_bayesian_model(collected_train_file_list,observation_stats,all_train_observations,all_test_observations,True,True)
    #print(f"{dead_model.mu}")
    #mean_covariance_overlay_plot(alive_model.mu, alive_model.cov_matrix, dead_model.mu, dead_model.cov_matrix)
    #dead_model=gmm_dead_model_training(collected_train_file_list,observation_stats,all_train_observations)
    
    #train_obs_probs_dead_model=calculate_class_probability(dead_model,collected_train_file_list,observation_stats,all_train_observations)
    #train_probs_predicted_for_dead_model,outlier_model_eval=outlier_model_threshold_selection(train_obs_probs_dead_model)
    '''
    plot_confusion_matrix(train_probs_predicted_for_dead_model, "Train","Blues", "Outlier")
    
    test_obs_probs_dead_model=calculate_class_probability(dead_model,collected_train_file_list,observation_stats,all_test_observations)
    
    test_probs_predicted_for_dead_model=outlier_model_eval.predict_probabilities_dictionary_update(test_obs_probs_dead_model)
    plot_confusion_matrix(test_probs_predicted_for_dead_model, "Test","Blues", "Outlier")
    
    alive_model=gmm_alive_model_training(collected_train_file_list,observation_stats,all_train_observations)
    train_obs_probs_alive_model=calculate_class_probability(alive_model,collected_train_file_list,observation_stats,all_train_observations)
    dead_train_obs_probs,alive_train_obs_probs,combined_obs_probs=combine_dictionary_dead_alive_probs(train_obs_probs_dead_model,train_obs_probs_alive_model)
    
    bayesian_model_without_threshold=BayesianModel()  
    bayesian_model_without_threshold.calculate_prior(dead_train_obs_probs,alive_train_obs_probs)
    train_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_obs_probs)
    #plot_confusion_matrix(train_probs_bayesin_model_without_threshold, "Train","Greens", "Bayesian")
    
    test_obs_probs_alive_model=calculate_class_probability(alive_model,collected_train_file_list,observation_stats,all_test_observations)
    dead_train_obs_probs,alive_train_obs_probs,combined_test_obs_probs=combine_dictionary_dead_alive_probs(test_obs_probs_dead_model,test_obs_probs_alive_model)
    test_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_test_obs_probs)
    '''
    #plot_confusion_matrix(test_probs_bayesin_model_without_threshold, "Test","Greens", "Bayesian")
    
    #first_key, first_value = next(iter(train_obs_probs_with_dead_model.items()))
    #print(f"{first_key}: {first_value}")
    #plot_trajectories_interactive(curr_train_obs)
    #dead_model,outlier_model_eval=run_outlier_model(collected_train_file_list,observation_stats,all_train_observations,all_test_observations,True)
    #dead_model,alive_model,bayesian_model_without_threshold=run_bayesian_model(collected_train_file_list,observation_stats,all_train_observations,all_test_observations,False,True)
    toxic_file_list=collect_tox_file_from_user_input()
    infer_observation_stats,_,all_infer_observations=prepare_train_infer_data(toxic_file_list,INFER)
    #all_infer_observations=infer_with_trained_model(toxic_file_list,infer_observation_stats,all_infer_observations,dead_model,alive_model,bayesian_model_without_threshold)
    #infer_with_GMM_bayesian_model(toxic_file_list,infer_observation_stats,all_infer_observations,dead_model,alive_model,bayesian_model_without_threshold)
    #infer_with_GMM_outlier_model(toxic_file_list,infer_observation_stats,all_infer_observations,dead_model,outlier_model_eval)
    
    
    