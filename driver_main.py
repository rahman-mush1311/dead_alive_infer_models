from driver_data_preprocessing import PreProcessingObservations
from driver_data_preprocessing_utils import get_visualization_ids,run_tracked_videos_by_filename,infer_with_trained_model
from visualize_object_trajectory import plot_object_trajectories

from driver_GridDisplacementModel import GridDisplacementModel
from GridOutlierModel import OutlierModelEvaluation
from GridBayesianModel import BayesianModel


import numpy

TRUE_LABELS = "true_labels"
LOG_PDFS="log_pdfs"

DEAD='d'
ALIVE='a'
MIXED='m'
TOX='t'

MOVING=1
NOTMOVING=0   

TRAIN="train"
INFER="infer"
SEARCH="search"

if __name__ == "__main__":

    #run_outlier_model()
    #run_bayesian_model()
    #run_tracked_videos_by_filename()
    user_test_performance=False  
    user_file_selected_mode=INFER
    user_selected_mode = input("Do you want to test on the toxic data? (y/n): ").strip().lower()
    
    if user_selected_mode == 'y':
    
        user_file_mode= int(input(
                """If you select see the performance of the test set on non toxic elmenet multiple sample press:
                1 → yes
                2 → no
                Enter your choice: """
            ))
            
        user_model_mode = int(input(
                """If you want to train model press:
                1 → outlier model trainning
                2 → bayesian model trainning
                3 → bayesian model trainning with boundary adjustment
                Enter your choice: """
            ))
            
        user_test_performance_mode = int(input(
                """If you want see the performance of the test set on non toxic elmenets press:
                1 → yes
                2 → no
                Enter your choice: """
            ))
            
        if user_test_performance_mode==1:
            user_test_performance=True
        else:
            user_test_performance=False
        if user_file_mode ==1:
            user_file_selected_mode = INFER
        else:
            user_file_selected_mode = SEARCH
        all_infer_obs_labeled=infer_with_trained_model(user_model_mode, user_test_performance,user_file_selected_mode)        
        user_visual_mode = input("Do you want visualize trajectory of the predicted tox data (y/n): ").strip().lower()
        
        if user_visual_mode== 'y':
            visualize_objects = int(input(
                """If you want to visualize:
                1 → Correctly predicted MOVING objects
                2 → Correctly predicted NOTMOVING objects
                3 → Falsely predicted objects
                Enter your choice: """
            ))

            if visualize_objects==1:
                moving_obs_ids= get_visualization_ids(all_infer_obs_labeled, MOVING, MOVING)
                plot_object_trajectories(all_infer_obs_labeled,moving_obs_ids,user_model_mode)
            elif visualize_objects==2:
                non_moving_obs_ids= get_visualization_ids(all_infer_obs_labeled, NOTMOVING, NOTMOVING)
                plot_object_trajectories(all_infer_obs_labeled,non_moving_obs_ids,user_model_mode)
            else:
                moving_mislabeled_obs_ids= get_visualization_ids(all_infer_obs_labeled, MOVING, NOTMOVING)
                plot_object_trajectories(all_infer_obs_labeled,moving_mislabeled_obs_ids,user_model_mode)
                non_moving_mislabeled_obs_ids= get_visualization_ids(all_infer_obs_labeled, NOTMOVING, MOVING)
                plot_object_trajectories(all_infer_obs_labeled,non_moving_mislabeled_obs_ids,user_model_mode)
        else:
            print(f"user doesn't want to see the predicted tox objects tracks!")
    else:
        print(f"user doesn't want to test on toxic data")
        
    
    
    
    
    