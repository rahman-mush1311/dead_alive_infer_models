from driver_data_preprocessing import PreProcessingObservations
from driver_training_inference import run_bayesian_model,fisher_feature_analysis,_calculate_train_acc_summary,analyze_feature_importance
from driver_train_infer_svm import run_complete_svm_pipeline,run_multiple_svm_experiments,run_feature_statistics_test,run_feature_statistics_test_per_file,feature_correlation_analysis
from visualize_object_trajectory import visualize_model_features_correlations,extract_correlations_from_model,plot_corr
#from GridBayesianModel import BayesianModel


import numpy
import os
import math
from collections import Counter
from PIL import Image
import matplotlib.pyplot
from collections import Counter
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

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
        if len(train_observations)>0:
            file_processor.compute_global_stats(train_observations)
            all_train_observations[text_file]=train_observations
            observation_stats[text_file]={'mu': file_processor.total_mu, 'cov': file_processor.total_cov_matrix}
        if len(test_observations)>0:
            all_test_observations[text_file]=test_observations
    '''    
        curr_train_motile_obs,curr_train_non_motile_obs=count_lables(train_observations)
        curr_test_motile_obs,curr_test_non_motile_obs=count_lables(test_observations)
        
        # --- accumulate counts ---
        total_train += len(train_observations)
        total_test  += len(test_observations)
        motile_train += curr_train_motile_obs
        nonmotile_train += curr_train_non_motile_obs
        motile_test += curr_test_motile_obs
        nonmotile_test += curr_test_non_motile_obs
        total_obs_size += len(labeled_observations)       
    
    # --- final summary ---
    print(f"\n==== Training/Test Summary ====")
    print(f"Total train objects: {total_train}")
    print(f"  Motile (alive):     {motile_train}")
    print(f"  Non-motile (dead):  {nonmotile_train}")
    print(f"Total test objects:  {total_test}")
    print(f"  Motile (alive):     {motile_test}")
    print(f"  Non-motile (dead):  {nonmotile_test}")
    print(f"===============================\n")
    '''   
    return observation_stats,all_train_observations,all_test_observations

def run_multiple_experiments(collected_train_txt_file_lists,observation_stats,all_train_observations, all_test_observations, n_runs=10):
    """
    Run the experiment multiple times and collect results
    """
    
    analyzer = MultiRunAnalyzer()
    
    for run_id in range(1, n_runs + 1):
        print(f"\n{'='*80}")
        print(f"RUN {run_id}/{n_runs}")
        print(f"{'='*80}\n")
        
        # Train models (with different random seeds if applicable)
        dead_model, alive_model, predictions = run_single_experiment(
            collected_train_txt_file_lists,
            observation_stats,
            all_train_observations,
            all_test_observations,
            run_id=run_id
        )
        
        # Add results to analyzer
        analyzer.add_run(f"Run_{run_id}", predictions)
    
    # Compute statistics
    summary = analyzer.compute_statistics()
    
    # Generate LaTeX table
    analyzer.generate_latex_table()
    analyzer.save_latex_table('results_table.tex')
    
    # Visualizations
    analyzer.plot_results(save_path='multi_run_boxplots.png')
    analyzer.plot_run_comparison(save_path='multi_run_comparison.png')
    
    # Export raw data
    analyzer.export_to_csv('all_runs_data.csv')
    
    # Get reporting strings
    print(f"\n{'='*80}")
    print("REPORTING STRINGS FOR PAPER")
    print(f"{'='*80}")
    print(f"Accuracy:  {analyzer.get_reporting_string('accuracy')}")
    print(f"F1-Score:  {analyzer.get_reporting_string('f1')}")
    print(f"Precision: {analyzer.get_reporting_string('precision')}")
    print(f"Recall:    {analyzer.get_reporting_string('recall')}")
    
    return analyzer




    
    
if __name__ == "__main__":
    #_calculate_train_acc_summary()
    collected_train_txt_file_lists=collect_files("train text files",".txt")
    collected_train_excel_file_lists=collect_files("train excel files",".xlsx")
    #run_multiple_svm_experiments(collected_train_txt_file_lists,collected_train_excel_file_lists)
    #run_feature_statistics_test(collected_train_txt_file_lists,collected_train_excel_file_lists)
    observation_stats,all_train_observations,all_test_observations=prepare_train_data(collected_train_txt_file_lists,collected_train_excel_file_lists)
    #dead_model,alive_model=run_bayesian_model(collected_train_txt_file_lists,observation_stats,all_train_observations,all_test_observations,True)
    #run_feature_statistics_test_per_file(collected_train_txt_file_lists,all_train_observations)
    #feature_correlation_analysis(collected_train_txt_file_lists,collected_train_excel_file_lists)
    analyze_feature_importance(collected_train_txt_file_lists,observation_stats,all_train_observations)
    '''
    print("\n---: Run Pipeline ---")
    svm_classifier, train_obs, test_obs, test_preds, metrics = run_complete_svm_pipeline(collected_train_txt_file_lists,collected_train_excel_file_lists)
    
    print("\n--- PIPELINE FINISHED ---")
    print(f"Final test accuracy: {metrics['accuracy']:.3f}")
    print(f"Final test F1-score: {metrics['f1']:.3f}")
    '''
    '''
    
    for i in range(5):
        observation_stats,all_train_observations,all_test_observations=prepare_train_data(collected_train_txt_file_lists,collected_train_excel_file_lists)
        dead_model,alive_model,bayesian_without_threshold=run_bayesian_model(collected_train_txt_file_lists,observation_stats,all_train_observations,all_test_observations,True)
    '''
    #visualize_model_features_correlations(dead_model)
    #visualize_model_features_correlations(alive_model)
    #alive_corrs = extract_correlations_from_model(alive_model)
    #dead_corrs  = extract_correlations_from_model(dead_model)
    #plot_corr(alive_corrs[(0,1)], "Motile Model Cell [0,1]")
    #plot_corr(dead_corrs[(0,1)],  "Non-motile Model Cell [0,1]")
    
    '''
    print(f"--------printing stats for dead model------------------")
    print(f"{dead_model.mu}")
    #print(f"{dead_model.cov_matrix}")
    
    print(f"--------printing stats for alive model------------------")
    print(f"{alive_model.mu}")
    #print(f"{alive_model.cov_matrix}")
    '''
    #fisher_feature_analysis(collected_train_txt_file_lists,collected_train_excel_file_lists)
    #_calculate_train_acc_summary()
    
    