from driver_data_preprocessing import PreProcessingObservations
from visualize_object_trajectory import take_user_input,find_object_ids,plot_object_trajectories

from driver_GridDisplacementModel import GridDisplacementModel
from GridModelEval import OutlierModelEvaluation
from GridBayesianModel import BayesianModel
from GridProbabilityCalculator import GridProbabilityCalculator

import numpy

TRUE_LABELS = "true_labels"
LOG_PDFS='log_pdfs'
DEAD='d'
ALIVE='a'
MIXED='m'
TOX='t'

def preparing_train_data():

    observation_stats ={}
    
    dead_train_obs={}
    dead_test_obs={}    
    alive_train_obs={}
    alive_test_obs={}
    
    all_train_obs={}
    all_test_obs={}
    
    file_loader = PreProcessingObservations()
    alive_file_lists = file_loader.load_files_from_folder(ALIVE,8)
    
    for files in alive_file_lists:
        
        prefix,extension,not_mixed_flag=file_loader.get_file_prefix(files)
        #$$$$$$$$$$$$ Sanity Checking $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        print(f" for {files} prefix is: {prefix},extension is: {extension},is not mixed: {not_mixed_flag}")
        #$$$$$$$$$$$$get observations & stats$$$$$$$$$$$$$$$$$$$$$$$$$$$
        observations=file_loader.load_observations(files)
        train_observations,test_observations=file_loader.prepare_train_test(observations,train_ratio=0.8)
        train_obs_mu,train_obs_cov=file_loader.compute_global_stats(train_observations)
        #store the stats for train set and store the dead alive obs
        observation_stats[files]= {'mu': train_obs_mu, 'cov': train_obs_cov}
        if len(train_observations)>0:               
            alive_train_obs[files]=train_observations
        if len(test_observations)>0:
            alive_test_obs[files]=test_observations
        #$$$$$$$$$$$$ Sanity Checking $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        print(f"from alive files total obs size: {len(observations)}")
        print(f"from alive files: train size for {files}: {len(alive_train_obs[files])}")
        print(f"from alive files: test size for {files}: {len(alive_test_obs[files])}")
        
    dead_file_lists = file_loader.load_files_from_folder(DEAD,8)
    for files in dead_file_lists:
        prefix,extension,not_mixed_flag=file_loader.get_file_prefix(files)
        #$$$$$$$$$$$$ Sanity Checking $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        print(f"for {files} prefix is: {prefix},extension is: {extension},is not mixed: {not_mixed_flag}")
        observations=file_loader.load_observations(files)
        train_observations,test_observations=file_loader.prepare_train_test(observations,train_ratio=0.8)
        train_obs_mu,train_obs_cov=file_loader.compute_global_stats(train_observations)
        #store the stats for train set and store the dead alive obs
        observation_stats[files]= {'mu': train_obs_mu, 'cov': train_obs_cov}
        if len(train_observations)>0:               
            dead_train_obs[files]=train_observations
        if len(test_observations)>0:
            dead_test_obs[files]=test_observations
        #$$$$$$$$$$$$ Sanity Checking $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        print(f"from dead files total obs size: {len(observations)}")
        print(f"from dead files: train size for {files}: {len(dead_train_obs[files])}")
        print(f"from dead files: test size for {files}: {len(dead_test_obs[files])}")
        
    
    mixed_file_lists = file_loader.load_files_from_folder(MIXED,7)
    
    for files in mixed_file_lists:

        prefix,extension,not_mixed=file_loader.get_file_prefix(files)
        #$$$$$$$$$$$$ Sanity Checking $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        print(f"for {files} prefix is: {prefix},extension is: {extension},is not mixed: {not_mixed}")
        observations=file_loader.load_observations(files)
        train_observations,test_observations=file_loader.prepare_train_test(observations,train_ratio=0.8)
        #get stats for the train file
        train_obs_mu,train_obs_cov=file_loader.compute_global_stats(train_observations)
        #store the stats for train set and store the dead alive obs
        observation_stats[files]= {'mu': train_obs_mu, 'cov': train_obs_cov} 
        #$$$$$$$$$$$$$$$get labels based on average-variance$$$$$$$$$$$$$$$$$$$
        train_dead_observations,train_alive_observations=file_loader.split_observations_by_average(train_observations,train_obs_mu,train_obs_cov,files)
        if len(train_dead_observations)>0:               
            dead_train_obs[files]=train_dead_observations
            print(f"from mixed files: train size for DEAD {files}: {len(dead_train_obs[files])}")
        else:
            print(f"!!!!WARNING!!!!! No dead train observation found by average-variance spliting {len(train_dead_observations)}")
        if len(train_alive_observations)>0:
            alive_train_obs[files]=train_alive_observations
            print(f"from mixed files: train size for ALIVE {files}: {len(alive_train_obs[files])}")
        else:
            print(f"!!!!WARNING!!!!! No alive train observation found by average-variance spliting {len(train_alive_observations)}")
        ############split the test into dead alive by average-variance################################
        test_dead_observations,test_alive_observations=file_loader.split_observations_by_average(test_observations,train_obs_mu,train_obs_cov,files)
        if len(test_dead_observations)>0:
            dead_test_obs[files]=test_dead_observations
            print(f"from mixed files: test size for  DEAD {files}: {len(dead_test_obs[files])}")
        else:
            print(f"!!!!WARNING!!!!! No dead test observation found by average-variance spliting {len(test_dead_observations)}")
        if len(test_alive_observations)>0:
            alive_test_obs[files]=test_alive_observations
            print(f"from mixed files: test size for  ALIVE {files}: {len(alive_test_obs[files])}")
        else:
            print(f"!!!!WARNING!!!!! No alive test observation found by average-variance spliting {len(test_alive_observations)}")
        #$$$$$$$$$$$$ Sanity Checking $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        print(f"from mixed files total obs size: {len(observations)}")
       
    return alive_file_lists,dead_file_lists,mixed_file_lists,alive_train_obs,dead_train_obs,alive_test_obs,dead_test_obs,observation_stats  

def combine_train_models(curr_file_lists, curr_train_obs, observation_stats):

    curr_models_params = {}
    
    for file in curr_file_lists:
    
        if file not in curr_train_obs:
            print(f"Warning: {file} not found in curr_train_obs in combine_train_models function.")
            continue  # Skip this file if not found
        else:
            train_curr_observations=curr_train_obs[file]
            print(f"from combine_train_models function {file} has {len(train_curr_observations)} observations")
        
        #####Start Calculating DEAD Model Parameters######
        train_observations_stats=observation_stats[file]
        if len(train_observations_stats)!=0:
        
            grid_displacement_model=GridDisplacementModel() 
            grid_displacement_model.total_mu=train_observations_stats['mu']
            grid_displacement_model.total_cov_matrix=train_observations_stats['cov']
        
            curr_grid_displacements=grid_displacement_model.calculate_displacements(train_curr_observations)
            curr_grid_model_parameters=grid_displacement_model.calculate_parameters(curr_grid_displacements)
        
            curr_models_params[file] = grid_displacement_model
            ########Sanity########
            '''
            print(f"from combine_train_models function {file} assigned parameters(normalization): {curr_models_params[file].total_mu}\n"
                f"{curr_models_params[file].total_cov_matrix} ")
            print (f"{file} stats are: {observation_stats[file]['mu']}, {observation_stats[file]['cov']}")
            print(f"from combine_train_models function {file} caluclated parameters: {curr_models_params[file].mu}\n"
                f"{curr_models_params[file].cov_matrix} ")
            '''
        else:
            print(f"!!!WARNING!!!! {file} doesn't contain any normalization contents")
    #####Combine the selected models######   
    combined_model = GridDisplacementModel()
    
    # Track the models
    calculated_models = []
    valid_file_size=0
    for i, files in enumerate(curr_file_lists):
        get_file = files
        
        if get_file not in curr_models_params:
            print(f"Warning: {get_file} not found in current_models_params.")
            continue  # Skip this file if not found
        else:
            # Store the models
            calculated_models.append(curr_models_params[get_file])
            valid_file_size+=1

    if valid_file_size>0:
        print(f"models to combine {valid_file_size}, {len(calculated_models)}.")
        combined_model = combined_model.add_models(*calculated_models)
    else:
        print(f"!!!!WARNING!!!! no valid models found {valid_file_size}, {len(calculated_models)}")
    #print(f"from combine model after combining all: {combined_model.mu}\n {combined_model.cov_matrix}")
    return combined_model

def compute_probabilities_with_combined_model(file_lists, curr_obs, observation_stats,combined_model,label):

    combined_model_probabilities={}
    
    for file in file_lists:
        if file not in curr_obs:
            print(f"Warning: {file} not found in curr_train_obs for calculating probability.")
            continue  # Skip this file if not found
        else:
            curr_observations_for_probability_calculation = curr_obs[file]
            curr_obs_stats= observation_stats[file]
        
            # Step 1: Get normalization statistics
            dx_norm, dy_norm = curr_obs_stats['mu']
            sx_norm, sy_norm = numpy.sqrt(numpy.diag(curr_obs_stats['cov']))
        
            # Step 2: Copy parameters into calculator
            calculator = GridProbabilityCalculator()
            calculator.mu = combined_model.mu
            calculator.cov_matrix = combined_model.cov_matrix
            calculator.n = combined_model.n
        
            # Step 3: Compute log-probabilities
            curr_log_pdf_dict, _ = calculator.compute_probabilities(curr_observations_for_probability_calculation, dx_norm, dy_norm, sx_norm, sy_norm)
        
            # Step 4: Merge into master dictionary
            combined_model_probabilities = calculator.combine_data_with_labels(curr_log_pdf_dict, combined_model_probabilities, label)
            print(f"from probability calculator {len(combined_model_probabilities)}")
    
    return combined_model_probabilities

def evaluate_outlier_model_with_threshold(curr_set):
    '''
    start performing the thresholds using the train set and prediction.
    Parameters:
    -curr_set: dictionary with object probabilities{object_id: {LOG_PDFS:list of log of probabilities}{TRUE_LABELS:d/a}}
    Returns:
    -curr_updated_set: a dictionary {object_id: {LOG_PDFS:list of log of probabilities}{TRUE_LABELS:d/a}{PREDICTED_LABELS: d/a}}
    -dead_outlier_model_eval: OutlierModelEvaluation() object with the calculated thresholds
    '''
    outlier_model_eval=OutlierModelEvaluation()
    
    window_sizes=[1,2,3,4,5,6,7,8,9,10]
    outlier_model_eval.evaluate_thresholds_window_sizes(curr_set, window_sizes)
    print(f"thresholds: {outlier_model_eval.best_accuracy_threshold, outlier_model_eval.best_precision_threshold, outlier_model_eval.window_size}")
    
    curr_updated_set=outlier_model_eval.predict_probabilities_dictionary_update(curr_set)
    
    return curr_updated_set,outlier_model_eval
    
def run_outlier_model():

    alive_file_lists,dead_file_lists,mixed_file_lists,alive_train_obs,dead_train_obs,alive_test_obs,dead_test_obs,observation_stats=preparing_train_data() 
    
    #$$$$$$$$$$$$$$$$$$$ OUTLIER TRAINNING WITH DEAD $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    all_dead_file_lists = dead_file_lists+mixed_file_lists
    outlier_model=combine_train_models(all_dead_file_lists, dead_train_obs, observation_stats)
    
    #$$$$$$$$$$$$$$$$$$$ Probability for dead $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    print(f"$$$$$$$$$$$$$$$for dead train$$$$$$$$$$$$$$")
    train_dead_with_dead_probs=compute_probabilities_with_combined_model(all_dead_file_lists, dead_train_obs, observation_stats,outlier_model,DEAD)
    
    #$$$$$$$$$$$$$$$$$$$ Probability for alive $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    print(f"$$$$$$$$$$$$$$$for alive train$$$$$$$$$$$$$$")
    all_alive_file_lists = alive_file_lists+mixed_file_lists
    train_dead_with_alive_probs=compute_probabilities_with_combined_model(all_alive_file_lists, alive_train_obs, observation_stats,outlier_model,ALIVE)
    
    #$$$$$$$$$$$$$$$$$$$$ Evaluation for outlier model $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    train_probablity_set=train_dead_with_dead_probs|train_dead_with_alive_probs
    predicted_train_set,dead_outlier_model_eval=evaluate_outlier_model_with_threshold(train_probablity_set)
    dead_outlier_model_eval.plot_confusion_matrix_outlier_model(predicted_train_set, "Train")
    
    ##########Testing##################
    print(f"$$$$$$$$$$$$$$$for dead test$$$$$$$$$$$$$$")
    test_dead_with_dead_probs=compute_probabilities_with_combined_model(all_dead_file_lists, dead_test_obs, observation_stats,outlier_model,DEAD)
    print(f"$$$$$$$$$$$$for alive test $$$$$$$$$$$$$$$$$$$$$")
    test_dead_with_alive_probs=compute_probabilities_with_combined_model(all_alive_file_lists, alive_test_obs, observation_stats,outlier_model,ALIVE)
    test_probablity_set=test_dead_with_dead_probs|test_dead_with_alive_probs
    predicted_test_probability_set=dead_outlier_model_eval.predict_probabilities_dictionary_update(test_probablity_set)
    dead_outlier_model_eval.plot_confusion_matrix_outlier_model(predicted_test_probability_set, "Test")
    
    return predicted_train_set,predicted_test_probability_set,outlier_model,dead_outlier_model_eval
    
def run_bayesian_model():
    alive_file_lists,dead_file_lists,mixed_file_lists,alive_train_obs,dead_train_obs,alive_test_obs,dead_test_obs,observation_stats=preparing_train_data() 
    
    all_file_lists= dead_file_lists+alive_file_lists
    
    #$$$$$$$$$$$$$$$$$$$ BAYESIANN TRAINNING WITH DEAD ALIVE $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    all_dead_file_lists = dead_file_lists+mixed_file_lists
    all_alive_file_lists = alive_file_lists+mixed_file_lists
    bayesian_dead_model=combine_train_models(all_dead_file_lists, dead_train_obs, observation_stats)
    bayesian_alive_model=combine_train_models(all_alive_file_lists, alive_train_obs, observation_stats)
    
    #$$$$$$$$$$$$$$$$$$$ Probability DEAD ALIVE with DEAD Model $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    print(f"$$$$$$$$$$$$$$$for train with dead model $$$$$$$$$$$$$$")
    train_dead_obs_with_dead_probs=compute_probabilities_with_combined_model(all_dead_file_lists, dead_train_obs, observation_stats,bayesian_dead_model,DEAD)
    train_alive_obs_with_dead_probs=compute_probabilities_with_combined_model(all_alive_file_lists, alive_train_obs, observation_stats,bayesian_dead_model,ALIVE)
    #$$$$$$$$$$$$$$$$$$$ Probability DEAD ALIVE with ALIVE Model $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    print(f"$$$$$$$$$$$$$$$for train with alive model $$$$$$$$$$$$$$")
    train_dead_obs_with_alive_probs=compute_probabilities_with_combined_model(all_dead_file_lists, dead_train_obs, observation_stats,bayesian_alive_model,DEAD)
    train_alive_obs_with_alive_probs=compute_probabilities_with_combined_model(all_alive_file_lists, alive_train_obs, observation_stats,bayesian_alive_model,ALIVE)
    
    train_with_dead_probablity_set=train_dead_obs_with_dead_probs|train_alive_obs_with_dead_probs
    train_with_alive_probability_set = train_dead_obs_with_alive_probs| train_alive_obs_with_alive_probs
    
    #################Bayesian#################
    bayesian_model_without_threshold = BayesianModel()
    bayesian_model_without_threshold.calculate_prior(train_dead_obs_with_dead_probs,train_alive_obs_with_dead_probs)
    ############Without Threshold#############
    predicted_train_obs=bayesian_model_without_threshold.sum_log_probabilities(train_with_dead_probablity_set, train_with_alive_probability_set)
    bayesian_model_without_threshold.plot_confusion_matrix(predicted_train_obs,"Train","Purples")
    ##########Testing##################
    print(f"$$$$$$$$$$$$for dead model test$$$$$$$$$$$$$$$$$$$$$")
    test_dead_obs_with_dead_probs=compute_probabilities_with_combined_model(all_dead_file_lists, dead_test_obs, observation_stats,bayesian_dead_model,DEAD)
    test_alive_obs_with_dead_probs=compute_probabilities_with_combined_model(all_alive_file_lists, alive_test_obs, observation_stats,bayesian_dead_model,ALIVE)
    print(f"$$$$$$$$$$$$for alive model test$$$$$$$$$$$$$$$$$$$$$")
    test_alive_obs_with_alive_probs=compute_probabilities_with_combined_model(all_dead_file_lists, dead_test_obs, observation_stats,bayesian_alive_model,DEAD)
    test_dead_obs_with_alive_probs=compute_probabilities_with_combined_model(all_alive_file_lists, alive_test_obs, observation_stats,bayesian_alive_model,ALIVE)
    
    test_with_dead_probablity_set=test_dead_obs_with_dead_probs|test_alive_obs_with_dead_probs
    test_with_alive_probability_set = test_dead_obs_with_alive_probs| test_alive_obs_with_alive_probs
    
    predicted_test_set=bayesian_model_without_threshold.sum_log_probabilities(test_with_dead_probablity_set, test_with_alive_probability_set)
    bayesian_model_without_threshold.plot_confusion_matrix(predicted_test_set,"Test","Purples")
    
    
    
    ####Thresholding#####
    '''
    bayesian_model_with_threshold = BayesianModel()
    bayesian_model_with_threshold.calculate_prior(train_dead_obs_with_dead_probs,train_alive_obs_with_dead_probs)
    bayesian_model_with_threshold.find_optimal_threshold(predicted_train_obs)
    predicted_train_obs_updated=bayesian_model_with_threshold.predict_with_bayesian_threshold(predicted_train_obs)
    bayesian_model_with_threshold.plot_confusion_matrix(predicted_train_obs_updated,"Train With Threshold","Greens")
    predicted_test_obs_updated=bayesian_model_with_threshold.predict_with_bayesian_threshold(predicted_test_obs)
    bayesian_model_with_threshold.plot_confusion_matrix(predicted_test_obs_updated,"Test With Threshold","Greens")
    '''
    #######Plot##########
    flattened_train_obs_dead= {obj_id: traj for file_data in dead_train_obs.values() for obj_id, traj in file_data.items()}
    flattened_train_obs_alive={obj_id: traj for file_data in alive_train_obs.values() for obj_id, traj in file_data.items()}
    all_train_obs=flattened_train_obs_dead|flattened_train_obs_alive
    
    flattened_test_obs_dead= {obj_id: traj for file_data in dead_test_obs.values() for obj_id, traj in file_data.items()}
    flattened_test_obs_alive={obj_id: traj for file_data in alive_test_obs.values() for obj_id, traj in file_data.items()}
    all_test_obs=flattened_test_obs_dead|flattened_test_obs_alive
   
    alive_predicted_alive_ids=find_object_ids(predicted_test_set, DEAD, DEAD)
    plot_object_trajectories(all_test_obs,alive_predicted_alive_ids,DEAD, DEAD)
    
    return bayesian_dead_model, bayesian_alive_model, bayesian_model_without_threshold

def prepare_infer_files(combined_bayesian_dead_model,combined_bayesian_alive_model,combined_bayesian_model_without_threshold,model_choice):

    file_loader = PreProcessingObservations()
    tox_file_lists = file_loader.load_files_from_folder(TOX,1)
    
    tox_observations={}
    observation_stats={}
    
    for files in tox_file_lists:
        prefix,extension,not_mixed_flag=file_loader.get_file_prefix(files)
        #$$$$$$$$$$$$ Sanity Checking $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        print(f" for {files} prefix is: {prefix},extension is: {extension},is not mixed: {not_mixed_flag}")
        #$$$$$$$$$$$$get observations & stats$$$$$$$$$$$$$$$$$$$$$$$$$$$
        observations=file_loader.load_observations(files)
        obs_mu,obs_cov=file_loader.compute_global_stats(observations)
        #store the stats for train set and store the dead alive obs
        observation_stats[files]= {'mu': obs_mu, 'cov': obs_cov}
        tox_observations[files]=observations
        
        #$$$$$$$$$$$$ Sanity Checking $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        print(f"from infer file function {len(observations)}")
        if model_choice==1:
            '''
            obs_probability=infer_with_outlier_model(combined_model,observations,obs_mu,obs_cov,ALIVE)
            obs_probability_with_prediction=combined_model_threshold.predict_probabilities_dictionary_update(obs_probability)
        
            combined_model_threshold.plot_confusion_matrix_outlier_model(obs_probability_with_prediction, files)
            alive_predicted_alive_ids=find_object_ids(obs_probability_with_prediction, ALIVE, ALIVE)
            plot_object_trajectories(observations,alive_predicted_alive_ids,ALIVE, ALIVE,files)
            '''
            print(f"comment things out!")
        else:
            obs_probability_with_prediction=infer_with_bayesian_model(combined_bayesian_dead_model,combined_bayesian_alive_model,combined_bayesian_model_without_threshold,observations,obs_mu,obs_cov,ALIVE)
            alive_predicted_alive_ids=find_object_ids(obs_probability_with_prediction, ALIVE, ALIVE)
            plot_object_trajectories(observations,alive_predicted_alive_ids,ALIVE, ALIVE,files)
    return tox_file_lists,tox_observations,observation_stats
    
def infer_with_outlier_model(combined_outlier_model,curr_tox_obs,curr_tox_obs_mu,curr_tox_obs_cov,true_label):

    tox_obs_probabilities={}
    
    calculator = GridProbabilityCalculator()
    calculator.mu = combined_outlier_model.mu
    calculator.cov_matrix = combined_outlier_model.cov_matrix
    calculator.n = combined_outlier_model.n
        
    dx_norm, dy_norm = curr_tox_obs_mu
    sx_norm, sy_norm = numpy.sqrt(numpy.diag(curr_tox_obs_cov))
    
    curr_log_pdf_dict, _ = calculator.compute_probabilities(curr_tox_obs, dx_norm, dy_norm, sx_norm, sy_norm)
    tox_obs_probabilities = calculator.combine_data_with_labels(curr_log_pdf_dict, tox_obs_probabilities, true_label)
    
    return tox_obs_probabilities

def infer_with_bayesian_model(combined_bayesian_dead_model,combined_bayesian_alive_model,bayesian_model_without_threshold,curr_tox_obs,curr_tox_obs_mu,curr_tox_obs_cov,true_label):

    tox_obs_probabilities={}
    tox_obs_dead_probabilities={}
    tox_obs_alive_probabilities={}
    
    dead_calculator = GridProbabilityCalculator()
    dead_calculator.mu = combined_bayesian_dead_model.mu
    dead_calculator.cov_matrix = combined_bayesian_dead_model.cov_matrix
    dead_calculator.n = combined_bayesian_dead_model.n
        
    dx_norm, dy_norm = curr_tox_obs_mu
    sx_norm, sy_norm = numpy.sqrt(numpy.diag(curr_tox_obs_cov))
    
    curr_dead_log_pdf_dict, _ = dead_calculator.compute_probabilities(curr_tox_obs, dx_norm, dy_norm, sx_norm, sy_norm)
    tox_obs_dead_probabilities = dead_calculator.combine_data_with_labels(curr_dead_log_pdf_dict, tox_obs_dead_probabilities, ALIVE)
    first_key = next(iter(tox_obs_dead_probabilities))
    '''
    print(f"First key for dead probs: {first_key}")

    # Get the first key-value pair as a tuple
    first_item = next(iter(tox_obs_dead_probabilities.items()))
    print(f"First item for dead probs: {first_item}")
    '''
    alive_calculator = GridProbabilityCalculator()
    alive_calculator.mu = combined_bayesian_alive_model.mu
    alive_calculator.cov_matrix = combined_bayesian_alive_model.cov_matrix
    alive_calculator.n = combined_bayesian_alive_model.n
        
    dx_norm, dy_norm = curr_tox_obs_mu
    sx_norm, sy_norm = numpy.sqrt(numpy.diag(curr_tox_obs_cov))
    
    curr_alive_log_pdf_dict, _ = alive_calculator.compute_probabilities(curr_tox_obs, dx_norm, dy_norm, sx_norm, sy_norm)
    tox_obs_alive_probabilities = alive_calculator.combine_data_with_labels(curr_alive_log_pdf_dict, tox_obs_alive_probabilities, ALIVE)
    '''
    first_key = next(iter(tox_obs_alive_probabilities))
    print(f"First key for alive probs: {first_key}")

    # Get the first key-value pair as a tuple
    first_item = next(iter(tox_obs_alive_probabilities.items()))
    print(f"First item for alive probs: {first_item}")
    '''
    tox_obs_probabilities=bayesian_model_without_threshold.sum_log_probabilities(tox_obs_dead_probabilities, tox_obs_alive_probabilities)
    '''
    first_key = next(iter(tox_obs_probabilities))
    print(f"First key for dead-alive probs: {first_key}")

    # Get the first key-value pair as a tuple
    first_item = next(iter(tox_obs_probabilities.items()))
    print(f"First item for dead-alive probs: {first_item}")
    '''
    return tox_obs_probabilities

if __name__ == "__main__":

    '''
    alive_file_lists,dead_file_lists,mixed_file_lists,alive_train_obs,dead_train_obs,alive_test_obs,dead_test_obs,observation_stats=preparing_train_data() 
    all_dead_file_lists = dead_file_lists+mixed_file_lists
    all_alive_file_lists = alive_file_lists+mixed_file_lists
    '''
    #outlier_train_probability_set, outlier_test_probability_set,outlier_model,dead_outlier_model_eval=run_outlier_model()
    bayesian_dead_model, bayesian_alive_model, bayesian_model_without_threshold=run_bayesian_model()
    #tox_file_lists,tox_obs,tox_obs_stats=prepare_infer_files(outlier_model,dead_outlier_model_eval,bayesian_dead_model, bayesian_alive_model, bayesian_model_without_threshold,2)
    #tox_file_lists,tox_obs,tox_obs_stats=prepare_infer_files(bayesian_dead_model, bayesian_alive_model, bayesian_model_without_threshold,2)

    #tox_probs_with_dead_model=compute_probabilities_with_combined_model(tox_file_lists, tox_obs, tox_obs_stats,outlier_model,ALIVE)
    #predicted_tox_probability_set=dead_outlier_model_eval.predict_probabilities_dictionary_update(tox_probs_with_dead_model)
    #dead_outlier_model_eval.plot_confusion_matrix_outlier_model(predicted_tox_probability_set, "All Tox")
    
    
    
    