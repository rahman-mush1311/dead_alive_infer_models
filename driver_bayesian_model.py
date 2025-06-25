from driver_observation_parser import ParsingObservations
from driver_GridDisplacementModel import GridDisplacementModel
from GridProbabilityCalculator import GridProbabilityCalculator
from GridModelEval import OutlierModelEvaluation
from GridBayesianModel import BayesianModel
from mean_covariance_plot import get_displacements,mean_covariance_plot,plot_histogram,mean_covariance_overlay_plot
import numpy

TRUE_LABELS = "true_labels"
LOG_PDFS='log_pdfs'
DEAD='d'
ALIVE='a'

def get_train_test_observation_stats():
    
    observation_stats ={}
    
    dead_train_obs={}
    dead_test_obs={}
    
    alive_train_obs={}
    alive_test_obs={}
    
    file_loader = ParsingObservations()
    file_loader.load_files_from_folder(ALIVE,5)
    
    for files in file_loader.filelists:
        #parse a single file
        print(f"filename is: {files}")
        observations=file_loader.load_observations(files)
        #split the single file into train test set
        train_observations,test_observations=file_loader.prepare_train_test(observations,train_ratio=0.8)
        #get stats for the train file
        train_obs_mu,train_obs_cov=file_loader.compute_global_stats(train_observations)
        #store the stats for train set and store the dead alive obs
        observation_stats[files]= {'mu': train_obs_mu, 'cov': train_obs_cov} 
        #get the labels for the train file if they belong to date-files all of them should be in dead_obs otherwise split them
        #train_dead_observations,train_alive_observations=file_loader.split_observations_by_displacements(train_observations,train_obs_cov,files)
        #train_dead_observations,train_alive_observations=file_loader.split_observations_by_filename(train_observations,files)
        #$$$$$$$$$$$$$$$get labels based on average$$$$$$$$$$$$$$$$$$$
        train_dead_observations,train_alive_observations=file_loader.split_observations_by_average(train_observations,train_obs_mu,train_obs_cov,files)
        if len(train_dead_observations)>0:               
            dead_train_obs[files]=train_dead_observations
        if len(train_alive_observations)>0:
            alive_train_obs[files]=train_alive_observations
        #split the test into dead and alive store them
        #test_dead_observations,test_alive_observations=file_loader.split_observations_by_displacements(test_observations,train_obs_cov,files)
        #test_dead_observations,test_alive_observations=file_loader.split_observations_by_filename(test_observations,files)
        ############split the test into dead alive by average################################
        test_dead_observations,test_alive_observations=file_loader.split_observations_by_average(test_observations,train_obs_mu,train_obs_cov,files)
        if len(test_dead_observations)>0:
            dead_test_obs[files]=test_dead_observations
        if len(test_alive_observations)>0:
            alive_test_obs[files]=test_alive_observations
        
    return file_loader,dead_train_obs,alive_train_obs,dead_test_obs,alive_test_obs,observation_stats
    
def compute_probabilities_with_models(file_loader, curr_train_obs, observation_stats,curr_model,label):

    curr_model_probabilities={}
    
    for file in file_loader.filelists:
        if file not in curr_train_obs:
            print(f"Warning: {file} not found in current_models_params.")
            continue  # Skip this file if not found
        else:
            curr_observations = curr_train_obs[file]
            curr_obs_stats= observation_stats[file]
        
            # Step 1: Get normalization statistics
            dx_norm, dy_norm = curr_obs_stats['mu']
            sx_norm, sy_norm = numpy.sqrt(numpy.diag(curr_obs_stats['cov']))
        
            # Step 2: Copy parameters into calculator
            calculator = GridProbabilityCalculator()
            calculator.mu = curr_model.mu
            calculator.cov_matrix = curr_model.cov_matrix
            calculator.n = curr_model.n
        
            # Step 3: Compute log-probabilities
            curr_log_pdf_dict, _ = calculator.compute_probabilities(curr_observations, dx_norm, dy_norm, sx_norm, sy_norm)
        
            # Step 4: Merge into master dictionary
            curr_model_probabilities = calculator.combine_data_with_labels(curr_log_pdf_dict, curr_model_probabilities, label)
            print(f"from probability calculator {len(curr_model_probabilities)}")
    
    return curr_model_probabilities    
def combine_train_models(file_loader, curr_train_obs, observation_stats):
    curr_models_params = {}
    for file in file_loader.filelists:
    
        if file not in curr_train_obs:
            print(f"Warning: {file} not found in current_models_params.")
            continue  # Skip this file if not found
        else:
            train_curr_observations=curr_train_obs[file]
            print(f"from bayesian model train {file} has {len(train_curr_observations)} observations")
        
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
            print(f"from combine dead model function {file} parameters: {dead_models_params[file].total_mu}\n"
                f"{dead_models_params[file].total_cov_matrix} ")
            print (f"{file} stats are: {observation_stats[file]['mu']}, {observation_stats[file]['cov']}")
            '''
        else:
            print(f"this {file} doesn't contain any normalization contents")
    #####Combine the dead models######   
    combined_model = GridDisplacementModel()
    
    # Track the models
    calculated_models = []
    valid_file_size=0
    for i, files in enumerate(file_loader.filelists):
        get_file = files
        
        if get_file not in curr_models_params:
            print(f"Warning: {get_file} not found in current_models_params.")
            continue  # Skip this file if not found
        else:
            # Store the models
            calculated_models.append(curr_models_params[get_file])
            valid_file_size+=1

    # When reaching the last file, combine the last two models
    if i == len(file_loader.filelists) - 1:  # Last file
        print(len(calculated_models))
        combined_model = combined_model.add_models(*calculated_models)  # Unpack list
    elif valid_file_size>0:
        print("models to combine {valid_file_size}.")
        combined_model = combined_model.add_models(*calculated_models)
    #print(f"from combine model after combining all: {combined_model.mu}\n {combined_model.cov_matrix}")
    return combined_model
if __name__ == "__main__":

    file_loader,dead_train_obs,alive_train_obs,dead_test_obs,alive_test_obs,observation_stats=get_train_test_observation_stats()
    
    bayesian_dead_model=combine_train_models(file_loader, dead_train_obs, observation_stats)
    bayesian_alive_model=combine_train_models(file_loader, alive_train_obs, observation_stats)
    
    
    print(f"$$$$$$$$$$$$$$$for dead model train$$$$$$$$$$$$$$")
    train_dead_obs_with_dead_probs=compute_probabilities_with_models(file_loader, dead_train_obs, observation_stats,bayesian_dead_model,DEAD)
    train_alive_obs_with_dead_probs=compute_probabilities_with_models(file_loader, alive_train_obs, observation_stats,bayesian_dead_model,ALIVE)
    print(f"$$$$$$$$$$$$for alive model train$$$$$$$$$$$$$$$$$$$$$")
    train_alive_obs_with_alive_probs=compute_probabilities_with_models(file_loader, alive_train_obs, observation_stats,bayesian_alive_model,ALIVE)
    train_dead_obs_with_alive_probs=compute_probabilities_with_models(file_loader, dead_train_obs, observation_stats,bayesian_alive_model,DEAD)
    train_with_dead_probablity_set=train_dead_obs_with_dead_probs|train_alive_obs_with_dead_probs
    train_with_alive_probability_set = train_dead_obs_with_alive_probs| train_alive_obs_with_alive_probs
    #################Bayesian#################
    bayesian_model_without_threshold = BayesianModel()
    bayesian_model_without_threshold.calculate_prior(train_dead_obs_with_dead_probs,train_alive_obs_with_dead_probs)
    ############Without Threshold#############
    predicted_train_obs=bayesian_model_without_threshold.sum_log_probabilities(train_with_dead_probablity_set, train_with_alive_probability_set)
    bayesian_model_without_threshold.plot_confusion_matrix(predicted_train_obs,"Train","Purples")
    ##########Testing##################
    test_dead_obs_with_dead_probs=compute_probabilities_with_models(file_loader, dead_test_obs, observation_stats,bayesian_dead_model,DEAD)
    test_alive_obs_with_dead_probs=compute_probabilities_with_models(file_loader, alive_test_obs, observation_stats,bayesian_dead_model,ALIVE)
    print(f"$$$$$$$$$$$$for alive model test$$$$$$$$$$$$$$$$$$$$$")
    test_alive_obs_with_alive_probs=compute_probabilities_with_models(file_loader, alive_test_obs, observation_stats,bayesian_alive_model,ALIVE)
    test_dead_obs_with_alive_probs=compute_probabilities_with_models(file_loader, dead_test_obs, observation_stats,bayesian_alive_model,DEAD)
    test_with_dead_probablity_set=test_dead_obs_with_dead_probs|test_alive_obs_with_dead_probs
    test_with_alive_probability_set = test_dead_obs_with_alive_probs| test_alive_obs_with_alive_probs
    
    predicted_test_obs=bayesian_model_without_threshold.sum_log_probabilities(test_with_dead_probablity_set, test_with_alive_probability_set)
    bayesian_model_without_threshold.plot_confusion_matrix(predicted_test_obs,"Test","Purples")
    ####Thresholding#####
    
    bayesian_model_with_threshold = BayesianModel()
    bayesian_model_with_threshold.calculate_prior(train_dead_obs_with_dead_probs,train_alive_obs_with_dead_probs)
    bayesian_model_with_threshold.find_optimal_threshold(predicted_train_obs)
    predicted_train_obs_updated=bayesian_model_with_threshold.predict_with_bayesian_threshold(predicted_train_obs)
    bayesian_model_with_threshold.plot_confusion_matrix(predicted_train_obs_updated,"Train With Threshold","Greens")
    predicted_test_obs_updated=bayesian_model_with_threshold.predict_with_bayesian_threshold(predicted_test_obs)
    bayesian_model_with_threshold.plot_confusion_matrix(predicted_test_obs_updated,"Test With Threshold","Greens")
    
    
    