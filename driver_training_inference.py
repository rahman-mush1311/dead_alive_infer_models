
#from driver_GridFeatureModel import GridFeatureModel
from driver_GridDisplacementModel import GridDisplacementModel
from GridBayesianModel import BayesianModel
from driver_data_preprocessing import PreProcessingObservations
from driver_GridFeatureAnalyzer import FeatureImportanceAnalyzer
from visualize_object_trajectory import plot_confusion_matrix,visualize_auc_score,_plot_top_ranked_pair

import os
import numpy
import matplotlib.pyplot
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


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
                '''
                grid_feature_model=GridFeatureModel() 
                grid_feature_model.total_mu=curr_obs_stats['mu']
                grid_feature_model.total_cov_matrix=curr_obs_stats['cov']
                print(f"$$$$ dead model training for single file {file}")
                curr_grid_features= grid_feature_model.calculate_displacements(filtered_curr_nonmoving_obs)
                curr_grid_model_parameters= grid_feature_model.calculate_parameters(curr_grid_features)
        
                dead_models_params[file] =  grid_feature_model
                '''
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
                '''
                grid_feature_model=GridFeatureModel() 
                grid_feature_model.total_mu=curr_obs_stats['mu']
                grid_feature_model.total_cov_matrix=curr_obs_stats['cov']
                print(f"$$$$ alive model training for single file {file}")
                curr_grid_features=grid_feature_model.calculate_displacements(filtered_curr_moving_obs)
                grid_feature_model.calculate_parameters(curr_grid_features)
        
                alive_models_params[file] = grid_feature_model
                '''
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

def combine_trained_models(collected_file_lists, curr_models_params):
    
    #combined_model = GridFeatureModel()
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
    
def calculate_class_probability(combined_model,collected_file_lists,observation_stats,curr_observations):
    
    feature_probabilities_labeled={}
    
    for file in collected_file_lists:
         
        if file in curr_observations and observation_stats:
            curr_obs_stats= observation_stats[file]
            curr_obs_for_probability_calculation=curr_observations[file]
            
            curr_tracking_obs=get_dictionary_of_tracking_data(curr_obs_for_probability_calculation)            
            contains_valid_stats, dx_norm, dy_norm, sx_norm, sy_norm = get_sample_file_stats(curr_obs_stats)

            if contains_valid_stats and curr_tracking_obs:
                '''     
                calculator = GridFeatureModel()
                calculator.mu = combined_model.mu
                calculator.cov_matrix = combined_model.cov_matrix
                calculator.n = combined_model.n
                
                curr_log_pdf_dict= calculator.compute_probabilities(curr_tracking_obs, dx_norm, dy_norm, sx_norm, sy_norm)
                '''
                calculator = GridDisplacementModel()
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
    train_acc,train_f1,train_rec,train_pre=plot_confusion_matrix(train_probs_bayesin_model_without_threshold, "Train","Greens", "Bayesian")
    
    #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$TESTING WITHOUT MARGIN###############################
    if test_performance==True:
        test_obs_probs_dead_model=calculate_class_probability(dead_model,collected_file_lists,obs_stats,all_test_obs)
        test_obs_probs_alive_model=calculate_class_probability(alive_model,collected_file_lists,obs_stats,all_test_obs)
        dead_train_obs_probs,alive_train_obs_probs,combined_test_obs_probs=combine_dictionary_nonmotile_motile_probs(test_obs_probs_dead_model,test_obs_probs_alive_model)
        test_probs_bayesin_model_without_threshold=bayesian_model_without_threshold.sum_log_probabilities(combined_test_obs_probs)
        test_acc,test_f1,test_rec,test_pre=plot_confusion_matrix(test_probs_bayesin_model_without_threshold, "Test","Greens", "Bayesian")
        
        return train_acc,train_f1,train_rec,train_pre,test_acc,test_f1,test_rec,test_pre
    else:
        print(f"user doesn't want the model to see the test set performance")
       
    #return dead_model,alive_model,bayesian_model_without_threshold
    '''
    return dead_model,alive_model
    '''
    
def fisher_score_1d(f, y):
    """Fisher score of one feature between class 0 and 1."""
    f0, f1 = f[y == 0], f[y == 1]
    if len(f0) < 2 or len(f1) < 2:
        return 0.0
    m0, m1 = f0.mean(), f1.mean()
    v0, v1 = f0.var(),  f1.var()
    return (m1 - m0)**2 / (v0 + v1 + 1e-8)

def fisher_score_2d(X_pair, y):
    """
    True 2-D Fisher score for a feature pair.
    """
    X0 = X_pair[y == 0]
    X1 = X_pair[y == 1]
    if len(X0) < 2 or len(X1) < 2:
        return 0.0

    mu0 = X0.mean(axis=0)
    mu1 = X1.mean(axis=0)
    cov0 = np.cov(X0, rowvar=False)
    cov1 = np.cov(X1, rowvar=False)

    num = np.sum((mu1 - mu0)**2)  # numerator: squared distance between class means
    den = np.trace(cov0 + cov1)   # denominator: total within-class scatter
    return num / (den + 1e-8)

    
def rank_feature_pairs_fisher(X, y, feature_names):
    """
    X: (N_samples, 6)
    y: (N_samples,)
    feature_names: list of 6 names
    Returns: list of (name_i, name_j, score) sorted high→low
    """
    n_feat = X.shape[1]
    # 1D Fisher scores
    fisher_1d = [fisher_score_1d(X[:, i], y) for i in range(n_feat)]

    results = []
    '''
    for i, j in combinations(range(n_feat), 2):
        pair_score = 0.5 * (fisher_1d[i] + fisher_1d[j])
        results.append((feature_names[i], feature_names[j], pair_score))
    '''
    for i, j in combinations(range(n_feat), 2):
        X_pair = X[:, [i, j]]
        pair_score = fisher_score_2d(X_pair, y)
        results.append((feature_names[i], feature_names[j], pair_score))

    results.sort(key=lambda x: x[2], reverse=True)
    return results
    
def prepare_data_for_feature_discrimination(collected_train_txt_file_lists,collected_train_excel_file_lists):
    feature_X_list = []
    feature_y_list = []

    for text_file, excel_file in zip(collected_train_txt_file_lists,collected_train_excel_file_lists):
        print(f" txt file is: {text_file},{excel_file}")
        file_processor=PreProcessingObservations()
        tracking_observations=file_processor.load_observations(text_file)
        labeles_loaded=file_processor.load_labels(excel_file)
        labeled_observations=file_processor.label_observations_by_expert_labels(text_file,excel_file,tracking_observations,labeles_loaded)

        # global dx/dy stats for THIS file
        file_processor.compute_global_stats(labeled_observations)
        mu_dx, mu_dy = file_processor.total_mu[0], file_processor.total_mu[1]
        std_dx = numpy.sqrt(file_processor.total_cov_matrix[0,0])
        std_dy = numpy.sqrt(file_processor.total_cov_matrix[1,1])

        # extract per-step features + labels for this file
        X_file, y_file = extract_feature_matrix_from_labeled_obs(labeled_observations, mu_dx, mu_dy, std_dx, std_dy)
        if X_file.shape[0] > 0:
            feature_X_list.append(X_file)
            feature_y_list.append(y_file)

    # ---- after the for-loop: build global matrix and rank ----
    if feature_X_list:
        X_all = numpy.vstack(feature_X_list)
        y_all = numpy.concatenate(feature_y_list)
    
    return X_all,y_all
        
def extract_feature_matrix_from_labeled_obs(labeled_obs, mu_dx, mu_dy, std_dx, std_dy):
    """
    Build a global feature matrix X (N_samples x 6) and label vector y (N_samples)
    using the same 6-D features as GridFeatureModel:
    [dx_norm, dy_norm, heading, turning, ax, ay]
    """
    X_list = []
    y_list = []

    # guard against zero std
    if std_dx < 1e-10: std_dx = 1.0
    if std_dy < 1e-10: std_dy = 1.0

    for obj_id, data in labeled_obs.items():
        obs   = data[TRACKING_DATA]   # list of (x,y,frame) or (frame,x,y) → adjust order!
        label = data[TRUE_LABEL]     # 0 = non-motile, 1 = motile

        if len(obs) < 3:
            continue

        # unpack according to your tuple order; here I assume (x,y,f)
        for i in range(2, len(obs)):
            x0, y0, f0 = obs[i-2]
            x1, y1, f1 = obs[i-1]
            x2, y2, f2 = obs[i]

            df1 = f1 - f0
            df2 = f2 - f1
            if df1 <= 0 or df2 <= 0:
                continue

            # raw displacements
            dx1_raw = (x1 - x0) / df1
            dy1_raw = (y1 - y0) / df1
            dx2_raw = (x2 - x1) / df2
            dy2_raw = (y2 - y1) / df2

            # normalized displacements (same as GridFeatureModel)
            dx1_norm = (dx1_raw - mu_dx) / std_dx
            dy1_norm = (dy1_raw - mu_dy) / std_dy
            dx2_norm = (dx2_raw - mu_dx) / std_dx
            dy2_norm = (dy2_raw - mu_dy) / std_dy

            # heading angle
            heading = numpy.arctan2(dy2_norm, dx2_norm)

            # turning angle
            d1 = numpy.array([dx1_norm, dy1_norm])
            d2 = numpy.array([dx2_norm, dy2_norm])
            n1 = numpy.linalg.norm(d1)
            n2 = numpy.linalg.norm(d2)
            if n1 == 0 or n2 == 0:
                turning = 0.0
            else:
                cos_ang = numpy.clip(numpy.dot(d1, d2) / (n1 * n2), -1.0, 1.0)
                turning = numpy.arccos(cos_ang)
                # signed using cross product
                cross = d1[0] * d2[1] - d1[1] * d2[0]
                if cross < 0:
                    turning = -turning

            # acceleration from normalized displacements
            total_dt = df1 + df2
            if total_dt == 0:
                ax = ay = 0.0
            else:
                ax, ay = (d2 - d1) / total_dt

            feature_vec = [dx2_norm, dy2_norm, heading, turning, ax, ay]
            X_list.append(feature_vec)
            y_list.append(label)

    X = numpy.array(X_list)   # shape (N_samples, 6)
    y = numpy.array(y_list)   # shape (N_samples,)
    return X, y

def auc_for_pair(X, y, i, j):
    X_pair = X[:, [i, j]]
    clf = LogisticRegression(max_iter=1000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    for train_idx, test_idx in cv.split(X_pair, y):
        clf.fit(X_pair[train_idx], y[train_idx])
        prob = clf.predict_proba(X_pair[test_idx])[:, 1]
        aucs.append(roc_auc_score(y[test_idx], prob))
    return numpy.mean(aucs)

def rank_feature_pairs_auc(X, y, feature_names):
    n_feat = X.shape[1]
    results = []
    for i, j in combinations(range(n_feat), 2):
        score = auc_for_pair(X, y, i, j)
        results.append((feature_names[i], feature_names[j], score))
    results.sort(key=lambda x: x[2], reverse=True)
    return results
    
def fisher_feature_analysis(collected_text_file_lists,collected_excel_file_lists):
    
    feature_names = ['dx', 'dy', 'heading', 'turning', 'ax', 'ay']
    X_all,y_all=prepare_data_for_feature_discrimination(collected_text_file_lists,collected_excel_file_lists)
    
    print(f"\nTotal feature samples: {X_all.shape[0]}")
    '''
    pair_ranking = rank_feature_pairs_fisher(X_all, y_all, feature_names)

    print("\nTop 5 feature pairs (Fisher score):")
    for n1, n2, score in pair_ranking[:5]:
        print(f"{n1:8s} – {n2:8s} : {score:.4f}")

    print("\nBottom 5 feature pairs (least discriminative):")
    for n1, n2, score in pair_ranking[-5:]:
        print(f"{n1:8s} – {n2:8s} : {score:.4f}")
    '''
    '''
    auc_pairs = rank_feature_pairs_auc(X_all, y_all, feature_names)
    for name_i, name_j, auc in auc_pairs:
        print(f"{name_i:8s} – {name_j:8s}: AUC = {auc:.3f}")
    visualize_auc_score(auc_pairs)
    '''
    _plot_top_ranked_pair(X_all,y_all)

def _calculate_train_acc_summary():
    train_acc=[89.4,89.8,88.8,90.0,88.2,88.6,88.9,90.0,88.2]
    test_acc=[86.7,89.0,87.9,85.0,89.0,88.4,89.0,89.6,89.6]
    train_pre=[84.6,84.9,82.5,84.3,81.3,82.4,83.9,84.8]
    test_pre=[81.0,83,9,84.6,77.2,84.0,79.4,78.5,83.5]
    train_f1=[83.4,83.9,82.5,84,381.3,82.2,82.5,84.0,81.1]
    test_f1=[77.7,83.2,86.7,77.2,84.0,83.3,84.3,85.0,83.9]
    train_rec=[82.3,82.9,82.3,84.3,81.3,82.0,81.2,83.3,82.9]
    test_rec=[74.1,82.5,77.2,77.2,83.3,87.7,91.1,86.4,83.9]
    mean = numpy.mean(test_acc)
    std = numpy.std(test_acc, ddof=1)  # Sample std (n-1 denominator)
    stderr = std / numpy.sqrt(10)
    print(f"Test Accuracy:")
    print(f"  Mean ± Std:        {mean:.4f} ± {std:.4f}")
    print(f"  Mean ± StdErr:     {mean:.4f} ± {stderr:.4f}")
    mean_precision_train = numpy.mean(train_pre)
    std_precision_train = numpy.std(train_pre, ddof=1)  # Sample std (n-1 denominator)
    stderr_precision_train = std_precision_train / numpy.sqrt(9)
    print(f"Train Precision:")
    print(f"  Mean ± Std:        {mean_precision_train:.4f} ± {std_precision_train:.4f}")
    print(f"  Mean ± StdErr:     {mean_precision_train:.4f} ± {stderr_precision_train:.4f}")
    mean_precision_test = numpy.mean(test_pre)
    std_precision_test = numpy.std(test_pre, ddof=1)  # Sample std (n-1 denominator)
    stderr_precision_test = std_precision_test / numpy.sqrt(9)
    print(f"Train Precision:")
    print(f"  Mean ± Std:        {mean_precision_test:.4f} ± {std_precision_test:.4f}")
    print(f"  Mean ± StdErr:     {mean_precision_test:.4f} ± {stderr_precision_test:.4f}")
    
    mean_f1_train = numpy.mean(train_f1)
    std_f1_train = numpy.std(train_f1, ddof=1)  # Sample std (n-1 denominator)
    stderr_f1_train = std_f1_train / numpy.sqrt(10)
    print(f"Train F1:")
    print(f"  Mean ± Std:        {mean_f1_train:.4f} ± {std_f1_train:.4f}")
    print(f"  Mean ± StdErr:     {mean_f1_train:.4f} ± {stderr_f1_train:.4f}")
    
    mean_f1_test = numpy.mean(test_f1)
    std_f1_test = numpy.std(test_f1, ddof=1)  # Sample std (n-1 denominator)
    stderr_f1_test = std_f1_test / numpy.sqrt(10)
    print(f"Test F1:")
    print(f"  Mean ± Std:        {mean_f1_test:.4f} ± {std_f1_test:.4f}")
    print(f"  Mean ± StdErr:     {mean_f1_test:.4f} ± {stderr_f1_test:.4f}")
    
    mean_recall_train = numpy.mean(train_rec)
    std_recall_train = numpy.std(train_rec, ddof=1)  # Sample std (n-1 denominator)
    stderr_recall_train = std_recall_train / numpy.sqrt(10)
    print(f"Train Recall:")
    print(f"  Mean ± Std:        {mean_recall_train:.4f} ± {std_recall_train:.4f}")
    print(f"  Mean ± StdErr:     {mean_recall_train:.4f} ± {stderr_recall_train:.4f}")
    
    mean_recall_test = numpy.mean(test_rec)
    std_recall_test = numpy.std(test_rec, ddof=1)  # Sample std (n-1 denominator)
    stderr_recall_test = std_recall_test / numpy.sqrt(10)
    print(f"Test Recall:")
    print(f"  Mean ± Std:        {mean_recall_test:.4f} ± {std_recall_test:.4f}")
    print(f"  Mean ± StdErr:     {mean_recall_test:.4f} ± {stderr_recall_test:.4f}")

def analyze_feature_importance(collected_file_lists, obs_stats,all_train_obs):
    """
    Analyze feature importance after training dead and alive models.
    
    Parameters:
    - collected_file_lists: list of file names
    - observation_stats: dict {file: {'mu': ..., 'cov': ...}}
    - train_observations: dict {file: {obj_id: {TRACKING_DATA: ..., TRUE_LABEL: ...}}}
    - dead_models_params: dict {file: GridFeatureModel}
    - alive_models_params: dict {file: GridFeatureModel}
    - output_dir: where to save visualizations
    
    Returns:
    - analyzer: FeatureImportanceAnalyzer object with results
    """
    
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    dead_model_params=dead_model_training(collected_file_lists,obs_stats,all_train_obs)
    print(f"$$$$Bayesian combined dead model training $$$$$$")
    combined_dead_model=combine_trained_models(collected_file_lists, dead_model_params)
    
    alive_model_params=alive_model_training(collected_file_lists,obs_stats,all_train_obs)
    print(f"$$$$Bayesian combined alive model training $$$$$$")
    combined_alive_model=combine_trained_models(collected_file_lists, alive_model_params)
    
    # Run analysis
    analyzer = FeatureImportanceAnalyzer()
    
    # Global importance
    print("\nComputing global feature importance...")
    global_importance = analyzer.compute_log_likelihood_contribution_global(
        dead_model=combined_dead_model,
        alive_model=combined_alive_model,
        observations=all_train_obs,
        observation_stats=obs_stats
    )
    analyzer.visualize_global_importance()
    
    '''
    # Per-grid importance
    print("\nComputing per-grid feature importance...")
    grid_importance = analyzer.compute_log_likelihood_contribution_per_grid(
        dead_model=combined_dead_model,
        alive_model=combined_alive_model,
        observations=all_train_obs,
        observation_stats=obs_stats
    )
    
    # Visualize per-grid (top 3 features)
    analyzer.visualize_grid_importance(top_k=3)
    
    # Optional: print specific grid cells
    print("\nExample grid cell analysis:")
    analyzer.print_grid_cell_summary(1, 1)  # Center cell
    '''
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60 + "\n")
    
    return 