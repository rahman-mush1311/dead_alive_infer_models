from driver_data_preprocessing import PreProcessingObservations
from driver_SVMClassifier import SVMTrajectoryClassifier
from visualize_object_trajectory import plot_confusion_matrix, _svm_feature_coorelation
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from multi_run_manager import MultiRunManager

import os
import collections
from collections import Counter
import random
from scipy import stats
import numpy as np

TRUE_LABEL = "true_label"
PREDICTED_LABEL = "predicted_label"
TRACKING_DATA = "tracking_data"

MOTILE = 1
NOTMOTILE = 0

def load_and_label_all_observations(text_file_lists, excel_file_lists):
    """
    Load observations from all text/excel file pairs and combine into one dictionary.
    
    Parameters:
    - text_file_lists: list of text file paths
    - excel_file_lists: list of excel file paths
    
    Returns:
    - all_labeled_observations: dict {obj_id: {TRACKING_DATA: tracks, TRUE_LABEL: label}}
    """
    
    if len(text_file_lists) != len(excel_file_lists):
        raise ValueError(f"Mismatch: {len(text_file_lists)} text files but {len(excel_file_lists)} excel files")
    
    all_labeled_observations = {}
    total_objects = 0
    total_motile = 0
    total_notmotile = 0
    
    print("\n" + "="*70)
    print("LOADING AND LABELING OBSERVATIONS FROM ALL FILES")
    print("="*70)
    
    for i, (text_file, excel_file) in enumerate(zip(text_file_lists, excel_file_lists), 1):
        print(f"\nProcessing file pair {i}/{len(text_file_lists)}:")
        print(f"  Text:  {os.path.basename(text_file)}")
        print(f"  Excel: {os.path.basename(excel_file)}")
        
        # Initialize file processor
        file_processor = PreProcessingObservations()
        
        # Load raw observations from text file
        tracking_observations = file_processor.load_observations(text_file)
        print(f"  Loaded {len(tracking_observations)} tracked objects")
        
        # Load expert labels from excel file
        loaded_labels = file_processor.load_labels(excel_file)
        print(f"  Loaded {len(loaded_labels)} labeled objects")
        
        # Combine tracking data with expert labels (and filter good tracks)
        labeled_observations = file_processor.label_observations_by_expert_labels(
            text_file, excel_file, tracking_observations, loaded_labels
        )
        
        # Count labels in this file
        file_motile = sum(1 for obs in labeled_observations.values() if obs[TRUE_LABEL] == MOTILE)
        file_notmotile = len(labeled_observations) - file_motile
        
        print(f"  Final labeled objects: {len(labeled_observations)}")
        print(f"    Motile: {file_motile}")
        print(f"    Not-motile: {file_notmotile}")
        
        # Add to combined dictionary
        # Check for duplicate object IDs across files
        duplicate_ids = set(all_labeled_observations.keys()) & set(labeled_observations.keys())
        if duplicate_ids:
            print(f"  WARNING: Found {len(duplicate_ids)} duplicate object IDs - they will be overwritten")
        
        all_labeled_observations.update(labeled_observations)
        
        # Update totals
        total_objects += len(labeled_observations)
        total_motile += file_motile
        total_notmotile += file_notmotile
    
    # Print summary
    print("\n" + "="*70)
    print("COMBINED DATASET SUMMARY")
    print("="*70)
    print(f"Total objects: {len(all_labeled_observations)}")
    print(f"  Motile:     {total_motile}")
    print(f"  Not-motile: {total_notmotile}")
    print(f"  Balance:    {100*total_motile/len(all_labeled_observations):.1f}% motile")
    print("="*70)
    
    return all_labeled_observations

def split_train_test(all_observations, train_ratio=0.8, random_seed=42):
    """
    Split combined observations into train and test sets.
    
    Parameters:
    - all_observations: dict {obj_id: {TRACKING_DATA: tracks, TRUE_LABEL: label}}
    - train_ratio: fraction of data for training (default 0.8)
    - random_seed: random seed for reproducibility
    
    Returns:
    - train_observations: dict with training data
    - test_observations: dict with test data
    """
    
    
    print("\n" + "="*70)
    print(f"SPLITTING DATA: {int(train_ratio*100)}% TRAIN / {int((1-train_ratio)*100)}% TEST")
    print("="*70)
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    
    # Get all object IDs and shuffle them
    all_obj_ids = list(all_observations.keys())
    random.shuffle(all_obj_ids)
    
    # Calculate split index
    split_index = int(len(all_obj_ids) * train_ratio)
    
    # Split object IDs
    train_obj_ids = all_obj_ids[:split_index]
    test_obj_ids = all_obj_ids[split_index:]
    
    # Create train and test dictionaries
    train_observations = {obj_id: all_observations[obj_id] for obj_id in train_obj_ids}
    test_observations = {obj_id: all_observations[obj_id] for obj_id in test_obj_ids}
    
    # Count labels in each set
    train_motile = sum(1 for obs in train_observations.values() if obs[TRUE_LABEL] == MOTILE)
    train_notmotile = len(train_observations) - train_motile
    
    test_motile = sum(1 for obs in test_observations.values() if obs[TRUE_LABEL] == MOTILE)
    test_notmotile = len(test_observations) - test_motile
    
    print(f"\nTraining set: {len(train_observations)} objects")
    print(f"  Motile:     {train_motile}")
    print(f"  Not-motile: {train_notmotile}")
    print(f"  Balance:    {100*train_motile/len(train_observations):.1f}% motile")
    
    print(f"\nTest set: {len(test_observations)} objects")
    print(f"  Motile:     {test_motile}")
    print(f"  Not-motile: {test_notmotile}")
    print(f"  Balance:    {100*test_motile/len(test_observations):.1f}% motile")
    print("="*70)
    
    return train_observations, test_observations

def train_svm_classifier(train_observations, 
                         kernel='rbf', 
                         C=10.0, 
                         gamma='scale',
                         perform_hyperparameter_search=False):
    """
    Train SVM classifier on training data.
    
    Parameters:
    - train_observations: dict {obj_id: {TRACKING_DATA: tracks, TRUE_LABEL: label}}
    - kernel: SVM kernel type
    - C: regularization parameter
    - gamma: kernel coefficient
    - perform_hyperparameter_search: whether to optimize hyperparameters
    
    Returns:
    - svm_classifier: trained SVMTrajectoryClassifier
    """
    
    print("\n" + "="*70)
    print("TRAINING SVM CLASSIFIER")
    print("="*70)
    
    # Initialize classifier
    svm_classifier = SVMTrajectoryClassifier(kernel=kernel, C=C, gamma=gamma)
    
    # Hyperparameter search if requested
    if perform_hyperparameter_search:
        print("\nPerforming hyperparameter search...")
        best_params = svm_classifier.hyperparameter_search(train_observations)
        print(f"Best parameters found: {best_params}")
    else:
        print(f"\nUsing specified parameters:")
        print(f"  Kernel: {kernel}")
        print(f"  C: {C}")
        print(f"  Gamma: {gamma}")
    
    # Train the model
    print("\nTraining SVM model...")
    train_accuracy = svm_classifier.train(train_observations)
    
    print(f"\nTraining complete! Training accuracy: {train_accuracy:.3f}")
    print("="*70)
    
    return svm_classifier


def evaluate_svm_classifier(svm_classifier, curr_observations, observation_type, plot_cm=True):
    """
    Evaluate SVM classifier on test data.
    
    Parameters:
    - svm_classifier: trained SVMTrajectoryClassifier
    - curr_observations: dict {obj_id: {TRACKING_DATA: tracks, TRUE_LABEL: label}}
    - observation_type: Train/Test
    - plot_cm: whether to plot confusion matrix
    
    Returns:
    - test_predictions: dict {obj_id: {TRUE_LABEL, PREDICTED_LABEL}}
    - metrics: dict with performance metrics
    """
    
    print("\n" + "="*70)
    print("EVALUATING SVM CLASSIFIER ON {observation_type} SET")
    print("="*70)
    
    # Make predictions
    curr_predictions = svm_classifier.predict(curr_observations)
    
    # Extract true and predicted labels
    y_true = [curr_predictions[obj_id][TRUE_LABEL] for obj_id in curr_predictions]
    y_pred = [curr_predictions[obj_id][PREDICTED_LABEL] for obj_id in curr_predictions]
    
    # Compute metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=MOTILE)
    recall = recall_score(y_true, y_pred, pos_label=MOTILE)
    precision = precision_score(y_true, y_pred, pos_label=MOTILE, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[NOTMOTILE, MOTILE])
    
    # Print results
    print(f" {observation_type} Set Performance:\n")
    print(f"  Accuracy:  {accuracy:.3f}")
    print(f"  F1-Score:  {f1:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  Precision: {precision:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"                Not-Motile  Motile")
    print(f"  Actual Not-Motile   {cm[0,0]:3d}      {cm[0,1]:3d}")
    print(f"         Motile       {cm[1,0]:3d}      {cm[1,1]:3d}")
    print("="*70)
    
    # Store metrics
    metrics = {
        'accuracy': accuracy,
        'f1': f1,
        'recall': recall,
        'precision': precision,
        'confusion_matrix': cm
    }
    
    # Plot confusion matrix
    if plot_cm:
        try:
            plot_confusion_matrix(curr_predictions, f"{observation_type} Set", "Blues", "SVM")
        except Exception as e:
            print(f"Could not plot confusion matrix: {e}")
    
    return curr_predictions, metrics


def run_complete_svm_pipeline(text_file_lists, 
                               excel_file_lists,
                               train_ratio=0.8,
                               random_seed=42,
                               kernel='rbf',
                               C=10.0,
                               gamma='scale',
                               perform_hyperparameter_search=False,
                               plot_results=True):
    """
    Complete SVM pipeline: load data, split, train, evaluate.
    
    Parameters:
    - text_file_lists: list of text file paths
    - excel_file_lists: list of excel file paths
    - train_ratio: fraction for training (default 0.8)
    - random_seed: random seed for splitting
    - kernel: SVM kernel type
    - C: regularization parameter
    - gamma: kernel coefficient
    - perform_hyperparameter_search: whether to optimize hyperparameters
    - plot_results: whether to plot confusion matrix
    
    Returns:
    - svm_classifier: trained classifier
    - train_observations: training data
    - test_observations: test data
    - test_predictions: predictions on test set
    - metrics: performance metrics
    """
    
    print("\n" + "="*70)
    print("COMPLETE SVM TRAJECTORY CLASSIFICATION PIPELINE")
    print("="*70)
    
    # Step 1: Load and label all observations
    all_labeled_observations = load_and_label_all_observations(
        text_file_lists, 
        excel_file_lists
    )
    
    # Step 2: Split into train and test
    train_observations, test_observations = split_train_test(
        all_labeled_observations,
        train_ratio=train_ratio,
        random_seed=random_seed
    )
    
    # Step 3: Train SVM classifier
    svm_classifier = train_svm_classifier(
        train_observations,
        kernel=kernel,
        C=C,
        gamma=gamma,
        perform_hyperparameter_search=perform_hyperparameter_search
    )
    
    # Step 4: Evaluate on test set and train set

    train_predictions, metrics = evaluate_svm_classifier(
        svm_classifier,
        train_observations,
        "Train",
        plot_cm=plot_results
    )
    test_predictions, metrics = evaluate_svm_classifier(
        svm_classifier,
        test_observations,
        "Test",
        plot_cm=plot_results
    )
    n_support_vectors = svm_classifier.svm_model.n_support_.sum()
    print(f"Number of support vectors: {n_support_vectors}")
    print(f"Total SVM parameters: {n_support_vectors * 23 + 1}")
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    '''
    svm_hyperSearch = SVMTrajectoryClassifier()
    best_params, results_df = svm_hyperSearch.hyperparameter_search(train_observations,show_all_results=True,max_detailed_results=None)
    '''
    return svm_classifier, train_observations, test_observations, test_predictions, metrics
    
def run_multiple_svm_experiments(text_file_lists,
                                   excel_file_lists,
                                   n_runs=10,
                                   train_ratio=0.8,
                                   kernel='rbf',
                                   C=1.0,
                                   gamma='auto',
                                   perform_hyperparameter_search=False,
                                   plot_results=False):
    """
    Run SVM pipeline multiple times with different random seeds and collect statistics.
    
    Parameters:
    - text_file_lists: list of text file paths
    - excel_file_lists: list of excel file paths
    - n_runs: number of experimental runs (default 10)
    - train_ratio: fraction for training (default 0.8)
    - kernel: SVM kernel type (default 'rbf')
    - C: regularization parameter (default 1.0)
    - gamma: kernel coefficient (default 'auto')
    - perform_hyperparameter_search: whether to optimize hyperparameters (default False)
    - plot_results: whether to plot confusion matrices (default False, to avoid too many plots)
    - save_results: whether to save results to files (default True)
    
    Returns:
    - manager: MultiRunManager object with all results
    - stats: dictionary with computed statistics
    """
    from multi_run_manager import MultiRunManager
    
    print("\n" + "="*80)
    print(f"RUNNING MULTIPLE SVM EXPERIMENTS: {n_runs} RUNS")
    print("="*80)
    print(f"Configuration:")
    print(f"  Train ratio: {train_ratio}")
    print(f"  Kernel: {kernel}")
    print(f"  C: {C}")
    print(f"  Gamma: {gamma}")
    print(f"  Hyperparameter search: {perform_hyperparameter_search}")
    print("="*80)
    
    # Initialize manager
    manager = MultiRunManager()
    
    # Run experiments
    for run_id in range(1, n_runs + 1):
        print(f"\n{'#'*80}")
        print(f"### RUN {run_id}/{n_runs} (Random Seed: {run_id})")
        print(f"{'#'*80}\n")
        
        # Run single experiment with unique random seed
        svm_classifier, train_obs, test_obs, test_predictions, test_metrics = run_complete_svm_pipeline(
            text_file_lists=text_file_lists,
            excel_file_lists=excel_file_lists,
            train_ratio=train_ratio,
            random_seed=run_id,  # Different seed for each run
            kernel=kernel,
            C=C,
            gamma=gamma,
            perform_hyperparameter_search=perform_hyperparameter_search,
            plot_results=plot_results
        )
        
        # Optionally get training metrics
        # For now, we'll extract them by re-evaluating on train set
        train_predictions, train_metrics = evaluate_svm_classifier(
            svm_classifier,
            train_obs,
            "Train",
            plot_cm=False  # Don't plot to avoid clutter
        )
        
        # Add to manager
        manager.add_run(run_id, test_metrics, train_metrics)
        
        print(f"\n>>> Run {run_id} complete: Test Accuracy = {test_metrics['accuracy']:.4f}")
    
    # Compute and print statistics
    manager.compute_and_print_statistics()
    
    return manager
    
def univariate_statistical_tests(observations):
    """
    Perform t-test and Mann-Whitney U test for each feature.
    
    Parameters:
    - observations: dict {obj_id: {TRACKING_DATA: tracks, TRUE_LABEL: label}}
    
    Returns:
    - results: dict with test statistics for each feature
    """
    
    
    # Extract features
    classifier = SVMTrajectoryClassifier()
    X, y, obj_ids = classifier.extract_features_from_observations(observations)
    
    # Separate motile and non-motile
    motile_mask = (y == MOTILE)
    nonmotile_mask = (y == NOTMOTILE)
    
    X_motile = X[motile_mask]
    X_nonmotile = X[nonmotile_mask]
    
    feature_names = [
        'mean_dx', 'mean_dy', 'std_dx', 'std_dy',
        'mean_speed', 'std_speed', 'max_speed',
        'total_distance', 'net_displacement', 'tortuosity',
        'mean_heading_cos', 'mean_heading_sin',
        'mean_abs_turn', 'std_abs_turn', 'max_abs_turn',
        'mean_turn_cos', 'mean_turn_sin',
        'mean_ax', 'mean_ay', 'std_ax', 'std_ay',
        'mean_accel_mag', 'std_accel_mag'
    ]
    
    results = []
    
    print("\n" + "="*90)
    print("UNIVARIATE STATISTICAL TESTS FOR FEATURE DISCRIMINATION")
    print("="*90)
    print(f"\nMotile samples: {len(X_motile)}")
    print(f"Non-motile samples: {len(X_nonmotile)}")
    print("\n" + "-"*90)
    print(f"{'Feature':<20} {'t-statistic':<15} {'t-test p-value':<15} {'U-statistic':<15} {'U-test p-value':<15}")
    print("-"*90)
    
    for i, feature_name in enumerate(feature_names):
        motile_values = X_motile[:, i]
        nonmotile_values = X_nonmotile[:, i]
        
        # t-test (parametric)
        t_stat, t_pval = stats.ttest_ind(motile_values, nonmotile_values)
        
        # Mann-Whitney U test (non-parametric)
        u_stat, u_pval = stats.mannwhitneyu(motile_values, nonmotile_values, alternative='two-sided')
        
        results.append({
            'feature': feature_name,
            't_statistic': t_stat,
            't_pvalue': t_pval,
            'u_statistic': u_stat,
            'u_pvalue': u_pval
        })
        
        print(f"{feature_name:<20} {t_stat:<15.4f} {t_pval:<15.4e} {u_stat:<15.1f} {u_pval:<15.4e}")
    
    print("="*90 + "\n")
    
    return results


def plot_univariate_tests(results, test_type='both', save_path=None):
    """
    Plot results of univariate statistical tests.
    
    Parameters:
    - results: list of dicts from univariate_statistical_tests()
    - test_type: 't-test', 'u-test', or 'both'
    - save_path: path to save figure (optional)
    """
    import matplotlib.pyplot as plt
    
    feature_names = [r['feature'] for r in results]
    
    if test_type == 'both':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # t-test p-values
        t_pvalues = [r['t_pvalue'] for r in results]
        ax1.bar(range(len(feature_names)), [-np.log10(p) for p in t_pvalues])
        ax1.axhline(y=-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
        ax1.axhline(y=-np.log10(0.01), color='orange', linestyle='--', label='p=0.01')
        ax1.set_xticks(range(len(feature_names)))
        ax1.set_xticklabels(feature_names, rotation=45, ha='right')
        ax1.set_ylabel('-log10(p-value)')
        ax1.set_title('t-test: Feature Discrimination Power')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Mann-Whitney U test p-values
        u_pvalues = [r['u_pvalue'] for r in results]
        ax2.bar(range(len(feature_names)), [-np.log10(p) for p in u_pvalues])
        ax2.axhline(y=-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
        ax2.axhline(y=-np.log10(0.01), color='orange', linestyle='--', label='p=0.01')
        ax2.set_xticks(range(len(feature_names)))
        ax2.set_xticklabels(feature_names, rotation=45, ha='right')
        ax2.set_ylabel('-log10(p-value)')
        ax2.set_xlabel('Features')
        ax2.set_title('Mann-Whitney U test: Feature Discrimination Power')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
    elif test_type == 't-test':
        fig, ax = plt.subplots(figsize=(14, 6))
        t_pvalues = [r['t_pvalue'] for r in results]
        ax.bar(range(len(feature_names)), [-np.log10(p) for p in t_pvalues])
        ax.axhline(y=-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
        ax.axhline(y=-np.log10(0.01), color='orange', linestyle='--', label='p=0.01')
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_ylabel('-log10(p-value)')
        ax.set_xlabel('Features')
        ax.set_title('t-test: Feature Discrimination Power')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
    elif test_type == 'u-test':
        fig, ax = plt.subplots(figsize=(14, 6))
        u_pvalues = [r['u_pvalue'] for r in results]
        ax.bar(range(len(feature_names)), [-np.log10(p) for p in u_pvalues])
        ax.axhline(y=-np.log10(0.05), color='r', linestyle='--', label='p=0.05')
        ax.axhline(y=-np.log10(0.01), color='orange', linestyle='--', label='p=0.01')
        ax.set_xticks(range(len(feature_names)))
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_ylabel('-log10(p-value)')
        ax.set_xlabel('Features')
        ax.set_title('Mann-Whitney U test: Feature Discrimination Power')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")
    
    plt.show()


def print_top_discriminative_features(results, top_n=10):
    """
    Print the top N most discriminative features based on p-values.
    
    Parameters:
    - results: list of dicts from univariate_statistical_tests()
    - top_n: number of top features to show
    """
    # Sort by t-test p-value
    sorted_by_t = sorted(results, key=lambda x: x['t_pvalue'])
    
    print("\n" + "="*80)
    print(f"TOP {top_n} MOST DISCRIMINATIVE FEATURES (by t-test p-value)")
    print("="*80)
    print(f"{'Rank':<6} {'Feature':<20} {'t-statistic':<15} {'p-value':<15}")
    print("-"*80)
    
    for rank, r in enumerate(sorted_by_t[:top_n], 1):
        print(f"{rank:<6} {r['feature']:<20} {r['t_statistic']:<15.4f} {r['t_pvalue']:<15.4e}")
    
    print("="*80)
    
    # Sort by U-test p-value
    sorted_by_u = sorted(results, key=lambda x: x['u_pvalue'])
    
    print("\n" + "="*80)
    print(f"TOP {top_n} MOST DISCRIMINATIVE FEATURES (by Mann-Whitney U p-value)")
    print("="*80)
    print(f"{'Rank':<6} {'Feature':<20} {'U-statistic':<15} {'p-value':<15}")
    print("-"*80)
    
    for rank, r in enumerate(sorted_by_u[:top_n], 1):
        print(f"{rank:<6} {r['feature']:<20} {r['u_statistic']:<15.1f} {r['u_pvalue']:<15.4e}")
    
    print("="*80 + "\n")

def run_feature_statistics_test(text_file_lists, excel_file_lists):

    all_observations=load_and_label_all_observations(text_file_lists, excel_file_lists)
    results = univariate_statistical_tests(observations)
    print_top_discriminative_features(results, top_n=10)
    plot_univariate_tests(results, test_type='both')
    

def run_feature_statistics_test_per_file(collected_file_lists,all_train_obs):

    print("\n=== FEATURE ANALYSIS PER WATERFLOW CONDITION ===")
    for text_file, observations in all_train_obs.items():
        print(f"\n--- File: {text_file} ---")
        results = univariate_statistical_tests(observations)
        print_top_discriminative_features(results, top_n=5)

def feature_correlation_analysis(text_file_lists, excel_file_lists):
    """
    Compute and visualize correlation matrix between features.
    
    Parameters:
    - observations: dict {obj_id: {TRACKING_DATA: tracks, TRUE_LABEL: label}}
    - save_path: path to save figure (optional)
    
    Returns:
    - correlation_matrix: numpy array with correlations
    - feature_names: list of feature names
    """
    
    # Extract features
    observations=load_and_label_all_observations(text_file_lists, excel_file_lists)
    classifier = SVMTrajectoryClassifier()
    X, y, obj_ids = classifier.extract_features_from_observations(observations)
    
    feature_names = [
        'mean_dx', 'mean_dy', 'std_dx', 'std_dy',
        'mean_speed', 'std_speed', 'max_speed',
        'total_distance', 'net_displacement', 'tortuosity',
        'mean_heading_cos', 'mean_heading_sin',
        'mean_abs_turn', 'std_abs_turn', 'max_abs_turn',
        'mean_turn_cos', 'mean_turn_sin',
        'mean_ax', 'mean_ay', 'std_ax', 'std_ay',
        'mean_accel_mag', 'std_accel_mag'
    ]
    _svm_feature_coorelation (X, feature_names)
    
    
    return


def find_highly_correlated_features(text_file_lists, excel_file_lists, threshold=0.8):
    """
    Find pairs of features with high correlation.
    
    Parameters:
    - observations: dict {obj_id: {TRACKING_DATA: tracks, TRUE_LABEL: label}}
    - threshold: correlation threshold (default 0.8)
    
    Returns:
    - high_corr_pairs: list of (feature1, feature2, correlation) tuples
    """
    # Extract features
    observations=load_and_label_all_observations(text_file_lists, excel_file_lists)
    classifier = SVMTrajectoryClassifier()
    X, y, obj_ids = classifier.extract_features_from_observations(observations)
    
    feature_names = [
        'mean_dx', 'mean_dy', 'std_dx', 'std_dy',
        'mean_speed', 'std_speed', 'max_speed',
        'total_distance', 'net_displacement', 'tortuosity',
        'mean_heading_cos', 'mean_heading_sin',
        'mean_abs_turn', 'std_abs_turn', 'max_abs_turn',
        'mean_turn_cos', 'mean_turn_sin',
        'mean_ax', 'mean_ay', 'std_ax', 'std_ay',
        'mean_accel_mag', 'std_accel_mag'
    ]
    
    # Compute correlation matrix
    correlation_matrix = np.corrcoef(X.T)
    
    # Find high correlations
    high_corr_pairs = []
    n_features = len(feature_names)
    
    for i in range(n_features):
        for j in range(i+1, n_features):  # Only upper triangle
            corr = correlation_matrix[i, j]
            if abs(corr) >= threshold:
                high_corr_pairs.append((feature_names[i], feature_names[j], corr))
    
    # Sort by absolute correlation
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    
    # Print results
    print("\n" + "="*80)
    print(f"HIGHLY CORRELATED FEATURE PAIRS (|correlation| >= {threshold})")
    print("="*80)
    print(f"{'Feature 1':<25} {'Feature 2':<25} {'Correlation':<15}")
    print("-"*80)
    
    if len(high_corr_pairs) == 0:
        print(f"No feature pairs with |correlation| >= {threshold}")
    else:
        for feat1, feat2, corr in high_corr_pairs:
            print(f"{feat1:<25} {feat2:<25} {corr:>14.3f}")
    
    print("="*80 + "\n")
    
    return high_corr_pairs