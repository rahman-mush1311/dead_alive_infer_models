import numpy as np
import math
from scipy import stats
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score
from sklearn.model_selection import GridSearchCV

# String literals to constants
TRUE_LABEL = "true_label"
PREDICTED_LABEL = "predicted_label"
TRACKING_DATA = "tracking_data"

MOTILE = 1
NOTMOTILE = 0

class SVMTrajectoryClassifier:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        Initialize SVM classifier for trajectory-based motion classification.
        
        Parameters:
        - kernel: SVM kernel type ('linear', 'rbf', 'poly', 'sigmoid')
        - C: regularization parameter
        - gamma: kernel coefficient
        """
        self.svm_model = SVC(kernel=kernel, C=C, gamma=gamma, probability=True)
        self.scaler = StandardScaler()
        self.feature_names = []
    
    def compute_displacement_sequence(self, tracks):
        """
        Compute frame-by-frame displacements.
        
        Parameters:
        - tracks: list of (x, y, frame) tuples
        
        Returns:
        - displacements: list of (dx, dy) tuples
        """
        displacements = []
        for i in range(len(tracks) - 1):
            dframe = tracks[i+1][2] - tracks[i][2]
            if dframe > 0:
                dx = (tracks[i+1][0] - tracks[i][0]) / dframe
                dy = (tracks[i+1][1] - tracks[i][1]) / dframe
                displacements.append((dx, dy))
        return displacements
    
    def compute_heading_angles(self, displacements):
        """
        Compute heading angle for each displacement vector.
        
        Parameters:
        - displacements: list of (dx, dy) tuples
        
        Returns:
        - heading_angles: list of angles in radians [-π, π]
        """
        heading_angles = []
        for dx, dy in displacements:
            angle = math.atan2(dy, dx)
            heading_angles.append(angle)
        return heading_angles
    
    def compute_turning_angles(self, heading_angles):
        """
        Compute turning angles between consecutive heading directions.
        
        Parameters:
        - heading_angles: list of heading angles in radians
        
        Returns:
        - turning_angles: list of turning angles in radians
        """
        turning_angles = []
        for i in range(len(heading_angles) - 1):
            # Calculate angular difference, handling wraparound
            delta_angle = heading_angles[i+1] - heading_angles[i]
            # Normalize to [-π, π]
            delta_angle = math.atan2(math.sin(delta_angle), math.cos(delta_angle))
            turning_angles.append(delta_angle)
        return turning_angles
    
    def compute_accelerations(self, displacements):
        """
        Compute 2D acceleration vectors from displacement sequence.
        
        Parameters:
        - displacements: list of (dx, dy) tuples (velocities)
        
        Returns:
        - accelerations: list of (ax, ay) tuples
        """
        accelerations = []
        for i in range(len(displacements) - 1):
            ax = displacements[i+1][0] - displacements[i][0]
            ay = displacements[i+1][1] - displacements[i][1]
            accelerations.append((ax, ay))
        return accelerations
    
    def compute_total_distance(self, tracks):
        """
        Compute total distance traveled along trajectory.
        
        Parameters:
        - tracks: list of (x, y, frame) tuples
        
        Returns:
        - total_distance: scalar
        """
        total_distance = 0.0
        for i in range(len(tracks) - 1):
            dframe = tracks[i+1][2] - tracks[i][2]
            if dframe > 0:
                dx = tracks[i+1][0] - tracks[i][0]
                dy = tracks[i+1][1] - tracks[i][1]
                dist = math.sqrt(dx**2 + dy**2)
                total_distance += dist
        return total_distance
    
    def extract_features_from_trajectory(self, tracks):
        """
        Extract comprehensive kinematic features from a single trajectory.
        
        Parameters:
        - tracks: list of (x, y, frame) tuples
        
        Returns:
        - features: numpy array of shape (n_features,)
        """
        if len(tracks) < 2:
            return None
        
        # 1. Compute raw displacements
        displacements = self.compute_displacement_sequence(tracks)
        
        if len(displacements) < 2:
            return None
        
        # Convert to numpy arrays for easier computation
        disp_array = np.array(displacements)
        
        # 2. Displacement statistics (normalized)
        mean_dx = np.mean(disp_array[:, 0])
        mean_dy = np.mean(disp_array[:, 1])
        std_dx = np.std(disp_array[:, 0])
        std_dy = np.std(disp_array[:, 1])
        
        # 3. Speed statistics
        speeds = np.sqrt(disp_array[:, 0]**2 + disp_array[:, 1]**2)
        mean_speed = np.mean(speeds)
        std_speed = np.std(speeds)
        max_speed = np.max(speeds)
        
        # 4. Total distance traveled
        total_distance = self.compute_total_distance(tracks)
        
        # 5. Net displacement (straight-line distance from start to end)
        net_displacement = math.sqrt(
            (tracks[-1][0] - tracks[0][0])**2 + 
            (tracks[-1][1] - tracks[0][1])**2
        )
        
        # 6. Tortuosity (ratio of path length to net displacement)
        tortuosity = total_distance / net_displacement if net_displacement > 0 else 0
        
        # 7. Heading angle statistics
        heading_angles = self.compute_heading_angles(displacements)
        
        # Convert heading angles to Cartesian coordinates to handle circularity
        heading_cos = np.array([math.cos(a) for a in heading_angles])
        heading_sin = np.array([math.sin(a) for a in heading_angles])
        mean_heading_cos = np.mean(heading_cos)
        mean_heading_sin = np.mean(heading_sin)
        
        # 8. Turning angle statistics
        if len(heading_angles) > 1:
            turning_angles = self.compute_turning_angles(heading_angles)
            turning_array = np.array(turning_angles)
            
            # Absolute turning statistics
            abs_turns = np.abs(turning_array)
            mean_abs_turn = np.mean(abs_turns)
            std_abs_turn = np.std(abs_turns)
            max_abs_turn = np.max(abs_turns)
            '''
            # Count sharp turns (> 30 degrees)
            sharp_turn_threshold = math.radians(30)
            n_sharp_turns = np.sum(abs_turns > sharp_turn_threshold)
            sharp_turn_fraction = n_sharp_turns / len(turning_angles)
            '''
            # Convert turning angles to Cartesian for mean direction
            turn_cos = np.array([math.cos(a) for a in turning_angles])
            turn_sin = np.array([math.sin(a) for a in turning_angles])
            mean_turn_cos = np.mean(turn_cos)
            mean_turn_sin = np.mean(turn_sin)
        else:
            mean_abs_turn = 0
            std_abs_turn = 0
            max_abs_turn = 0
            #n_sharp_turns = 0
            #sharp_turn_fraction = 0
            mean_turn_cos = 0
            mean_turn_sin = 0
        
        # 9. Acceleration statistics
        if len(displacements) > 1:
            accelerations = self.compute_accelerations(displacements)
            accel_array = np.array(accelerations)
            
            mean_ax = np.mean(accel_array[:, 0])
            mean_ay = np.mean(accel_array[:, 1])
            std_ax = np.std(accel_array[:, 0])
            std_ay = np.std(accel_array[:, 1])
            
            # Acceleration magnitude
            accel_mag = np.sqrt(accel_array[:, 0]**2 + accel_array[:, 1]**2)
            mean_accel_mag = np.mean(accel_mag)
            std_accel_mag = np.std(accel_mag)
        else:
            mean_ax = 0
            mean_ay = 0
            std_ax = 0
            std_ay = 0
            mean_accel_mag = 0
            std_accel_mag = 0
        
        # 10. Directional persistence (correlation between consecutive displacements)
        if len(displacements) > 1:
            directional_correlations = []
            for i in range(len(displacements) - 1):
                v1 = np.array(displacements[i])
                v2 = np.array(displacements[i+1])
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                if norm1 > 0 and norm2 > 0:
                    correlation = np.dot(v1, v2) / (norm1 * norm2)
                    directional_correlations.append(correlation)
            mean_dir_persistence = np.mean(directional_correlations) if directional_correlations else 0
        else:
            mean_dir_persistence = 0
        
        # Compile all features into a single array
        features = np.array([
            mean_dx,                # 0: mean displacement X
            mean_dy,                # 1: mean displacement Y
            std_dx,                 # 2: std displacement X
            std_dy,                 # 3: std displacement Y
            mean_speed,             # 4: mean speed
            std_speed,              # 5: std speed
            max_speed,              # 6: max speed
            total_distance,         # 7: total distance
            net_displacement,       # 8: net displacement
            tortuosity,             # 9: tortuosity
            mean_heading_cos,       # 10: mean heading (cos component)
            mean_heading_sin,       # 11: mean heading (sin component)
            mean_abs_turn,          # 12: mean absolute turning angle
            std_abs_turn,           # 13: std absolute turning angle
            max_abs_turn,           # 14: max absolute turning angle
            mean_turn_cos,          # 15: mean turn (cos component)
            mean_turn_sin,          # 16: mean turn (sin component)
            mean_ax,                # 17: mean acceleration X
            mean_ay,                # 18: mean acceleration Y
            std_ax,                 # 19: std acceleration X
            std_ay,                 # 20: std acceleration Y
            mean_accel_mag,         # 21: mean acceleration magnitude
            std_accel_mag,          # 22: std acceleration magnitude
        ])
        
        return features
    
    def extract_features_from_observations(self, observations):
        """
        Extract features from all observations in a dictionary.
        
        Parameters:
        - observations: dict {obj_id: {TRACKING_DATA: tracks, TRUE_LABEL: label}}
        
        Returns:
        - X: feature matrix (n_objects, n_features)
        - y: label vector (n_objects,)
        - valid_obj_ids: list of object IDs with valid features
        """
        X_list = []
        y_list = []
        valid_obj_ids = []
        
        for obj_id, data in observations.items():
            tracks = data[TRACKING_DATA]
            label = data[TRUE_LABEL]
            
            features = self.extract_features_from_trajectory(tracks)
            
            if features is not None and not np.any(np.isnan(features)) and not np.any(np.isinf(features)):
                X_list.append(features)
                y_list.append(label)
                valid_obj_ids.append(obj_id)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        return X, y, valid_obj_ids
    
    def train(self, train_observations):
        """
        Train the SVM model on training observations.
        
        Parameters:
        - train_observations: dict {obj_id: {TRACKING_DATA: tracks, TRUE_LABEL: label}}
        
        Returns:
        - training accuracy
        """
        print("Extracting features from training data...")
        X_train, y_train, train_obj_ids = self.extract_features_from_observations(train_observations)
        
        print(f"Training on {len(X_train)} objects with {X_train.shape[1]} features each")
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train SVM
        print("Training SVM model...")
        self.svm_model.fit(X_train_scaled, y_train)
        
        # Compute training accuracy
        y_train_pred = self.svm_model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        print(f"Training accuracy: {train_accuracy:.3f}")
        
        return train_accuracy
    
    def predict(self, test_observations):
        """
        Predict labels for test observations.
        
        Parameters:
        - test_observations: dict {obj_id: {TRACKING_DATA: tracks, TRUE_LABEL: label}}
        
        Returns:
        - predictions: dict {obj_id: {TRUE_LABEL: label, PREDICTED_LABEL: pred}}
        """
        print("Extracting features from test data...")
        X_test, y_test, test_obj_ids = self.extract_features_from_observations(test_observations)
        
        print(f"Predicting on {len(X_test)} objects...")
        
        # Normalize features using training scaler
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predict
        y_pred = self.svm_model.predict(X_test_scaled)
        
        # Compile results
        predictions = {}
        for i, obj_id in enumerate(test_obj_ids):
            predictions[obj_id] = {
                TRUE_LABEL: test_observations[obj_id][TRUE_LABEL],
                PREDICTED_LABEL: int(y_pred[i])
            }
        
        return predictions
    
    def evaluate(self, test_observations):
        """
        Evaluate SVM model on test data and print metrics.
        
        Parameters:
        - test_observations: dict {obj_id: {TRACKING_DATA: tracks, TRUE_LABEL: label}}
        
        Returns:
        - metrics: dict with accuracy, f1, recall, precision
        """
        predictions = self.predict(test_observations)
        
        y_true = [predictions[obj_id][TRUE_LABEL] for obj_id in predictions]
        y_pred = [predictions[obj_id][PREDICTED_LABEL] for obj_id in predictions]
        
        # Compute metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, pos_label=MOTILE)
        recall = recall_score(y_true, y_pred, pos_label=MOTILE)
        precision = precision_score(y_true, y_pred, pos_label=MOTILE, zero_division=0)
        
        print(f"\n{'='*50}")
        print(f"SVM Model Evaluation")
        print(f"{'='*50}")
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"F1-Score:  {f1:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"{'='*50}\n")
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[NOTMOTILE, MOTILE])
        print("Confusion Matrix:")
        print(f"              Predicted")
        print(f"              Not-Motile  Motile")
        print(f"Actual Not-Motile   {cm[0,0]:3d}      {cm[0,1]:3d}")
        print(f"       Motile       {cm[1,0]:3d}      {cm[1,1]:3d}")
        print()
        
        metrics = {
            'accuracy': accuracy,
            'f1': f1,
            'recall': recall,
            'precision': precision,
            'confusion_matrix': cm
        }
        
        return metrics, predictions
    '''
    def hyperparameter_search(self, train_observations, param_grid=None):
        """
        Perform grid search for optimal SVM hyperparameters.
        
        Parameters:
        - train_observations: dict {obj_id: {TRACKING_DATA: tracks, TRUE_LABEL: label}}
        - param_grid: dict of parameters to search (optional)
        
        Returns:
        - best_params: dict of best parameters found
        """
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }
        
        print("Extracting features from training data...")
        X_train, y_train, train_obj_ids = self.extract_features_from_observations(train_observations)
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print("Performing grid search for optimal hyperparameters...")
        grid_search = GridSearchCV(
            SVC(probability=True),
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        print(f"\nBest parameters: {grid_search.best_params_}")
        print(f"Best cross-validation F1 score: {grid_search.best_score_:.3f}")
        
        # Update model with best parameters
        self.svm_model = grid_search.best_estimator_
        
        return grid_search.best_params_
    '''
    def hyperparameter_search(self, train_observations, param_grid=None, show_all_results=True,max_detailed_results=None):
        """
        Perform grid search for optimal SVM hyperparameters.
        
        Parameters:
        - train_observations: dict {obj_id: {TRACKING_DATA: tracks, TRUE_LABEL: label}}
        - param_grid: dict of parameters to search (optional)
        - show_all_results: if True, display results for all parameter combinations
        
        Returns:
        - best_params: dict of best parameters found
        - results_df: pandas DataFrame with all results (if pandas available)
        """
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
                'kernel': ['rbf', 'linear']
            }
        
        print("Extracting features from training data...")
        X_train, y_train, train_obj_ids = self.extract_features_from_observations(train_observations)
        
        # Normalize features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        print(f"\nPerforming grid search for optimal hyperparameters...")
        print(f"Parameter grid:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        # Calculate total combinations
        total_combinations = 1
        for values in param_grid.values():
            total_combinations *= len(values)
        print(f"\nTotal combinations to test: {total_combinations}")
        print(f"Using 5-fold cross-validation\n")
        
        grid_search = GridSearchCV(
            SVC(probability=True),
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1,
            return_train_score=True  # Also return training scores
        )
        
        grid_search.fit(X_train_scaled, y_train)
        
        # Display all results if requested
        if show_all_results:
            print("\n" + "="*80)
            print("DETAILED RESULTS FOR ALL HYPERPARAMETER COMBINATIONS")
            print("="*80)
            
            # Extract results
            results = grid_search.cv_results_
            
            # Sort by rank
            sorted_indices = np.argsort(results['rank_test_score'])
            
            print(f"\n{'Rank':<6} {'Kernel':<8} {'C':<10} {'Gamma':<10} {'Mean F1':<10} {'Std F1':<10} {'Mean Train F1':<14}")
            print("-" * 80)
            
            for idx in sorted_indices:
                rank = results['rank_test_score'][idx]
                params = results['params'][idx]
                mean_test_score = results['mean_test_score'][idx]
                std_test_score = results['std_test_score'][idx]
                mean_train_score = results['mean_train_score'][idx]
                
                kernel = params.get('kernel', 'N/A')
                C = params.get('C', 'N/A')
                gamma = params.get('gamma', 'N/A')
                
                # Format gamma for display
                if isinstance(gamma, float):
                    gamma_str = f"{gamma:.4f}"
                else:
                    gamma_str = str(gamma)
                
                print(f"{int(rank):<6} {kernel:<8} {C:<10} {gamma_str:<10} "
                      f"{mean_test_score:<10.4f} {std_test_score:<10.4f} {mean_train_score:<14.4f}")
            
            print("="*80)
            
            # Show combinations in detail
            if max_detailed_results is None:
                num_to_show = len(sorted_indices)
                header_text = f"ALL {num_to_show} PARAMETER COMBINATIONS (Detailed)"
            else:
                num_to_show = min(max_detailed_results, len(sorted_indices))
                header_text = f"TOP {num_to_show} PARAMETER COMBINATIONS (Detailed)"
            
            print("\n" + "="*80)
            print(header_text)
            print("="*80)
            '''
            # Show top 5 combinations in detail
            print("\n" + "="*80)
            print("TOP 5 PARAMETER COMBINATIONS (Detailed)")
            print("="*80)
            '''
            for i, idx in enumerate(sorted_indices[:num_to_show], 1):
                params = results['params'][idx]
                mean_test_score = results['mean_test_score'][idx]
                std_test_score = results['std_test_score'][idx]
                mean_train_score = results['mean_train_score'][idx]
                
                print(f"\n#{i} Rank {int(results['rank_test_score'][idx])}")
                print(f"  Parameters: {params}")
                print(f"  Cross-validation F1 Score: {mean_test_score:.4f} (+/- {std_test_score:.4f})")
                print(f"  Training F1 Score:          {mean_train_score:.4f}")
                print(f"  Overfitting gap:            {mean_train_score - mean_test_score:.4f}")
            
            print("="*80)
            
            # Try to create pandas DataFrame for easier analysis
            try:
                
                
                # Create DataFrame with results
                results_data = []
                for idx in range(len(results['params'])):
                    row = {
                        'rank': int(results['rank_test_score'][idx]),
                        'kernel': results['params'][idx].get('kernel', 'N/A'),
                        'C': results['params'][idx].get('C', 'N/A'),
                        'gamma': results['params'][idx].get('gamma', 'N/A'),
                        'mean_test_f1': results['mean_test_score'][idx],
                        'std_test_f1': results['std_test_score'][idx],
                        'mean_train_f1': results['mean_train_score'][idx],
                        'overfitting_gap': results['mean_train_score'][idx] - results['mean_test_score'][idx]
                    }
                    results_data.append(row)
                
                results_df = pd.DataFrame(results_data).sort_values('rank')
                
                print("\nResults saved as DataFrame (accessible as return value)")
                
            except ImportError:
                results_df = None
                print("\nNote: Install pandas to get results as DataFrame")
        else:
            results_df = None
        
        print(f"\n{'='*80}")
        print("BEST PARAMETERS FOUND")
        print(f"{'='*80}")
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation F1 score: {grid_search.best_score_:.3f}")
        print(f"{'='*80}\n")
        
        # Update model with best parameters
        self.svm_model = grid_search.best_estimator_
        
        if show_all_results and results_df is not None:
            return grid_search.best_params_, results_df
        else:
            return grid_search.best_params_
    
    






