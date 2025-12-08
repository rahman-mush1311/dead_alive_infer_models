import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from collections import defaultdict

TRUE_LABEL = "true_label"
TRACKING_DATA = "tracking_data"
MOTILE = 1
NOTMOTILE = 0

class FeatureImportanceAnalyzer:
    def __init__(self, feature_names=['dx', 'dy', 'heading', 'turning', 'ax', 'ay']):
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        
        # Global importance scores
        self.global_importance = None
        
        # Per-grid importance scores
        self.grid_importance = None
        
    def compute_log_likelihood_contribution_global(self, dead_model, alive_model, 
                                                     observations, observation_stats):
        """
        Compute how much each feature contributes to log-likelihood ratio globally.
        
        Method: 
        - For each observation, compute log P(alive) - log P(dead)
        - Remove one feature at a time (marginalize) and recompute
        - Contribution = original_log_ratio - reduced_log_ratio
        
        Returns: dict with feature names as keys, importance scores as values
        """
        
        feature_contributions = {name: [] for name in self.feature_names}
        
        for file, obs_dict in observations.items():
            if file not in observation_stats:
                continue
                
            curr_stats = observation_stats[file]
            mu_dx, mu_dy = curr_stats['mu'][0], curr_stats['mu'][1]
            std_dx = np.sqrt(curr_stats['cov'][0,0])
            std_dy = np.sqrt(curr_stats['cov'][1,1])
            
            if std_dx < 1e-10: std_dx = 1.0
            if std_dy < 1e-10: std_dy = 1.0
            
            for obj_id, obj_data in obs_dict.items():
                obs = obj_data[TRACKING_DATA]
                
                if len(obs) < 3:
                    continue
                
                for i in range(2, len(obs)):
                    # Extract feature vector
                    feature_vec = self._extract_feature_vector(obs, i, mu_dx, mu_dy, std_dx, std_dy)
                    if feature_vec is None:
                        continue
                    
                    x, y = obs[i-1][0], obs[i-1][1]
                    
                    # Full log-likelihood ratio
                    log_p_alive = self._compute_log_prob(alive_model, x, y, feature_vec)
                    log_p_dead = self._compute_log_prob(dead_model, x, y, feature_vec)
                    full_log_ratio = log_p_alive - log_p_dead
                    
                    # Compute contribution of each feature
                    for feat_idx in range(self.n_features):
                        # Create reduced feature vector (marginalize out this feature)
                        reduced_log_ratio = self._compute_reduced_log_ratio(
                            dead_model, alive_model, x, y, feature_vec, feat_idx
                        )
                        
                        # Contribution = how much log-ratio changes without this feature
                        contribution = abs(full_log_ratio - reduced_log_ratio)
                        feature_contributions[self.feature_names[feat_idx]].append(contribution)
        
        # Aggregate contributions
        self.global_importance = {
            name: np.mean(scores) if scores else 0.0 
            for name, scores in feature_contributions.items()
        }
        
        return self.global_importance
    
    def compute_log_likelihood_contribution_per_grid(self, dead_model, alive_model, 
                                                      observations, observation_stats):
        """
        Compute feature importance for each grid cell separately.
        
        Returns: dict {(grid_row, grid_col): {feature_name: importance}}
        """
        
        grid_rows = dead_model.num_rows()
        grid_cols = dead_model.num_cols()
        
        # Initialize storage for each grid cell
        grid_contributions = defaultdict(lambda: {name: [] for name in self.feature_names})
        
        for file, obs_dict in observations.items():
            if file not in observation_stats:
                continue
                
            curr_stats = observation_stats[file]
            mu_dx, mu_dy = curr_stats['mu'][0], curr_stats['mu'][1]
            std_dx = np.sqrt(curr_stats['cov'][0,0])
            std_dy = np.sqrt(curr_stats['cov'][1,1])
            
            if std_dx < 1e-10: std_dx = 1.0
            if std_dy < 1e-10: std_dy = 1.0
            
            for obj_id, obj_data in obs_dict.items():
                obs = obj_data[TRACKING_DATA]
                
                if len(obs) < 3:
                    continue
                
                for i in range(2, len(obs)):
                    feature_vec = self._extract_feature_vector(obs, i, mu_dx, mu_dy, std_dx, std_dy)
                    if feature_vec is None:
                        continue
                    
                    x, y = obs[i-1][0], obs[i-1][1]
                    grid_row, grid_col = dead_model.find_grid_cell(x, y)
                    
                    # Full log-likelihood ratio
                    log_p_alive = self._compute_log_prob(alive_model, x, y, feature_vec)
                    log_p_dead = self._compute_log_prob(dead_model, x, y, feature_vec)
                    full_log_ratio = log_p_alive - log_p_dead
                    
                    # Compute contribution of each feature
                    for feat_idx in range(self.n_features):
                        reduced_log_ratio = self._compute_reduced_log_ratio(
                            dead_model, alive_model, x, y, feature_vec, feat_idx
                        )
                        
                        contribution = abs(full_log_ratio - reduced_log_ratio)
                        grid_contributions[(grid_row, grid_col)][self.feature_names[feat_idx]].append(contribution)
        
        # Aggregate per grid
        self.grid_importance = {}
        for (row, col), feat_dict in grid_contributions.items():
            self.grid_importance[(row, col)] = {
                name: np.mean(scores) if scores else 0.0 
                for name, scores in feat_dict.items()
            }
        
        return self.grid_importance
    
    def _extract_feature_vector(self, obs, i, mu_dx, mu_dy, std_dx, std_dy):
        """Extract 6D feature vector [dx, dy, heading, turning, ax, ay]"""
        x0, y0, f0 = obs[i-2]
        x1, y1, f1 = obs[i-1]
        x2, y2, f2 = obs[i]
        
        df1 = f1 - f0
        df2 = f2 - f1
        
        if df1 <= 0 or df2 <= 0:
            return None
        
        # Raw displacements
        dx1_raw = (x1 - x0) / df1
        dy1_raw = (y1 - y0) / df1
        dx2_raw = (x2 - x1) / df2
        dy2_raw = (y2 - y1) / df2
        
        # Normalized
        dx1_norm = (dx1_raw - mu_dx) / std_dx
        dy1_norm = (dy1_raw - mu_dy) / std_dy
        dx2_norm = (dx2_raw - mu_dx) / std_dx
        dy2_norm = (dy2_raw - mu_dy) / std_dy
        
        # Heading
        heading = np.arctan2(dy2_norm, dx2_norm)
        
        # Turning
        d1 = np.array([dx1_norm, dy1_norm])
        d2 = np.array([dx2_norm, dy2_norm])
        n1 = np.linalg.norm(d1)
        n2 = np.linalg.norm(d2)
        
        if n1 == 0 or n2 == 0:
            turning = 0.0
        else:
            cos_ang = np.clip(np.dot(d1, d2) / (n1 * n2), -1.0, 1.0)
            turning = np.arccos(cos_ang)
            cross = d1[0] * d2[1] - d1[1] * d2[0]
            if cross < 0:
                turning = -turning
        
        # Acceleration
        total_dt = df1 + df2
        if total_dt == 0:
            ax = ay = 0.0
        else:
            ax, ay = (d2 - d1) / total_dt
        
        return np.array([dx2_norm, dy2_norm, heading, turning, ax, ay])
    
    def _compute_log_prob(self, model, x, y, feature_vec):
        """Compute log probability using the model"""
        grid_row, grid_col = model.find_grid_cell(x, y)
        cell_mu = np.array(model.mu[grid_row][grid_col])
        cell_cov = model.cov_matrix[grid_row][grid_col]
        n = model.n[grid_row][grid_col]
        
        if n >= 1:
            try:
                mvn = scipy.stats.multivariate_normal(mean=cell_mu, cov=cell_cov)
                prob = mvn.pdf(feature_vec)
                if prob > 0:
                    return np.log(prob)
            except:
                pass
        
        return -1e10  # Very small log probability
    
    def _compute_reduced_log_ratio(self, dead_model, alive_model, x, y, feature_vec, exclude_idx):
        """
        Compute log-likelihood ratio with one feature marginalized out.
        
        Method: Set the excluded feature to the mean value (least informative)
        """
        reduced_feature = feature_vec.copy()
        
        grid_row, grid_col = dead_model.find_grid_cell(x, y)
        
        # Use mean from alive model for the excluded feature
        alive_mu = np.array(alive_model.mu[grid_row][grid_col])
        reduced_feature[exclude_idx] = alive_mu[exclude_idx]
        
        log_p_alive = self._compute_log_prob(alive_model, x, y, reduced_feature)
        log_p_dead = self._compute_log_prob(dead_model, x, y, reduced_feature)
        
        return log_p_alive - log_p_dead
    
    def visualize_global_importance(self, save_path=None):
        """Bar plot of global feature importance"""
        if self.global_importance is None:
            print("Run compute_log_likelihood_contribution_global first")
            return
        
        names = list(self.global_importance.keys())
        values = list(self.global_importance.values())
        
        # Sort by importance
        sorted_pairs = sorted(zip(names, values), key=lambda x: x[1], reverse=True)
        names, values = zip(*sorted_pairs)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, values, color='steelblue', edgecolor='black')
        
        # Highlight top 3
        for i in range(min(3, len(bars))):
            bars[i].set_color('coral')
        
        plt.xlabel('Feature', fontsize=12)
        plt.ylabel('Log-Likelihood Contribution', fontsize=12)
        plt.title('Feature Importance (Log-Likelihood Contribution)', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print values
        print("\n=== Global Feature Importance ===")
        for name, val in sorted_pairs:
            print(f"{name:10s}: {val:.4f}")
    
    def visualize_grid_importance(self, top_k=3, save_path=None):
        """Heatmap showing importance of top-k features per grid cell"""
        if self.grid_importance is None:
            print("Run compute_log_likelihood_contribution_per_grid first")
            return
        
        # Find top-k most important features globally
        all_scores = defaultdict(list)
        for grid_cell, feat_dict in self.grid_importance.items():
            for feat_name, score in feat_dict.items():
                all_scores[feat_name].append(score)
        
        avg_scores = {name: np.mean(scores) for name, scores in all_scores.items()}
        top_features = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        top_feature_names = [name for name, _ in top_features]
        
        # Get grid dimensions
        grid_rows = max(row for row, col in self.grid_importance.keys()) + 1
        grid_cols = max(col for row, col in self.grid_importance.keys()) + 1
        
        # Create subplots for top features
        fig, axes = plt.subplots(1, top_k, figsize=(5*top_k, 4))
        if top_k == 1:
            axes = [axes]
        
        for idx, feat_name in enumerate(top_feature_names):
            # Create heatmap matrix
            heatmap = np.zeros((grid_rows, grid_cols))
            for (row, col), feat_dict in self.grid_importance.items():
                heatmap[row, col] = feat_dict[feat_name]
            
            im = axes[idx].imshow(heatmap, cmap='YlOrRd', aspect='auto')
            axes[idx].set_title(f'{feat_name}', fontsize=12)
            axes[idx].set_xlabel('Grid Column')
            axes[idx].set_ylabel('Grid Row')
            
            # Add values to cells
            for i in range(grid_rows):
                for j in range(grid_cols):
                    text = axes[idx].text(j, i, f'{heatmap[i, j]:.2f}',
                                         ha="center", va="center", color="black", fontsize=9)
            
            plt.colorbar(im, ax=axes[idx])
        
        plt.suptitle(f'Top {top_k} Features: Spatial Importance', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print summary
        print(f"\n=== Top {top_k} Features (Global Average) ===")
        for name, score in top_features:
            print(f"{name:10s}: {score:.4f}")
    
    def print_grid_cell_summary(self, grid_row, grid_col):
        """Print feature importance for a specific grid cell"""
        if self.grid_importance is None:
            print("Run compute_log_likelihood_contribution_per_grid first")
            return
        
        if (grid_row, grid_col) not in self.grid_importance:
            print(f"Grid cell ({grid_row}, {grid_col}) has no data")
            return
        
        feat_dict = self.grid_importance[(grid_row, grid_col)]
        sorted_feats = sorted(feat_dict.items(), key=lambda x: x[1], reverse=True)
        
        print(f"\n=== Grid Cell ({grid_row}, {grid_col}) Feature Importance ===")
        for name, score in sorted_feats:
            print(f"{name:10s}: {score:.4f}")