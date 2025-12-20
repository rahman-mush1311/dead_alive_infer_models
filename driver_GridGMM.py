import numpy
import scipy.stats
from sklearn.mixture import GaussianMixture
from collections import Counter
import warnings

# String literals to constants
TRUE_LABEL = "true_label"
LOG_PDFS = "log_pdfs"
TRACKING_DATA = "tracking_data"

MOTILE = 1
NOTMOTILE = 0


class GridGMMModel:
    """
    Grid-based Gaussian Mixture Model for 2D displacement modeling.
    
    Each grid cell contains a GMM with automatically selected number of components (via BIC/AIC).
    Supports weighted averaging of GMMs from multiple files.
    """
    
    def __init__(self, grid_rows=3, grid_cols=3, max_x=4128, max_y=2196, 
                 min_samples_per_component=10, max_components=3):
        """
        Initialize GridGMMModel.
        
        Parameters:
        -----------
        grid_rows : int
            Number of grid rows (default 5)
        grid_cols : int
            Number of grid columns (default 5)
        max_x : int
            Maximum x coordinate (default 4128)
        max_y : int
            Maximum y coordinate (default 2196)
        min_samples_per_component : int
            Minimum samples needed per component (default 10)
            Total samples needed = n_components * min_samples_per_component
        max_components : int
            Maximum number of GMM components to try (default 5)
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.max_x = max_x
        self.max_y = max_y
        self.min_samples_per_component = min_samples_per_component
        self.max_components = max_components
        
        # Storage for each cell: GMM parameters
        # Each cell stores dict: {
        #   'n_components': K, 
        #   'weights': array of shape (K,),
        #   'means': array of shape (K, 2),
        #   'covariances': array of shape (K, 2, 2)
        # }
        self.gmm_params = [[None for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        # Track number of observations per cell
        self.n = [[0 for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        # Normalization statistics (set from outside, per-file)
        self.total_mu = numpy.array([0.0, 0.0])
        self.total_cov_matrix = numpy.zeros((2, 2))
    
    def calculate_displacements(self, observations):
        """
        Calculate normalized displacements and assign to grid cells.
        
        Parameters:
        -----------
        observations : dict
            {obj_id: [(x, y, frame), ...]}
            
        Returns:
        --------
        grid_displacements : list of list of lists
            [rows][cols] -> list of (dx_norm, dy_norm) tuples
        """
        grid_dis = [[[] for _ in range(self.num_cols())] for _ in range(self.num_rows())]
        
        # Get normalization parameters
        mu_x, mu_y = self.total_mu[0], self.total_mu[1]
        std_x = numpy.sqrt(self.total_cov_matrix[0, 0])
        std_y = numpy.sqrt(self.total_cov_matrix[1, 1])
        
        # Prevent division by zero
        if std_x < 1e-10:
            std_x = 1.0
        if std_y < 1e-10:
            std_y = 1.0
        
        points = []
        
        for obj_id, obs in observations.items():
            for i in range(len(obs) - 1):
                dframe = obs[i+1][2] - obs[i][2]
                
                if dframe > 0:
                    # Raw displacement
                    dx = (obs[i+1][0] - obs[i][0]) / dframe
                    dy = (obs[i+1][1] - obs[i][1]) / dframe
                    
                    # Normalize
                    dx_norm = (dx - mu_x) / std_x
                    dy_norm = (dy - mu_y) / std_y
                    
                    # Find grid cell based on starting position
                    grid_row, grid_col = self.find_grid_cell(obs[i][0], obs[i][1])
                    
                    # Store
                    grid_dis[grid_row][grid_col].append((dx_norm, dy_norm))
                    self.n[grid_row][grid_col] += 1
                    points.append((dx_norm, dy_norm))
                else:
                    print(f"Warning: Invalid frame distance {dframe} for object {obj_id}")
        
        if len(points) < 1:
            print("Warning: No valid displacements found")
        
        return grid_dis
    
    def select_optimal_n_components(self, data, max_components=None):
        """
        Select optimal number of GMM components using BIC.
        
        Parameters:
        -----------
        data : array-like, shape (n_samples, 2)
            Data to fit
        max_components : int, optional
            Maximum number of components to try (default: self.max_components)
            
        Returns:
        --------
        optimal_k : int
            Optimal number of components (1 to max_components)
        """
        if max_components is None:
            max_components = self.max_components
        
        n_samples = len(data)
        
        # Limit max_components based on available data
        # Rule of thumb: need at least min_samples_per_component per component
        max_feasible = max(1, n_samples // self.min_samples_per_component)
        max_components = min(max_components, max_feasible)
        
        if max_components < 1:
            return 1
        
        best_bic = numpy.inf
        best_k = 1
        
        for k in range(1, max_components + 1):
            try:
                gmm = GaussianMixture(n_components=k, covariance_type='full', 
                                     random_state=42, max_iter=100)
                gmm.fit(data)
                bic = gmm.bic(data)
                
                if bic < best_bic:
                    best_bic = bic
                    best_k = k
            except Exception as e:
                # If fitting fails, skip this k
                warnings.warn(f"GMM fitting failed for k={k}: {e}")
                continue
        
        return best_k
    
    def fit_gmm_per_cell(self, grid_displacements):
        """
        Fit GMM for each grid cell with optimal component selection.
        
        Parameters:
        -----------
        grid_displacements : list of list of lists
            [rows][cols] -> list of (dx_norm, dy_norm) tuples
            
        Returns:
        --------
        None (updates self.gmm_params)
        """
        for row in range(self.num_rows()):
            for col in range(self.num_cols()):
                n_obs = self.n[row][col]
                
                if n_obs < self.min_samples_per_component:
                    print(f"Cell [{row}][{col}]: Insufficient data ({n_obs} obs), skipping GMM fit")
                    self.gmm_params[row][col] = None
                    continue
                
                # Convert to numpy array
                data = numpy.array(grid_displacements[row][col])
                
                # Select optimal number of components
                optimal_k = self.select_optimal_n_components(data)
                
                # Fit GMM with optimal K
                try:
                    gmm = GaussianMixture(n_components=optimal_k, covariance_type='full',
                                         random_state=42, max_iter=100)
                    gmm.fit(data)
                    
                    # Store parameters
                    self.gmm_params[row][col] = {
                        'n_components': optimal_k,
                        'weights': gmm.weights_.copy(),
                        'means': gmm.means_.copy(),
                        'covariances': gmm.covariances_.copy()
                    }
                    
                    print(f"Cell [{row}][{col}]: Fitted GMM with K={optimal_k} components ({n_obs} obs)")
                    
                except Exception as e:
                    print(f"Cell [{row}][{col}]: GMM fitting failed - {e}")
                    self.gmm_params[row][col] = None
    
    def calculate_parameters(self, grid_displacements):
        """
        Calculate GMM parameters for each grid cell.
        Wrapper for fit_gmm_per_cell() to match API of other models.
        
        Parameters:
        -----------
        grid_displacements : list of list of lists
            [rows][cols] -> list of (dx_norm, dy_norm) tuples
        """
        self.fit_gmm_per_cell(grid_displacements)
    
    def match_components_euclidean(self, means1, means2):
        """
        Match GMM components between two models using Euclidean distance.
        Uses greedy assignment: each component in means1 matched to closest in means2.
        
        Parameters:
        -----------
        means1 : array, shape (K1, 2)
            Means from first GMM
        means2 : array, shape (K2, 2)
            Means from second GMM
            
        Returns:
        --------
        matching : list of tuples
            [(idx1, idx2), ...] where idx1 is component in means1, 
            idx2 is matched component in means2
            If K1 != K2, unmatched components paired with None
        """
        K1 = len(means1)
        K2 = len(means2)
        
        # Greedy matching: for each component in means1, find closest in means2
        used_j = set()
        matching = []
        
        for i in range(K1):
            min_dist = numpy.inf
            best_j = None
            
            for j in range(K2):
                if j not in used_j:
                    dist = numpy.linalg.norm(means1[i] - means2[j])
                    if dist < min_dist:
                        min_dist = dist
                        best_j = j
            
            if best_j is not None:
                matching.append((i, best_j))
                used_j.add(best_j)
            else:
                matching.append((i, None))
        
        # Handle unmatched components in means2
        for j in range(K2):
            if j not in used_j:
                matching.append((None, j))
        
        return matching
    
    def select_target_k_weighted_mode(self, models_list):
        """
        Select target number of components using weighted mode.
        
        Parameters:
        -----------
        models_list : list of GridGMMModel
            List of models to combine
            
        Returns:
        --------
        target_k_per_cell : list of list of int
            [rows][cols] -> target K for that cell
        """
        target_k = [[0 for _ in range(self.num_cols())] for _ in range(self.num_rows())]
        
        for row in range(self.num_rows()):
            for col in range(self.num_cols()):
                # Collect (K, weight) pairs from all models
                k_weights = []
                
                for model in models_list:
                    if model.gmm_params[row][col] is not None:
                        k = model.gmm_params[row][col]['n_components']
                        weight = model.n[row][col]
                        k_weights.append((k, weight))
                
                if not k_weights:
                    target_k[row][col] = 0
                    continue
                
                # Find weighted mode
                k_vote = {}
                for k, w in k_weights:
                    k_vote[k] = k_vote.get(k, 0) + w
                
                # Select K with highest total weight
                best_k = max(k_vote.keys(), key=lambda k: k_vote[k])
                target_k[row][col] = best_k
        
        return target_k
    
    def pad_or_merge_to_target_k(self, gmm_param, target_k):
        """
        Adjust GMM to have target_k components.
        
        If current K < target_k: pad with zero-weight dummy components
        If current K > target_k: merge closest components (simple approach: keep top-K by weight)
        If current K == target_k: return as is
        
        Parameters:
        -----------
        gmm_param : dict
            GMM parameters {'n_components', 'weights', 'means', 'covariances'}
        target_k : int
            Target number of components
            
        Returns:
        --------
        adjusted_param : dict
            GMM parameters with target_k components
        """
        current_k = gmm_param['n_components']
        
        if current_k == target_k:
            return gmm_param
        
        elif current_k < target_k:
            # Pad with zero-weight components
            n_pad = target_k - current_k
            
            weights = numpy.concatenate([gmm_param['weights'], numpy.zeros(n_pad)])
            
            # Dummy means: just use origin (won't affect probability due to zero weight)
            dummy_means = numpy.zeros((n_pad, 2))
            means = numpy.vstack([gmm_param['means'], dummy_means])
            
            # Dummy covariances: identity matrices
            dummy_covs = numpy.array([numpy.eye(2) for _ in range(n_pad)])
            covariances = numpy.concatenate([gmm_param['covariances'], dummy_covs], axis=0)
            
            return {
                'n_components': target_k,
                'weights': weights,
                'means': means,
                'covariances': covariances
            }
        
        else:  # current_k > target_k
            # Keep top-K components by weight
            indices = numpy.argsort(gmm_param['weights'])[::-1][:target_k]
            
            weights = gmm_param['weights'][indices]
            weights = weights / weights.sum()  # Renormalize
            
            means = gmm_param['means'][indices]
            covariances = gmm_param['covariances'][indices]
            
            return {
                'n_components': target_k,
                'weights': weights,
                'means': means,
                'covariances': covariances
            }
    
    def add_models(self, *others):
        """
        Combine multiple GridGMMModels using weighted averaging.
        
        Strategy:
        1. Select target K per cell (weighted mode)
        2. Pad/merge each model's components to target K
        3. Match components across models (Euclidean distance)
        4. Weighted average of matched components
        
        Parameters:
        -----------
        *others : GridGMMModel instances
            Other models to combine with self
            
        Returns:
        --------
        combined : GridGMMModel
            New model with combined parameters
        """
        # Verify compatibility
        for o in others:
            assert self.grid_rows == o.grid_rows, "Grid rows must match"
            assert self.grid_cols == o.grid_cols, "Grid cols must match"
            assert self.max_x == o.max_x, "max_x must match"
            assert self.max_y == o.max_y, "max_y must match"
        
        combined = GridGMMModel(self.grid_rows, self.grid_cols, self.max_x, self.max_y,
                               self.min_samples_per_component, self.max_components)
        
        models = [self] + list(others)
        
        # Select target K per cell
        target_k_grid = self.select_target_k_weighted_mode(models)
        
        print("\n" + "="*70)
        print("COMBINING GMM MODELS")
        print("="*70)
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                target_k = target_k_grid[row][col]
                
                if target_k == 0:
                    print(f"Cell [{row}][{col}]: No valid models to combine")
                    combined.gmm_params[row][col] = None
                    combined.n[row][col] = 0
                    continue
                
                # Filter valid models for this cell
                valid_models = [m for m in models if m.gmm_params[row][col] is not None]
                
                if not valid_models:
                    combined.gmm_params[row][col] = None
                    combined.n[row][col] = 0
                    continue
                
                # Adjust all models to target_k
                adjusted_params = []
                weights_per_model = []
                
                for m in valid_models:
                    adjusted = self.pad_or_merge_to_target_k(m.gmm_params[row][col], target_k)
                    adjusted_params.append(adjusted)
                    weights_per_model.append(m.n[row][col])
                
                # Total observations
                n_total = sum(weights_per_model)
                p_values = [w / n_total for w in weights_per_model]
                
                # Initialize combined parameters (all models now have target_k components)
                # Use first model as reference for component ordering
                ref_means = adjusted_params[0]['means']
                
                combined_weights = numpy.zeros(target_k)
                combined_means = numpy.zeros((target_k, 2))
                combined_covs = numpy.zeros((target_k, 2, 2))
                
                # Match components using greedy assignment PER MODEL to prevent reuse
                # For each model, match its K components to reference K components
                matched_components = [[] for _ in range(target_k)]  # matched_components[k] = list of (model_idx, comp_idx)
                
                for model_idx, param in enumerate(adjusted_params):
                    # Compute distance matrix: ref_comp x model_comp
                    distance_matrix = numpy.zeros((target_k, target_k))
                    for ref_k in range(target_k):
                        for model_k in range(target_k):
                            distance_matrix[ref_k, model_k] = numpy.linalg.norm(
                                ref_means[ref_k] - param['means'][model_k]
                            )
                    
                    # Greedy assignment: for each reference component, find best available model component
                    used_model_comps = set()
                    assignments = {}  # ref_k -> model_k
                    
                    for ref_k in range(target_k):
                        # Find closest unused model component
                        best_model_k = None
                        best_dist = float('inf')
                        
                        for model_k in range(target_k):
                            if model_k not in used_model_comps:
                                dist = distance_matrix[ref_k, model_k]
                                if dist < best_dist:
                                    best_dist = dist
                                    best_model_k = model_k
                        
                        if best_model_k is not None:
                            assignments[ref_k] = best_model_k
                            used_model_comps.add(best_model_k)
                            matched_components[ref_k].append((model_idx, best_model_k))
                    
                    # Sanity check: should have K assignments
                    if len(assignments) != target_k:
                        print(f"Warning: Model {model_idx} only matched {len(assignments)}/{target_k} components")
                
                # Now average matched components
                for k in range(target_k):
                    matches = matched_components[k]  # List of (model_idx, comp_idx) tuples
                    
                    # Weighted average using ONLY model weights (not component weights)
                    # This treats all components equally, weighted only by model size
                    mean_sum = numpy.zeros(2)
                    cov_sum = numpy.zeros((2, 2))
                    weight_sum_unnorm = 0.0  # Sum of component weights (for renormalization)
                    
                    for model_idx, comp_idx in matches:
                        param = adjusted_params[model_idx]
                        model_weight = p_values[model_idx]  # Model's weight by n_obs
                        comp_weight = param['weights'][comp_idx]  # Component's weight in GMM
                        
                        # Accumulate for mean/cov using model weight only
                        mean_sum += model_weight * param['means'][comp_idx]
                        cov_sum += model_weight * param['covariances'][comp_idx]
                        
                        # Accumulate component weights for final GMM weight
                        weight_sum_unnorm += model_weight * comp_weight
                    
                    # Store averaged parameters
                    combined_means[k] = mean_sum  # Already weighted by p_values (sum to 1)
                    combined_covs[k] = cov_sum    # Already weighted by p_values (sum to 1)
                    combined_weights[k] = weight_sum_unnorm  # Will be renormalized later
                
                # Renormalize weights
                if combined_weights.sum() > 0:
                    combined_weights = combined_weights / combined_weights.sum()
                
                combined.gmm_params[row][col] = {
                    'n_components': target_k,
                    'weights': combined_weights,
                    'means': combined_means,
                    'covariances': combined_covs
                }
                combined.n[row][col] = n_total
                
                print(f"Cell [{row}][{col}]: Combined {len(valid_models)} models -> K={target_k} ({n_total} total obs)")
        
        print("="*70 + "\n")
        
        return combined
    
    def compute_probabilities(self, observations, dx_norm, dy_norm, sx_norm, sy_norm):
        """
        Compute log probabilities for all observations using GMM.
        
        Parameters:
        -----------
        observations : dict
            {obj_id: [(x, y, frame), ...]}
        dx_norm, dy_norm : float
            Mean displacements for normalization
        sx_norm, sy_norm : float
            Std deviations for normalization
            
        Returns:
        --------
        probabilities : dict
            {obj_id: {LOG_PDFS: [log_probs]}}
        """
        probabilities = {}
        empty_obs = 0
        
        for obj_id, obs in observations.items():
            obj_probabilities = []
            
            for i in range(len(obs) - 1):
                x, y = obs[i][0], obs[i][1]
                dframe = obs[i+1][2] - obs[i][2]
                
                if dframe > 0:
                    # Raw displacement
                    dx = (obs[i+1][0] - obs[i][0]) / dframe
                    dy = (obs[i+1][1] - obs[i][1]) / dframe
                    
                    # Normalize
                    norm_dx = (dx - dx_norm) / sx_norm
                    norm_dy = (dy - dy_norm) / sy_norm
                    
                    # Compute probability
                    prob = self.probability(x, y, norm_dx, norm_dy)
                    obj_probabilities.append(prob)
                else:
                    print(f"Warning: Invalid frame distance {dframe} for object {obj_id} at position ({x},{y})")
            
            if len(obs) - 1 <= 0:
                empty_obs += 1
                print(f"Warning: Object {obj_id} has {len(obs)} observations, no displacements")
            else:
                assert len(obj_probabilities) == len(obs) - 1, \
                    f"Mismatch: {obj_id} has {len(obj_probabilities)} probabilities but {len(obs)-1} displacements"
            
            log_obj_probabilities = self.log_probability(obj_probabilities)
            
            if len(log_obj_probabilities) >= 1:
                probabilities[obj_id] = {LOG_PDFS: log_obj_probabilities}
        
        return probabilities
    
    def probability(self, x, y, dx_norm, dy_norm):
        """
        Calculate probability using GMM for the grid cell.
        
        Parameters:
        -----------
        x, y : float
            Position coordinates (to determine grid cell)
        dx_norm, dy_norm : float
            Normalized displacement
            
        Returns:
        --------
        prob : float
            Probability value
        """
        grid_row, grid_col = self.find_grid_cell(x, y)
        cell_gmm = self.gmm_params[grid_row][grid_col]
        n = self.n[grid_row][grid_col]
        
        if cell_gmm is None or n < 1:
            # Don't print warning every time, just return default
            return 1e-10  # Small non-zero value
        
        # GMM probability: sum over all components
        point = numpy.array([dx_norm, dy_norm]).reshape(1, -1)
        
        total_prob = 0.0
        component_probs = []
        
        for k in range(cell_gmm['n_components']):
            weight = cell_gmm['weights'][k]
            mean = cell_gmm['means'][k]
            cov = cell_gmm['covariances'][k]
            
            if weight > 1e-10:  # Skip zero-weight components
                try:
                    # Check for singular covariance
                    if numpy.linalg.det(cov) < 1e-10:
                        # Regularize singular covariance
                        cov = cov + numpy.eye(2) * 1e-6
                    
                    mvn = scipy.stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True)
                    component_prob = mvn.pdf(point.flatten())
                    weighted_prob = weight * component_prob
                    total_prob += weighted_prob
                    component_probs.append(weighted_prob)
                except Exception as e:
                    # If this component fails, skip it
                    continue
        
        # Ensure we return a reasonable value
        if total_prob <= 0 or numpy.isnan(total_prob) or numpy.isinf(total_prob):
            return 1e-10
        
        return total_prob
    
    def log_probability(self, curr_pdf_list):
        """
        Convert probabilities to log probabilities.
        
        Parameters:
        -----------
        curr_pdf_list : list
            List of probability values
            
        Returns:
        --------
        log_values : list
            List of log probabilities
        """
        log_values = []
        
        for x in curr_pdf_list:
            if x <= 0:
                # Skip invalid probabilities
                continue
            else:
                log_values.append(numpy.log(x))
        
        return log_values
    
    def combine_computed_probability_with_labels(self, curr_log_pdf_dict, 
                                                  dis_prob_with_label, 
                                                  obs_dict_with_labels):
        """
        Combine log probabilities with true labels.
        
        Parameters:
        -----------
        curr_log_pdf_dict : dict
            {obj_id: {LOG_PDFS: [...]}}
        dis_prob_with_label : dict
            Accumulator dictionary
        obs_dict_with_labels : dict
            {obj_id: {TRACKING_DATA: ..., TRUE_LABEL: ...}}
            
        Returns:
        --------
        dis_prob_with_label : dict
            Updated dictionary
        """
        for obj_id, values in curr_log_pdf_dict.items():
            if obj_id not in dis_prob_with_label:
                dis_prob_with_label[obj_id] = {}
            
            dis_prob_with_label[obj_id][LOG_PDFS] = values[LOG_PDFS]
            dis_prob_with_label[obj_id][TRUE_LABEL] = obs_dict_with_labels[obj_id][TRUE_LABEL]
        
        return dis_prob_with_label
    
    def find_grid_cell(self, x, y):
        """
        Find grid cell for a given (x, y) position.
        
        Parameters:
        -----------
        x, y : float
            Position coordinates
            
        Returns:
        --------
        grid_row, grid_col : int, int
            Grid cell indices
        """
        grid_row = int(y * self.num_rows() // self.max_y)
        grid_col = int(x * self.num_cols() // self.max_x)
        
        # Handle boundary cases
        grid_row = min(grid_row, self.num_rows() - 1)
        grid_col = min(grid_col, self.num_cols() - 1)
        
        return grid_row, grid_col
    
    def num_rows(self):
        """Return number of grid rows."""
        return self.grid_rows
    
    def num_cols(self):
        """Return number of grid columns."""
        return self.grid_cols