import numpy 
import scipy.stats 
import math 
import sklearn.preprocessing 

#String literals to constants

TRUE_LABELS = "true_labels"
LOG_PDFS="log_pdfs"

MOVING=1
NOTMOVING=0

TRAIN="train"
INFER="infer"

class GridFeatureModel:
    def __init__(self, grid_rows=7, grid_cols=7, max_x=4128, max_y=2196):
        # self.n represents the number of observations for each cell
        self.n = [[ 0 for _ in range(grid_cols)] for _ in range(grid_rows)]

        # self.mu represents mu_x,mu_y for each cell
        self.mu = [[(0, 0, 0, 0, 0, 0) for _ in range(grid_cols)] for _ in range(grid_rows)]

        # self.cov_mat represents the covariance for the each cell
        self.cov_matrix = [[numpy.zeros((6, 6)) for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        self.max_x = max_x
        self.max_y = max_y
        
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        # TODO: add statistics for normalizing standard deviation
        self.total_mu=[(0,0)]
        self.total_cov_matrix=numpy.zeros((2, 2))
    
    def calculate_displacements(self,observations):
    
        grid_features = [[[] for _ in range(self.num_rows())] for _ in range(self.num_cols())]
        
        all_features = []
        
        mu_dx = self.total_mu[0]
        mu_dy = self.total_mu[1]
        std_dx = numpy.sqrt(self.total_cov_matrix[0, 0])
        std_dy = numpy.sqrt(self.total_cov_matrix[1, 1])
        
        # Prevent division by zero
        if std_dx < 1e-10:
            std_dx = 1.0
        if std_dy < 1e-10:
            std_dy = 1.0
        
        for obj_id, obs in observations.items():
            if len(obs) < 3:  # Need at least 3 points for all features
                continue
                
            for i in range(2, len(obs)):
                # Get three consecutive points
                x0, y0, f0 = obs[i-2]
                x1, y1, f1 = obs[i-1]
                x2, y2, f2 = obs[i]
                
                # Frame differences
                df1 = f1 - f0
                df2 = f2 - f1
                
                if df1 <= 0 or df2<=0:
                    print(f"Invalid frame sequence for {obj_id}: frames {f0}, {f1}, {f2}")
                    continue
                
                # Step 1: Compute RAW displacements
                dx1_raw = (x1 - x0) / df1
                dy1_raw = (y1 - y0) / df1
                dx2_raw = (x2 - x1) / df2
                dy2_raw = (y2 - y1) / df2
                
                # Step 2: NORMALIZE the displacements
                dx1_norm = (dx1_raw - mu_dx) / std_dx
                dy1_norm = (dy1_raw - mu_dy) / std_dy
                dx2_norm = (dx2_raw - mu_dx) / std_dx
                dy2_norm = (dy2_raw - mu_dy) / std_dy
                
                heading=self.compute_heading_angle(dy2_norm, dx2_norm)
                
                # Turning angle (from normalized displacements)
                turning = self.compute_turning_angle([dx1_norm, dy1_norm], [dx2_norm, dy2_norm])
                
                # Acceleration (from normalized displacements)
                accel = self.compute_acceleration([dx1_norm, dy1_norm], [dx2_norm, dy2_norm], df1, df2)
                ax, ay = accel[0], accel[1]
                
                # Find grid cell based on middle position
                grid_row, grid_col = self.find_grid_cell(x1, y1)
                
                # Store features [dx_norm, dy_norm, heading, turning, accel]
                feature_vector = [dx2_norm, dy2_norm, heading, turning,  ax, ay]
                #feature_vector = [dx2_norm, dy2_norm, heading,turning]
                grid_features[grid_row][grid_col].append(feature_vector)
                all_features.append(feature_vector)
                
                self.n[grid_row][grid_col] += 1
        
        if len(all_features) < 1:
            print(f"No valid observations for feature extraction")
                
        return grid_features
    '''
    def calculate_features(self, observations):
        """
        Extract features with proper angle handling
        
        Returns:
        - grid_features: [rows][cols] lists of feature vectors
        """
        mu_dx = self.total_mu[0]
        mu_dy = self.total_mu[1]
        std_dx = numpy.sqrt(self.total_cov_matrix[0, 0])
        std_dy = numpy.sqrt(self.total_cov_matrix[1, 1])
        
        # Prevent division by zero
        if std_dx < 1e-10:
            std_dx = 1.0
        if std_dy < 1e-10:
            std_dy = 1.0
            
        grid_features = [[[] for _ in range(self.num_cols())] 
                        for _ in range(self.num_rows())]
        
        for obj_id, obs in observations.items():
            # Need at least 3 points for all features
            if len(obs) < 3:
                continue
            
            for i in range(2, len(obs)):
                # Get grid cell from starting position
                grid_row, grid_col = self.find_grid_cell(obs[i-1][0], obs[i-1][1])
                
                # ========== DISPLACEMENT ==========
                dframe1 = obs[i][2] - obs[i-1][2]
                dframe0 = obs[i-1][2] - obs[i-2][2]
                
                if dframe1 <= 0 or dframe0 <= 0:
                    continue
                
                # Current displacement
                dx_curr = (obs[i][0] - obs[i-1][0]) / dframe1
                dy_curr = (obs[i][1] - obs[i-1][1]) / dframe1
                
                #current displacement normalized
                dx_curr_norm=(dx_curr-mu_dx)/std_dx
                dy_curr_norm=(dy_curr-mu_dy)/std_dy
                
                # Previous displacement
                dx_prev = (obs[i-1][0] - obs[i-2][0]) / dframe0
                dy_prev = (obs[i-1][1] - obs[i-2][1]) / dframe0
                
                #previous displacement normalized
                dx_prev_norm=(dx_prev-mu_dx)/std_dx
                dy_prev_norm=(dy_prev-mu_dy)/std_dy
                
                # ========== HEADING ANGLE (as unit circle) ==========
                theta = numpy.arctan2(dy_curr_norm, dx_curr_norm)
                cos_theta = numpy.cos(theta)
                sin_theta = numpy.sin(theta)
                
                # ========== TURNING ANGLE (as unit circle) ==========
                d_curr = numpy.array([dx_curr_norm, dy_curr_norm])
                d_prev = numpy.array([dx_prev_norm, dy_prev_norm])
                
                norm_curr = numpy.linalg.norm(d_curr)
                norm_prev = numpy.linalg.norm(d_prev)
                
                if norm_curr > 0 and norm_prev > 0:
                    cos_phi_raw = numpy.dot(d_curr, d_prev) / (norm_curr * norm_prev)
                    cos_phi_raw = numpy.clip(cos_phi_raw, -1, 1)
                    phi = numpy.arccos(cos_phi_raw)
                    
                    # Convert to unit circle
                    cos_phi = numpy.cos(phi)
                    sin_phi = numpy.sin(phi)
                else:
                    cos_phi = 1.0
                    sin_phi = 0.0
                
                # ========== ACCELERATION ==========
                dt = obs[i][2] - obs[i-2][2]
                if dt > 0:
                    ax = (dx_curr_norm - dx_prev_norm) / dt
                    ay = (dy_curr_norm - dy_prev_norm) / dt
                else:
                    continue
                
                # ========== ASSEMBLE FEATURE VECTOR ==========
                # Order: [dx, dy, cos_θ, sin_θ, cos_φ, sin_φ, ax, ay]
                feature_vector = [dx_prev_norm, dy_prev_norm, cos_theta, sin_theta, 
                                cos_phi, sin_phi, ax, ay]
                self.n[grid_row][grid_col] += 1
                
                grid_features[grid_row][grid_col].append(feature_vector)
        
        return grid_features
        
    def _compute_single_feature_vector(self, obs, obs_type, mu_dx, mu_dy, std_dx, std_dy):
        """
        Compute feature vector for observation
        
        Returns:
        - feature_vector: [dx, dy, cos_θ, sin_θ, cos_φ, sin_φ, ax, ay]
        """
        # Frame differences
        dframe1 = obs[i][2] - obs[i-1][2]
        dframe0 = obs[i-1][2] - obs[i-2][2]
        
        if dframe1 <= 0 or dframe0 <= 0:
            return None
        
        # ========== DISPLACEMENT (Linear) ==========
        dx_curr = (obs[i][0] - obs[i-1][0]) / dframe1
        dy_curr = (obs[i][1] - obs[i-1][1]) / dframe1
        
        dx_prev = (obs[i-1][0] - obs[i-2][0]) / dframe0
        dy_prev = (obs[i-1][1] - obs[i-2][1]) / dframe0
        
        #current displacement normalized
        dx_curr_norm=(dx_curr-mu_dx)/std_dx
        dy_curr_norm=(dy_curr-mu_dy)/std_dy
        
        #previous displacement normalized
        dx_prev_norm=(dx_prev-mu_dx)/std_dx
        dy_prev_norm=(dy_prev-mu_dy)/std_dy
        
        # ========== HEADING ANGLE (Angular → Cartesian) ==========
        theta = numpy.arctan2(dy_curr_norm, dx_curr_norm)
        cos_theta = numpy.cos(theta)
        sin_theta = numpy.sin(theta)
        
        # ========== TURNING ANGLE (Angular → Cartesian) ==========
        d_curr = numpy.array([dx_curr_norm, dy_curr_norm])
        d_prev = numpy.array([dx_prev_norm, dy_prev_norm])
        
        norm_curr = numpy.linalg.norm(d_curr)
        norm_prev = numpy.linalg.norm(d_prev)
        
        if norm_curr > 1e-6 and norm_prev > 1e-6:
            cos_phi_raw = numpy.dot(d_curr, d_prev) / (norm_curr * norm_prev)
            cos_phi_raw = numpy.clip(cos_phi_raw, -1, 1)
            phi = numpy.arccos(cos_phi_raw)
            cos_phi = numpy.cos(phi)
            sin_phi = numpy.sin(phi)
        else:
            cos_phi = 1.0
            sin_phi = 0.0
        
        # ========== ACCELERATION (Linear) ==========
        dt = obs[i][2] - obs[i-2][2]
        if dt > 0:
            ax = (dx_curr_norm - dx_prev_norm) / dt
            ay = (dy_curr_norm - dy_prev_norm) / dt
        else:
            return None
        
        # Assemble feature vector
        # Order: [dx, dy, cos_θ, sin_θ, cos_φ, sin_φ, ax, ay]
        return numpy.array([dx_curr, dy_curr, cos_theta, sin_theta, 
                        cos_phi, sin_phi, ax, ay])
    '''    
    def compute_heading_angle(self, dy_norm, dx_norm):
        """
        Compute heading angle between two consecutive points
        θ_i = arctan2((y_i - y_{i-1}) / (x_i - x_{i-1}))
        
        Returns angle in (-pi, pi]
        """
        return numpy.arctan2(dy_norm, dx_norm)
    
    def compute_turning_angle(self, d1_norm, d2_norm):
        """
        Compute turning angle between two displacement vectors
        φ_i = arccos(d_i · d_{i-1} / (|d_i| |d_{i-1}|))
        
        Parameters:
        - d1: displacement vector [dx1_norm, dy1_norm]
        - d2: displacement vector [dx2_norm, dy2_norm]
        
        Returns angle in (-pi, pi]
        """
        d1 = numpy.array(d1_norm)
        d2 = numpy.array(d2_norm)
        
        norm1 = numpy.linalg.norm(d1)
        norm2 = numpy.linalg.norm(d2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute dot product and clamp to [-1, 1] for numerical stability
        cos_angle = numpy.dot(d1, d2) / (norm1 * norm2)
        cos_angle = numpy.clip(cos_angle, -1.0, 1.0)
        
        # Get unsigned angle
        angle = numpy.arccos(cos_angle)
        
        # Determine sign using cross product (2D: z-component only)
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        if cross < 0:
            angle = -angle
            
        return angle
    
    def compute_acceleration(self, d1, d2, dt1, dt2):
        """
        Compute acceleration as change in displacement vector
        a_i = (d_i - d_{i-1}) / (t_i - t_{i-2})
        
        Returns 2D acceleration vector [ax, ay]
        
        Parameters:
        - d1: first displacement vector [dx1, dy1]
        - d2: second displacement vector [dx2, dy2]
        - dt1: time difference for first displacement
        - dt2: time difference for second displacement
        
        Returns:
        - acceleration vector [ax, ay]
        """
        d1 = numpy.array(d1)
        d2 = numpy.array(d2)
        
        total_dt = dt1 + dt2
        if total_dt == 0:
            return numpy.array([0.0, 0.0])
            
        # Acceleration vector: change in displacement divided by time
        accel = (d2 - d1) / total_dt
        
        return accel  # Returns [ax, ay]
    
    def calculate_parameters(self, grid_features):
        """
        Calculate mean and covariance for each grid cell
        
        NOTE: Features are already normalized (displacements normalized in calculate_displacements_extended)
        
        Parameters:
        - grid_features: 6D feature vectors [dx_norm, dy_norm, heading, turning, ax, ay]
        
        Returns: None (updates self.mu and self.cov_matrix)
        """
        # Features already have normalized displacements, so compute statistics directly
        for row in range(self.num_rows()):
            for col in range(self.num_cols()):
                n = self.n[row][col]
                
                if n > 1:
                    if n < 30:
                        print(f"Grid [{row}][{col}] has {n} observations (less than 30)")
                    
                    feature_array = numpy.array(grid_features[row][col])
                    
                    # Calculate mean (6D vector) directly on features
                    cell_mu = numpy.mean(feature_array, axis=0)
                    self.mu[row][col] = cell_mu
                    
                    # Calculate covariance (6x6 matrix) directly on features
                    cell_cov = numpy.cov(feature_array.T)
                    self.cov_matrix[row][col] = cell_cov
                else:
                    print(f"Grid [{row}][{col}] has insufficient observations: {n}")
        
        return
    
    def add_models(self, *others):
        """
        Combine multiple GridDisplacementModelExtended instances
        
        Parameters:
        - *others: tuple of model instances
        
        Returns:
        - combined: new GridDisplacementModelExtended with weighted averages
        """
        for o in others:
            assert self.grid_rows == o.grid_rows
            assert self.grid_cols == o.grid_cols
            assert self.max_x == o.max_x
            assert self.max_y == o.max_y

        combined = GridFeatureModel(
            self.grid_rows, self.grid_cols, self.max_x, self.max_y
        )
        models = [self] + list(others)
        
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                valid_models = [m for m in models if m.n[row][col] > 0]
                
                if not valid_models:
                    print(f"No valid observations for cell [{row}][{col}]")
                    continue
                
                n_values = [m.n[row][col] for m in valid_models]
                mu_values = [numpy.array(m.mu[row][col]) for m in valid_models]
                cov_values = [numpy.array(m.cov_matrix[row][col]) for m in valid_models]
                
                # Total observations
                n_total = sum(n_values)
                
                # Weighted probabilities
                p_values = [n / n_total for n in n_values]
                
                # Weighted mean
                mu_est = sum(p * mu for p, mu in zip(p_values, mu_values))
                
                # Weighted covariance
                cov_est = sum(p * cov for p, cov in zip(p_values, cov_values)) + \
                          (sum(p * numpy.outer(mu, mu) for p, mu in zip(p_values, mu_values)) - \
                           numpy.outer(mu_est, mu_est))
                
                combined.n[row][col] = n_total
                combined.mu[row][col] = mu_est
                combined.cov_matrix[row][col] = cov_est
        
        return combined
        
    def compute_probabilities(self, observations, mu_dx, mu_dy, std_dx, std_dy):
        """
        Compute log probabilities for all objects using extended features
        
        Process (SAME as training):
        1. Compute RAW displacements
        2. NORMALIZE displacements 
        3. Compute heading, turning, accel (2D vector) from NORMALIZED displacements
        4. Compute probability
        
        Parameters:
        - observations: dictionary {object_id: [(x, y, frame), ...]}
        - norm_params: dict with 'mu' and 'std' for normalization (6D, but only first 2 used for displacements)
        
        Returns:
        - probabilities: {object_id: {LOG_PDFS: [log_probs]}}
        """
        probabilities = {}
        
        # Prevent division by zero
        if std_dx < 1e-10:
            std_dx = 1.0
        if std_dy < 1e-10:
            std_dy = 1.0
        
        for obj_id, obs in observations.items():
            if len(obs) < 3:
                print(f"Object {obj_id} has insufficient points: {len(obs)}")
                continue
            
            obj_probabilities = []
            
            for i in range(2, len(obs)):
                # Get three consecutive points
                x0, y0, f0 = obs[i-2]
                x1, y1, f1 = obs[i-1]
                x2, y2, f2 = obs[i]
                
                df1 = f1 - f0
                df2 = f2 - f1
                
                if df1 <= 0 or df2<=0:
                    continue
                
                # Step 1: Calculate RAW displacements
                dx1_raw = (x1 - x0) / df1
                dy1_raw = (y1 - y0) / df1
                dx2_raw = (x2 - x1) / df2
                dy2_raw = (y2 - y1) / df2
                
                # Step 2: NORMALIZE displacements
                dx1_norm = (dx1_raw - mu_dx) / std_dx
                dy1_norm = (dy1_raw - mu_dy) / std_dy
                dx2_norm = (dx2_raw - mu_dx) / std_dx
                dy2_norm = (dy2_raw - mu_dy) / std_dy
                
                # Step 3: Compute other features from NORMALIZED displacements
                heading = numpy.arctan2(dy2_norm, dx2_norm)
                turning = self.compute_turning_angle([dx1_norm, dy1_norm], [dx2_norm, dy2_norm])
                accel = self.compute_acceleration([dx1_norm, dy1_norm], [dx2_norm, dy2_norm], df1, df2)
                ax, ay = accel[0], accel[1]
                
                # Create 6D feature vector [dx_norm, dy_norm, heading, turning, ax, ay]
                features = numpy.array([dx2_norm, dy2_norm, heading, turning, ax, ay])
                #features = numpy.array([dx2_norm, dy2_norm, heading,turning])
                
                # Compute probability using these features
                prob = self.probability(x1, y1, features)
                obj_probabilities.append(prob)
            
            if len(obj_probabilities) > 0:
                log_probs = self.log_probability(obj_probabilities)
                probabilities[obj_id] = {LOG_PDFS: log_probs}
            else:
                print(f"No valid probabilities for {obj_id}")
        
        return probabilities
    '''
    
    def compute_probabilities(self, observations,mu_dx, mu_dy, std_dx, std_dy):
        """
        Compute log probabilities for all observations
        
        Parameters:
        - observations: dict {obj_id: [(x, y, frame), ...]}
        
        Returns:
        - probabilities: dict {obj_id: {'log_pdfs': [...]}}
        """
        probabilities = {}
        
        for obj_id, obs in observations.items():
            if len(obs) < 3:
                continue
            
            obj_probabilities = []
            
            for i in range(2, len(obs)):
                # Get starting position
                x, y = obs[i-1][0], obs[i-1][1]
                
                # Compute normalized feature vector 
                feature_vector = self._compute_single_feature_vector(obs, i, mu_dx, mu_dy, std_dx, std_dy)
                
                if feature_vector is not None:
                    prob = self.probability(x, y, feature_vector)
                    #print(f"{feature_vector} {prob}")
                    obj_probabilities.append(prob)
            if len(obj_probabilities) > 0:
                log_probs = self.log_probability(obj_probabilities)
                probabilities[obj_id] = {LOG_PDFS: log_probs}
            else:
                print(f"No valid probabilities for {obj_id}")
        
        return probabilities
    '''    
    def log_probability(self, curr_pdf_list):
        """Convert probabilities to log probabilities"""
        log_values = []
        
        for x in curr_pdf_list:
            if x <= 0:
                #print(f"Warning: invalid probability {x}")
                continue
            else:
                log_values.append(math.log(x))
        
        return log_values
    
    def probability(self, x, y, features):
        """
        Calculate probability using 6D multivariate normal
        
        Parameters:
        - x, y: spatial coordinates for grid cell lookup
        - features: 6D feature vector [dx_norm, dy_norm, heading, turning, ax, ay]
                   (displacements already normalized, angles/accel computed from normalized displacements)
        
        Returns:
        - probability value
        """
        grid_row, grid_col = self.find_grid_cell(x, y)
        cell_mu = numpy.array(self.mu[grid_row][grid_col])
        cell_cov = self.cov_matrix[grid_row][grid_col]
        n = self.n[grid_row][grid_col]
        
        if n >= 1:
            # Create 6D multivariate normal
            mvn = scipy.stats.multivariate_normal(mean=cell_mu, cov=cell_cov)
            prob = mvn.pdf(features)
            return prob
        else:
            print(f"Cell [{grid_row}][{grid_col}] has no observations")
            return 1e-10  # Small non-zero value
    
    def combine_computed_probability_with_labels(self, curr_log_pdf_dict, 
                                                   dis_prob_with_label, 
                                                   obs_dict_with_labels):
        """
        Combine log probabilities with true labels
        
        Parameters:
        - curr_log_pdf_dict: {obj_id: {LOG_PDFS: [...]}}
        - dis_prob_with_label: accumulator dictionary
        - obs_dict_with_labels: {obj_id: {TRACKING_DATA: ..., TRUE_LABELS: ...}}
        
        Returns:
        - updated dis_prob_with_label
        """
        for obj_id, values in curr_log_pdf_dict.items():
            if obj_id not in dis_prob_with_label:
                dis_prob_with_label[obj_id] = {}
            
            dis_prob_with_label[obj_id][LOG_PDFS] = values[LOG_PDFS]
            dis_prob_with_label[obj_id][TRUE_LABELS] = \
                obs_dict_with_labels[obj_id][TRUE_LABELS]
        
        return dis_prob_with_label   
    def find_grid_cell(self, x, y):
        '''
        find the where a particular displacement dx/dy should be assigned but it uses the starting x,y to calculate them.
        Parameters:
        x - int value of x-axis coordinate
        y - int value of y-axis coordinate
        Returns:
        grid_row,grid_col- int value of 0<=grid_row, grid_col<5
        '''
        grid_row = y * self.num_rows() // self.max_y
        grid_col = x * self.num_cols() // self.max_x
        return grid_row, grid_col

    def num_rows(self):
        '''
        returns the number of grid rows in the model.
        
        Returns:
        -len(self.n): int rows in the grid model.
        '''
        return len(self.n)

    def num_cols(self):
        '''
        returns the number of grid collums in the model.
        
        Returns:
        len(self.n[0]): int cols in the grid model.
        '''
        return len(self.n[0])