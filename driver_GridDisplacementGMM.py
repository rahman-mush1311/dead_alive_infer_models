import numpy 
import scipy.stats 
import math 
from sklearn.mixture import GaussianMixture

# String literals to constants
TRUE_LABELS = "true_labels"
LOG_PDFS = "log_pdfs"

MOVING = 1
NOTMOVING = 0

class GMMDisplacementModel:
    def __init__(self, grid_rows=3, grid_cols=3, max_x=4128, max_y=2196, n_components=2):
        """
        GMM-based displacement model with spatial grid context.
        
        Parameters:
        - grid_rows: number of rows in spatial grid
        - grid_cols: number of columns in spatial grid  
        - max_x: maximum x coordinate
        - max_y: maximum y coordinate
        - n_components: number of GMM components per cell (fixed)
        """
        # Grid for spatial context (to track where displacements occur)
        self.n = [[0 for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        # GMM model per cell (learned via EM)
        self.gmm = [[None for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        self.max_x = max_x
        self.max_y = max_y
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.n_components = n_components
        
        # Global normalization parameters
        self.total_mu = numpy.array([0.0, 0.0])
        self.total_cov_matrix = numpy.zeros((2, 2))
        
    def collect_displacements(self, observations):
        """
        Collects all displacements from observations and tracks spatial distribution.
        
        Parameters:
        - observations: {obj_id: [(frame, x, y), ...]}
        
        Returns:
        - displacements: numpy array of shape (N, 2) with [dx, dy] values
        - grid_counts: updated self.n tracking where displacements occur
        """
        #to keep the displacements in the grid formats
        grid_dis = [[[] for _ in range(self.num_rows())] for _ in range(self.num_cols())]
        
        for obj_id, obs in observations.items():
            for i in range(len(obs) - 1):
                dframe = obs[i+1][2] - obs[i][2]
                #to do: dframe<=0 continue logging error
                if dframe>0:
                
                    dx = obs[i+1][0] - obs[i][0]
                    dy = obs[i+1][1] - obs[i][1]

                    grid_row, grid_cell = self.find_grid_cell(obs[i][0],
                                                      obs[i][1])
                    grid_pos=grid_dis[grid_row][grid_cell]
                    
                    self.n[grid_row][grid_cell] += 1
                    
                    dx=dx/dframe
                    dy=dy/dframe
                    grid_pos.append((dx,dy))
                    
                else:
                    print(f"distance of frame is getting invalid values for calculation: {dframe}")
                    
        return grid_dis
    
    def set_normalization_params(self, mu, cov):
        """
        Set global normalization parameters.
        
        Parameters:
        - mu: mean [dx_mean, dy_mean]
        - std: standard deviation [dx_std, dy_std]
        """
        self.total_mu = numpy.array(mu)
        self.total_cov_matrix = numpy.array(cov)
    
    def apply_normalization(self,grid_displacements):
        '''
        we apply the normalization to each points located in the grid displacements lists.
        - Parameters:
        grid_displacements: grid_row*grid_col [5X5] lists containing all the displacements
        - Returns:
        - grid_displacements: normalized dx,dy for all the cells.
        '''
        mu_x,mu_y=self.total_mu
        std_x, std_y = numpy.sqrt(numpy.diag(self.total_cov_matrix))
        
        for row in range(len(grid_displacements)):
            for col in range(len(grid_displacements[row])):
                
                if grid_displacements[row][col]:  # Only normalize if the cell is not empty
                    grid_displacements[row][col] = [((dx - mu_x) / std_x, (dy - mu_y) / std_y) for dx, dy in grid_displacements[row][col]]
                
                else:
                    print(f"[{row}][{col}] doesn't contain any element to normalize from apply_normalization function {len(grid_displacements[row][col])}")
                                
        return grid_displacements
    
    
    
    def calculate_GMM_parameters(self, grid_normalized_displacements):
        # Fit GMM for each cell
        print("\n=== Fitting GMMs per Cell ===")
        for row in range(self.num_rows()):
            for col in range(self.num_cols()):
                n_samples = len(grid_normalized_displacements[row][col])
                
                if n_samples >= self.n_components * 5:  # Need at least 5 samples per component
                    try:
                        # Convert to numpy array
                        cell_data = numpy.array(grid_normalized_displacements[row][col])
                        
                        # Fit GMM
                        gmm = GaussianMixture(
                            n_components=self.n_components,
                            covariance_type='full',
                            max_iter=100,
                            n_init=10,
                            random_state=42
                        )
                        gmm.fit(cell_data)
                        
                        self.gmm[row][col] = gmm
                        print(f"  Cell [{row}][{col}]: Fitted GMM with {n_samples} samples, "
                              f"weights={gmm.weights_}")
                    
                    except Exception as e:
                        print(f"  Cell [{row}][{col}]: Failed to fit GMM - {e}")
                        self.gmm[row][col] = None
                else:
                    print(f"  Cell [{row}][{col}]: Not enough samples ({n_samples}), "
                          f"need at least {self.n_components * 5}")
                    self.gmm[row][col] = None
        
        print("=== GMM Fitting Complete ===\n")
    
    def compute_probabilities(self, observations):
        """
        Compute log probabilities for observations using per-cell fitted GMMs.
        
        Parameters:
        - observations: {obj_id: [(frame, x, y), ...]}
        
        Returns:
        - probabilities: {obj_id: {LOG_PDFS: [log_prob1, log_prob2, ...]}}
        """
        probabilities = {}
        
        for obj_id, obs in observations.items():
            obj_log_probs = []
            
            for i in range(len(obs) - 1):
                dframe = obs[i+1][2] - obs[i][2]
                
                if dframe > 0:
                    # Calculate displacement
                    dx = (obs[i+1][0] - obs[i][0]) / dframe
                    dy = (obs[i+1][1] - obs[i][1]) / dframe
                    
                    # Find grid cell
                    x, y = obs[i][0], obs[i][1]
                    grid_row, grid_col = self.find_grid_cell(x, y)
                    
                    # Get GMM for this cell
                    cell_gmm = self.gmm[grid_row][grid_col]
                    
                    if cell_gmm is not None:
                        # Normalize displacement
                        total_std = numpy.sqrt(numpy.diag(self.total_cov_matrix))
                        norm_displacement = (numpy.array([dx, dy]) - self.total_mu) / total_std
                        
                        # Compute log probability using cell's GMM
                        log_prob = cell_gmm.score_samples(norm_displacement.reshape(1, -1))[0]
                        obj_log_probs.append(log_prob)
                    else:
                        # Cell has no GMM (not enough data), use small probability
                        print(f"Warning: Cell [{grid_row}][{grid_col}] has no GMM for obj {obj_id}")
                        obj_log_probs.append(-100.0)  # Very low log probability
                else:
                    print(f"Warning: invalid dframe={dframe} for obj_id={obj_id}")
            
            if len(obj_log_probs) >= 1:
                probabilities[obj_id] = {LOG_PDFS: obj_log_probs}
            else:
                print(f"Warning: No valid probabilities for obj_id={obj_id}")
        
        return probabilities
    
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
    
    def print_summary(self):
        """Print model summary."""
        print("\n=== GMM Model Summary ===")
        print(f"Grid size: {self.grid_rows}x{self.grid_cols}")
        print(f"Number of GMM components: {self.n_components}")
        print(f"Normalization mu: {self.total_mu}")
        print(f"Normalization std: {self.total_std}")
        
        if self.gmm is not None:
            print(f"\nGMM Component Weights:")
            for i, weight in enumerate(self.gmm.weights_):
                print(f"  Component {i}: {weight:.4f}")
            
            print(f"\nSpatial distribution of displacements:")
            total_displacements = sum(sum(row) for row in self.n)
            print(f"Total displacements: {total_displacements}")
            for row in range(self.grid_rows):
                row_str = " ".join(f"{self.n[row][col]:5d}" for col in range(self.grid_cols))
                print(f"  Row {row}: {row_str}")