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

class GridDisplacementModel:
    def __init__(self, grid_rows=5, grid_cols=5, max_x=4128, max_y=2196):
        # self.n represents the number of observations for each cell
        self.n = [[ 0 for _ in range(grid_cols)] for _ in range(grid_rows)]

        # self.mu represents mu_x,mu_y for each cell
        self.mu = [[(0, 0) for _ in range(grid_cols)] for _ in range(grid_rows)]

        # self.cov_mat represents the covariance for the each cell
        self.cov_matrix = [[numpy.zeros((2, 2)) for _ in range(grid_cols)] for _ in range(grid_rows)]
        
        self.max_x = max_x
        self.max_y = max_y
        
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        # TODO: add statistics for normalizing standard deviation
        self.total_mu=[(0,0)]
        self.total_cov_matrix=numpy.zeros((2, 2))
        
    def calculate_displacements(self, observations,calculate_normalization=True):
        '''
        this creates a grid_row*grid col[5X5]size (grid_dis) by processing a dictionary of observations where each grid cell contains the displacements
        the calculation of displacement across 2 axes is displacmenet along dx= x2-x1/frame_distance; dy=y2-y1/frame_distance
        and the cell where the displacement belongs is calculate by using find_grid_cell() method
        additionally, use_normalization tells us to calculate the normalization variables(mu, std_x, std_y) necessary to normalize
    
        Parameters:
        - observation: dictionary containing object's observations(object id: (frame,x_cordinate,y_coordinate))
        - calculate_normalization: boolean True or False indicate the normalization variables needed to be calculated or not? for train set we perform the calculation 
        Returns:
        - returns a 5*5 list of the displacements(dx,dy) might not be necessary.
        '''
        #to keep the displacements in the grid formats
        grid_dis = [[[] for _ in range(self.num_rows())] for _ in range(self.num_cols())]
        
        points=[]
        for obj_id, obs in observations.items():
            for i in range(len(obs) - 1):
                dframe = obs[i+1][0] - obs[i][0]
                #to do: dframe<=0 continue logging error
                if dframe>0:
                
                    dx = obs[i+1][1] - obs[i][1]
                    dy = obs[i+1][2] - obs[i][2]

                    grid_row, grid_cell = self.find_grid_cell(obs[i][1],
                                                      obs[i][2])
                    grid_pos=grid_dis[grid_row][grid_cell]
                    
                    self.n[grid_row][grid_cell] += 1
                    
                    dx=dx/dframe
                    dy=dy/dframe
                    grid_pos.append((dx,dy))
                    points.append((dx,dy))
                    
                else:
                    print(f"distance of frame is getting invalid values for calculation: {dframe}")
        
        if len(points)>=1:
            grid_dis=self.apply_normalization(grid_dis)
        else:
            print(f"it doesn't contain any observations")
                    
        return grid_dis
      
        
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
    
    def calculate_parameters(self,grid_displacements):
        '''
        we calculate the each cell's mu & covariance matrices. traverse over each cell one by one then calculates mu, covariance using the points in that cell. 
        - Parameters:
        grid_displacements: normalized displacements [5X5]list. 
        -Returns:
        -N/A
        '''
        for row in range(self.num_rows()):
            for col in range(self.num_cols()):
                n=self.n[row][col]
                
                if n>1:
                    if n<30:
                        print(f"at grid {row}{col} obs are: {n} less than 30")
                        assert n == len(grid_displacements[row][col]), f"Mismatch: {n} is but items are: {len(grid_displacements[row][col])}"        
                          
                    dxdy_items = numpy.array(grid_displacements[row][col])
                        
                    cell_mu = numpy.mean(dxdy_items, axis=0)
                    self.mu[row][col]=cell_mu
                    
                
                    cell_cov_matrix = numpy.cov(dxdy_items.T)
                    self.cov_matrix[row][col]=cell_cov_matrix
                                
                                
                else:
                    print(f"at grid {row}{col} obs are: {n} not enough to calculate")
                
        return 
        
    def add_models(self, *others):
        '''
        we add all the GridDisplacementModels initialized with their own grid_mu's and grid covaraince matrices. 
        Parameters:
        - *others: tuples of models.
        Returns:
        combined: GridDisplacementModel initialized with adding all the model's mu and covariaces; this will be used to calculate the probabilities     
        '''
        for o in others:
            assert self.grid_rows == o.grid_rows
            assert self.grid_cols == o.grid_cols
            assert self.max_x == o.max_x
            assert self.max_y == o.max_y

        combined = GridDisplacementModel(self.grid_rows, self.grid_cols, self.max_x, self.max_y)
        models = [self] + list(others)  # Include self and all other models       
        for row in range(self.grid_rows):
            for col in range(self.grid_cols):
                valid_models = [m for m in models if m.n[row][col] > 0]  # Filter out empty models

                # If no valid models exist, skip this cell
                if not valid_models:
                    print(f"No valid observations for cell while calculating weighted average [{row}][{col}]")
                    continue  
                else:
                    n_values = [m.n[row][col] for m in valid_models]
                    mu_values = [numpy.array(m.mu[row][col]) for m in valid_models]
                    cov_values = [numpy.array(m.cov_matrix[row][col]) for m in valid_models]

                    # Compute total observations
                    n_total = sum(n_values)
                    # Compute p_values and mu_est
                    p_values = [n / n_total for n in n_values]
                    mu_est = sum(p * mu for p, mu in zip(p_values, mu_values))
                    # Compute estimated covariance matrix
                    cov_est = sum(p * cov for p, cov in zip(p_values, cov_values)) + (sum(p * numpy.outer(mu, mu) for p, mu in zip(p_values, mu_values)) - numpy.outer(mu_est, mu_est))
                    '''
                    #Sanity checking
                    #print(f" for cell [{row}][{col}] n is: {n_values} mu is: {mu_values} covariance: {cov_values} and total_n for cell is {n_total}")
                    #print(f"p_values are: {p_values}")
                    #print(f"mu are: {mu_est}")
                    #print(f"cov_est: {cov_est}")
                    '''
                    # Store computed values in the new model
                    combined.n[row][col] = n_total
                    combined.mu[row][col] = mu_est
                    combined.cov_matrix[row][col] = cov_est
                    
        
        return combined
    
    def compute_probabilities(self, observations,dx_norm, dy_norm, sx_norm, sy_norm):
        '''
        this calculates the probability of all objects using the given dead/alive model the calculation happens in probability(). for sanity checking purpose we don't consider if any object has only one set of coordinates.
        Parameters:
        -observations: dictionary containing object's observations(object id: (frame,x_cordinate,y_coordinate))
        -dx_norm: float value of mean of x_cordinates
        -dy_norm: float value of mean of y_coordinates
        -sx_norm: float value of variance of x_cordinates
        -sy_norm: float value of variance of y_coordinates
        
        Returns:
        -probabilities: a dictionary containing {object_id: {LOG_PDFS:list of log of probabilities}
        '''
        probabilities={}
        empty_obs=0
        
        for obj_id, obs in observations.items():
            obj_probabilities=[]
            for i in range(len(obs) - 1):
                x,y=obs[i][1],obs[i][2]
                dframe = obs[i+1][0] - obs[i][0]
                dx = obs[i+1][1] - obs[i][1]
                dy = obs[i+1][2] - obs[i][2]
                if dframe>0:
                    dx,dy=(dx/dframe),(dy/dframe)
                    norm_dx = (dx - dx_norm) / sx_norm 
                    norm_dy = (dy - dy_norm) / sy_norm
                    probs=self.probability(x, y, norm_dx, norm_dy)    
                    obj_probabilities.append(probs)
                
                else:
                    print(f"!!!WARNING!!! invalid frame distance {dframe} for {x,y} for {obj_id}")
                    
            if len(obs)-1<=0:
                empty_obs+=1
                print(f"!!! WARNING!!! {obj_id} has {len(obs)} therefore empty probs!!")
            else:
                assert len(obj_probabilities) == len(obs)-1 and len(obs)>0, f"Mismatch: {obj_id} has {len(obj_probabilities)} probabilities but {len(obs)-1} observations!"
            
            
            log_obj_probabilities=self.log_probability(obj_probabilities)
            
            if len(log_obj_probabilities)>=1:
                probabilities[obj_id]={LOG_PDFS:log_obj_probabilities}
                
        #print(f"emptys are for the current sample file is: {empty_obs}")  
        
        return probabilities
        
    def log_probability(self, curr_pdf_list):
        '''
        takes the list of log probabilities applies log transformations to that
        Parameters:
        -cuur_pdf_list: a list containing one object's probabilities
        Returns:
        log_values: a list of log applied probabilities
        '''
        log_values = []
        
        for x in curr_pdf_list:
            if x <=0 :
                print(f"!!!Warning !!!! invalid or zero probability encountered: {x}")
            else:
                log_values.append(math.log(x))
    
        return log_values
        
    def probability(self, x, y, dx_norm, dy_norm):
        '''
        this calculates the probability of one objects using the particular cell's mu & covariance. we use the find_grid_cell() for that
        Parameters:
        -x: int value of x_cordinate 
        -y: int value of y_coordinates
        -dx: normalized displacement of x_cordinate
        -dy: normalized displacement of y_coordinates
        
        Returns:
        -curr_probability: float containing the calculated probabilities
        '''
        grid_row, grid_col = self.find_grid_cell(x, y)
        cell_mu = self.mu[grid_row][grid_col]
        cell_cov_matrix = self.cov_matrix[grid_row][grid_col]
        n = self.n[grid_row][grid_col]
        
        if n>=1:
            
            # TODO: create a 2-dimensional Gaussian distribution and use it to calculate a probability for (dx, dy)
            mvn = scipy.stats.multivariate_normal(mean=cell_mu , cov=cell_cov_matrix) #to do use the library name
            curr_probability=mvn.pdf((dx_norm,dy_norm))
            
            #print(f"current cell [{grid_row}][{grid_col}] mu is: {cell_mu} covariance is: {cell_cov_matrix}  probabilities: {curr_probability} for {dx,dy}")
            return curr_probability
        else:
            print(f"current cell [{grid_row}][{grid_col}] mu is: {cell_mu} sigma is: {cell_cov_matrix}  probabilities: empty for cord {x,y} displacements{dx_norm,dy_norm}")
            return 0.0
    
    def combine_computed_probability_with_labels(self,curr_log_pdf_dict,dis_prob_with_label,obs_dict_with_labels):
        """
        Combines log-probability values from a current dictionary into a master dictionary with true labels.    
        Parameters:
        - curr_log_pdf_dict: {obj_id: {LOG_PDFS: [...]}}, from one file
        - dis_prob_with_label: master dict accumulating all log PDFs and labels
        - obs_dict_with_labels: {obj_id: {obs: [...], true_labels}}, from one file
    
        Returns:
        - dis_prob_with_label: updated with new entries or extended values
        """
        for obj_id, values in curr_log_pdf_dict.items():
            if obj_id not in dis_prob_with_label:
               dis_prob_with_label[obj_id] = {} 
        
            dis_prob_with_label[obj_id][LOG_PDFS] = values[LOG_PDFS]
            dis_prob_with_label[obj_id][TRUE_LABELS] = obs_dict_with_labels[obj_id][TRUE_LABELS]

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