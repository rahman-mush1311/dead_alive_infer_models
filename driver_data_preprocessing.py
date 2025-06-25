import re
import collections
import glob
import os 
import random
import numpy
import matplotlib.pyplot as plt


MOVING=1
NOTMOVING=0

TRAIN="train"
INFER="infer"

TRACKING_DATA = "tracking_data"
TRUE_LABELS = "true_labels"
PREDICTED_LABELS= "predicted_labels"

class PreProcessingObservations:
    def __init__(self):
        self.total_mu=[(0,0)]
        self.total_cov_matrix=numpy.zeros((2, 2))
        self.total_obs=0
    def load_observations(self,filename):
        """
        Processes the input files and parses them to extract object ID, frame, x, and y coordinates.
        Parameters:
        -filenames: list of filenames to parse
        Returns:
        -observations: a dictionary (object id: (frame,x_cordinate,y_coordinate)).
        """
        pattern = re.compile(r'''
        \s*(?P<object_id>\d+),
        \s*(?P<within_frame_id>\d+),
        \s*'(?P<file_path>[^']+)',
        \s*cX\s*=\s*(?P<x>\d+),
        \s*cY\s*=\s*(?P<y>\d+),
        \s*Frame\s*=\s*(?P<frame>\d+)
        ''', re.VERBOSE)
        
        observations = collections.defaultdict(list)

        with open(filename) as object_xys:
            prefix,extension,obs_type=self.get_file_prefix(filename)
            for line in object_xys:
                m = pattern.match(line)
                if m:
                    obj_id = int(m.group('object_id'))
                    frame = int(m.group('frame'))
                    cX = int(m.group('x'))
                    cY = int(m.group('y'))
                    obj_id = f"{prefix}_{obj_id}_{extension}"
                    observations[obj_id].append((frame, cX, cY))

        # Ensure observations are sorted by frame
        
        for object_id in observations:
            observations[object_id].sort()
        
        for object_id, items in observations.items():
            assert all(items[i][0] <= items[i + 1][0] for i in range(len(items) - 1)), f"Items for {object_id} are not sorted by frame"    
        
        return observations,obs_type
    
    def get_file_prefix(self, filename):
        '''
        extract the filename/dataset name to append it to object_id, since each dataset starts with 1... appending to same dictionaries will cause issues.
        Parameters:
        -filename: a str containing dataset/filename
        Returns:
        str matching re patterns
        '''

        '''
        Extracts (date, image_id) from a filename if it contains 'ObjectXYs'.
    
        Returns:
        (date_str, image_id) if valid; otherwise None
        '''
        # Ensure it contains 'ObjectXYs'
        if "ObjectXYs" not in filename:
            raise ValueError(f"Filename isn't valid: {filename}")
            
        if filename.endswith("AliveObjectXYs.txt"):
            obs_type = 1
        elif filename.endswith("DeadObjectXYs.txt"):
            obs_type = 2
        else:
            obs_type = 0
        
        # Regex pattern: match DATE, IMAGE info before ObjectXYs        
        pattern = re.compile(r'(?P<date>\d{1,2}-\d{1,2}-\d{2})_(?P<image_id>.+)ObjectXYs\.txt$')
        match = pattern.search(filename)
    
        if match:
            date_str = match.group('date')
            image_id = match.group('image_id')
                      
            return date_str, image_id, obs_type
        else:
            raise ValueError(f"Filename pattern mismatch: {filename}")

    
    def compute_global_stats(self, curr_obs):
    
        """
        Computes global mean and covariance of dx/dy for all objects and stores in self.total_mu and self.total_cov_matrix.
        Parameters:
        -curr_obs: one dictionary of sample parsed observation {pbject id: [(frame1,x1,y1)..(framen,xn,yn)]}
        Returns:
        N/A
        """
        
        all_dx_dy = []
        tracking_only_obs = {obj_id: obj_data[TRACKING_DATA]for obj_id, obj_data in curr_obs.items()}
        # First pass: compute dx/dy for all object
        for obj_id, obs in tracking_only_obs.items():                
            for i in range(len(obs) - 1):
                dframe = obs[i+1][0] - obs[i][0]
                if dframe > 0:
                    dx = (obs[i+1][1] - obs[i][1]) / dframe
                    dy = (obs[i+1][2] - obs[i][2]) / dframe                  
                    all_dx_dy.append([dx,dy])
                    
                else:
                    print(f"!!!!dframe has invalid value while computing the global stats: {dframe}")
                    
        if all_dx_dy:
            # Global averages of dx and dy across all objects
            all_dx_dy_np=numpy.array(all_dx_dy)
            all_dx_dy_mu=numpy.mean(all_dx_dy_np, axis=0)
            all_dx_dy_cov=numpy.cov(all_dx_dy_np.T)
            
            self.total_mu = all_dx_dy_mu.tolist()        
            self.total_cov_matrix = all_dx_dy_cov
            self.total_obs=len(tracking_only_obs)
            ##########SANITY CHECKING#########################
            #print(f"current sample files stats mu are: {self.total_mu[0]:.2f},{self.total_mu[1]:.2f}\n"
                    #f"and cov is: {self.total_cov_matrix}")
        return
    
    def observations_labeling_by_average_variance(self, curr_obs, file_type, compute_stats):
        """
        seperates the dictionary observation into dead alive set, if it comes from mixed observation we apply automatic labeling stategy or else 
        to the corresponding file type
        Parameters: one dictionary of sample parsed observation {pbject id: [(frame1,x1,y1)..(framen,xn,yn)]}
        -curr_obs: 
        """
        all_dx_dy=[]
        object_avg = {}
        labeled_obs = collections.defaultdict(list)
              
        if file_type==0:
            for obj_id, obs in curr_obs.items():
                curr_obj_dx_dy=[]   
                for i in range(len(obs) - 1):
                    dframe = obs[i+1][0] - obs[i][0]
                    if dframe > 0:
                        dx = (obs[i+1][1] - obs[i][1]) / dframe
                        dy = (obs[i+1][2] - obs[i][2]) / dframe
                        curr_obj_dx_dy.append([dx,dy])
                        all_dx_dy.append([dx,dy])
                        
                    else:
                        print(f"dframe has invalid value: {dframe}")
                if curr_obj_dx_dy:
                    curr_obj_dx_dy_np = numpy.array(curr_obj_dx_dy)
                    curr_obj_dx_dy_mu = numpy.mean(curr_obj_dx_dy_np , axis=0)
                    curr_obj_dx_dy_var = numpy.var(curr_obj_dx_dy_np , axis=0)
                    avg_dx,avg_dy=curr_obj_dx_dy_mu[0],curr_obj_dx_dy_mu[1]
                    object_avg[obj_id] = (curr_obj_dx_dy_mu[0], curr_obj_dx_dy_mu[1], curr_obj_dx_dy_var[0], curr_obj_dx_dy_var[1])
                
            if all_dx_dy:
               
                all_dx_dy_np=numpy.array(all_dx_dy)
                all_dx_dy_mu=numpy.mean(all_dx_dy_np, axis=0)
                all_dx_dy_cov=numpy.cov(all_dx_dy_np.T)
                
                global_avg_dx=all_dx_dy_mu[0]
                global_avg_dy=all_dx_dy_mu[1]
        
                global_var_dx = all_dx_dy_cov[0][0]
                global_var_dy = all_dx_dy_cov[1][1]
            
            else:
                print(f"!!!WARNING!!! current dictionary has no observations")
                return 
            
            for obj_id, (avg_dx, avg_dy,var_dx,var_dy) in object_avg.items():
                obs = curr_obs[obj_id]
                if len(obs) > 5 and ((avg_dx > global_avg_dx and var_dx > global_var_dx) or (avg_dy > global_avg_dy and var_dy > global_var_dy)):
                    labeled_obs[obj_id] = {
                        TRACKING_DATA: obs,
                        TRUE_LABELS: MOVING
                    }
                    
                elif len(obs) > 5:
                    labeled_obs[obj_id] = {
                        TRACKING_DATA: obs,
                        TRUE_LABELS: NOTMOVING
                    }
                   
            
        elif file_type==2:
            for obj_id, obs in curr_obs.items():
                if len(obs)>=5:
                        labeled_obs[obj_id]={
                        TRACKING_DATA: obs,
                        TRUE_LABELS: NOTMOVING
                    }
        else:
            for obj_id, obs in curr_obs.items():
                if len(obs)>=5:
                        labeled_obs[obj_id]={
                        TRACKING_DATA: obs,
                        TRUE_LABELS: MOVING
                    }
        
        if compute_stats == True:
            self.compute_global_stats(labeled_obs)
        return labeled_obs
    
    def prepare_train_test(self,curr_obs,train_ratio=0.8):
        """
        Splits a dictionary into train and test sets based on a specified ratio.
    
        Parameters:
        -curr_obs (dict): The input dictionary with keys as object IDs and values as observations (e.g., lists of log PDFs).
        -train_ratio (float): The ratio of the data to include in the training set (e.g., 0.8 for 80% train and 20% test).
    
        Returns:
        - train_dict: The training set dictionary.
        - test_dict: The test set dictionary.
        """
        TRAIN_RATIO=train_ratio
        keys = list(curr_obs.keys())
        random.shuffle(keys)

        # Calculate split index
        split_index = int(len(keys) * train_ratio)

        # Split keys and sort them
        train_keys = sorted(keys[:split_index])
        test_keys = sorted(keys[split_index:])

        # Create sorted train and test dictionaries
        train_dict = {key: curr_obs[key] for key in train_keys}
        test_dict = {key: curr_obs[key] for key in test_keys}

        return train_dict,test_dict
    
    