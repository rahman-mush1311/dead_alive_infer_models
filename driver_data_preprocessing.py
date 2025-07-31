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
SCORES = "scores"

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
            
    def is_starting_or_ending_near_edge(self,track, width=4096, height=2160, margin_ratio=0.25):
    
        x_start, y_start = track[0][1], track[0][2]  # Starting coordinates
        x_end, y_end = track[-1][1], track[-1][2]    # Ending coordinates

        margin_x = margin_ratio * width
        margin_y = margin_ratio * height
       
        valid_entry = (x_start <= margin_x or y_start <= margin_y)
        valid_exit = (x_end >= (width - margin_x) or y_end >= (height - margin_y))
        
        if valid_entry and valid_exit:
            return True
        else:
            return False
            
    def trajectory_quality_analysis(self,curr_obs):
        
        truncated_observations=collections.defaultdict(list)
        for obj_id,obs in curr_obs.items():
            is_valid=self.is_starting_or_ending_near_edge(obs)
            if is_valid==True:
                truncated_observations[obj_id]=obs
            else:
                print(f"{obj_id} is starting late or ending early!!")
        return truncated_observations        
    
    def get_displacement_sequence(self,curr_obs):
        """
        Computes the displacement sequence for the objects and returns a modified dictionary
        Parameters:
        -curr_obs: one dictionary of sample parsed observation {object id: [(frame1,x1,y1)..(framen,xn,yn)]}
        Returns:
        - curr_obs_displacements: one dictionary of observation sequences {object id: [(dx1,dy1)..(,dxn-1,dyn-1)]}
        """
        curr_obs_displacements = collections.defaultdict(list)
        for obj_id, obs in curr_obs.items(): 
            curr_obj_dx_dy=[]
            for i in range(len(obs) - 1):
                dframe = obs[i+1][0] - obs[i][0]
                if dframe > 0:
                    dx = (obs[i+1][1] - obs[i][1]) / dframe
                    dy = (obs[i+1][2] - obs[i][2]) / dframe                  
                    curr_obj_dx_dy.append([dx,dy])
                    
                else:
                    print(f"!!!!dframe has invalid value while computing the global stats: {dframe}")
            if curr_obj_dx_dy:
                curr_obs_displacements[obj_id]=curr_obj_dx_dy
            else:
                print(f"Displacements couldn't be calculated lack of observations,size is: {len(obs)}")
        return curr_obs_displacements
        
    def compute_global_stats(self, curr_obs):
    
        """
        Computes global mean and covariance of dx/dy for all objects and stores in self.total_mu and self.total_cov_matrix.
        Parameters:
        -curr_obs: one dictionary of sample parsed observation {object id: [(frame1,x1,y1)..(framen,xn,yn)]}
        Returns:
        N/A
        """
        
        all_dx_dy = []
        #tracking_only_obs = {obj_id: obj_data[TRACKING_DATA]for obj_id, obj_data in curr_obs.items()}
        #gets the displacement sequence
        curr_obs_displacements=self.get_displacement_sequence(curr_obs)
       
        for obj_id, dis in curr_obs_displacements.items():                
            if len(dis)>1:
                all_dx_dy.extend(dis)
            else:
                print(f"object id {obj_id}: displacement sequence lenght is {len(dis)}")
                    
        if all_dx_dy:
            # Global averages of dx and dy across all objects
            all_dx_dy_np=numpy.array(all_dx_dy)
            all_dx_dy_mu=numpy.mean(all_dx_dy_np, axis=0)
            all_dx_dy_cov=numpy.cov(all_dx_dy_np.T)
            
            self.total_mu = all_dx_dy_mu.tolist()        
            self.total_cov_matrix = all_dx_dy_cov
            self.total_obs=len(curr_obs_displacements)
            ##########SANITY CHECKING#########################
            print(f"current sample files stats mu are: {self.total_mu[0]:.2f},{self.total_mu[1]:.2f}\n"
                    f"and cov is: {self.total_cov_matrix}")
        return
    
    def observations_labeling_by_average_variance(self,curr_obs, file_type,enable_global_stats):
        """
        label the dictionary with 0/1. Each object's displacements statistics is compared against global statistics.
        if any object's mean and variance in any direction is greater than mean then we label it is as MOVING otherwise NOTMOVING
        and if object's trajectory data is less than 5 we discard those objects
        
        Parameters:
        -curr_obs: one dictionary of sample parsed observation {object id: [(frame1,x1,y1)..(framen,xn,yn)]}
        - file_type: 0/1/2 denoting mixed observations, non-ostracods and ostracods
        - enable_global_stats: True/False if it is from train observations then we compute global stats otherwise we don't
        
        Returns:
        - labeled_obs: a dictionary of the parsed observations and their true labels; {object id: {TRACKING_DATA: [(frame1,x1,y1)..(framen,xn,yn)], TRUE_LABELS: 0/1}}
        """
        
        labeled_obs=collections.defaultdict(list)
        
        if enable_global_stats==True:
            self.compute_global_stats(curr_obs)
        else:
            print(f"It is a test set no need compute stats!")
        
        curr_obs_displacements=self.get_displacement_sequence(curr_obs)
        
        if file_type==0:
        
            global_mean_dx=self.total_mu[0]
            global_mean_dy=self.total_mu[1]
            global_std_dx = numpy.sqrt(self.total_cov_matrix[0][0])
            global_std_dy = numpy.sqrt(self.total_cov_matrix[1][1])
            
            for obj_id,displacements in curr_obs_displacements.items():
                if len(displacements)>5:
                    
                    obs=curr_obs[obj_id]
                    curr_obj_dx_dy=numpy.array(displacements)
                    
                    obj_mu=numpy.mean(curr_obj_dx_dy,axis=0)
                    obj_cov=numpy.cov(curr_obj_dx_dy.T)
                    obj_var_dx=numpy.sqrt(obj_cov[0][0])
                    obj_var_dy=numpy.sqrt(obj_cov[1][1])

                    if (obj_mu[0]>global_mean_dx and obj_var_dx>global_std_dx) or (obj_mu[1]>global_mean_dy and obj_var_dy>global_std_dy):
                        
                        labeled_obs[obj_id]={ TRACKING_DATA: obs,
                                      SCORES: obj_mu,
                                      TRUE_LABELS: MOVING
                        }
                    else:
                        labeled_obs[obj_id]={ TRACKING_DATA: obs,
                                      SCORES: obj_mu,
                                      TRUE_LABELS: NOTMOVING
                        }
        elif file_type==1:
            for obj_id,obs in curr_obs.items():
                if len(obs)>5:
                     labeled_obs[obj_id]={ TRACKING_DATA: obs,
                                      TRUE_LABELS: MOVING
                        }
        else:
            for obj_id,obs in curr_obs.items():
                if len(obs)>5:
                     labeled_obs[obj_id]={ TRACKING_DATA: obs,
                                      TRUE_LABELS: NOTMOVING
                        }
            
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
    
    