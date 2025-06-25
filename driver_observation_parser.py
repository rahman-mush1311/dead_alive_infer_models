import re
import collections
import glob
import os 
import random
import numpy
import matplotlib.pyplot as plt
from bisect import bisect_left


DEAD='d'
ALIVE='a'

class ParsingObservations:
    def __init__(self):
        self.filelists = []
    
    def load_files_from_folder(self,filetype,filecount):
        '''
        in here we get the list of files/datasets from a particular folders (DEAD/ALIVE folders)
        Parameters:
        -filetype: which subfolder to go to
        -filecount: int how many files/datasets to consider
        '''
        counter_flag=0
        
        if filetype==DEAD:
            subfolder = "dead_files"  
        else:
            subfolder = "alive_files"
            
        for root, dirs, files in os.walk(subfolder):
            for file in files:
                if file.endswith(".txt"):  # Only add .txt files
                    self.filelists.append(os.path.join(root, file))
                    
                if len(self.filelists)>=filecount: #this is to give how many files to work with
                    counter_flag=1
                    break
            if counter_flag==1:
                break
        #print(self.filelists)            
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
            prefix,extension=self.get_file_prefix(filename)
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
        
        return observations
    
    def get_file_prefix(self, filename):
        '''
        extract the filename/dataset name to append it to object_id, since each dataset starts with 1... appending to same dictionaries will cause issues.
        Parameters:
        -filename: a str containing dataset/filename
        Returns:
        str matching re patterns
        '''
        if re.search(r"DeadObjectXYs\.txt", filename):
            return 'D', ''
        else:
            file_pattern = re.compile(r'''(\d{1,2}-\d{1,2}-\d{2})_(\d+)_ObjectXYs\.txt|AliveObjectXYs(\w+)\.txt''')
            match = file_pattern.search(filename)
            if match.group(1)and match.group(2):
                return (match.group(1), match.group(2)) 
            else:
                return ('Alive',match.group(3))
        return '',''
    
    def compute_global_stats(self, curr_obs):
    
        all_dx_dy = []
        
        # First pass: compute dx/dy for all object
        for obj_id, obs in curr_obs.items():
                
            for i in range(len(obs) - 1):
                dframe = obs[i+1][0] - obs[i][0]
                if dframe > 0:
                    dx = (obs[i+1][1] - obs[i][1]) / dframe
                    dy = (obs[i+1][2] - obs[i][2]) / dframe                  
                    all_dx_dy.append([dx,dy])
                    
                else:
                    print(f"dframe has invalid value: {dframe}")

        # Global averages of dx and dy across all objects
        all_dx_dy_np=numpy.array(all_dx_dy)
        all_dx_dy_mu=numpy.mean(all_dx_dy_np, axis=0)
        all_dx_dy_var=numpy.var(all_dx_dy_np, axis=0)
        all_dx_dy_cov=numpy.cov(all_dx_dy_np.T)
        global_avg_dx,global_avg_dy = all_dx_dy_mu[0],all_dx_dy_mu[1]
        
        print(f"Global avg var: {all_dx_dy_var[0]:.2f}, dy: {all_dx_dy_var[1]:.2f}")
        return all_dx_dy_mu,all_dx_dy_cov
    def split_observations_by_filename(self, curr_obs,curr_filename):
    
        prefix,extension = self.get_file_prefix(curr_filename)
        print(prefix, extension)
        
        dead_obs = collections.defaultdict(list)
        alive_obs = collections.defaultdict(list)
        
        if prefix.startswith("A"): 
            for obj_id, obs in curr_obs.items():
                if len(obs)>=5:
                        alive_obs[obj_id]=obs
            
        else:
            for obj_id, obs in curr_obs.items():
                if len(obs)>=5:
                        dead_obs[obj_id]=obs
        print(f"from filewise name label function: total obs len is {len(curr_obs)}, dead obs len is {len(dead_obs)} and alive obs len is {len(alive_obs)}")
        
        return dead_obs,alive_obs
    def split_observations_by_displacements(self, curr_obs, global_dx_dy_cov,curr_filename):
        #need to subtract mu, take the absoulate value in subtraction values
        prefix,extension = self.get_file_prefix(curr_filename)
        print(prefix, extension)
        
        global_std_dx = numpy.sqrt(global_dx_dy_cov[0][0])
        global_std_dy = numpy.sqrt(global_dx_dy_cov[1][1])
        
        dead_obs = collections.defaultdict(list)
        alive_obs = collections.defaultdict(list)
        if prefix.startswith("A"):
            # First pass: compute max dx/dy per object
            for obj_id, obs in curr_obs.items():
                #curr_obj_dx=[]
                #curr_obj_dy=[]
                both_dxdy_indices=[]
                for i in range(len(obs) - 1):
                    dframe = obs[i+1][0] - obs[i][0]
                    if dframe > 0:
                        dx = (obs[i+1][1] - obs[i][1]) / dframe
                        dy = (obs[i+1][2] - obs[i][2]) / dframe
                        #curr_obj_dx.append(dx)
                        #curr_obj_dy.append(dy)
                        if dx > global_std_dx and dy > global_std_dy:
                            both_dxdy_indices.append(i)
                    
                    else:
                        print(f"dframe has invalid value: {dframe}")
                if  both_dxdy_indices and len(obs)>=5:
                    '''
                    curr_obj_dx.sort()
                    curr_obj_dy.sort()
                
                    low_dx_count = bisect_left(curr_obj_dx, global_std_dx)
                    low_dy_count = bisect_left(curr_obj_dy, global_std_dy)
                
                    if low_dx_count >=5 and low_dy_count >=5 and len(obs) >=5:
                        dead_obs[obj_id]=obs
                    elif len(obs)>=5:
                        alive_obs[obj_id]=obs
                    '''
                    indices = sorted(set(both_dxdy_indices)) #don't need that
                    flag=False
                    window_size=3
                    ############SANITY CHECKING###############
                    #print(f"for obj_id: {obj_id}, indices are: {indices} ")
                    
                    for k in range(len(indices) - 2):                        
                        window = indices[k:k + window_size]
                        ############SANITY CHECKING###############
                        #print(f"for obj_id: {obj_id}, current indices are: {window} ")
                        if all(window[x] + 1 == window[x + 1] for x in range(len(window) - 1)):
                            flag=True
                            alive_obs[obj_id]=obs
                            ############SANITY CHECKING###############
                            #print(f"{obj_id} goes to alive")
                            break
                            
                        if flag==True:
                            break
                        else:
                            continue
                    else:
                        dead_obs[obj_id]=obs
                        ############SANITY CHECKING###############
                        #print(f"{obj_id} goes to dead")
                else:
                    print(f"for {obj_id}: len of obs is {len(obs)}, or dx, dy empty")
                       
            ###########SAVING TRAJECTORY###############            
            #self.visualize_object_trajectory(dead_obs)        
        else:
            dead_obs=curr_obs
        print(f"from split function: total obs len is {len(curr_obs)}, dead obs len is {len(dead_obs)} and alive obs len is {len(alive_obs)}")
        
        return dead_obs,alive_obs
    
    def split_observations_by_average(self, curr_obs, global_dx_dy_mu,global_dx_dy_cov,curr_filename):
    
        prefix,extension = self.get_file_prefix(curr_filename)
        print(prefix, extension)
        
        object_avg = {}

        dead_obs = collections.defaultdict(list)
        alive_obs = collections.defaultdict(list)
        
        global_avg_dx=global_dx_dy_mu[0]
        global_avg_dy=global_dx_dy_mu[1]
        global_std_dx = numpy.sqrt(global_dx_dy_cov[0][0])
        global_std_dy = numpy.sqrt(global_dx_dy_cov[1][1])
        global_var_dx = global_dx_dy_cov[0][0]
        global_var_dy = global_dx_dy_cov[1][1]
        
        
        if prefix.startswith("A"):
            # First pass: compute max dx/dy per object
            for obj_id, obs in curr_obs.items():
                curr_obj_dx_dy=[]   
                for i in range(len(obs) - 1):
                    dframe = obs[i+1][0] - obs[i][0]
                    if dframe > 0:
                        dx = (obs[i+1][1] - obs[i][1]) / dframe
                        dy = (obs[i+1][2] - obs[i][2]) / dframe
                        curr_obj_dx_dy.append([dx,dy]) 
                    
                    else:
                        print(f"dframe has invalid value: {dframe}")
                if curr_obj_dx_dy:
                    curr_obj_dx_dy_np = numpy.array(curr_obj_dx_dy)
                    curr_obj_dx_dy_mu = numpy.mean(curr_obj_dx_dy_np , axis=0)
                    curr_obj_dx_dy_var = numpy.var(curr_obj_dx_dy_np , axis=0)
                    avg_dx,avg_dy=curr_obj_dx_dy_mu[0],curr_obj_dx_dy_mu[1]
                    object_avg[obj_id] = (avg_dx, avg_dy,curr_obj_dx_dy_var[0],curr_obj_dx_dy_var[1])
                
            
            for obj_id, (avg_dx, avg_dy,var_dx,var_dy) in object_avg.items():
                obs = curr_obs[obj_id]
                z_dx=(avg_dx-global_avg_dx)
                z_dy=(avg_dy-global_avg_dy)
                if len(obs) > 5 and ((avg_dx > global_avg_dx or avg_dy >global_avg_dy) or (var_dx > global_var_dx or var_dy > global_var_dy)):
                    alive_obs[obj_id] = obs
                    
                elif len(obs) > 5:
                    dead_obs[obj_id] = obs
                   
            
        else:
            for obj_id, obs in curr_obs.items():
                if len(obs)>=5:
                        dead_obs[obj_id]=obs
       
        print(f"from average split function: total obs len is {len(curr_obs)}, dead obs len is {len(dead_obs)} and alive obs len is {len(alive_obs)}")
        
        return dead_obs,alive_obs
    
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
    
    def visualize_labeled_objects(self,alive_points,dead_points,global_avg_dx,global_avg_dy,filename):
    
        alive_x, alive_y = zip(*alive_points) if alive_points else ([], [])
        dead_x, dead_y = zip(*dead_points) if dead_points else ([], [])

        # Plotting
        plt.figure(figsize=(8, 6))
        plt.scatter(dead_x, dead_y, color='red', alpha=0.6, label='Non-moving')
        plt.scatter(alive_x, alive_y, color='green', alpha=0.8, label='Moving')

        # Threshold lines
        plt.axvline(global_avg_dx, color='blue', linestyle='--', linewidth=2, label='Global Mean dx')
        plt.axhline(global_avg_dy, color='orange', linestyle='--', linewidth=2, label='Global Mean dy')

        plt.xlabel('Average dx')
        plt.ylabel('Average dy')
        plt.title('Average & Variance-Based Labeling')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    