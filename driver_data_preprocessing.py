import re
import collections
import glob
import os 
import random
import numpy
import math
import pandas
from collections import Counter
import matplotlib.pyplot as plt


MOTILE=1
NOTMOTILE=0 

TRAIN="train"
INFER="infer"

TRACKING_DATA = "tracking_data"
TRUE_LABEL = "true_label"
PREDICTED_LABEL= "predicted_label"
SCORES = "scores"


class PreProcessingObservations:
    def __init__(self):
        self.total_mu=[(0,0)]
        self.total_cov_matrix=numpy.zeros((2, 2))
        self.total_obs=0
    def load_observations(self,filename):
        """
        Processes the input file and parses them to extract object ID, frame, x, and y coordinates.
        Parameters:
        -filename: Text File(with their directory and filename) to parse
        Returns:
        -observations: a dictionary (object id: [(x_cordinate_1,y_coordinate_1,frame_1),...,(x_cordinate_n,y_coordinate_n,frame_n)].
        """
        pattern = re.compile('''^[ ]*(?P<objectid>[0-9]+),[ ]*(?P<occurrence>[0-9]+),[ ]'[^']+',[ ]cX=[ ]*(?P<cx>[0-9]+),[ ]cY=[ ]*(?P<cy>[0-9]+),[ ]Frame=[ ]*(?P<frame>[0-9]+)''')
        
        observations = collections.defaultdict(list)
        with open(filename) as file_input:
            for line in file_input:
                #print(f, line)
                m = pattern.match(line)
                assert m, (f, line)
                objectid = int(m.group('objectid'))
                #occurrence = int(m.group('occurrence'))
                x = int(m.group('cx'))
                y = int(m.group('cy'))
                frame = int(m.group('frame'))

                observations[objectid].append((x, y, frame))

        for objectid in observations:
            observations[objectid].sort(key=lambda t: t[2])  # sort by frame index


        return observations
    
    def load_labels(self,filename):
        """
        Processes the input excel file and extracts corresponding object's label.The function checks if the object id was tracked and it was a good track
        Parameters:
        -filename: Excel File(with their directory and filename) to extract the label
        Returns:
        -loaded_labels: a dictionary (object id: 1/0).
        """
        loaded_labels = {}
        
        cols_to_use = [
        "Object Id",
        "Tracked Object",
        "Motile Organism",
        "Good Track?"
        ]
        df=pandas.read_excel(filename,skiprows=1, usecols=cols_to_use)
        df["Tracked Object"] = df["Tracked Object"].apply(lambda x: 1 if str(x).strip().upper() == "YES" else 0)
        df["Good Track?"] = df["Good Track?"].apply(lambda x: 1 if str(x).strip().upper() == "YES" else 0)
        df["Motile Organism"] = df["Motile Organism"].apply(lambda x: 1 if str(x) == "X" else 0)
        
        for _, row in df.iterrows():
            obj_id = int(row["Object Id"])
            tracked = int(row["Tracked Object"])
            is_motile = int(row["Motile Organism"])
            good_track = int(row["Good Track?"])

            if tracked == 1:  # only take tracked objects
                loaded_labels[obj_id] = MOTILE if is_motile == 1 else NOTMOTILE
        #print(df.head())
        return loaded_labels
    
    def check_for_short_track(self,tracks):

        if (len(tracks)<10):
            return True
        else:
            return False
    
    def check_for_starting_track_late(self,track, width=4128, height=2196, margin_ratio=0.25):
    
        x_start, y_start = track[0][0], track[0][1]
        margin_x = margin_ratio * width
        margin_y = margin_ratio * height
       
        valid_entry = (x_start <= margin_x)
        if valid_entry:
            return True
        else:
            return False
    
    def check_for_ending_track_early(self,track, width=4128, height=2196, margin_ratio=0.25):
    
        x_end, y_end = track[-1][0], track[-1][1]
        margin_x = margin_ratio * width
        #margin_y = margin_ratio * height
        valid_exit = (x_end >= (width - margin_x) )
        if valid_exit:
            return True
        else:
            return False
            
    def filter_tracks(self,tracks):
    
        is_valid_entry=True
        is_valid_exit=True
        is_skipping_frame=False
        if(len(tracks)<10):
            return False
        
        else:
            is_valid_entry=self.check_for_starting_track_late(tracks)
            is_valid_exit=self.check_for_ending_track_early(tracks)
            is_skipping_frame=self.is_missing_frame(tracks)
            if is_valid_entry==True and is_valid_exit==True and is_skipping_frame==False:
                return True
            else:
                return False
        
    
    def match_txt_excel_file_prefix(self,curr_text_filepath, curr_excel_filepath):
        """
        Check if the text file and excel file names match up to 'Image_X'.
    
        Parameters:
        - text_filename (str): name of the text file (e.g., 3-19-25_Image_2_trackStore.txt)
        - excel_filename (str): name of the excel file (e.g., 3-19-25_Image_2.xlsx)
    
        Returns:
        - bool: True if they match up to Image_X, False otherwise
        """
        text_filename = os.path.basename(curr_text_filepath)
        excel_filename= os.path.basename(curr_excel_filepath)
        # Remove extensions
        text_name = os.path.splitext(text_filename)[0]
        excel_name = os.path.splitext(excel_filename)[0]
    
        # Find the cutoff at "Image_"
        if "Image_" in text_name and "Image_" in excel_name:
            text_parts = text_name.split("_")
            excel_parts = excel_name.split("_")

            # Get prefix up to the number after Image_
            text_index = [i for i, p in enumerate(text_parts) if p.startswith("Image")]
            excel_index = [i for i, p in enumerate(excel_parts) if p.startswith("Image")]
            
            if text_index and excel_index:
                text_prefix = "_".join(text_parts[:text_index[0]+2])
                excel_prefix = "_".join(excel_parts[:excel_index[0]+2])
                return text_prefix == excel_prefix

        return False
    
    def label_observations_by_expert_labels(self,text_filepath,excel_filepath,observations,loaded_labels):
        """
        Merges two dictionary into one dictionary with their tracking data and their expert labels. Additionally the program parses the text file name to modify the object id
        modifying object id is neccesary because the text file and excel file starts their object_id with 1..n and in different files have different population of objects tracked
        Parameters:
        -filename: Text File(with their directory and filename) to parse the name only for modifying the object_id
        -observations: a dictionary (object id: [(x_cordinate_1,y_coordinate_1,frame_1),...,(x_cordinate_n,y_coordinate_n,frame_n)]
        -loaded_labels: a dictionary (object id: 1/0)
        Returns:
        -labeled_observations: a dictionary {object id: TRACKING_DATA: [(x_cordinate_1,y_coordinate_1,frame_1),...,(x_cordinate_n,y_coordinate_n,frame_n)],
                                                        TRUE_LABELS: 0/1                                                                                                    }.
        """
        
        labeled_observations = collections.defaultdict(list)
        date_str,image_id_str = self.get_file_prefix(text_filepath)
        if self.match_txt_excel_file_prefix(text_filepath,excel_filepath)==True:
            for obj_id,tracks in observations.items():
                good_track_flag=self.filter_tracks(tracks)
                if obj_id in loaded_labels and good_track_flag==True:
                    object_id=f"{date_str}_{obj_id}_{image_id_str}"
                    labeled_observations[object_id]={TRACKING_DATA: tracks,
                                                TRUE_LABEL: loaded_labels[obj_id]
                    }
                '''
                else:
                    print(f"{obj_id} is not good track")
                '''
        else:
            print(f"!!!ERROR!!! Text filename and Excel filenames don't match")
        return labeled_observations
    
        
    def get_file_prefix(self, filepath):
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
        filename = os.path.basename(filepath)
        print(filename)
        # Ensure it contains 'trackStore'
        if "trackStore" not in filename:
            raise ValueError(f"Filename isn't valid: {filename}")
            
        # Regex pattern: match DATE, IMAGE info before ObjectXYs        
        pattern = re.compile(r'(?P<date>\d{1,2}-\d{1,2}-\d{2})_(?P<image_id>.+)trackStore\.txt$')
        match = pattern.search(filename)
    
        if match:
            date_str = match.group('date')
            image_id = match.group('image_id')
                      
            return date_str, image_id
        else:
            raise ValueError(f"Filename pattern mismatch: {filename}")
    
    
    def is_missing_frame(self,o):
    
        frames = [r[2] for r in o]  # actual frames observed
        expected_frames = list(range(min(frames), min(frames) + len(frames)))  # expected consecutive frames
    
        missing = sorted(set(expected_frames) - set(frames))
        extra = sorted(set(frames) - set(expected_frames))  # if needed for debug

        missing_frame_flag = False
        if len(missing) > 2:
            missing_frame_flag = True

        return missing_frame_flag
    
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
                dframe = obs[i+1][2] - obs[i][2]
                if dframe > 0:
                    dx = (obs[i+1][0] - obs[i][0]) / dframe
                    dy = (obs[i+1][1] - obs[i][1]) / dframe                  
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
        tracking_only_obs = {obj_id: obj_data[TRACKING_DATA]for obj_id, obj_data in curr_obs.items()}
        #gets the displacement sequence
        curr_obs_displacements=self.get_displacement_sequence(tracking_only_obs)
       
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
            '''
            print(f"current sample files stats mu are: {self.total_mu[0]:.4f},{self.total_mu[1]:.4f}\n"
                    f"and cov is: {self.total_cov_matrix}")
            '''
        return
    
        
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
    
    def stats(self,x):
        n = len(x)
        s = sum(x)
        mu = s / n if n else 0
        std = math.sqrt(sum([(xi - mu) ** 2 for xi in x]))
        max_x = max(x) if n else 0
        min_x = min(x) if n else 0
        print(f'{n=} {s=} {mu=} {std=} {max_x=} {min_x=}')
    
    def analyze_object(self,o):
        #occurrences = [r[0] for r in o]
        #assert occurrences == list(range(1, len(o) + 1)) # occurrences are sequential
        dx = []
        dy = []
        for i in range(1, len(o)):
            dx.append(o[i][1] - o[i-1][1])
            dy.append(o[i][2] - o[i-1][2])
        print('stats dx')
        self.stats(dx)
        print('stats dy')
        self.stats(dy)
        frames = [r[2] for r in o]
        if frames != list(range(o[0][2], o[0][2] + len(o))):
            print(f'**** FRAMES ARE NOT SEQUENTIAL: {frames}') # frame numbers are sequential
    
    def analyze(self,objects):
        for objectid in objects:
            print()
            print(f'analyzing {objectid} with {len(objects[objectid])} objects')
            self.analyze_object(objects[objectid])
    
    
    
    