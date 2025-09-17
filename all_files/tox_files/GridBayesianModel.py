import numpy 
import scipy.stats 
import math 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score,roc_curve,ConfusionMatrixDisplay, auc

#String literals to constants
TRUE_LABELS = "true_labels"
PREDICTED_LABELS= "predicted_labels"


MOVING=1
NOTMOVING=0

LOG_PDFS="log_pdfs"
DEAD_PDFS="dead_log_sum_pdfs"
ALIVE_PDFS="alive_log_sum_pdfs"

class BayesianModel:
    def __init__(self):
        
        self.prior_dead=0.0
        self.prior_alive=0.0
        
        self.filtered_thresholds = []
        self.best_accuracy_threshold = None
        self.best_precision_threshold = None
        self.best_recall_threshold= None
        self.optimal_threshold = None
        
        self.best_classify = -float('inf')       
        self.best_accuracy = -float('inf')
        self.best_precision = -float('inf')
        self.best_recall = -float('inf')
    
    def calculate_prior(self,dead_train_obs,alive_train_obs):
        """
        calculate the prior probabilities according to the number of points, the forumla is: #of alive or dead objects/total number of object's (according to trainning)
    
        Parameters:
        -alive_train_obs: The dictionary with alive keys as object IDs and values as observations (e.g., lists of log PDFs).
        -dead_train_obs: The dictionary with dead keys as object IDs and values as observations (e.g., lists of log PDFs).
    
        """
        self.prior_dead=len(dead_train_obs)/(len(dead_train_obs)+len(alive_train_obs))
        self.prior_alive=len(alive_train_obs)/(len(dead_train_obs)+len(alive_train_obs))
       
        return
    
    def sum_log_probabilities(self,curr_obs_with_probs):
        
        curr_likelihood={}
        
        for obj_id in curr_obs_with_probs:
            cls=NOTMOVING
            valid_dead_log_pdfs = [v for v in curr_obs_with_probs[obj_id][DEAD_PDFS] if v != 0]
            valid_alive_log_pdfs = [v for v in curr_obs_with_probs[obj_id][ALIVE_PDFS] if v != 0]        
            if not valid_dead_log_pdfs or not valid_alive_log_pdfs:
                print(f"Warning: Invalid log_pdfs for obj_id {obj_id}. {valid_dead_log_pdfs} {valid_alive_log_pdfs}")
                continue
       
            # Compute the log posterior probabilities
            dead_log_sum_pdf = numpy.sum(valid_dead_log_pdfs) + numpy.log(self.prior_dead)
            alive_log_sum_pdf = numpy.sum(valid_alive_log_pdfs) + numpy.log(self.prior_alive)
        
            #print(f"dead_log_sum is {dead_log_sum_pdf}, alive_log_sum_pdf: {alive_log_sum_pdf}")
            if dead_log_sum_pdf>alive_log_sum_pdf:
                cls=NOTMOVING
            else:
                cls=MOVING
                
            curr_likelihood[obj_id] = {
                DEAD_PDFS: dead_log_sum_pdf,
                ALIVE_PDFS: alive_log_sum_pdf,
                TRUE_LABELS: curr_obs_with_probs[obj_id][TRUE_LABELS],
                PREDICTED_LABELS: cls
            }          
        
        return curr_likelihood
        
    def find_optimal_threshold(self, curr_likelihood_without_threshold):
    
        true_labels = []
        
        for obj_id in curr_likelihood_without_threshold:
            dead_logs_sum = curr_likelihood_without_threshold[obj_id][DEAD_PDFS]
            alive_logs_sum = curr_likelihood_without_threshold[obj_id][ALIVE_PDFS] 
        
            threshold = dead_logs_sum - alive_logs_sum
            self.filtered_thresholds.append(threshold)
            
            curr_true_label = curr_likelihood_without_threshold[obj_id][TRUE_LABELS]
            true_labels.append(curr_true_label)
        
        for i in range(len(self.filtered_thresholds)):
            delta=self.filtered_thresholds[i]
            for obj_id in curr_likelihood_without_threshold:
                dead_logs_sum = curr_likelihood_without_threshold[obj_id][DEAD_PDFS]
                alive_logs_sum = curr_likelihood_without_threshold[obj_id][ALIVE_PDFS]
                if dead_logs_sum>alive_logs_sum+delta:
                    cls=NOTMOVING
                    curr_likelihood_without_threshold[obj_id][PREDICTED_LABELS]=cls
                else:
                    cls=MOVING
                    curr_likelihood_without_threshold[obj_id][PREDICTED_LABELS]=cls
        
            true_label = [curr_likelihood_without_threshold[obj_id][TRUE_LABELS] for obj_id in curr_likelihood_without_threshold]
            predicted_label= [curr_likelihood_without_threshold[obj_id][PREDICTED_LABELS] for obj_id in curr_likelihood_without_threshold]
        
            # Create the confusion matrix
            cm = confusion_matrix(true_label, predicted_label, labels=[NOTMOVING, MOVING])
            accuracy = accuracy_score(true_label, predicted_label)
            f1 = f1_score(true_label, predicted_label, pos_label=1, average='binary')
            recall = recall_score(true_label, predicted_label, pos_label=1, average='binary')
            precision = precision_score(true_label, predicted_label, pos_label=1, average='binary')
            classify = cm[0, 0] + cm[1, 1]
            #print(f"for theshold: {delta} {accuracy:<10.3f}{f1:<10.3f}{recall:<10.3f}{precision:<10.3f}")
            
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.best_accuracy_threshold = delta
            if self.best_precision < precision:
                self.best_precision_threshold = delta
                self.best_precision = precision 
            if recall > self.best_recall:
                self.best_recall = recall
                self.best_recall_threshold = delta
            
            if classify>self.best_classify and precision>=self.best_precision:
                self.best_classify=classify
                self.optimal_threshold=delta
        '''        
        print(f"Accuracy Threshold: {self.best_accuracy_threshold}, Accuracy: {self.best_accuracy},\n" 
                f"Best Precision: {self.best_precision}, Precision Threshold: {self.best_precision_threshold},\n"
                f"Best Recall: {self.best_recall}, Recall Threshold: {self.best_recall_threshold}\n"
                f"Best Classify: {self.best_classify}, Optimal Threshold: {self.optimal_threshold}\n")
        '''
        return 
        
    def predict_with_bayesian_threshold(self, curr_likelihood):
        
        for obj_id in curr_likelihood:
            dead_logs_sum = curr_likelihood[obj_id][DEAD_PDFS]
            alive_logs_sum = curr_likelihood[obj_id][ALIVE_PDFS]
            if dead_logs_sum>alive_logs_sum+self.optimal_threshold:
                cls=NOTMOVING
                curr_likelihood[obj_id][PREDICTED_LABELS]=cls
            else:
                cls=MOVING
                curr_likelihood[obj_id][PREDICTED_LABELS]=cls
        
        return curr_likelihood 
        
    def find_obj_ids(self,curr_obs,label_true,label_predicted):
        '''
        we find the list of ids to the analysis.
        Parameters:
        -curr_obs:{object_id: {LOG_PDFS:list of log of probabilities}{TRUE_LABELS:d/a}{PREDICTED_LABELS: d/a}}
        -label_true: DEAD/ALIVE
        -label_predicted: DEAD/ALIVE
        Returns:
        -ids: list of ids where the labels meet the given conditions
        '''
        ids = [ obj_id for obj_id, details in curr_obs.items()
        if details[TRUE_LABELS] == label_true and details[PREDICTED_LABELS] == label_predicted
        ]
        print(len(ids))
        
        return ids
        
    def get_prefix(self,obj_id):
        '''
        parsers the object_id for the purpose of which dataset it belongs to. it needs more work
        Parameters:
        -obj_id: str containing obj_id
        Returns:
        -D/A/date
        '''
        if obj_id.startswith("D"):
            return "D"
        elif obj_id.endswith("a") or obj_id.endswith("p") or obj_id.endswith("t"):
            return "A"
        elif "1-6-25" in obj_id:
            return "1-6-25"
            
        else:
            return "12-27-24"
            
    def plot_extracted_obj(self, extracted_ids, observations,label_true,label_predicted):
        '''
        we analysis the object's trajectory. label_true and label_predicted could be matching or not matching depending upon our analysis.
        
        Parameters:
        
        -extracted_ids: list of ids
        -observations: dictionary containing object's observations(object id: (frame,x_cordinate,y_coordinate)) 
        -label_predicted: DEAD/ALIVE
        -label_predicted: DEAD/ALIVE
        Returns:
        -N/A
        '''
        prefix_colors = {
        "D": "red",
        "A": "blue",
        "1-6-25": "green",
        "12-27-24": "pink"
        }
        i=0
        for obj_id in extracted_ids:
            if obj_id in observations:              
                points = observations[obj_id]
                x = [p[1] for p in points]  # Extract x-coordinates
                y = [p[2] for p in points]  # Extract y-coordinates
                   
                prefix = self.get_prefix(obj_id)  # Determine prefix
                color = prefix_colors.get(prefix, "black")  # Assign color (default to black if unknown)
    
                plt.plot(x, y, marker="o", linestyle="-", color=color, label=prefix if prefix not in plt.gca().get_legend_handles_labels()[1] else "")  # Plot trajectory

                # Labels & Formatting
                plt.xlabel("X Coordinate")
                plt.ylabel("Y Coordinate")
                plt.title(f"{label_true} Object Labels Predicted {label_predicted} Train Set {i}")
                plt.legend(title=f"Object id {obj_id}")
                plt.grid(True, linestyle="--", alpha=0.6)
                i+=1
                # Show the plot
                plt.show() 
                    