import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score,roc_curve,ConfusionMatrixDisplay,auc

TRUE_LABELS = "true_labels"
PREDICTED_LABELS= "predicted_labels"
LOG_PDFS="log_pdfs"

MOVING=1
NOTMOVING=0

class OutlierModelEvaluation:
    def __init__(self, window=3):
    
        self.window_size = window
        self.filtered_thresholds = []
        
        self.best_accuracy_threshold = None
        self.best_precision_threshold = None
        self.optimal_threshold = None
        
        self.best_classify = -float('inf')
        self.best_precision = -float('inf')
        self.best_accuracy = -float('inf')
       
    
    def get_thresholds_from_roc(self,curr_obs):
        '''
        it filters furthers when the tpr improves fpr decreases, it rounds the fpr, tpr values to 2 digits get more relevant thresholds. assigns to the class's self.filtered_thresholds list
        '''
        true_labels=[]
        log_pdf_values=[]
        seen = set()

        for obj_data in curr_obs.values():            
            for log_pdf in obj_data[LOG_PDFS]:
                if log_pdf not in seen:
                    seen.add(log_pdf)
                    log_pdf_values.append(log_pdf)
                    true_labels.append(obj_data[TRUE_LABELS])
                    
        true_labels=np.array(true_labels)
        log_pdf_values=np.array(log_pdf_values)
        print(len(true_labels),len(log_pdf_values))
        fpr, tpr, roc_thresholds = roc_curve(true_labels, log_pdf_values)
         
        for i in range(1, len(roc_thresholds)):
            if round(tpr[i],2) > round(tpr[i - 1],2) or round(fpr[i],2)<round(fpr[i-1],2):
                self.filtered_thresholds.append(roc_thresholds[i])
        return 
    
    def predict_probabilities_dictionary_update(self,curr_obs):
        '''
        this function final classification is done with best threshold which gives maximum classification. It prints the confusion matrices also.
        Parameters:
        - curr_obs :dictionary {object_id: {LOG_PDFS:list of log of probabilities}{TRUE_LABELS:d/a}}
        Returns:
        - curr_obs :dictionary {object_id: {LOG_PDFS:list of log of probabilities}{TRUE_LABELS:d/a}{PREDICTED_LABELS: d/a}}
        '''
        for obj_id in curr_obs:
            cls = NOTMOVING
            for i in range(len(curr_obs[obj_id][LOG_PDFS]) - self.window_size + 1):
                w = curr_obs[obj_id][LOG_PDFS][i:i+self.window_size]
                if all([p <= self.optimal_threshold for p in w]):
                    cls = MOVING
                    break

        # Update the dictionary with predicted and true labels
            curr_obs[obj_id] = {
                LOG_PDFS: curr_obs[obj_id][LOG_PDFS],  # Original log PDF values
                TRUE_LABELS: curr_obs[obj_id][TRUE_LABELS],
                PREDICTED_LABELS: cls
            }
        return curr_obs
        
    def plot_confusion_matrix_outlier_model(self,curr_obs, typeofset):
    
        true_labels = [curr_obs[obj_id][TRUE_LABELS] for obj_id in curr_obs]
        predicted_labels = [curr_obs[obj_id][PREDICTED_LABELS] for obj_id in curr_obs]

        # Create the confusion matrix
        cm = confusion_matrix(true_labels, predicted_labels, labels=[NOTMOVING, MOVING])
        accuracy = accuracy_score(true_labels, predicted_labels)
        f1 = f1_score(true_labels, predicted_labels, pos_label='a', average='binary')
        recall = recall_score(true_labels, predicted_labels, pos_label='a', average='binary')
        precision = precision_score(true_labels, predicted_labels, pos_label='a', average='binary',zero_division=0)
        print(f"{accuracy:<10.3f}{f1:<10.3f}{recall:<10.3f}{precision:<10.3f}")
        
        # Display confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Moving(0)", "Moving (1)"])
        disp.plot(cmap="Blues")
        disp.ax_.set_title(f"Confusion Matrix Using {typeofset}")
        disp.ax_.set_xlabel("Predicted Labels")
        disp.ax_.set_ylabel("True Labels")
    
        metrics_text = (
            f"Accuracy: {accuracy:.3f}\n"
            f"F1-Score: {f1:.3f}\n"
            f"Recall: {recall:.3f}\n"
            f"Precision: {precision:.3f}\n"
            
        )
        #f"Threshold: {self.best_accuracy_threshold:.3f}"
        disp.ax_.legend(
            handles=[
                plt.Line2D([], [], color='white', label=metrics_text)
            ],
            loc='lower right',
            fontsize=10,
            frameon=False
        )
    
        
        plt.show()
        return 
        
    def evaluate_thresholds_window_sizes(self,curr_obs, window_sizes):
        """
        this function selects thresholds from curr_obs dictionary using get_thresholds_from_roc(). Additionally, this function does the classification according to the thresholds got from roc_curve, it looks at different window to consider the classification
        the metrics.
        Parameters:
        - curr_obs dictionary {object_id: {LOG_PDFS:list of log of probabilities}{TRUE_LABELS:d/a}}
        - window sizes : list of window to explore [1-10]
        Returns:
        - N/A
    
        """
        self.get_thresholds_from_roc(curr_obs)
        
        for window in window_sizes:
            #print(f"for window {window} thresholds to explore: {len(self.filtered_thresholds)}")
            
            for threshold in self.filtered_thresholds:
                true_labels = []
                predictions = []
 
                for obj_id, values in curr_obs.items():
                    cls = NOTMOVING
                    log_values = values[LOG_PDFS]
                    for i in range(len(log_values) - window + 1):
                        w = log_values[i:i + window]
                        if all(p <= threshold for p in w):
                            cls = MOVING
                            break
                    
                    predictions.append(cls)
                    true_labels.append(values[TRUE_LABELS])
                           
                
                cm = confusion_matrix(true_labels, predictions, labels=[0, 1])
                accuracy = accuracy_score(true_labels, predictions)
                f1 = f1_score(true_labels, predictions)
                recall = recall_score(true_labels, predictions,zero_division=0)
                precision = precision_score(true_labels, predictions,zero_division=0)
                classify = cm[0, 0] + cm[1, 1]  # TP + TN
            
                if classify > self.best_classify:
                    self.best_accuracy = accuracy
                    self.best_accuracy_threshold = threshold
                    self.best_classify=classify                   
                    
                if precision > self.best_precision:
                    self.best_precision = precision
                    self.best_precision_threshold = threshold
                
                if classify >= self.best_classify and precision >= self.best_precision:
                    self.window_size=window
                    self.optimal_threshold= threshold
                    
                    #print(f"window_reset to {self.window_size} classify {self.best_classify} {accuracy:<10.3f}{f1:<10.3f}{recall:<10.3f}{precision:<10.3f}\n"
                            #f"Optimal Threshold: {self.optimal_threshold}")
                            
            #print(f"for threshold: {threshold} {accuracy:<10.3f}{f1:<10.3f}{recall:<10.3f}{precision:<10.3f}")   
            