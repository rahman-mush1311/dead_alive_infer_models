import numpy 
import scipy.stats 
import math 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, precision_score,roc_curve,ConfusionMatrixDisplay, auc, precision_recall_curve, auc, average_precision_score

#String literals to constants
TRUE_LABEL = "true_label"
PREDICTED_LABEL= "predicted_label"


MOTILE=1
NOTMOTILE=0 

LOG_PDFS="log_pdfs"
DEAD_PDFS="dead_log_sum_pdfs"
ALIVE_PDFS="alive_log_sum_pdfs"

class BayesianModel:
    def __init__(self):
        
        self.prior_dead=0.5
        self.prior_alive=0.5
        
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
            cls=NOTMOTILE
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
                cls=NOTMOTILE
            else:
                cls=MOTILE
                
            curr_likelihood[obj_id] = {
                DEAD_PDFS: dead_log_sum_pdf,
                ALIVE_PDFS: alive_log_sum_pdf,
                TRUE_LABEL: curr_obs_with_probs[obj_id][TRUE_LABEL],
                PREDICTED_LABEL: cls
            }          
        
        #self.calculate_auc_pr(curr_likelihood)
        return curr_likelihood
        
    def find_optimal_threshold(self, curr_likelihood_without_threshold):
    
        true_labels = []
        
        for obj_id in curr_likelihood_without_threshold:
            dead_logs_sum = curr_likelihood_without_threshold[obj_id][DEAD_PDFS]
            alive_logs_sum = curr_likelihood_without_threshold[obj_id][ALIVE_PDFS] 
        
            threshold = dead_logs_sum - alive_logs_sum
            self.filtered_thresholds.append(threshold)
            
            curr_true_label = curr_likelihood_without_threshold[obj_id][TRUE_LABEL]
            true_labels.append(curr_true_label)
        
        for i in range(len(self.filtered_thresholds)):
            delta=self.filtered_thresholds[i]
            for obj_id in curr_likelihood_without_threshold:
                dead_logs_sum = curr_likelihood_without_threshold[obj_id][DEAD_PDFS]
                alive_logs_sum = curr_likelihood_without_threshold[obj_id][ALIVE_PDFS]
                if dead_logs_sum>alive_logs_sum+delta:
                    cls=NOTMOTILE
                    curr_likelihood_without_threshold[obj_id][PREDICTED_LABEL]=cls
                else:
                    cls=MOTILE
                    curr_likelihood_without_threshold[obj_id][PREDICTED_LABEL]=cls
        
            true_label = [curr_likelihood_without_threshold[obj_id][TRUE_LABEL] for obj_id in curr_likelihood_without_threshold]
            predicted_label= [curr_likelihood_without_threshold[obj_id][PREDICTED_LABEL] for obj_id in curr_likelihood_without_threshold]
        
            # Create the confusion matrix
            cm = confusion_matrix(true_label, predicted_label, labels=[NOTMOTILE, MOTILE])
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
                cls=NOTMOTILE
                curr_likelihood[obj_id][PREDICTED_LABEL]=cls
            else:
                cls=MOTILE
                curr_likelihood[obj_id][PREDICTED_LABEL]=cls
        
        return curr_likelihood 
    
    def calculate_auc_pr(self, predictions_dict):
        """
        Calculate and plot the Precision-Recall curve and AUC-PR score.
        
        What does AUC-PR quantify?
        --------------------------------
        - Precision-Recall (PR) curve shows the trade-off between precision and recall
          at different classification thresholds
        - Precision: Of all objects predicted as MOTILE, what fraction are actually motile?
          Formula: TP / (TP + FP)
        - Recall: Of all truly MOTILE objects, what fraction did we correctly identify?
          Formula: TP / (TP + FN)
        - AUC-PR: Area under the PR curve (ranges 0-1, higher is better)
        - Unlike ROC-AUC, PR curves are better for imbalanced datasets
        
        Parameters:
        -----------
        predictions_dict : dict
            Dictionary with structure:
            {obj_id: {
                DEAD_PDFS: float (log probability of dead/non-motile),
                ALIVE_PDFS: float (log probability of alive/motile),
                TRUE_LABEL: int (0 or 1),
                PREDICTED_LABEL: int (0 or 1)
            }}
        ALIVE_PDFS : str
            Key name for alive log probabilities
        DEAD_PDFS : str
            Key name for dead log probabilities
        TRUE_LABEL : str
            Key name for true labels
        MOTILE : int, default=1
            The label value for the positive class (motile organisms)
        
        Returns:
        --------
        auc_pr_score : float
            Area under the Precision-Recall curve
        precision : np.array
            Precision values at different thresholds
        recall : np.array
            Recall values at different thresholds
        avg_precision : float
            Average precision score (alternative AUC-PR calculation)
        
        What data do you need?
        ----------------------
        For each object, you need:
        1. True label (ground truth): 0 (not motile) or 1 (motile)
        2. Prediction scores: log probabilities from your model
           - Score = ALIVE_PDFS - DEAD_PDFS (log probability ratio)
           - Higher score = more confident the object is motile
           - Lower score = more confident the object is non-motile
        """
        
        # Extract true labels and calculate scores
        true_labels = []
        prediction_scores = []
        
        for obj_id, data in predictions_dict.items():
            # True label (0 or 1)
            true_labels.append(data[TRUE_LABEL])
            
            # Score calculation: log probability ratio (Option B)
            # Higher score = more likely to be ALIVE/MOTILE
            score = data[ALIVE_PDFS] - data[DEAD_PDFS]
            prediction_scores.append(score)
        
        # Convert to numpy arrays
        true_labels = numpy.array(true_labels)
        prediction_scores = numpy.array(prediction_scores)
        
        # Calculate precision-recall curve
        precision, recall, thresholds = precision_recall_curve(true_labels, prediction_scores, pos_label=MOTILE)
        
        # Calculate AUC-PR
        auc_pr_score = auc(recall, precision)
        
        # Alternative: Average Precision (AP) - similar to AUC-PR but weighted differently
        avg_precision = average_precision_score(true_labels, prediction_scores, pos_label=MOTILE)
        
        # Plot the PR curve
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, 'b-', linewidth=2, label=f'PR Curve (AUC = {auc_pr_score:.3f})')
        plt.plot(recall, precision, 'bo', markersize=4, alpha=0.3)
        
        # Add baseline (random classifier)
        # For a balanced dataset, baseline would be 0.5
        # For imbalanced, baseline = fraction of positive samples
        baseline = numpy.sum(true_labels == MOTILE) / len(true_labels)
        plt.plot([0, 1], [baseline, baseline], 'r--', linewidth=2, label=f'Baseline (Random) = {baseline:.3f}')
        
        plt.xlabel('Recall (True Positive Rate)', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve for Motility Classification', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        # Add text box with interpretation
        textstr = f'AUC-PR: {auc_pr_score:.3f}\nAvg Precision: {avg_precision:.3f}\nBaseline: {baseline:.3f}\n\n'
        textstr += f'Total samples: {len(true_labels)}\n'
        textstr += f'Motile (positive): {numpy.sum(true_labels == MOTILE)}\n'
        textstr += f'Non-motile (negative): {numpy.sum(true_labels != MOTILE)}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        #plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
        #print(f"Precision-Recall curve saved as 'precision_recall_curve.png'")
        plt.show()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("PRECISION-RECALL ANALYSIS SUMMARY")
        print("="*60)
        print(f"AUC-PR Score:        {auc_pr_score:.4f}")
        print(f"Average Precision:   {avg_precision:.4f}")
        print(f"Baseline (Random):   {baseline:.4f}")
        print(f"\nInterpretation:")
        if auc_pr_score > 0.9:
            print("  ✓ EXCELLENT - Model has very strong predictive power")
        elif auc_pr_score > 0.7:
            print("  ✓ GOOD - Model performs well")
        elif auc_pr_score > baseline + 0.1:
            print("  ✓ FAIR - Model is better than random, but has room for improvement")
        else:
            print("  ✗ POOR - Model performs close to random guessing")
        
        print(f"\nDataset Balance:")
        print(f"  Total objects: {len(true_labels)}")
        print(f"  Motile objects: {numpy.sum(true_labels == MOTILE)} ({100*numpy.sum(true_labels == MOTILE)/len(true_labels):.1f}%)")
        print(f"  Non-motile objects: {numpy.sum(true_labels != MOTILE)} ({100*numpy.sum(true_labels != MOTILE)/len(true_labels):.1f}%)")
        print("="*60 + "\n")
        
        return 
        