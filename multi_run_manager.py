"""
Multi-Run Manager for SVM Classification - Simplified version
Only prints statistics to console
"""

import numpy as np


class MultiRunManager:
    """Manages multiple experimental runs and computes statistics."""
    
    def __init__(self):
        self.runs_data = []
        
    def add_run(self, run_id, test_metrics, train_metrics=None):
        """Add results from a single run."""
        run_data = {
            'run_id': run_id,
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall']
        }
        
        if train_metrics is not None:
            run_data['train_accuracy'] = train_metrics['accuracy']
            run_data['train_f1'] = train_metrics['f1']
            run_data['train_precision'] = train_metrics['precision']
            run_data['train_recall'] = train_metrics['recall']
        
        self.runs_data.append(run_data)
        
    def compute_and_print_statistics(self):
        """Compute and print statistics."""
        if len(self.runs_data) == 0:
            print("No runs data available.")
            return
        
        n_runs = len(self.runs_data)
        
        # Extract test metrics
        test_acc = [r['test_accuracy'] for r in self.runs_data]
        test_f1 = [r['test_f1'] for r in self.runs_data]
        test_prec = [r['test_precision'] for r in self.runs_data]
        test_rec = [r['test_recall'] for r in self.runs_data]
        
        print("\n" + "="*80)
        print(f"RESULTS FROM {n_runs} RUNS")
        print("="*80)
        print("\nTEST SET METRICS:")
        print("-"*80)
        
        self._print_metric("Accuracy", test_acc)
        self._print_metric("F1-Score", test_f1)
        self._print_metric("Precision", test_prec)
        self._print_metric("Recall", test_rec)
        
        # Training metrics if available
        if 'train_accuracy' in self.runs_data[0]:
            train_acc = [r['train_accuracy'] for r in self.runs_data]
            train_f1 = [r['train_f1'] for r in self.runs_data]
            train_prec = [r['train_precision'] for r in self.runs_data]
            train_rec = [r['train_recall'] for r in self.runs_data]
            
            print("\nTRAINING SET METRICS:")
            print("-"*80)
            self._print_metric("Accuracy", train_acc)
            self._print_metric("F1-Score", train_f1)
            self._print_metric("Precision", train_prec)
            self._print_metric("Recall", train_rec)
        
        print("="*80 + "\n")
    
    def _print_metric(self, name, values):
        """Helper to print a single metric's statistics."""
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        stderr = std / np.sqrt(len(values))
        
        print(f"\n{name}:")
        print(f"  {mean:.4f} ± {std:.4f} (SE: {stderr:.4f})")
        print(f"  Range: [{np.min(values):.4f}, {np.max(values):.4f}]")