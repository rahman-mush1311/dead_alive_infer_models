import os
import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Import your existing preprocessing class
import sys
sys.path.append('/mnt/project')
from driver_data_preprocessing import PreProcessingObservations

# Constants
TRACKING_DATA = "tracking_data"
TRUE_LABEL = "true_label"
MOTILE = 1
NOTMOTILE = 0


class TrackStatisticsCollector:
    """
    Collects and saves track statistics from multiple populations for visualization.
    """
    
    def __init__(self):
        self.processor = PreProcessingObservations()
        self.all_data = {
            'week-old': {},
            'day-old': {},
            'mixed': {}
        }
    
    def process_single_population(self, text_dir, excel_dir, population_name):
        """
        Process all files in a population directory and collect statistics.
        
        Parameters:
        - text_dir: path to text files directory
        - excel_dir: path to excel files directory  
        - population_name: 'week-old', 'day-old', or 'mixed'
        
        Returns:
        - population_data: dictionary with all statistics and raw data
        """
        print(f"\n{'='*70}")
        print(f"Processing {population_name.upper()} population...")
        print(f"{'='*70}")
        
        # Get all text files
        text_files = sorted(glob.glob(os.path.join(text_dir, "*_trackStore.txt")))
        excel_files = sorted(glob.glob(os.path.join(excel_dir, "*.xlsx")))
        
        print(f"Found {len(text_files)} text files and {len(excel_files)} excel files")
        
        population_data = {
            'files': [],
            'all_track_lengths': [],  # All track lengths combined
            'all_track_lengths_motile': [],  # Only motile
            'all_track_lengths_notmotile': [],  # Only non-motile
            'file_statistics': {},  # Per-file statistics
            'overall_statistics': {},  # Overall statistics
            'labeled_observations': {}  # Store all labeled observations
        }
        
        # Process each file pair
        for text_file in text_files:
            # Find matching excel file
            text_basename = os.path.basename(text_file)
            # Extract the base name (remove _trackStore.txt)
            base_name = text_basename.replace('_trackStore.txt', '')
            
            # Find matching excel file
            matching_excel = None
            for excel_file in excel_files:
                excel_basename = os.path.basename(excel_file).replace('.xlsx', '')
                if excel_basename == base_name:
                    matching_excel = excel_file
                    break
            
            if matching_excel is None:
                print(f"Warning: No matching excel file for {text_basename}, skipping...")
                continue
            
            print(f"\nProcessing: {text_basename}")
            
            try:
                # Load observations and labels
                observations = self.processor.load_observations(text_file)
                labels = self.processor.load_labels(matching_excel)
                labeled_obs = self.processor.label_observations_by_expert_labels(
                    text_file, matching_excel, observations, labels
                )
                
                # Get statistics for this file
                overall_stats = self.processor.get_track_length_statistics(labeled_obs)
                by_label_stats = self.processor.get_track_statistics_by_label(labeled_obs)
                
                # Store file-level data
                file_data = {
                    'text_file': text_file,
                    'excel_file': matching_excel,
                    'overall_stats': overall_stats,
                    'by_label_stats': by_label_stats,
                    'track_lengths': overall_stats['all_lengths']
                }
                
                population_data['files'].append(text_basename)
                population_data['file_statistics'][text_basename] = file_data
                
                # Accumulate all track lengths
                population_data['all_track_lengths'].extend(overall_stats['all_lengths'])
                
                # Separate by label
                for obj_id, data in labeled_obs.items():
                    track_length = len(data[TRACKING_DATA])
                    if data[TRUE_LABEL] == MOTILE:
                        population_data['all_track_lengths_motile'].append(track_length)
                    else:
                        population_data['all_track_lengths_notmotile'].append(track_length)
                
                # Store labeled observations
                population_data['labeled_observations'].update(labeled_obs)
                
                # Print file statistics
                self.processor.print_track_statistics(labeled_obs, text_basename)
                self.processor.print_track_statistics_by_label(labeled_obs, text_basename)
                
            except Exception as e:
                print(f"Error processing {text_basename}: {e}")
                continue
        
        # Calculate overall population statistics
        if population_data['all_track_lengths']:
            population_data['overall_statistics'] = {
                'total_objects': len(population_data['all_track_lengths']),
                'total_motile': len(population_data['all_track_lengths_motile']),
                'total_notmotile': len(population_data['all_track_lengths_notmotile']),
                'max_length': int(np.max(population_data['all_track_lengths'])),
                'min_length': int(np.min(population_data['all_track_lengths'])),
                'mean_length': float(np.mean(population_data['all_track_lengths'])),
                'median_length': float(np.median(population_data['all_track_lengths'])),
                'std_length': float(np.std(population_data['all_track_lengths'])),
                'q25': float(np.percentile(population_data['all_track_lengths'], 25)),
                'q75': float(np.percentile(population_data['all_track_lengths'], 75)),
            }
            
            # Motile statistics
            if population_data['all_track_lengths_motile']:
                population_data['overall_statistics']['motile'] = {
                    'count': len(population_data['all_track_lengths_motile']),
                    'mean': float(np.mean(population_data['all_track_lengths_motile'])),
                    'median': float(np.median(population_data['all_track_lengths_motile'])),
                    'std': float(np.std(population_data['all_track_lengths_motile'])),
                    'max': int(np.max(population_data['all_track_lengths_motile'])),
                    'min': int(np.min(population_data['all_track_lengths_motile'])),
                }
            
            # Non-motile statistics
            if population_data['all_track_lengths_notmotile']:
                population_data['overall_statistics']['notmotile'] = {
                    'count': len(population_data['all_track_lengths_notmotile']),
                    'mean': float(np.mean(population_data['all_track_lengths_notmotile'])),
                    'median': float(np.median(population_data['all_track_lengths_notmotile'])),
                    'std': float(np.std(population_data['all_track_lengths_notmotile'])),
                    'max': int(np.max(population_data['all_track_lengths_notmotile'])),
                    'min': int(np.min(population_data['all_track_lengths_notmotile'])),
                }
        
        # Print overall population summary
        self._print_population_summary(population_name, population_data)
        
        return population_data
    
    def _print_population_summary(self, population_name, population_data):
        """Print summary statistics for the entire population."""
        stats = population_data['overall_statistics']
        
        print(f"\n{'='*70}")
        print(f"{population_name.upper()} POPULATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total objects:      {stats['total_objects']}")
        print(f"  Motile:           {stats['total_motile']} ({100*stats['total_motile']/stats['total_objects']:.1f}%)")
        print(f"  Non-motile:       {stats['total_notmotile']} ({100*stats['total_notmotile']/stats['total_objects']:.1f}%)")
        print(f"\nTrack Length Statistics:")
        print(f"  Mean:             {stats['mean_length']:.2f}")
        print(f"  Median:           {stats['median_length']:.2f}")
        print(f"  Std Dev:          {stats['std_length']:.2f}")
        print(f"  Min:              {stats['min_length']}")
        print(f"  Max:              {stats['max_length']}")
        print(f"  Q25-Q75:          {stats['q25']:.1f} - {stats['q75']:.1f}")
        print(f"{'='*70}\n")
    
    def collect_all_populations(self, week_old_dirs, day_old_dirs, mixed_dirs):
        """
        Collect statistics from all three populations.
        
        Parameters:
        - week_old_dirs: dict with 'text' and 'excel' directory paths
        - day_old_dirs: dict with 'text' and 'excel' directory paths
        - mixed_dirs: dict with 'text' and 'excel' directory paths
        
        Returns:
        - all_data: dictionary with data from all populations
        """
        # Process week-old
        self.all_data['week-old'] = self.process_single_population(
            week_old_dirs['text'], 
            week_old_dirs['excel'], 
            'week-old'
        )
        
        # Process day-old
        self.all_data['day-old'] = self.process_single_population(
            day_old_dirs['text'], 
            day_old_dirs['excel'], 
            'day-old'
        )
        
        # Process mixed
        self.all_data['mixed'] = self.process_single_population(
            mixed_dirs['text'], 
            mixed_dirs['excel'], 
            'mixed'
        )
        
        return self.all_data
    
    def save_data(self, output_dir, save_formats=['json', 'pickle', 'csv']):
        """
        Save collected data in multiple formats.
        
        Parameters:
        - output_dir: directory to save files
        - save_formats: list of formats ('json', 'pickle', 'csv')
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"SAVING DATA TO: {output_dir}")
        print(f"{'='*70}\n")
        
        # 1. Save as JSON (for human-readable inspection)
        if 'json' in save_formats:
            json_file = os.path.join(output_dir, 'track_statistics.json')
            
            # Prepare JSON-serializable version (remove labeled_observations)
            json_data = {}
            for pop_name, pop_data in self.all_data.items():
                json_data[pop_name] = {
                    'files': pop_data['files'],
                    'all_track_lengths': pop_data['all_track_lengths'],
                    'all_track_lengths_motile': pop_data['all_track_lengths_motile'],
                    'all_track_lengths_notmotile': pop_data['all_track_lengths_notmotile'],
                    'overall_statistics': pop_data['overall_statistics'],
                }
                # Add file statistics without labeled observations
                json_data[pop_name]['file_statistics'] = {}
                for file_name, file_data in pop_data['file_statistics'].items():
                    json_data[pop_name]['file_statistics'][file_name] = {
                        'overall_stats': {
                            'total_objects': file_data['overall_stats']['total_objects'],
                            'max_length': file_data['overall_stats']['max_length'],
                            'min_length': file_data['overall_stats']['min_length'],
                            'mean_length': file_data['overall_stats']['mean_length'],
                            'median_length': file_data['overall_stats']['median_length'],
                        },
                        'by_label_stats': file_data['by_label_stats'],
                        'track_lengths': file_data['track_lengths']
                    }
            
            with open(json_file, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"✓ Saved JSON: {json_file}")
        
        # 2. Save as Pickle (for complete Python objects including labeled_observations)
        if 'pickle' in save_formats:
            pickle_file = os.path.join(output_dir, 'track_statistics.pkl')
            with open(pickle_file, 'wb') as f:
                pickle.dump(self.all_data, f)
            print(f"✓ Saved Pickle: {pickle_file}")
        
        # 3. Save as CSV (for easy inspection in Excel)
        if 'csv' in save_formats:
            # Overall summary CSV
            summary_data = []
            for pop_name, pop_data in self.all_data.items():
                stats = pop_data['overall_statistics']
                summary_data.append({
                    'population': pop_name,
                    'total_objects': stats['total_objects'],
                    'total_motile': stats['total_motile'],
                    'total_notmotile': stats['total_notmotile'],
                    'motile_percentage': 100 * stats['total_motile'] / stats['total_objects'],
                    'mean_length': stats['mean_length'],
                    'median_length': stats['median_length'],
                    'std_length': stats['std_length'],
                    'min_length': stats['min_length'],
                    'max_length': stats['max_length'],
                    'q25': stats['q25'],
                    'q75': stats['q75'],
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_csv = os.path.join(output_dir, 'population_summary.csv')
            summary_df.to_csv(summary_csv, index=False)
            print(f"✓ Saved Summary CSV: {summary_csv}")
            
            # Per-file CSV
            file_data = []
            for pop_name, pop_data in self.all_data.items():
                for file_name, file_stats in pop_data['file_statistics'].items():
                    file_data.append({
                        'population': pop_name,
                        'file': file_name,
                        'total_objects': file_stats['overall_stats']['total_objects'],
                        'motile_count': file_stats['by_label_stats']['motile']['count'],
                        'notmotile_count': file_stats['by_label_stats']['notmotile']['count'],
                        'mean_length': file_stats['overall_stats']['mean_length'],
                        'median_length': file_stats['overall_stats']['median_length'],
                        'max_length': file_stats['overall_stats']['max_length'],
                        'min_length': file_stats['overall_stats']['min_length'],
                    })
            
            files_df = pd.DataFrame(file_data)
            files_csv = os.path.join(output_dir, 'per_file_statistics.csv')
            files_df.to_csv(files_csv, index=False)
            print(f"✓ Saved Per-File CSV: {files_csv}")
            
            # Raw track lengths CSV (for plotting)
            raw_data = []
            for pop_name, pop_data in self.all_data.items():
                for length in pop_data['all_track_lengths']:
                    raw_data.append({
                        'population': pop_name,
                        'track_length': length
                    })
            
            raw_df = pd.DataFrame(raw_data)
            raw_csv = os.path.join(output_dir, 'raw_track_lengths.csv')
            raw_df.to_csv(raw_csv, index=False)
            print(f"✓ Saved Raw Track Lengths CSV: {raw_csv}")
        
        print(f"\n{'='*70}")
        print("DATA SAVED SUCCESSFULLY!")
        print(f"{'='*70}\n")
    
    def print_summary_statistics(self,population_data):
        """
        Print summary statistics for all populations.
        """
        print("\n" + "="*70)
        print("SUMMARY STATISTICS")
        print("="*70)
        
        for pop_name, lengths in population_data.items():
            print(f"\n{pop_name.upper()}:")
            print(f"  Total objects:  {len(lengths)}")
            print(f"  Mean:           {np.mean(lengths):.2f}")
            print(f"  Median:         {np.median(lengths):.2f}")
            print(f"  Std Dev:        {np.std(lengths):.2f}")
            print(f"  Min:            {np.min(lengths)}")
            print(f"  Max:            {np.max(lengths)}")
            print(f"  Q25:            {np.percentile(lengths, 25):.1f}")
            print(f"  Q75:            {np.percentile(lengths, 75):.1f}")
        
        print("="*70 + "\n")
    
    def load_track_data_from_pickle(self,pickle_file):
        """
        Load track statistics from saved pickle file.
        
        Parameters:
        - pickle_file: path to the pickle file
        
        Returns:
        - data: dictionary with all population data
        """
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        return data


    def load_track_data_from_csv(self,csv_file):
        """
        Load raw track lengths from CSV file.
        
        Parameters:
        - csv_file: path to raw_track_lengths.csv
        
        Returns:
        - data: dictionary {population_name: list_of_track_lengths}
        """
        df = pd.read_csv(csv_file)
        
        data = {}
        for pop_name in df['population'].unique():
            pop_df = df[df['population'] == pop_name]
            data[pop_name] = pop_df['track_length'].tolist()
        
        return data


    def plot_overlaid_histograms(self,population_data, bins=50, alpha=0.6, figsize=(6, 6)):
        """
        Create overlaid histograms for track length distributions.
        
        Parameters:
        - population_data: dict {population_name: list_of_track_lengths}
        - bins: number of bins
        - alpha: transparency
        - figsize: figure size
        
        Returns:
        - fig, ax: matplotlib objects
        """
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 11
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = {
            'week-old': '#2E86AB',
            'day-old': '#A23B72',
            'organics': '#F18F01'
        }
        
        for pop_name, track_lengths in population_data.items():
            color = colors.get(pop_name, 'gray')
            
            ax.hist(track_lengths, bins=bins, alpha=alpha,
                    label=f'{pop_name}',
                    color=color, edgecolor='black', linewidth=0.5)
        
        ax.set_xlabel('Track Length (number of frames)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Frequency (count)', fontsize=13, fontweight='bold')
        ax.set_title('Distribution of Track Lengths Across Populations',
                     fontsize=15, fontweight='bold', pad=20)
        
        # Enhanced legend
        legend_labels = []
        for pop_name, track_lengths in population_data.items():
            mean_val = np.mean(track_lengths)
            #median_val = np.median(track_lengths)
            std_val = np.std(track_lengths)
            legend_labels.append(
                f'{pop_name} (n={len(track_lengths)})\n'
                f'  μ={mean_val:.1f}, σ={std_val:.1f}'
            )
        
        ax.legend(legend_labels, fontsize=10, loc='upper right', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        return fig, ax
    
    def visualize_from_saved_data(self,data_source, source_type, output_dir):
        """
        Main visualization function that loads from saved data.
        
        Parameters:
        - data_source: path to pickle file or CSV file
        - source_type: 'pickle' or 'csv'
        - output_dir: where to save figures (optional)
        
        Returns:
        - population_data: loaded data
        - figures: tuple of matplotlib figures
        """
        print("Loading data...")
        
        if source_type == 'pickle':
            all_data = self.load_track_data_from_pickle(data_source)
            # Extract just the track lengths for plotting
            population_data = {
                'week-old': all_data['week-old']['all_track_lengths'],
                'day-old': all_data['day-old']['all_track_lengths'],
                'organics': all_data['mixed']['all_track_lengths']
            }
            
            population_data_full = {
                'week-old': all_data['week-old'],
                'day-old': all_data['day-old'],
                'organics': all_data['mixed']  # Full data structure, just renamed key
            }
            
            
        elif source_type == 'csv':
            population_data = self.load_track_data_from_csv(data_source)
        else:
            raise ValueError("source_type must be 'pickle' or 'csv'")
        
        print("Data loaded successfully!")
        '''
        # Print summary
        self.print_summary_statistics(population_data)
        
        # Generate visualizations
        print("Generating visualizations...")
        
        fig1, ax1 = self.plot_overlaid_histograms(population_data, bins=60, alpha=0.6)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig1.savefig(os.path.join(output_dir, 'histogram_overlaid_frame_stats.png'),
                        dpi=300, bbox_inches='tight')
            print(f"✓ Saved: histogram_overlaid_frame_stats.png")
        
        plt.show()
        
        return population_data, (fig1)
        '''
        # 2. Sample counts
        print("4. Creating sample count bar chart...")
        fig2, ax2 = self.plot_grouped_bar_sample_counts(population_data_full)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            fig2.savefig(os.path.join(output_dir, 'grouped_bar_counts_sample_count_groupwise.png'),
                    dpi=300, bbox_inches='tight')
            print("   ✓ Saved: grouped_bar_counts_sample_count_groupwise.png")
        plt.show()
        
        return population_data, (fig2)
            
    def plot_grouped_bar_sample_counts(self, population_data_full, figsize=(8, 6)):
        """
        Create grouped bar chart showing object counts (Motile vs Non-Motile).
        
        X-axis: Population groups
        Y-axis: Count (number of objects)
        Bars: Motile and Non-Motile counts side-by-side
        
        Parameters:
        - population_data_full: full data dict from pickle
        - figsize: figure size
        
        Returns:
        - fig, ax: matplotlib objects
        """
        sns.set_style("whitegrid")
        
        populations = []
        motile_counts = []
        notmotile_counts = []
        
        for pop_name in ['week-old', 'day-old', 'organics']:
            if pop_name in population_data_full:
                populations.append(pop_name)
                stats = population_data_full[pop_name]['overall_statistics']
                motile_counts.append(stats['total_motile'])
                notmotile_counts.append(stats['total_notmotile'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(populations))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, motile_counts, width,
                       label='Motile', color='#06A77D',
                       edgecolor='black', linewidth=1.5, alpha=0.85)
        
        bars2 = ax.bar(x + width/2, notmotile_counts, width,
                       label='Non-Motile', color='#D62246',
                       edgecolor='black', linewidth=1.5, alpha=0.85)
        
        ax.set_xlabel('Population', fontsize=8, fontweight='bold')
        ax.set_ylabel('Number of Objects', fontsize=8, fontweight='bold')
        ax.set_title('Sample Size: Motile vs Non-Motile Objects',
                     fontsize=8, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(populations, fontsize=8)
        ax.legend(fontsize=8, loc='upper left')
        
        # ✅ FIXED: Add value labels and percentages with proper indexing
        for bars, counts in [(bars1, motile_counts), (bars2, notmotile_counts)]:
            for idx, (bar, count) in enumerate(zip(bars, counts)):
                height = bar.get_height()
                total = motile_counts[idx] + notmotile_counts[idx]
                percentage = 100 * count / total if total > 0 else 0
                
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(count)}\n({percentage:.1f}%)',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_axisbelow(True)
        
        plt.tight_layout()
        return fig, ax
    
    def collector_frame_stat_data():
        """
        Main function to collect and save track statistics from all populations.
        """
        # Initialize collector
        collector = self()
        
        # Define your directory paths
        # UPDATE THESE TO YOUR ACTUAL PATHS
        base_dir = "C:/Users/mushfika_rahman1/Documents/dead_alive_infer_models/all_files/other_files"
        
        week_old_dirs = {
            'text': os.path.join(base_dir, "week-old/text"),
            'excel': os.path.join(base_dir, "week-old/excel")
        }
        
        day_old_dirs = {
            'text': os.path.join(base_dir, "day-old/text"),
            'excel': os.path.join(base_dir, "day-old/excel")
        }
        
        mixed_dirs = {
            'text': os.path.join(base_dir, "mixed-organics/text"),
            'excel': os.path.join(base_dir, "mixed-organics/excel")
        }
        
        # Collect data from all populations
        all_data = collector.collect_all_populations(week_old_dirs, day_old_dirs, mixed_dirs)
        
        # Save data
        output_dir = "C:/Users/mushfika_rahman1/Documents/dead_alive_infer_models/results/frame_stat"
        collector.save_data(output_dir, save_formats=['json', 'pickle', 'csv'])
        
        print("Done! Data is ready for visualization.")
        
        return collector, all_data
    
    def call_frame_stat_visualizor():

        pickle_file = "C:/Users/mushfika_rahman1/Documents/dead_alive_infer_models/results/frame_stat/track_statistics.pkl"
        
        # Option 2: Load from CSV (lighter, just track lengths)
        # csv_file = "/home/claude/track_statistics_data/raw_track_lengths.csv"
        
        output_dir = "C:/Users/mushfika_rahman1/Documents/dead_alive_infer_models/results/frame_stat"
        visualizor = self()
        # Run visualization
        print("Starting visualization from saved data...")
        population_data, figures = visualizor.visualize_from_saved_data(pickle_file,'pickle', output_dir)
        return




