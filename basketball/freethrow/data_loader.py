import os
import json
import pandas as pd
import numpy as np

class FreeThrowDataLoader:
    def __init__(self, base_path):
        """
        Initialize the data loader with the base path to the data directory.
        
        Args:
            base_path (str): Path to the directory containing the data files
        """
        self.base_path = base_path
        
    def load_participant_data(self, participant_id):
        """
        Load all trials for a specific participant.
        
        Args:
            participant_id (str): Identifier for the participant (e.g., 'P0001')
            
        Returns:
            List of dictionaries containing trial data
        """
        # Construct path to participant's data directory
        participant_path = os.path.join(self.base_path, "data", participant_id)
        
        if not os.path.exists(participant_path):
            raise ValueError(f"No data found for participant {participant_id}")
            
        # Load trials
        trials_data = []
        trial_files = sorted([f for f in os.listdir(participant_path) if f.endswith('.json')])
        
        for trial_file in trial_files:
            trial_path = os.path.join(participant_path, trial_file)
            try:
                with open(trial_path, 'r') as f:
                    trial_data = json.load(f)
                    # Add participant_id to each trial
                    trial_data['participant_id'] = participant_id
                    trials_data.append(trial_data)
            except Exception as e:
                print(f"Error loading {trial_file}: {str(e)}")
        
        return trials_data
    
    def extract_trial_features(self, trial_data):
        """
        Extract relevant features from a single trial.
        
        Args:
            trial_data (dict): Raw trial data
            
        Returns:
            dict: Extracted features
        """
        features = {
            'trial_id': trial_data.get('trial_id', ''),
            'result': trial_data.get('result', 'unknown'),
            'tracking_data': []
        }
        
        # Extract tracking data
        for frame in trial_data.get('tracking', []):
            if 'data' in frame and 'player' in frame['data']:
                features['tracking_data'].append({
                    'time': frame.get('time', 0),
                    'joints': frame['data']['player']
                })
        
        return features
    
    def process_trials_batch(self, participant_data, batch_size=None):
        """
        Process multiple trials in batches.
        
        Args:
            participant_data (dict): Data for a participant
            batch_size (int, optional): Number of trials to process at once
            
        Returns:
            list: List of processed trial data
        """
        trials = participant_data['trials']
        if batch_size is None:
            batch_size = len(trials)
            
        processed_trials = []
        
        for i in range(0, len(trials), batch_size):
            batch = trials[i:i + batch_size]
            batch_processed = [self.extract_trial_features(trial) for trial in batch]
            processed_trials.extend(batch_processed)
            
        return processed_trials
    
    def calculate_trial_statistics(self, trial_data):
        """
        Calculate statistics for a single trial.
        
        Args:
            trial_data (dict): Processed trial data
            
        Returns:
            dict: Trial statistics
        """
        stats = {
            'trial_id': trial_data['trial_id'],
            'result': trial_data['result']
        }
        
        if trial_data['tracking_data']:
            try:
                # Calculate temporal statistics
                durations = np.diff([frame['time'] for frame in trial_data['tracking_data']])
                stats['duration_mean'] = np.mean(durations) if len(durations) > 0 else 0
                stats['duration_std'] = np.std(durations) if len(durations) > 0 else 0
                
                # Calculate spatial statistics
                r_wrist_positions = [
                    frame['joints']['R_WRIST'] 
                    for frame in trial_data['tracking_data']
                    if 'R_WRIST' in frame['joints']
                ]
                
                if r_wrist_positions:
                    r_wrist_positions = np.array(r_wrist_positions)
                    stats['wrist_displacement'] = np.linalg.norm(
                        r_wrist_positions[-1] - r_wrist_positions[0]
                    )
                    stats['wrist_path_length'] = np.sum(
                        np.linalg.norm(np.diff(r_wrist_positions, axis=0), axis=1)
                    )
            except Exception as e:
                print(f"Error calculating statistics for trial {trial_data['trial_id']}: {str(e)}")
                stats.update({
                    'duration_mean': 0,
                    'duration_std': 0,
                    'wrist_displacement': 0,
                    'wrist_path_length': 0
                })
        
        return stats
    
    def get_trial_summary(self, participant_data):
        """
        Generate a summary of all trials for a participant.
        
        Args:
            participant_data (dict): Data for a participant
            
        Returns:
            pd.DataFrame: Summary statistics for all trials
        """
        processed_trials = self.process_trials_batch(participant_data)
        trial_stats = [
            self.calculate_trial_statistics(trial) 
            for trial in processed_trials
        ]
        
        summary_df = pd.DataFrame(trial_stats)
        
        # Calculate additional summary statistics
        if not summary_df.empty:
            made_shots = summary_df['result'] == 'made'
            summary_stats = {
                'total_trials': len(summary_df),
                'made_shots': sum(made_shots),
                'shooting_percentage': (sum(made_shots) / len(summary_df)) * 100 if len(summary_df) > 0 else 0,
                'avg_wrist_displacement': summary_df['wrist_displacement'].mean(),
                'avg_path_length': summary_df['wrist_path_length'].mean()
            }
            
            # Add to summary DataFrame
            stats_df = pd.DataFrame([summary_stats])
            return summary_df, stats_df
        
        return summary_df, pd.DataFrame()

def main():
    """Test the data loader with directory structure checking"""
    # Set up base path
    base_path = r"C:\Users\16476\Downloads\SPL-Open-Data-main\basketball\freethrow"
    
    print("Checking directory structure:")
    print("\nBase directory contents:")
    print(os.listdir(base_path))
    
    data_dir = os.path.join(base_path, "data")
    if os.path.exists(data_dir):
        print("\nData directory contents:")
        print(os.listdir(data_dir))
        
        # Check P0001 directory
        p0001_dir = os.path.join(data_dir, "P0001")
        if os.path.exists(p0001_dir):
            print("\nP0001 directory contents:")
            print(os.listdir(p0001_dir))
    else:
        print("\nData directory not found!")
    
    # Initialize loader and process data
    try:
        print("\nInitializing data loader...")
        loader = FreeThrowDataLoader(base_path)
        
        print("Loading participant data...")
        participant_data = loader.load_participant_data('P0001')
        
        print(f"Found {len(participant_data['trials'])} trials")
        
        print("\nProcessing trials...")
        trial_summary, summary_stats = loader.get_trial_summary(participant_data)
        
        print("\nTrial Summary:")
        print(trial_summary.describe())
        
        print("\nOverall Statistics:")
        print(summary_stats)
        
    except Exception as e:
        print(f"\nError processing data: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()