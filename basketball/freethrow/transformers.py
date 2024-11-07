import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

class AdvancedShotClustering:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.shot_data = self.load_data()
        self.scaled_data = None
        self.phase_labels = None

    def load_data(self):
        all_data = []
        for file_name in os.listdir(self.data_dir):
            file_path = os.path.join(self.data_dir, file_name)
            if file_name.endswith('.json'):
                with open(file_path, 'r') as file:
                    shot = json.load(file)
                    
                    participant_id = shot['participant_id']
                    trial_id = shot['trial_id']
                    result = shot['result']
                    
                    for frame in shot['tracking']:
                        frame_data = frame['data']['player']
                        time = frame['time']
                        
                        frame_row = {
                            'participant_id': participant_id,
                            'trial_id': trial_id,
                            'result': result,
                            'time': time
                        }
                        
                        for joint, coords in frame_data.items():
                            frame_row[f"{joint}_x"] = coords[0]
                            frame_row[f"{joint}_y"] = coords[1]
                            frame_row[f"{joint}_z"] = coords[2]
                        
                        all_data.append(frame_row)

        return pd.DataFrame(all_data)

    def calculate_velocity(self):
        """
        Calculate velocity for key joints (wrist, elbow, shoulder) along each axis.
        """
        for joint in ['R_WRIST', 'L_WRIST', 'R_ELBOW', 'L_ELBOW', 'R_SHOULDER', 'L_SHOULDER']:
            for axis in ['x', 'y', 'z']:
                position = self.shot_data[f"{joint}_{axis}"]
                self.shot_data[f"{joint}_{axis}_vel"] = position.diff() / self.shot_data['time'].diff()

    def calculate_angles(self):
        """
        Calculate angles between key joints (e.g., shoulder-elbow-wrist) to capture arm orientation.
        """
        def calculate_angle(v1, v2):
            """Calculate the angle between two vectors."""
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            return np.arccos(dot_product / (norm_v1 * norm_v2)) * (180.0 / np.pi)

        # Calculate angles for right and left arms
        for side in ['R', 'L']:
            shoulder = self.shot_data[[f"{side}_SHOULDER_x", f"{side}_SHOULDER_y", f"{side}_SHOULDER_z"]].values
            elbow = self.shot_data[[f"{side}_ELBOW_x", f"{side}_ELBOW_y", f"{side}_ELBOW_z"]].values
            wrist = self.shot_data[[f"{side}_WRIST_x", f"{side}_WRIST_y", f"{side}_WRIST_z"]].values
            
            # Calculate vectors
            shoulder_to_elbow = elbow - shoulder
            elbow_to_wrist = wrist - elbow
            
            # Calculate angle between vectors at the elbow
            angles = [calculate_angle(shoulder_to_elbow[i], elbow_to_wrist[i]) for i in range(len(shoulder_to_elbow))]
            self.shot_data[f"{side}_ELBOW_angle"] = angles

        # Check if the angle columns were added successfully
        if 'R_ELBOW_angle' not in self.shot_data.columns:
            print("Error: 'R_ELBOW_angle' column was not added.")
        if 'L_ELBOW_angle' not in self.shot_data.columns:
            print("Error: 'L_ELBOW_angle' column was not added.")

    def calculate_temporal_features(self):
        """
        Calculate additional temporal features such as velocity ratios and acceleration.
        """
        self.calculate_velocity()
        self.calculate_angles()
        
        # Calculate velocity ratios between key joints
        self.shot_data['wrist_elbow_velocity_ratio'] = (
            self.shot_data['R_WRIST_z_vel'] / (self.shot_data['R_ELBOW_z_vel'] + 1e-5)
        )
        self.shot_data['wrist_shoulder_velocity_ratio'] = (
            self.shot_data['R_WRIST_z_vel'] / (self.shot_data['R_SHOULDER_z_vel'] + 1e-5)
        )

    def preprocess_data(self):
        self.calculate_temporal_features()
        
        # Impute and scale the data
        imputer = SimpleImputer(strategy='mean')
        numeric_data = self.shot_data.select_dtypes(include=[np.number])
        numeric_data_imputed = imputer.fit_transform(numeric_data)
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(numeric_data_imputed)
        return self.scaled_data

    def apply_clustering(self, n_clusters=8):
        """
        Cluster shot data into phases using K-Means with added features.
        """
        imputer = SimpleImputer(strategy='mean')
        numeric_data = self.shot_data.select_dtypes(include=[np.number])
        numeric_data_imputed = imputer.fit_transform(numeric_data)
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(numeric_data_imputed)

        # Apply KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        self.phase_labels = kmeans.fit_predict(scaled_data)
        self.shot_data['phase'] = self.phase_labels
        return self.shot_data

    def analyze_phase_relationships(self):
        """
        Examine relationships between phases and biomechanical features within each cluster.
        """
        phase_descriptions = {}
        phase_groups = self.shot_data.groupby('phase')

        for phase, group in phase_groups:
            mean_wrist_elbow_ratio = group['wrist_elbow_velocity_ratio'].mean()
            mean_wrist_shoulder_ratio = group['wrist_shoulder_velocity_ratio'].mean()
            mean_wrist_z_vel = group['R_WRIST_z_vel'].mean()
            mean_elbow_angle = group['R_ELBOW_angle'].mean()

            phase_descriptions[phase] = {
                'mean_wrist_elbow_ratio': mean_wrist_elbow_ratio,
                'mean_wrist_shoulder_ratio': mean_wrist_shoulder_ratio,
                'mean_wrist_z_velocity': mean_wrist_z_vel,
                'mean_elbow_angle': mean_elbow_angle
            }

        # Print out relationships for each phase
        print("Phase Relationship Summary:")
        for phase, description in phase_descriptions.items():
            print(f"Phase {phase}: "
                  f"Wrist-Elbow Ratio: {description['mean_wrist_elbow_ratio']}, "
                  f"Wrist-Shoulder Ratio: {description['mean_wrist_shoulder_ratio']}, "
                  f"Wrist Velocity: {description['mean_wrist_z_velocity']}, "
                  f"Elbow Angle: {description['mean_elbow_angle']}")

        return phase_descriptions

    def visualize_temporal_transitions(self, separate_plots=True, smoothing_window=5):
        """
        Visualize phase transitions over time for successful vs. missed shots, with options for separate plots and smoothing.
        """
        self.shot_data['is_successful'] = self.shot_data['result'].apply(lambda x: 1 if x == 'made' else 0)

        if separate_plots:
            fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True, sharey=True)
            for result_type, data in self.shot_data.groupby('is_successful'):
                label = 'Made' if result_type == 1 else 'Missed'
                
                # Smooth the phase transitions over a rolling window
                data['smoothed_phase'] = data['phase'].rolling(window=smoothing_window, min_periods=1).mean()
                
                ax = axes[result_type]
                ax.plot(data['time'], data['smoothed_phase'], label=label, alpha=0.7)
                ax.set_title(f'Temporal Transitions of Phases ({label} Shots)')
                ax.set_xlabel('Time')
                ax.set_ylabel('Phase')
                ax.legend()
        else:
            # Overlay plot for comparison (original approach)
            plt.figure(figsize=(12, 6))
            for result_type, data in self.shot_data.groupby('is_successful'):
                label = 'Made' if result_type == 1 else 'Missed'
                
                # Smooth the phase transitions over a rolling window
                data['smoothed_phase'] = data['phase'].rolling(window=smoothing_window, min_periods=1).mean()
                
                plt.plot(data['time'], data['smoothed_phase'], label=label, alpha=0.7)
            plt.xlabel('Time')
            plt.ylabel('Phase')
            plt.title('Temporal Transitions of Phases (Made vs. Missed Shots)')
            plt.legend()
        
        plt.show()

# Usage
clustering = AdvancedShotClustering('C:/Users/16476/Downloads/SPL-Open-Data-main/basketball/freethrow/data/P0001')
clustering.preprocess_data()
clustering.apply_clustering()
phase_relationships = clustering.analyze_phase_relationships()
clustering.visualize_temporal_transitions(separate_plots=True, smoothing_window=10)
