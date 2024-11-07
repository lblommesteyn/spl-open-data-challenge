import os
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

class ShotClustering:
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

    def preprocess_data(self):
        self.calculate_velocity()
        self.calculate_angles()
        
        # Impute and scale the data
        imputer = SimpleImputer(strategy='mean')
        numeric_data = self.shot_data.select_dtypes(include=[np.number])
        numeric_data_imputed = imputer.fit_transform(numeric_data)
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(numeric_data_imputed)
        return self.scaled_data

    def segment_shot_phases(self, n_clusters=8):
        """
        Cluster shot data into phases using K-Means with added features.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        self.phase_labels = kmeans.fit_predict(self.scaled_data)
        self.shot_data['phase'] = self.phase_labels
        return self.shot_data

    def interpret_phases(self):
        """
        Interpret each phase by calculating the mean of key features.
        """
        phase_descriptions = {}
        phase_groups = self.shot_data.groupby('phase')

        for phase, group in phase_groups:
            mean_wrist_z_vel = group['R_WRIST_z_vel'].mean()
            mean_elbow_angle = group['R_ELBOW_angle'].mean()
            mean_time = group['time'].mean()
            
            # Determine the phase based on observed characteristics
            if mean_wrist_z_vel < 0.05 and mean_elbow_angle < 45:
                phase_name = 'Preparation'
            elif mean_wrist_z_vel > 0.05 and mean_wrist_z_vel < 0.3 and mean_elbow_angle > 45:
                phase_name = 'Loading'
            elif mean_wrist_z_vel > 0.3 and mean_elbow_angle > 80:
                phase_name = 'Release'
            else:
                phase_name = 'Follow-Through'
            
            # Store the interpreted phase information
            phase_descriptions[phase] = {
                'phase_name': phase_name,
                'mean_wrist_z_velocity': mean_wrist_z_vel,
                'mean_elbow_angle': mean_elbow_angle,
                'mean_time': mean_time
            }
        
        # Output phase interpretations
        print("Phase Interpretation Summary:")
        for phase, description in phase_descriptions.items():
            print(f"Phase {phase}: {description['phase_name']} - "
                f"Wrist Velocity: {description['mean_wrist_z_velocity']}, "
                f"Elbow Angle: {description['mean_elbow_angle']}, "
                f"Time: {description['mean_time']}")
        
        return phase_descriptions


    def visualize_phases(self):
        plt.figure(figsize=(10, 6))
        plt.scatter(self.shot_data['time'], self.shot_data['R_WRIST_z'], c=self.phase_labels, cmap='viridis')
        plt.xlabel('Time')
        plt.ylabel('R_Wrist Z Coordinate')
        plt.title('Shot Phases Segmentation')
        plt.colorbar(label='Phase')
        plt.show()

# Usage
clustering = ShotClustering('C:/Users/16476/Downloads/SPL-Open-Data-main/basketball/freethrow/data/P0001')
clustering.preprocess_data()
clustering.segment_shot_phases()
phase_descriptions = clustering.interpret_phases()
clustering.visualize_phases()
