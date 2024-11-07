import os
import json
import pandas as pd
import numpy as np
from scipy import stats, signal
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportion_confint
from statsmodels.tsa.seasonal import seasonal_decompose

class FreeThrowDataLoader:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        
    def load_participant_data(self, participant_id: str) -> List[Dict]:
        participant_path = self.base_path / participant_id
        trials = []
        
        for trial_file in sorted(participant_path.glob('BB_FT_*.json')):
            with open(trial_file, 'r') as f:
                trial_data = json.load(f)
                trials.append(trial_data)
                
        return trials

class AdvancedFreeThrowAnalyzer:
    def __init__(self, trials_data: List[Dict]):
        self.trials_data = trials_data
        self.participant_id = trials_data[0]['participant_id']
        self.base_analysis = self._compute_base_metrics()
        
    def _compute_base_metrics(self) -> pd.DataFrame:
        trials_metrics = []
        
        for trial_idx, trial in enumerate(self.trials_data, 1):
            metrics = self._analyze_single_trial(trial)
            metrics['trial_number'] = trial_idx
            metrics['trial_id'] = trial['trial_id']
            metrics['result'] = trial['result']
            trials_metrics.append(metrics)
            
        return pd.DataFrame(trials_metrics)
    
    def _analyze_single_trial(self, trial: Dict) -> Dict[str, Any]:
        frames_data = []
        
        for frame in trial['tracking']:
            player = frame['data']['player']
            
            joints = {
                'r_shoulder': np.array(player['R_SHOULDER']),
                'l_shoulder': np.array(player['L_SHOULDER']),
                'r_elbow': np.array(player['R_ELBOW']),
                'r_wrist': np.array(player['R_WRIST']),
                'r_hand': np.mean([
                    np.array(player['R_1STFINGER']),
                    np.array(player['R_5THFINGER'])
                ], axis=0)
            }
            
            elbow_angle = self._calculate_joint_angle(
                joints['r_shoulder'],
                joints['r_elbow'],
                joints['r_wrist']
            )
            
            wrist_angle = self._calculate_joint_angle(
                joints['r_elbow'],
                joints['r_wrist'],
                joints['r_hand']
            )
            
            if frames_data:
                prev_frame = frames_data[-1]
                dt = frame['time'] - prev_frame['time']
                
                hand_velocity = np.linalg.norm(
                    (joints['r_hand'] - prev_frame['hand_pos']) / dt
                )
                wrist_velocity = np.linalg.norm(
                    (joints['r_wrist'] - prev_frame['wrist_pos']) / dt
                )
            else:
                hand_velocity = 0.0
                wrist_velocity = 0.0
            
            frames_data.append({
                'time': frame['time'],
                'elbow_angle': elbow_angle,
                'wrist_angle': wrist_angle,
                'hand_pos': joints['r_hand'],
                'wrist_pos': joints['r_wrist'],
                'hand_velocity': hand_velocity,
                'wrist_velocity': wrist_velocity
            })
        
        frames_df = pd.DataFrame(frames_data)
        phases = self._identify_shot_phases(frames_df)
        
        metrics = {
            'prep_duration': float(phases['preparation_duration']),
            'shot_duration': float(phases['shot_duration']),
            'release_angle': float(phases['release_angle']),
            'max_hand_speed': float(frames_df['hand_velocity'].max()),
            'avg_elbow_angle': float(frames_df['elbow_angle'].mean()),
            'std_elbow_angle': float(frames_df['elbow_angle'].std()),
            'release_height': float(phases['release_height']),
            'motion_smoothness': float(self._calculate_smoothness(frames_df['hand_velocity'].values))
        }
        
        return metrics
    
    def _calculate_joint_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
    
    def _identify_shot_phases(self, frames_df: pd.DataFrame) -> Dict[str, Any]:
        speeds = frames_df['hand_velocity'].values
        
        prep_threshold = 0.1 * np.max(speeds)
        prep_end = np.where(speeds > prep_threshold)[0][0]
        
        release_idx = np.argmax(speeds)
        
        return {
            'preparation_duration': frames_df.iloc[prep_end]['time'] - frames_df.iloc[0]['time'],
            'shot_duration': frames_df.iloc[release_idx]['time'] - frames_df.iloc[prep_end]['time'],
            'release_angle': frames_df.iloc[release_idx]['wrist_angle'],
            'release_height': frames_df.iloc[release_idx]['hand_pos'][2]
        }
    
    def _calculate_smoothness(self, velocities: np.ndarray) -> float:
        jerks = np.diff(velocities)
        return float(-np.log(np.mean(np.square(jerks))))