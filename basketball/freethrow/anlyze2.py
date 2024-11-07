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
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import euclidean

class AdvancedFreeThrowAnalyzer:
    def __init__(self, trials_data: List[Dict]):
        """
        Initialize analyzer with trials data
        
        Args:
            trials_data: List of trial dictionaries
        """
        self.trials_data = trials_data
        self.participant_id = trials_data[0]['participant_id']
        self.base_analysis = self._compute_base_metrics()
        
    def _compute_base_metrics(self) -> pd.DataFrame:
        """Compute basic metrics for each trial"""
        trials_metrics = []
        
        for trial_idx, trial in enumerate(self.trials_data, 1):
            metrics = self._analyze_single_trial(trial)
            metrics['trial_number'] = trial_idx
            metrics['trial_id'] = trial['trial_id']
            metrics['result'] = trial['result']
            trials_metrics.append(metrics)
            
        return pd.DataFrame(trials_metrics)
    
    def _analyze_single_trial(self, trial: Dict) -> Dict[str, Any]:
        """Analyze a single trial in detail"""
        frames_data = []
        
        for frame in trial['tracking']:
            player = frame['data']['player']
            
            # Get joint positions
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
            
            # Calculate joint angles
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
            
            # Calculate velocities
            if frames_data:
                prev_frame = frames_data[-1]
                dt = frame['time'] - prev_frame['time']
                
                hand_velocity_vec = (joints['r_hand'] - prev_frame['hand_pos']) / dt
                wrist_velocity_vec = (joints['r_wrist'] - prev_frame['wrist_pos']) / dt
                
                hand_velocity = np.linalg.norm(hand_velocity_vec)
                wrist_velocity = np.linalg.norm(wrist_velocity_vec)
            else:
                hand_velocity = 0.0
                wrist_velocity = 0.0
            
            frames_data.append({
                'time': frame['time'],
                'elbow_angle': elbow_angle,
                'wrist_angle': wrist_angle,
                'hand_pos': joints['r_hand'],
                'wrist_pos': joints['r_wrist'],
                'hand_velocity': hand_velocity,  # Now storing scalar value
                'wrist_velocity': wrist_velocity  # Now storing scalar value
            })
        
        # Convert to DataFrame for easier analysis
        frames_df = pd.DataFrame(frames_data)
        
        # Identify shot phases
        phases = self._identify_shot_phases(frames_df)
        
        # Calculate summary metrics
        metrics = {
            'prep_duration': float(phases['preparation_duration']),  # Ensure float
            'shot_duration': float(phases['shot_duration']),
            'release_angle': float(phases['release_angle']),
            'max_hand_speed': float(np.max(frames_df['hand_velocity'])),
            'avg_elbow_angle': float(frames_df['elbow_angle'].mean()),
            'std_elbow_angle': float(frames_df['elbow_angle'].std()),
            'release_height': float(phases['release_height']),
            'motion_smoothness': self._calculate_smoothness(frames_df['hand_velocity'].values)
        }
        
        return metrics
    
    def _calculate_smoothness(self, velocities: np.ndarray) -> float:
        """Calculate movement smoothness using normalized jerk"""
        jerks = np.diff(velocities)
        return float(-np.log(np.mean(np.square(jerks))))
    
    def _identify_shot_phases(self, frames_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify key phases of the shot"""
        # Calculate speed profile
        speeds = frames_df['hand_velocity'].values
        
        # Find preparation phase (before main movement)
        prep_threshold = 0.1 * np.max(speeds)
        prep_end = np.where(speeds > prep_threshold)[0][0]
        
        # Find release point (peak speed)
        release_idx = np.argmax(speeds)
        
        return {
            'preparation_duration': frames_df.iloc[prep_end]['time'] - frames_df.iloc[0]['time'],
            'shot_duration': frames_df.iloc[release_idx]['time'] - frames_df.iloc[prep_end]['time'],
            'release_angle': frames_df.iloc[release_idx]['wrist_angle'],
            'release_height': frames_df.iloc[release_idx]['hand_pos'][2]
        }
    
    def _calculate_joint_angle(self, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
        """Calculate angle between three points in degrees"""
        v1 = p1 - p2
        v2 = p3 - p2
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    
    def _identify_shot_phases(self, frames_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify key phases of the shot"""
        # Calculate speed profile
        speeds = np.array([np.linalg.norm(v) for v in frames_df['hand_velocity']])
        
        # Find preparation phase (before main movement)
        prep_threshold = 0.1 * np.max(speeds)
        prep_end = np.where(speeds > prep_threshold)[0][0]
        
        # Find release point (peak speed)
        release_idx = np.argmax(speeds)
        
        return {
            'preparation_duration': frames_df.iloc[prep_end]['time'] - frames_df.iloc[0]['time'],
            'shot_duration': frames_df.iloc[release_idx]['time'] - frames_df.iloc[prep_end]['time'],
            'release_angle': frames_df.iloc[release_idx]['wrist_angle'],
            'release_height': frames_df.iloc[release_idx]['hand_pos'][2]
        }
    
    def _calculate_smoothness(self, velocities: np.ndarray) -> float:
        """Calculate movement smoothness using normalized jerk"""
        jerks = np.diff(velocities, axis=0)
        return -np.log(np.mean(np.square(jerks)))
    
    def analyze_shot_consistency(self) -> Dict[str, Any]:
        """Analyze consistency of shooting motion across trials"""
        # Cluster shots based on key metrics
        features = ['prep_duration', 'shot_duration', 'release_angle', 
                   'max_hand_speed', 'avg_elbow_angle', 'release_height']
        
        X = StandardScaler().fit_transform(self.base_analysis[features])
        
        # Determine optimal number of clusters
        max_clusters = min(5, len(self.base_analysis) // 10)
        inertias = []
        
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        
        # Use elbow method to find optimal clusters
        optimal_clusters = 2  # Default
        for i in range(1, len(inertias)-1):
            if (inertias[i-1] - inertias[i]) / (inertias[i] - inertias[i+1]) < 1.5:
                optimal_clusters = i + 1
                break
        
        # Cluster the shots
        kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
        self.base_analysis['shot_cluster'] = kmeans.fit_predict(X)
        
        # Analyze success rate by cluster
        cluster_success = self.base_analysis.groupby('shot_cluster').agg({
            'result': lambda x: (x == 'made').mean(),
            'trial_number': 'count'
        }).rename(columns={'result': 'success_rate', 'trial_number': 'count'})
        
        return {
            'cluster_analysis': cluster_success.to_dict(),
            'optimal_cluster': cluster_success.success_rate.idxmax(),
            'cluster_characteristics': self._get_cluster_characteristics(optimal_clusters)
        }
    
    def _get_cluster_characteristics(self, n_clusters: int) -> Dict[str, Any]:
        """Get characteristics of each shot cluster"""
        features = ['prep_duration', 'shot_duration', 'release_angle', 
                   'max_hand_speed', 'avg_elbow_angle', 'release_height']
        
        characteristics = {}
        
        for cluster in range(n_clusters):
            cluster_data = self.base_analysis[
                self.base_analysis['shot_cluster'] == cluster
            ]
            
            characteristics[f'cluster_{cluster}'] = {
                'mean_values': cluster_data[features].mean().to_dict(),
                'std_values': cluster_data[features].std().to_dict(),
                'success_rate': (cluster_data['result'] == 'made').mean()
            }
            
        return characteristics
    
    def analyze_fatigue_effects(self) -> Dict[str, Any]:
        """Analyze detailed fatigue effects"""
        window_size = min(10, len(self.base_analysis) // 5)
        
        # Calculate rolling statistics
        rolling_stats = {}
        metrics = ['release_angle', 'max_hand_speed', 'motion_smoothness']
        
        for metric in metrics:
            series = self.base_analysis[metric]
            rolling_mean = series.rolling(window=window_size).mean()
            rolling_std = series.rolling(window=window_size).std()
            
            rolling_stats[metric] = {
                'trend': rolling_mean.tolist(),
                'variability': rolling_std.tolist(),
                'degradation_rate': np.polyfit(
                    range(len(series)),
                    series.fillna(method='ffill'),
                    1
                )[0]
            }
        
        # Analyze success rate changes
        success_series = (self.base_analysis['result'] == 'made').astype(int)
        success_trend = success_series.rolling(window=window_size).mean()
        
        return {
            'metric_trends': rolling_stats,
            'success_trend': success_trend.tolist(),
            'fatigue_correlation': {
                metric: stats.pearsonr(
                    range(len(self.base_analysis)),
                    self.base_analysis[metric].fillna(method='ffill')
                )[0]
                for metric in metrics
            }
        }
    
    def analyze_rhythm_patterns(self) -> Dict[str, Any]:
        """Analyze shooting rhythm and timing patterns"""
        # Calculate interval consistency
        prep_durations = self.base_analysis['prep_duration']
        shot_durations = self.base_analysis['shot_duration']
        
        rhythm_metrics = {
            'prep_consistency': 1 - (prep_durations.std() / prep_durations.mean()),
            'shot_consistency': 1 - (shot_durations.std() / shot_durations.mean()),
            'optimal_prep_duration': prep_durations[
                self.base_analysis['result'] == 'made'
            ].mean(),
            'optimal_shot_duration': shot_durations[
                self.base_analysis['result'] == 'made'
            ].mean()
        }
        
        # Analyze timing patterns of successful shots
        successful_patterns = self.base_analysis[
            self.base_analysis['result'] == 'made'
        ][['prep_duration', 'shot_duration']]
        
        return {
            'rhythm_metrics': rhythm_metrics,
            'successful_patterns': successful_patterns.to_dict(orient='records'),
            'timing_correlation': stats.pearsonr(
                prep_durations,
                shot_durations
            )[0]
        }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        consistency = self.analyze_shot_consistency()
        fatigue = self.analyze_fatigue_effects()
        rhythm = self.analyze_rhythm_patterns()
        
        # Calculate overall performance metrics
        success_rate = (self.base_analysis['result'] == 'made').mean()
        ci_success = proportion_confint(
            count=(self.base_analysis['result'] == 'made').sum(),
            nobs=len(self.base_analysis),
            alpha=0.05,
            method='wilson'
        )
        
        return {
            'overall_performance': {
                'success_rate': success_rate,
                'confidence_interval': ci_success,
                'total_trials': len(self.base_analysis),
                'best_cluster': consistency['optimal_cluster']
            },
            'consistency_analysis': consistency,
            'fatigue_analysis': fatigue,
            'rhythm_analysis': rhythm,
            'recommendations': self._generate_recommendations(
                consistency, fatigue, rhythm
            )
        }
    
    def _generate_recommendations(
        self,
        consistency: Dict[str, Any],
        fatigue: Dict[str, Any],
        rhythm: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Consistency recommendations
        best_cluster = consistency['optimal_cluster']
        best_characteristics = consistency['cluster_characteristics'][f'cluster_{best_cluster}']
        
        recommendations.append(
            f"Optimal release angle: {best_characteristics['mean_values']['release_angle']:.1f}°"
        )
        
        # Fatigue recommendations
        for metric, trend in fatigue['fatigue_correlation'].items():
            if abs(trend) > 0.2:
                recommendations.append(
                    f"Consider rest periods to maintain {metric.replace('_', ' ')} consistency"
                )
        
        # Rhythm recommendations
        if rhythm['rhythm_metrics']['prep_consistency'] < 0.8:
            recommendations.append(
                "Work on maintaining consistent preparation timing"
            )
        
        return recommendations
class FreeThrowDataLoader:
    def __init__(self, base_path: str):
        """
        Initialize the data loader with the base path to the data directory
        
        Args:
            base_path: Path to the main data directory
        """
        self.base_path = Path(base_path)
        
    def load_participant_data(self, participant_id: str) -> List[Dict]:
        """
        Load all trials for a specific participant
        
        Args:
            participant_id: Participant ID (e.g., 'P0001')
            
        Returns:
            List of dictionaries containing trial data
        """
        participant_path = self.base_path / participant_id
        trials = []
        
        # Load each trial file
        for trial_file in sorted(participant_path.glob('BB_FT_*.json')):
            with open(trial_file, 'r') as f:
                trial_data = json.load(f)
                trials.append(trial_data)
                
        return trials

def main():
    base_path = r"C:\Users\16476\Downloads\SPL-Open-Data-main\basketball\freethrow\data"
    loader = FreeThrowDataLoader(base_path)
    participant_data = loader.load_participant_data('P0001')
    
    analyzer = AdvancedFreeThrowAnalyzer(participant_data)
    report = analyzer.generate_comprehensive_report()
    
    # 1. Overall Performance
    print("\n=== OVERALL PERFORMANCE ===")
    print(f"Success Rate: {report['overall_performance']['success_rate']:.1%}")
    print(f"Total Trials: {report['overall_performance']['total_trials']}")
    ci = report['overall_performance']['confidence_interval']
    print(f"Confidence Interval: ({ci[0]:.1%}, {ci[1]:.1%})")
    
    # 2. Shot Clusters Analysis
    print("\n=== SHOT CLUSTERS ANALYSIS ===")
    consistency = report['consistency_analysis']
    best_cluster = consistency['optimal_cluster']
    print(f"Best Performing Cluster: {best_cluster}")
    
    for cluster, chars in consistency['cluster_characteristics'].items():
        print(f"\nCluster {cluster}:")
        print(f"Success Rate: {chars['success_rate']:.1%}")
        print("\nMean Values:")
        for metric, value in chars['mean_values'].items():
            print(f"- {metric}: {value:.2f}")
    
    # 3. Fatigue Analysis
    print("\n=== FATIGUE ANALYSIS ===")
    fatigue = report['fatigue_analysis']
    print("\nFatigue Correlations:")
    for metric, corr in fatigue['fatigue_correlation'].items():
        print(f"{metric}: {corr:.3f}")
    
    # 4. Rhythm Analysis
    print("\n=== RHYTHM ANALYSIS ===")
    rhythm = report['rhythm_analysis']
    print("\nRhythm Metrics:")
    for metric, value in rhythm['rhythm_metrics'].items():
        print(f"{metric}: {value:.3f}")
    print(f"\nTiming Correlation: {rhythm['timing_correlation']:.3f}")
    
    # 5. Recommendations
    print("\n=== RECOMMENDATIONS ===")
    for rec in report['recommendations']:
        print(f"- {rec}")

    # 6. Additional Statistics
    print("\n=== DETAILED STATISTICS ===")
    base_metrics = analyzer.base_analysis
    
    print("\nShot Metrics (Mean ± Std):")
    for column in ['release_angle', 'max_hand_speed', 'avg_elbow_angle', 'release_height']:
        mean = base_metrics[column].mean()
        std = base_metrics[column].std()
        print(f"{column}: {mean:.2f} ± {std:.2f}")
    
    # 7. Success Rate Trends
    print("\nSuccess Rate by Quarter:")
    n_trials = len(base_metrics)
    quarter_size = n_trials // 4
    for i in range(4):
        start_idx = i * quarter_size
        end_idx = (i + 1) * quarter_size if i < 3 else None
        quarter_success = (base_metrics['result'][start_idx:end_idx] == 'made').mean()
        print(f"Q{i+1}: {quarter_success:.1%}")
    
    # 8. Streak Analysis
    print("\nStreak Analysis:")
    results = (base_metrics['result'] == 'made').astype(int).values
    current_streak = 1
    max_make_streak = 0
    max_miss_streak = 0
    current_type = results[0]
    
    for i in range(1, len(results)):
        if results[i] == results[i-1]:
            current_streak += 1
        else:
            if current_type == 1:
                max_make_streak = max(max_make_streak, current_streak)
            else:
                max_miss_streak = max(max_miss_streak, current_streak)
            current_streak = 1
            current_type = results[i]
    
    print(f"Longest Make Streak: {max_make_streak}")
    print(f"Longest Miss Streak: {max_miss_streak}")

if __name__ == "__main__":
    main()