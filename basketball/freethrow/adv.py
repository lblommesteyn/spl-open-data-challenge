import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    balanced_accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
import json
import glob
from datetime import datetime

class EnhancedFreeThrowAnalyzer:
    def __init__(self, base_path):
        """
        Initialize the analyzer with path to save results and define markers/features
        
        Args:
            base_path (str): Base directory for saving results
        """
        self.base_path = base_path
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(base_path, f"analysis_{self.timestamp}")
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Define key body markers
        self.markers = {
            'shooting_hand': ['R_WRIST', 'R_1STFINGER', 'R_5THFINGER'],
            'shooting_arm': ['R_SHOULDER', 'R_ELBOW', 'R_WRIST'],
            'torso': ['R_SHOULDER', 'L_SHOULDER', 'R_HIP', 'L_HIP'],
            'legs': ['R_HIP', 'R_KNEE', 'R_ANKLE']
        }
        
        # Define feature sets
        self.feature_sets = {
            'release_features': [
                'release_height',
                'release_velocity',
                'release_angle'
            ],
            'deviation_features': [
                'max_lateral_deviation',
                'mean_lateral_deviation',
                'vertical_deviation_consistency',
                'path_deviation_score'
            ],
            'rhythm_features': [
                'propulsion_rhythm',
                'tempo_consistency',
                'phase_timing',
                'acceleration_profile'
            ],
            'coordination_features': [
                'joint_synchronization',
                'kinetic_chain_score',
                'joint_angle_consistency',
                'multi_joint_coordination'
            ],
            'smoothness_features': [
                'normalized_jerk',
                'velocity_smoothness',
                'acceleration_smoothness',
                'overall_motion_quality'
            ]
        }

    def calculate_body_center(self, frames):
        """Calculate body center line using hips and shoulders"""
        try:
            r_hip = np.array([pos for pos in frames['R_HIP']])
            l_hip = np.array([pos for pos in frames['L_HIP']])
            r_shoulder = np.array([pos for pos in frames['R_SHOULDER']])
            l_shoulder = np.array([pos for pos in frames['L_SHOULDER']])
            
            hip_center = (r_hip + l_hip) / 2
            shoulder_center = (r_shoulder + l_shoulder) / 2
            
            return hip_center, shoulder_center
        except Exception as e:
            print(f"Error calculating body center: {str(e)}")
            return None, None

    def calculate_release_features(self, trial):
        """Calculate ball release parameters"""
        try:
            frames = pd.DataFrame([frame['data']['player'] for frame in trial['tracking']])
            
            # Get hand markers
            wrist = np.array([pos for pos in frames['R_WRIST']])
            finger1 = np.array([pos for pos in frames['R_1STFINGER']])
            finger5 = np.array([pos for pos in frames['R_5THFINGER']])
            
            # Calculate hand center
            hand_center = (finger1 + finger5) / 2
            
            # Calculate velocities
            hand_velocity = np.diff(hand_center, axis=0)
            hand_speed = np.linalg.norm(hand_velocity, axis=1)
            
            # Find release frame (peak velocity)
            release_frame = np.argmax(hand_speed)
            
            return {
                'release_height': hand_center[release_frame, 1],
                'release_velocity': hand_speed[release_frame],
                'release_angle': np.degrees(np.arctan2(
                    hand_velocity[release_frame, 1],
                    hand_velocity[release_frame, 0]
                ))
            }
        except Exception as e:
            print(f"Error calculating release features: {str(e)}")
            return dict.fromkeys(['release_height', 'release_velocity', 'release_angle'], np.nan)

    def calculate_deviation_features(self, trial):
        """Calculate deviation metrics from ideal shooting path"""
        try:
            frames = pd.DataFrame([frame['data']['player'] for frame in trial['tracking']])
            
            # Calculate body center
            hip_center, shoulder_center = self.calculate_body_center(frames)
            if hip_center is None:
                raise ValueError("Could not calculate body center")
            
            # Get shooting arm positions
            wrist_pos = np.array([pos for pos in frames['R_WRIST']])
            
            # Calculate midline (vertical line from hip center)
            midline_x = hip_center[:, 0]
            
            # Calculate lateral deviation from midline
            lateral_deviation = np.abs(wrist_pos[:, 0] - midline_x)
            
            # Calculate vertical path deviation
            start_height = wrist_pos[0, 1]
            release_height = wrist_pos[-1, 1]
            ideal_path = np.linspace(start_height, release_height, len(wrist_pos))
            vertical_deviation = np.abs(wrist_pos[:, 1] - ideal_path)
            
            # Calculate path efficiency
            path_length = np.sum(np.sqrt(np.sum(np.diff(wrist_pos, axis=0)**2, axis=1)))
            straight_line_dist = np.sqrt(np.sum((wrist_pos[-1] - wrist_pos[0])**2))
            
            return {
                'max_lateral_deviation': np.max(lateral_deviation),
                'mean_lateral_deviation': np.mean(lateral_deviation),
                'vertical_deviation_consistency': np.std(vertical_deviation),
                'path_deviation_score': path_length / straight_line_dist if straight_line_dist > 0 else np.nan
            }
        except Exception as e:
            print(f"Error calculating deviation features: {str(e)}")
            return dict.fromkeys([
                'max_lateral_deviation', 'mean_lateral_deviation',
                'vertical_deviation_consistency', 'path_deviation_score'
            ], np.nan)

    def calculate_rhythm_features(self, trial):
        """Calculate rhythm and timing metrics during shot"""
        try:
            frames = pd.DataFrame([frame['data']['player'] for frame in trial['tracking']])
            wrist_pos = np.array([pos for pos in frames['R_WRIST']])
            
            # Calculate velocities and accelerations
            velocity = np.diff(wrist_pos, axis=0)
            acceleration = np.diff(velocity, axis=0)
            
            # Focus on vertical component for shot rhythm
            vertical_vel = velocity[:, 1]
            vertical_acc = acceleration[:, 1]
            
            # Identify propulsion phase (upward movement)
            propulsion_start = np.where(vertical_vel > 0)[0][0]
            propulsion_phase = vertical_acc[propulsion_start:]
            
            # Calculate rhythm metrics
            phase_changes = np.where(np.diff(np.signbit(vertical_acc)))[0]
            phase_durations = np.diff(phase_changes)
            
            return {
                'propulsion_rhythm': np.mean(propulsion_phase),
                'tempo_consistency': np.std(phase_durations) if len(phase_durations) > 0 else np.nan,
                'phase_timing': len(propulsion_phase) / len(acceleration),
                'acceleration_profile': np.max(propulsion_phase) / np.mean(np.abs(propulsion_phase))
            }
        except Exception as e:
            print(f"Error calculating rhythm features: {str(e)}")
            return dict.fromkeys([
                'propulsion_rhythm', 'tempo_consistency',
                'phase_timing', 'acceleration_profile'
            ], np.nan)

    def _calculate_joint_angles(self, point1, point2, point3):
        """Calculate angle between three points"""
        try:
            v1 = point1 - point2
            v2 = point3 - point2
            
            dot_product = np.sum(v1 * v2, axis=1)
            norms = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
            
            # Handle numerical errors
            cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
            angles = np.arccos(cos_angle)
            
            return np.degrees(angles)
        except Exception as e:
            print(f"Error calculating joint angles: {str(e)}")
            return np.array([np.nan] * len(point1))
        
    def calculate_coordination_features(self, trial):
        """Calculate coordination metrics between joints"""
        try:
            frames = pd.DataFrame([frame['data']['player'] for frame in trial['tracking']])
            
            # Get joint positions
            shoulder = np.array([pos for pos in frames['R_SHOULDER']])
            elbow = np.array([pos for pos in frames['R_ELBOW']])
            wrist = np.array([pos for pos in frames['R_WRIST']])
            
            # Calculate joint angles
            elbow_angles = self._calculate_joint_angles(shoulder, elbow, wrist)
            
            # Calculate joint velocities
            shoulder_vel = np.diff(shoulder, axis=0)
            elbow_vel = np.diff(elbow, axis=0)
            wrist_vel = np.diff(wrist, axis=0)
            
            # Calculate velocity magnitudes
            shoulder_speed = np.linalg.norm(shoulder_vel, axis=1)
            elbow_speed = np.linalg.norm(elbow_vel, axis=1)
            wrist_speed = np.linalg.norm(wrist_vel, axis=1)
            
            # Find peak timing
            shoulder_peak = np.argmax(shoulder_speed)
            elbow_peak = np.argmax(elbow_speed)
            wrist_peak = np.argmax(wrist_speed)
            
            # Calculate coordination scores
            joint_sync = np.corrcoef(elbow_speed, wrist_speed)[0, 1]
            proper_sequence = (shoulder_peak < elbow_peak < wrist_peak)
            timing_gaps = [elbow_peak - shoulder_peak, wrist_peak - elbow_peak]
            
            return {
                'joint_synchronization': joint_sync,
                'kinetic_chain_score': 1 / (1 + np.std(timing_gaps)) if proper_sequence else 0,
                'joint_angle_consistency': np.std(elbow_angles),
                'multi_joint_coordination': np.mean([
                    np.corrcoef(shoulder_speed, elbow_speed)[0, 1],
                    np.corrcoef(elbow_speed, wrist_speed)[0, 1]
                ])
            }
        except Exception as e:
            print(f"Error calculating coordination features: {str(e)}")
            return dict.fromkeys([
                'joint_synchronization', 'kinetic_chain_score',
                'joint_angle_consistency', 'multi_joint_coordination'
            ], np.nan)

    def calculate_smoothness_features(self, trial):
        """Calculate smoothness metrics for the shooting motion"""
        try:
            frames = pd.DataFrame([frame['data']['player'] for frame in trial['tracking']])
            wrist_pos = np.array([pos for pos in frames['R_WRIST']])
            
            # Calculate derivatives
            velocity = np.diff(wrist_pos, axis=0)
            acceleration = np.diff(velocity, axis=0)
            jerk = np.diff(acceleration, axis=0)
            
            # Calculate speed profiles
            speed = np.linalg.norm(velocity, axis=1)
            acc_magnitude = np.linalg.norm(acceleration, axis=1)
            jerk_magnitude = np.linalg.norm(jerk, axis=1)
            
            # Calculate smoothness metrics
            movement_duration = len(frames) / 30  # Assuming 30fps
            movement_distance = np.sum(speed)
            
            normalized_jerk = np.sqrt(np.mean(jerk_magnitude**2) * 
                                    (movement_duration**5 / movement_distance**2))
            
            return {
                'normalized_jerk': -normalized_jerk,  # Negative so higher is better
                'velocity_smoothness': 1 / (1 + np.std(speed)),
                'acceleration_smoothness': 1 / (1 + np.std(acc_magnitude)),
                'overall_motion_quality': 1 / (1 + np.mean(jerk_magnitude))
            }
        except Exception as e:
            print(f"Error calculating smoothness features: {str(e)}")
            return dict.fromkeys([
                'normalized_jerk', 'velocity_smoothness',
                'acceleration_smoothness', 'overall_motion_quality'
            ], np.nan)

    def extract_all_features(self, trial):
        """Extract all features for a single trial"""
        try:
            features = {}
            
            # Extract each feature set
            features.update(self.calculate_release_features(trial))
            features.update(self.calculate_deviation_features(trial))
            features.update(self.calculate_rhythm_features(trial))
            features.update(self.calculate_coordination_features(trial))
            features.update(self.calculate_smoothness_features(trial))
            
            # Add trial metadata
            features['trial_id'] = trial['trial_id']
            features['shot_result'] = 1 if trial['result'] == 'made' else 0
            
            return features
            
        except Exception as e:
            print(f"Error extracting features for trial {trial.get('trial_id', 'unknown')}: {str(e)}")
            # Return dictionary with all features as NaN
            features = {name: np.nan for feature_set in self.feature_sets.values() 
                      for name in feature_set}
            features['trial_id'] = trial.get('trial_id', 'unknown')
            features['shot_result'] = np.nan
            return features

    def prepare_feature_combinations(self, trials_data):
        """Prepare different combinations of feature sets for analysis"""
        print("\nExtracting features from all trials...")
        
        # Extract features for all trials
        all_features = []
        total_trials = len(trials_data)
        
        for i, trial in enumerate(trials_data, 1):
            features = self.extract_all_features(trial)
            all_features.append(features)
            if i % 10 == 0:  # Progress update every 10 trials
                print(f"Processed {i}/{total_trials} trials")
        
        # Convert to DataFrame
        features_df = pd.DataFrame(all_features)
        print(f"\nExtracted {len(features_df.columns) - 2} features")  # -2 for trial_id and shot_result
        
        # Create feature set combinations
        feature_combos = self._generate_feature_combinations()
        
        return features_df, feature_combos

    def _generate_feature_combinations(self):
        """Generate different combinations of feature sets"""
        feature_sets = list(self.feature_sets.keys())
        combinations = []
        
        # Single feature sets
        for fs in feature_sets:
            combinations.append([fs])
        
        # Pairs of feature sets
        for i in range(len(feature_sets)):
            for j in range(i+1, len(feature_sets)):
                combinations.append([feature_sets[i], feature_sets[j]])
        
        # Triplets
        for i in range(len(feature_sets)):
            for j in range(i+1, len(feature_sets)):
                for k in range(j+1, len(feature_sets)):
                    combinations.append([feature_sets[i], feature_sets[j], feature_sets[k]])
        
        # All features
        combinations.append(feature_sets)
        
        print(f"Generated {len(combinations)} feature combinations")
        return combinations
    
    def train_and_evaluate_models(self, features_df, feature_combos):
        """Train and evaluate models for each feature combination"""
        results = {}
        
        for combo in feature_combos:
            combo_name = '_'.join(combo)
            print(f"\nEvaluating feature combination: {combo_name}")
            
            try:
                # Create output directory for this combination
                combo_dir = os.path.join(self.results_dir, combo_name)
                os.makedirs(combo_dir, exist_ok=True)
                
                # Get features for this combination
                selected_features = []
                for feature_set in combo:
                    selected_features.extend(self.feature_sets[feature_set])
                
                # Filter for available features
                available_features = [f for f in selected_features if f in features_df.columns]
                if not available_features:
                    print(f"No features available for combination {combo_name}")
                    continue
                
                # Prepare data
                X = features_df[available_features]
                y = features_df['shot_result']
                
                # Remove rows with all NaN
                valid_rows = ~X.isna().all(axis=1)
                X = X[valid_rows]
                y = y[valid_rows]
                
                if len(X) < 10:  # Minimum sample size check
                    print(f"Insufficient samples for {combo_name}")
                    continue
                
                # Train and evaluate model
                model_results = self._train_models(X, y)
                
                # Save results
                results[combo_name] = model_results
                
                # Generate visualizations
                self._generate_visualizations(X, y, model_results, combo_dir, combo_name)
                
                # Save feature importance
                if 'feature_importance' in model_results:
                    self._plot_feature_importance(model_results['feature_importance'], combo_dir)
                
                # Save performance metrics
                self._save_performance_metrics(model_results, combo_dir)
                
            except Exception as e:
                print(f"Error processing combination {combo_name}: {str(e)}")
                continue
        
        # Generate comparative analysis
        if results:
            self._generate_comparative_analysis(results)
        
        return results

    def _train_models(self, X, y):
        """Train and evaluate machine learning models"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create pipeline with preprocessing and model
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),  # Use median for robustness
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                class_weight='balanced',  # Handle class imbalance
                random_state=42
            ))
        ])
        
        # Train model
        pipeline.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(
            pipeline, X_train, y_train, 
            cv=min(5, len(X_train)), 
            scoring='balanced_accuracy'
        )
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate probabilities for ROC curve
        y_prob = pipeline.predict_proba(X_test)
        
        # Get feature importance
        feature_importance = pd.Series(
            pipeline.named_steps['classifier'].feature_importances_,
            index=X.columns
        ).sort_values(ascending=False)
        
        return {
            'pipeline': pipeline,
            'cv_scores': cv_scores,
            'classification_report': classification_report(y_test, y_pred),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'feature_importance': feature_importance,
            'balanced_accuracy': balanced_accuracy_score(y_test, y_pred),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_prob': y_prob,
            'feature_names': list(X.columns)
        }

    def _save_performance_metrics(self, model_results, output_dir):
        """Save detailed performance metrics to file"""
        with open(os.path.join(output_dir, 'performance_metrics.txt'), 'w') as f:
            f.write("Model Performance Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Classification Report
            f.write("Classification Report:\n")
            f.write("-" * 20 + "\n")
            f.write(model_results['classification_report'])
            f.write("\n\n")
            
            # Cross-validation Scores
            f.write("Cross-validation Scores:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Mean: {model_results['cv_scores'].mean():.3f}\n")
            f.write(f"Std: {model_results['cv_scores'].std():.3f}\n")
            f.write(f"Scores: {', '.join([f'{score:.3f}' for score in model_results['cv_scores']])}\n\n")
            
            # Overall Performance
            f.write("Overall Performance:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Balanced Accuracy: {model_results['balanced_accuracy']:.3f}\n")
            
            # Feature Importance
            f.write("\nTop 10 Most Important Features:\n")
            f.write("-" * 20 + "\n")
            top_features = model_results['feature_importance'].head(10)
            for feature, importance in top_features.items():
                f.write(f"{feature}: {importance:.3f}\n")

    def _calculate_combined_score(self, results):
        """Calculate a combined score for each model (accuracy and stability)"""
        return {
            name: {
                'combined_score': result['balanced_accuracy'] * 0.7 + 
                                (1 - result['cv_scores'].std()) * 0.3,
                'accuracy': result['balanced_accuracy'],
                'cv_std': result['cv_scores'].std()
            }
            for name, result in results.items()
        }
    
    def _generate_visualizations(self, X, y, model_results, output_dir, combo_name):
        """Generate comprehensive visualizations for analysis results"""
        plt.style.use('default')
        
        # Create feature distributions plot
        self._plot_feature_distributions(X, y, output_dir)
        
        # Create correlation matrix
        self._plot_correlation_matrix(X, output_dir)
        
        # Create PCA analysis plots
        self._plot_pca_analysis(X, y, output_dir)
        
        # Create model performance plots
        self._plot_model_performance(model_results, output_dir)
            
        plt.close('all')

    def _plot_feature_distributions(self, X, y, output_dir):
        """Plot distributions of each feature split by shot result"""
        n_features = X.shape[1]
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig = plt.figure(figsize=(15, 5*n_rows))
        
        for i, feature in enumerate(X.columns, 1):
            plt.subplot(n_rows, n_cols, i)
            
            # Plot distributions for made and missed shots
            for result, label in [(0, 'Missed'), (1, 'Made')]:
                sns.kdeplot(
                    data=X[feature][y == result],
                    label=label,
                    fill=True,
                    alpha=0.5
                )
            
            plt.title(f'{feature} Distribution')
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_correlation_matrix(self, X, output_dir):
        """Plot feature correlation matrix with hierarchical clustering"""
        plt.figure(figsize=(12, 10))
        
        # Calculate correlations
        correlations = X.corr()
        
        # Create clustered heatmap
        sns.clustermap(
            correlations,
            annot=True,
            cmap='coolwarm',
            center=0,
            fmt='.2f',
            figsize=(15, 15),
            dendrogram_ratio=0.1,
            cbar_pos=(0.02, 0.8, 0.03, 0.2)
        )
        
        plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_pca_analysis(self, X, y, output_dir):
        """Perform and plot PCA analysis"""
        # Prepare data
        X_imputed = SimpleImputer(strategy='median').fit_transform(X)
        X_scaled = StandardScaler().fit_transform(X_imputed)
        
        # Perform PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        plt.plot(range(1, len(explained_variance) + 1), cumulative_variance, 'bo-')
        plt.axhline(y=0.9, color='r', linestyle='--', label='90% Explained Variance')
        plt.title('Cumulative Explained Variance Ratio')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.legend()
        
        plt.savefig(os.path.join(output_dir, 'pca_explained_variance.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot first two components
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            X_pca[:, 0], X_pca[:, 1],
            c=y,
            cmap='coolwarm',
            alpha=0.6,
            s=100
        )
        
        plt.title('PCA: First Two Components')
        plt.xlabel(f'PC1 ({explained_variance[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2%} variance)')
        
        # Add feature vectors
        feature_vectors = pca.components_[:2].T
        for i, feature in enumerate(X.columns):
            plt.arrow(
                0, 0,
                feature_vectors[i, 0] * 3,
                feature_vectors[i, 1] * 3,
                color='k',
                alpha=0.5,
                head_width=0.05
            )
            plt.text(
                feature_vectors[i, 0] * 3.2,
                feature_vectors[i, 1] * 3.2,
                feature,
                color='k',
                ha='center',
                va='center'
            )
        
        plt.colorbar(scatter, label='Shot Result')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'pca_components.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_model_performance(self, model_results, output_dir):
        """Plot detailed model performance metrics"""
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            model_results['confusion_matrix'],
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Missed', 'Made'],
            yticklabels=['Missed', 'Made']
        )
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Importance
        plt.figure(figsize=(12, 8))
        importances = model_results['feature_importance']
        importances.sort_values().plot(kind='barh')
        plt.title('Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def _generate_comparative_analysis(self, all_results):
        """Generate comparative analysis of all feature combinations"""
        scores = self._calculate_combined_score(all_results)
        comparison_df = pd.DataFrame(scores).T
        
        # Plot comparative performance
        plt.figure(figsize=(15, 8))
        comparison_df['combined_score'].sort_values().plot(kind='barh')
        plt.title('Model Performance Comparison')
        plt.xlabel('Combined Score (Accuracy & Stability)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate detailed report
        report = self._generate_analysis_report(comparison_df, all_results)
        
        # Save report
        with open(os.path.join(self.results_dir, 'analysis_report.txt'), 'w') as f:
            f.write(report)
        
        # Save detailed results
        comparison_df.to_csv(os.path.join(self.results_dir, 'model_comparison.csv'))
from advancedml import AdvancedMLAnalyzer
from anlyze2 import AdvancedFreeThrowAnalyzer, FreeThrowDataLoader

def main():
    # Load your data and initialize analyzers
    base_path = r"C:\Users\16476\Downloads\SPL-Open-Data-main\basketball\freethrow\data"
    loader = FreeThrowDataLoader(base_path)
    participant_data = loader.load_participant_data('P0001')
    
    # Create analyzers
    ft_analyzer = AdvancedFreeThrowAnalyzer(participant_data)
    ml_analyzer = AdvancedMLAnalyzer(ft_analyzer)
    
    # Print available columns for debugging
    print("\nColumns in base_df:")
    print(ml_analyzer.base_df.columns.tolist())
    print("\nColumns in sequence_features:")
    print(ml_analyzer.sequence_features.columns.tolist())
    print("\nColumns in combined_df:")
    print(ml_analyzer.combined_df.columns.tolist())
    
    # Run analyses
    try:
        print("\nTraining ensemble models...")
        ensemble_results = ml_analyzer.train_ensemble_models()
        
        print("\nAnalyzing shot patterns...")
        pattern_analysis = ml_analyzer.analyze_shot_patterns()
        
        print("\nGenerating visualizations...")
        ml_analyzer.plot_analysis_results(ensemble_results, pattern_analysis)
        
        # Print results
        print("\n=== MODEL PERFORMANCE ===")
        for name, result in ensemble_results['results'].items():
            print(f"\n{name.upper()} RESULTS:")
            print(result['classification_report'])
        
        print("\n=== FEATURE IMPORTANCE ===")
        for name, importance in ensemble_results['feature_importance'].items():
            print(f"\n{name.upper()} TOP 5 FEATURES:")
            print(importance.sort_values(ascending=False))
            
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()