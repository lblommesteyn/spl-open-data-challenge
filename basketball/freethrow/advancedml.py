import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from anlyze2 import AdvancedFreeThrowAnalyzer, FreeThrowDataLoader
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

class AdvancedMLAnalyzer:
    def __init__(self, analyzer: AdvancedFreeThrowAnalyzer):
        self.analyzer = analyzer
        self.base_df = analyzer.base_analysis
        self.sequence_features = self._extract_sequence_features()
        # Merge base_df and sequence_features right after extraction
        self.combined_df = pd.merge(
            self.base_df,
            self.sequence_features,
            on='trial_id',
            suffixes=('', '_seq')  # Avoid _x, _y suffixes
        )

    def _extract_sequence_features(self) -> pd.DataFrame:
        """Extract sequential features from each trial"""
        sequence_data = []
        
        for trial in self.analyzer.trials_data:
            try:
                # Extract temporal features from the trial data
                features = {
                    'trial_id': trial['trial_id'],
                    'motion_smoothness': self._calculate_motion_smoothness(trial),
                    'movement_coordination': self._calculate_coordination(trial),
                    'peak_velocity': self._calculate_peak_velocity(trial),
                    'velocity_consistency': self._calculate_velocity_consistency(trial)
                }
                sequence_data.append(features)
            except (KeyError, ValueError, IndexError) as e:
                print(f"Error processing trial {trial['trial_id']}: {str(e)}")
                features = {
                    'trial_id': trial['trial_id'],
                    'motion_smoothness': np.nan,
                    'movement_coordination': np.nan,
                    'peak_velocity': np.nan,
                    'velocity_consistency': np.nan
                }
                sequence_data.append(features)
        
        return pd.DataFrame(sequence_data)

    def _calculate_motion_smoothness(self, trial):
        """Calculate motion smoothness from trial data"""
        try:
            # Extract wrist position data
            frames = pd.DataFrame([frame['data']['player'] for frame in trial['tracking']])
            wrist_pos = np.array([pos for pos in frames['R_WRIST']])
            
            # Calculate velocity and acceleration
            velocity = np.diff(wrist_pos, axis=0)
            acceleration = np.diff(velocity, axis=0)
            
            # Calculate smoothness (negative mean squared jerk)
            smoothness = -np.mean(np.square(acceleration))
            return smoothness
        except:
            return np.nan

    def _calculate_coordination(self, trial):
        """Calculate movement coordination between joints"""
        try:
            frames = pd.DataFrame([frame['data']['player'] for frame in trial['tracking']])
            
            # Get joint positions
            wrist_pos = np.array([pos for pos in frames['R_WRIST']])
            elbow_pos = np.array([pos for pos in frames['R_ELBOW']])
            
            # Calculate velocities
            wrist_vel = np.diff(wrist_pos, axis=0)
            elbow_vel = np.diff(elbow_pos, axis=0)
            
            # Calculate correlation between joint velocities
            coordination = np.corrcoef(
                np.linalg.norm(wrist_vel, axis=1),
                np.linalg.norm(elbow_vel, axis=1)
            )[0, 1]
            
            return coordination
        except:
            return np.nan

    def _calculate_peak_velocity(self, trial):
        """Calculate peak velocity of the wrist"""
        try:
            frames = pd.DataFrame([frame['data']['player'] for frame in trial['tracking']])
            wrist_pos = np.array([pos for pos in frames['R_WRIST']])
            
            # Calculate velocity magnitude
            velocity = np.diff(wrist_pos, axis=0)
            velocity_magnitude = np.linalg.norm(velocity, axis=1)
            
            return np.max(velocity_magnitude)
        except:
            return np.nan

    def _calculate_velocity_consistency(self, trial):
        """Calculate consistency of velocity profile"""
        try:
            frames = pd.DataFrame([frame['data']['player'] for frame in trial['tracking']])
            wrist_pos = np.array([pos for pos in frames['R_WRIST']])
            
            # Calculate velocity magnitude
            velocity = np.diff(wrist_pos, axis=0)
            velocity_magnitude = np.linalg.norm(velocity, axis=1)
            
            # Calculate coefficient of variation
            return np.std(velocity_magnitude) / np.mean(velocity_magnitude)
        except:
            return np.nan

    def create_interaction_features(self, X):
        """Create interaction features between key measurements"""
        X_interaction = X.copy()
        
        # Add interaction features
        if all(col in X.columns for col in ['peak_velocity', 'motion_smoothness']):
            X_interaction['velocity_smoothness_interaction'] = X['peak_velocity'] * X['motion_smoothness']
        
        if all(col in X.columns for col in ['movement_coordination', 'peak_velocity']):
            X_interaction['coordination_velocity'] = X['movement_coordination'] * X['peak_velocity']
        
        if all(col in X.columns for col in ['peak_velocity', 'release_height']):
            X_interaction['velocity_height_ratio'] = X['peak_velocity'] / (X['release_height'] + 1e-6)
        
        if all(col in X.columns for col in ['motion_smoothness', 'movement_coordination']):
            X_interaction['smoothness_coordination'] = X['motion_smoothness'] * X['movement_coordination']
        
        return X_interaction

    def analyze_temporal_patterns(self):
        """Analyze how features change over consecutive shots"""
        temporal_features = pd.DataFrame()
        temporal_features['shot_number'] = range(len(self.combined_df))
        
        # Define features that exist in combined_df
        available_features = [
            col for col in ['peak_velocity', 'movement_coordination', 'motion_smoothness']
            if col in self.combined_df.columns
        ]
        
        # Calculate rolling statistics
        window_sizes = [3, 5, 7]
        
        for window in window_sizes:
            for feature in available_features:
                # Rolling mean
                temporal_features[f'{feature}_rolling_mean_{window}'] = (
                    self.combined_df[feature].rolling(window).mean()
                )
                # Rolling std
                temporal_features[f'{feature}_rolling_std_{window}'] = (
                    self.combined_df[feature].rolling(window).std()
                )
                # Rolling trend
                temporal_features[f'{feature}_rolling_trend_{window}'] = (
                    self.combined_df[feature].rolling(window)
                    .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] 
                          if len(x) == window else np.nan)
                )
        
        return temporal_features

    def prepare_ml_features(self):
        """Prepare features for ML models"""
        # Use the already combined DataFrame
        features_df = self.combined_df.copy()
        
        # Get available feature columns
        base_features = [
            'prep_duration', 'shot_duration', 'release_angle',
            'max_hand_speed', 'avg_elbow_angle', 'release_height'
        ]
        
        sequence_features = [
            'motion_smoothness', 'movement_coordination',
            'peak_velocity', 'velocity_consistency'
        ]
        
        # Filter for features that actually exist in the DataFrame
        available_features = [col for col in base_features + sequence_features 
                            if col in features_df.columns]
        
        # Add temporal features
        temporal_features = self.analyze_temporal_patterns()
        
        # Combine all features
        X = pd.concat([
            features_df[available_features],
            temporal_features.drop('shot_number', axis=1)
        ], axis=1)
        
        # Create target variable
        y = (features_df['result'] == 'made').astype(int)
        
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def analyze_shot_patterns(self):
        """Analyze shot patterns using dimensionality reduction"""
        X_train, _, y_train, _ = self.prepare_ml_features()
        
        # Handle missing values first
        imputer = SimpleImputer(strategy='mean')
        X_clean = imputer.fit_transform(X_train)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_clean)
        
        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        
        # t-SNE with optimized parameters
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            n_iter=1000,
            learning_rate='auto',
            init='pca',
            random_state=42
        )
        tsne_result = tsne.fit_transform(X_scaled)
        
        return {
            'pca': {
                'result': pca_result,
                'explained_variance': pca.explained_variance_ratio_,
                'components': pca.components_,
                'feature_names': X_train.columns
            },
            'tsne': {
                'result': tsne_result
            },
            'labels': y_train
        }

    def train_ensemble_models(self):
        """Train and evaluate ensemble models"""
        X_train, X_test, y_train, y_test = self.prepare_ml_features()
        
        # Calculate class weights
        n_samples = len(y_train)
        n_classes = len(np.unique(y_train))
        class_weights = dict(zip(
            np.unique(y_train),
            n_samples / (n_classes * np.bincount(y_train))
        ))
        
        # Define pipelines for each model
        pipelines = {
            'random_forest': Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('classifier', RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    class_weight=class_weights,
                    random_state=42
                ))
            ]),
            'hist_gradient_boosting': Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', HistGradientBoostingClassifier(
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                ))
            ])
        }
        
        results = {}
        feature_importance = {}
        cv_scores = {}
        
        for name, pipeline in pipelines.items():
            print(f"\nTraining {name}...")
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores[name] = cross_val_score(
                pipeline, X_train, y_train, cv=5, scoring='accuracy'
            )
            
            # Evaluate
            y_pred = pipeline.predict(X_test)
            results[name] = {
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            # Feature importance (handle differently for each model type)
            if name == 'random_forest':
                feature_importance[name] = pd.Series(
                    pipeline.named_steps['classifier'].feature_importances_,
                    index=X_train.columns
                )
            else:
                # For HistGradientBoostingClassifier, only return if available
                if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
                    feature_importance[name] = pd.Series(
                        pipeline.named_steps['classifier'].feature_importances_,
                        index=X_train.columns
                    )
        
        return {
            'pipelines': pipelines,
            'results': results,
            'feature_importance': feature_importance,
            'cv_scores': cv_scores
        }

    def plot_analysis_results(self, ensemble_results, pattern_analysis):
        """Create comprehensive visualizations"""
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Feature Importance
        plt.subplot(3, 2, 1)
        feature_imp = ensemble_results['feature_importance']['random_forest'].sort_values()
        feature_imp.plot(kind='barh')
        plt.title('Feature Importance (Random Forest)', fontsize=12, pad=20)
        plt.xlabel('Importance Score')
        
        # 2. PCA Visualization
        plt.subplot(3, 2, 2)
        pca_result = pattern_analysis['pca']['result']
        scatter = plt.scatter(
            pca_result[:, 0],
            pca_result[:, 1],
            c=pattern_analysis['labels'],
            cmap='coolwarm',
            alpha=0.6
        )
        plt.title('Shot Patterns (PCA)', fontsize=12, pad=20)
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.colorbar(scatter, label='Made Shot')
        
        # 3. Model Performance Comparison
        plt.subplot(3, 2, 3)
        cv_scores = pd.DataFrame(ensemble_results['cv_scores'])
        sns.boxplot(data=cv_scores)
        plt.title('Model Performance Comparison', fontsize=12, pad=20)
        plt.ylabel('Cross-validation Accuracy')
        plt.xticks(rotation=45)
        
        # 4. t-SNE Visualization
        plt.subplot(3, 2, 4)
        tsne_result = pattern_analysis['tsne']['result']
        scatter = plt.scatter(
            tsne_result[:, 0],
            tsne_result[:, 1],
            c=pattern_analysis['labels'],
            cmap='coolwarm',
            alpha=0.6
        )
        plt.title('Shot Patterns (t-SNE)', fontsize=12, pad=20)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.colorbar(scatter, label='Made Shot')
        
        # 5. Temporal Patterns
        plt.subplot(3, 2, 5)
        temporal_features = self.analyze_temporal_patterns()
        
        # Get first available feature for plotting
        available_features = [col for col in ['peak_velocity', 'movement_coordination', 'motion_smoothness']
                            if col in self.combined_df.columns]
        
        if available_features:
            feature = available_features[0]
            plt.plot(temporal_features['shot_number'],
                    temporal_features[f'{feature}_rolling_mean_5'],
                    label=feature.replace('_', ' ').title())
            plt.title(f'Temporal Patterns (5-shot rolling mean)', fontsize=12, pad=20)
            plt.xlabel('Shot Number')
            plt.ylabel('Feature Value')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig('advanced_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.close()

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
            print(importance.sort_values(ascending=False).head())
            
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()