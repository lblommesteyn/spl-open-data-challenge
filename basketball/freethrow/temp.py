import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from data_loader import FreeThrowDataLoader
from advancedml import AdvancedFreeThrowAnalyzer, AdvancedMLAnalyzer
from anlyze2 import AdvancedFreeThrowAnalyzer

class TemporalFeatureImportanceAnalysis:
    def __init__(self, ml_analyzer):
        self.ml_analyzer = ml_analyzer
    
    def calculate_feature_importance_over_time(self, window_size=5):
        """Calculate moving feature importance using sliding windows"""
        # Initialize DataFrame for storing feature importances
        importance_over_time = []
        analysis_data = []  # Store additional analysis info
        
        # Get total number of shots
        n_shots = len(self.ml_analyzer.combined_df)
        
        # Get initial feature importance for reference
        ensemble_results = self.ml_analyzer.train_ensemble_models()
        feature_names = ensemble_results['feature_importance']['random_forest'].index
        
        # Calculate feature importance for each window
        for i in range(0, n_shots - window_size + 1):
            # Get window indices
            window_indices = range(i, i + window_size)
            
            # Train model on window data and get feature importance
            X = self.ml_analyzer.combined_df.iloc[window_indices]
            y = (X['result'] == 'made').astype(int)
            
            # Create and train Random Forest directly
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            features_for_training = X.drop(['result', 'trial_id'] if 'trial_id' in X.columns else ['result'], axis=1)
            
            # Handle potential missing values
            features_for_training = features_for_training.fillna(features_for_training.mean())
            
            rf.fit(features_for_training, y)
            
            # Get feature importance for this window
            importance = pd.Series(
                rf.feature_importances_,
                index=features_for_training.columns
            )
            
            # Add shot number (middle of window) and feature importances to results
            result = {'shot_number': i + window_size//2}
            result.update(importance.to_dict())
            importance_over_time.append(result)
            
            # Store additional analysis information
            analysis_info = {
                'window_start': i,
                'window_end': i + window_size,
                'shot_number': i + window_size//2,
                'window_accuracy': y.mean(),
                'top_feature': importance.idxmax(),
                'top_feature_importance': importance.max(),
                'feature_importance_std': importance.std(),
                'n_features': len(features_for_training.columns)
            }
            analysis_data.append(analysis_info)
        
        self.importance_df = pd.DataFrame(importance_over_time)
        self.analysis_df = pd.DataFrame(analysis_data)
        return self.importance_df

    def save_analysis_results(self, output_dir='feature_importance'):
        """Save all analysis results to files"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Save feature importance data
        self.importance_df.to_csv(os.path.join(output_dir, 'feature_importance_over_time.csv'), index=False)
        
        # Save analysis data
        self.analysis_df.to_csv(os.path.join(output_dir, 'temporal_analysis_metrics.csv'), index=False)
        
        # Generate and save summary statistics
        summary_stats = {
            'importance_summary': self.importance_df.describe(),
            'analysis_summary': self.analysis_df.describe(),
            'top_features_by_window': self.analysis_df.groupby('top_feature').size().sort_values(ascending=False),
            'average_accuracy': self.analysis_df['window_accuracy'].mean(),
            'accuracy_trend': self.analysis_df['window_accuracy'].rolling(window=5).mean()
        }
        
        # Save summary statistics
        with pd.ExcelWriter(os.path.join(output_dir, 'analysis_summary.xlsx')) as writer:
            summary_stats['importance_summary'].to_excel(writer, sheet_name='Importance Summary')
            summary_stats['analysis_summary'].to_excel(writer, sheet_name='Analysis Summary')
            summary_stats['top_features_by_window'].to_excel(writer, sheet_name='Top Features')
            pd.DataFrame({
                'average_accuracy': [summary_stats['average_accuracy']],
                'accuracy_std': [self.analysis_df['window_accuracy'].std()],
                'min_accuracy': [self.analysis_df['window_accuracy'].min()],
                'max_accuracy': [self.analysis_df['window_accuracy'].max()]
            }).to_excel(writer, sheet_name='Accuracy Metrics')

    def plot_and_save_importance_trends(self, importance_df, output_dir='feature_importance'):
        """Plot and save the temporal trends of feature importance with enhanced visualization"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
                
        # Get feature columns (all except shot_number)
        features = [col for col in importance_df.columns if col != 'shot_number']
        
        # Set style for better-looking plots
        plt.style.use('default')
        
        # Create individual plots for each feature
        for feature in features:
            # Create figure with specific size and DPI
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1], dpi=100)
            fig.suptitle(f'Feature Importance Analysis: {feature.replace("_", " ").title()}', 
                        fontsize=16, y=0.95)
            
            # Main plot with feature importance
            ax1.plot(importance_df['shot_number'], 
                    importance_df[feature], 
                    color='royalblue',
                    marker='o', 
                    markersize=4,
                    linestyle='-',
                    linewidth=1,
                    alpha=0.6,
                    label='Raw importance')
            
            # Add rolling average with shaded confidence interval
            window = 5
            rolling_avg = importance_df[feature].rolling(window=window, center=True).mean()
            rolling_std = importance_df[feature].rolling(window=window, center=True).std()
            
            ax1.plot(importance_df['shot_number'],
                    rolling_avg,
                    color='red',
                    linestyle='-',
                    linewidth=2,
                    label=f'{window}-shot moving average')
            
            # Add confidence interval
            ax1.fill_between(importance_df['shot_number'],
                            rolling_avg - rolling_std,
                            rolling_avg + rolling_std,
                            color='red',
                            alpha=0.1)
            
            # Distribution plot at the bottom
            sns.kdeplot(data=importance_df[feature], ax=ax2, color='royalblue', fill=True)
            ax2.axvline(importance_df[feature].mean(), color='red', linestyle='--', 
                        label=f'Mean: {importance_df[feature].mean():.3f}')
            
            # Customize main plot
            ax1.set_xlabel('Shot Number')
            ax1.set_ylabel('Feature Importance')
            ax1.grid(True, linestyle='--', alpha=0.7)
            ax1.legend(loc='upper right')
            
            # Add statistics annotations
            stats_text = f'Mean: {importance_df[feature].mean():.3f}\n'
            stats_text += f'Std Dev: {importance_df[feature].std():.3f}\n'
            stats_text += f'Max: {importance_df[feature].max():.3f}\n'
            stats_text += f'Min: {importance_df[feature].min():.3f}'
            
            # Add text box with statistics
            ax1.text(0.02, 0.98, stats_text,
                    transform=ax1.transAxes,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Customize distribution plot
            ax2.set_xlabel('Feature Importance Distribution')
            ax2.set_ylabel('Density')
            ax2.legend()
            
            plt.tight_layout()
            
            # Save individual feature plot
            plt.savefig(os.path.join(output_dir, f'feature_importance_{feature}.png'))
            plt.close()
            
        # Create correlation heatmap of features over time
        plt.figure(figsize=(12, 8))
        feature_corr = importance_df[features].corr()
        sns.heatmap(feature_corr, 
                    annot=True, 
                    cmap='RdBu_r', 
                    center=0,
                    fmt='.2f',
                    square=True)
        plt.title('Feature Importance Correlation Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_correlation_heatmap.png'))
        plt.close()

def main():
    # Set up path and load data
    base_path = r"C:\Users\16476\Downloads\SPL-Open-Data-main\basketball\freethrow"
    loader = FreeThrowDataLoader(base_path)
    participant_data = loader.load_participant_data('P0001')
    
    # Create analyzers
    ft_analyzer = AdvancedFreeThrowAnalyzer(participant_data)
    ml_analyzer = AdvancedMLAnalyzer(ft_analyzer)

    # Analyze temporal feature importance
    temporal_importance = TemporalFeatureImportanceAnalysis(ml_analyzer)
    importance_over_time = temporal_importance.calculate_feature_importance_over_time(window_size=5)
    
    # Save all analysis results
    temporal_importance.save_analysis_results()
    temporal_importance.plot_and_save_importance_trends(importance_over_time)

if __name__ == "__main__":
    main()