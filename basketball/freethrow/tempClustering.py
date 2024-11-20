import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import json


# FreeThrowDataLoader to Load JSON Data
class FreeThrowDataLoader:
    def __init__(self, base_path):
        self.base_path = base_path

    def load_participant_data(self, participant_id):
        """
        Load and combine all JSON files for a given participant ID.
        Files are expected to be in the directory: base_path/{participant_id}/BB_FT_{participant_id}_T*.json
        """
        # Define the directory for the participant
        participant_dir = os.path.join(self.base_path, participant_id)
        search_pattern = os.path.join(participant_dir, f"BB_FT_{participant_id}_T*.json")
        files = glob.glob(search_pattern)

        if not files:
            raise FileNotFoundError(f"No files found for participant ID {participant_id} in {participant_dir}")

        # Combine all matching JSON files into one DataFrame
        data_frames = []
        for file_path in files:
            with open(file_path, "r") as f:
                data = json.load(f)
                df = pd.json_normalize(data)
                data_frames.append(df)

        # Concatenate all the data
        combined_data = pd.concat(data_frames, ignore_index=True)

        # Convert 'result' column to binary
        combined_data['made'] = (combined_data['result'] == 'made').astype(int)

        # Drop the 'result' column (we already converted it to 'made')
        combined_data.drop(columns=['result'], inplace=True)

        return combined_data


# AdvancedFreeThrowAnalyzer
class AdvancedFreeThrowAnalyzer:
    def __init__(self, participant_data):
        self.participant_data = participant_data


# AdvancedMLAnalyzer
class AdvancedMLAnalyzer:
    def __init__(self, ft_analyzer):
        self.ft_analyzer = ft_analyzer
        self.combined_df = ft_analyzer.participant_data

    def train_ensemble_models(self):
        # Mock feature importance output
        feature_importance = {
            'random_forest': pd.Series(
                np.random.rand(5),
                index=['velocity_consistency', 'motion_smoothness', 'peak_velocity', 'release_angle', 'release_height']
            )
        }
        return {'feature_importance': feature_importance}


# Fatigue Analysis Class
class AdvancedFatigueAnalysis:
    def __init__(self, ml_analyzer):
        self.ml_analyzer = ml_analyzer

    def group_analysis_by_performance(self):
        """Group shots by performance (e.g., makes vs misses) and analyze feature importance."""
        df = self.ml_analyzer.combined_df.copy()
        
        # Group by makes and misses
        makes = df[df['made'] == 1]
        misses = df[df['made'] == 0]
        
        # Train separate models for makes and misses
        self._train_and_store_feature_importance(makes, 'Makes')
        self._train_and_store_feature_importance(misses, 'Misses')
        
        return makes, misses

    def _train_and_store_feature_importance(self, data, group_name):
        """Train a RandomForestClassifier and store feature importance."""
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Drop non-numeric columns
        features = data.drop(['trial_id', 'made'], axis=1, errors='ignore')
        numeric_features = features.select_dtypes(include=[np.number])
        
        # Fill missing values with column means
        numeric_features = numeric_features.fillna(numeric_features.mean())
        
        # Train the Random Forest classifier
        rf.fit(numeric_features, data['made'])
        
        # Get feature importance
        importance = pd.Series(rf.feature_importances_, index=numeric_features.columns)
        importance.sort_values(ascending=False, inplace=True)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importance.values, y=importance.index, palette='viridis')
        plt.title(f'Feature Importance for {group_name}')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig(f'feature_importance_{group_name}.png')
        plt.close()

    def fatigue_clustering(self, n_clusters=3):
        """Cluster shots based on fatigue-related features."""
        df = self.ml_analyzer.combined_df.copy()
        
        # Define potential fatigue features
        fatigue_features = ['velocity_consistency', 'motion_smoothness', 'peak_velocity']
        
        # Check which features are present in the dataset
        available_features = [feature for feature in fatigue_features if feature in df.columns]
        
        if not available_features:
            raise KeyError(f"No fatigue-related features found in the dataset. Available columns: {df.columns.tolist()}")
        
        # Use only the available features for clustering
        df = df[available_features].fillna(df.mean())
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        df['fatigue_cluster'] = kmeans.fit_predict(df)
        
        # Analyze feature importance within each cluster
        cluster_results = {}
        for cluster in range(n_clusters):
            cluster_data = df[df['fatigue_cluster'] == cluster]
            self._train_and_store_feature_importance(cluster_data, f'Cluster {cluster}')
            cluster_results[cluster] = cluster_data
        
        return cluster_results

    def fatigue_correlation_analysis(self):
        """Analyze correlation between fatigue indicators and makes/misses."""
        df = self.ml_analyzer.combined_df.copy()
        
        fatigue_features = ['velocity_consistency', 'motion_smoothness', 'peak_velocity']
        correlation_results = {}
        
        for feature in fatigue_features:
            if feature in df.columns:
                correlation = df[feature].corr(df['made'])
                correlation_results[feature] = correlation
        
        # Plot correlations
        plt.figure(figsize=(8, 6))
        sns.barplot(x=list(correlation_results.keys()), y=list(correlation_results.values()), palette='coolwarm')
        plt.title('Correlation Between Fatigue Features and Makes')
        plt.xlabel('Fatigue Feature')
        plt.ylabel('Correlation with Makes')
        plt.tight_layout()
        plt.savefig('fatigue_feature_correlation.png')
        plt.close()
        
        return correlation_results

    def fatigue_progression_plot(self):
        """Visualize fatigue progression over time."""
        df = self.ml_analyzer.combined_df.copy()
        fatigue_features = ['velocity_consistency', 'motion_smoothness', 'peak_velocity']
        
        available_features = [feature for feature in fatigue_features if feature in df.columns]
        if not available_features:
            raise KeyError("No fatigue-related features available for progression plot.")
        
        df['shot_number'] = range(len(df))
        
        plt.figure(figsize=(14, 8))
        for feature in available_features:
            plt.plot(df['shot_number'], df[feature], label=feature, alpha=0.8)
        
        plt.title('Fatigue Features Progression Over Time')
        plt.xlabel('Shot Number')
        plt.ylabel('Feature Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('fatigue_progression.png')
        plt.close()

    def logistic_regression_analysis(self):
        """Logistic regression to analyze how fatigue affects makes/misses."""
        df = self.ml_analyzer.combined_df.copy()
        fatigue_features = ['velocity_consistency', 'motion_smoothness', 'peak_velocity']
        
        available_features = [feature for feature in fatigue_features if feature in df.columns]
        if not available_features:
            raise KeyError("No fatigue-related features available for logistic regression.")
        
        # Train logistic regression
        X = df[available_features].fillna(df.mean())
        y = df['made']
        
        lr = LogisticRegression(random_state=42, max_iter=1000)
        lr.fit(X, y)
        
        # Display coefficients
        coefficients = pd.Series(lr.coef_[0], index=available_features)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=coefficients.values, y=coefficients.index, palette='coolwarm')
        plt.title('Logistic Regression Coefficients')
        plt.xlabel('Effect on Makes')
        plt.ylabel('Fatigue Feature')
        plt.tight_layout()
        plt.savefig('logistic_regression_coefficients.png')
        plt.close()
        
        return coefficients


# Main Function
def main():
    # Set up the path to the data folder
    base_path = r"C:\Users\16476\Downloads\SPL-Open-Data-main\basketball\freethrow\data"
    
    # Load data for participant P0001
    loader = FreeThrowDataLoader(base_path)
    participant_data = loader.load_participant_data('P0001')
    
    # Print available columns for debugging
    print("Loaded Data Columns:", participant_data.columns)
    
    # Create analyzers
    ft_analyzer = AdvancedFreeThrowAnalyzer(participant_data)
    ml_analyzer = AdvancedMLAnalyzer(ft_analyzer)
    
    # Perform advanced fatigue analysis
    fatigue_analysis = AdvancedFatigueAnalysis(ml_analyzer)
    makes, misses = fatigue_analysis.group_analysis_by_performance()
    cluster_results = fatigue_analysis.fatigue_clustering()
    correlation_results = fatigue_analysis.fatigue_correlation_analysis()
    fatigue_analysis.fatigue_progression_plot()
    coefficients = fatigue_analysis.logistic_regression_analysis()
    
    print("Fatigue Analysis Completed")
    print("Correlation Results:", correlation_results)
    print("Logistic Regression Coefficients:", coefficients)

if __name__ == "__main__":
    main()
