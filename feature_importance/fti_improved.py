import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read data
df = pd.read_csv(r'C:\Users\16476\Downloads\SPL-Open-Data-main\feature_importance\feature_importance_over_time.csv')

# Create a mask for non-zero shots
non_zero_mask = df.iloc[:, 1:].sum(axis=1) != 0

# Calculate correlations over time using rolling windows
window_size = 10
correlations_over_time = []
for i in range(0, len(df)-window_size, window_size):
   window_data = df.iloc[i:i+window_size, 1:]  # Exclude shot_number
   corr = window_data.corr()
   correlations_over_time.append(corr)

# Plot feature importance with better visualization
plt.figure(figsize=(15, 8))
for column in df.columns[1:]:  # Skip shot_number
   plt.plot(df['shot_number'], df[column], label=column, alpha=0.7, linewidth=2)
   
   # Add markers for non-zero points
   non_zero_points = df[non_zero_mask]
   plt.scatter(non_zero_points['shot_number'], non_zero_points[column], alpha=0.3)

plt.xlabel('Shot Number')
plt.ylabel('Feature Importance')
plt.title('Feature Importance Over Shots (Successful Shots Only)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_by_shot_enhanced.png')

# Print interesting correlations and patterns
print("\nKey Statistics for Successful Shots:")
successful_shots = df[non_zero_mask]
print("\nFeature Correlations:")
corr_matrix = successful_shots.iloc[:, 1:].corr()


# Print strongest correlations
threshold = 0.5
strong_corrs = []
for i in range(len(corr_matrix.columns)):
   for j in range(i+1, len(corr_matrix.columns)):
       if abs(corr_matrix.iloc[i,j]) >= threshold:
           strong_corrs.append((corr_matrix.columns[i], 
                              corr_matrix.columns[j], 
                              corr_matrix.iloc[i,j]))

print("\nStrongest Feature Correlations (|r| >= 0.5):")
for feat1, feat2, corr in sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True):
   print(f"{feat1} - {feat2}: {corr:.3f}")

# Plot correlation heatmap for successful shots
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, fmt='.2f')
plt.title('Feature Correlations for Successful Shots')
plt.tight_layout()
plt.savefig('feature_correlations_heatmap.png')