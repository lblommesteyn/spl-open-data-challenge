import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Read data
df = pd.read_csv(r'C:\Users\16476\Downloads\SPL-Open-Data-main\feature_importance\feature_importance_over_time.csv')

# Remove rows where all feature importances are 0 (misses)
df = df[df.iloc[:, 1:].sum(axis=1) != 0]

# Group by every 5 shots and take mean
df['shot_group'] = df['shot_number'] // 5
grouped_df = df.groupby('shot_group').mean()

# Plot smoothed feature importance
plt.figure(figsize=(15, 8))
for column in df.columns[1:-1]:  # Skip shot_number and shot_group
   plt.plot(grouped_df.index * 5, grouped_df[column], 
            label=column, alpha=0.7, linewidth=2)

plt.xlabel('Shot Number (Grouped by 5)')
plt.ylabel('Average Feature Importance')
plt.title('Feature Importance Over Successful Shots (5-Shot Moving Average)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('feature_importance_smoothed.png')

# Print strongest correlations for successful shots
corr_matrix = df.iloc[:, 1:-1].corr()
threshold = 0.5
strong_corrs = [(i, j, corr_matrix.iloc[i,j]) 
               for i in range(len(corr_matrix.columns)) 
               for j in range(i+1, len(corr_matrix.columns))
               if abs(corr_matrix.iloc[i,j]) >= threshold]

print("\nStrongest Feature Correlations (|r| >= 0.5):")
for i, j, corr in sorted(strong_corrs, key=lambda x: abs(x[2]), reverse=True):
   print(f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr:.3f}")