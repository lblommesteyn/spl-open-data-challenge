import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
df = pd.read_csv(r'C:\Users\16476\Downloads\SPL-Open-Data-main\feature_importance\feature_importance_over_time.csv')

# Calculate average importance for each feature
avg_importance = df.mean()

# Plot 1: Feature importance by shot
plt.figure(figsize=(15, 8))
for column in df.columns:
   if column != 'shot_number':
       plt.plot(df['shot_number'], df[column], label=column, alpha=0.7)
plt.xlabel('Shot Number')
plt.ylabel('Feature Importance')
plt.title('Feature Importance Over Shots')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('feature_importance_by_shot.png')
plt.close()

# Plot 2: Average feature importance
plt.figure(figsize=(12, 6))
avg_importance_sorted = avg_importance.sort_values(ascending=False)
avg_importance_sorted = avg_importance_sorted.drop('shot_number')
sns.barplot(x=avg_importance_sorted.index, y=avg_importance_sorted.values)
plt.xticks(rotation=45, ha='right')
plt.xlabel('Features')
plt.ylabel('Feature Importance')
plt.title('Feature Importance')
plt.tight_layout()
plt.savefig('average_feature_importance.png')
plt.close()

# Print average importance values
print("\nAverage Feature Importance:")
for feature, importance in avg_importance_sorted.items():
      print(f"{feature}: {importance:.4f}")