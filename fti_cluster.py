import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import AgglomerativeClustering

# Read data and remove misses
df = pd.read_csv(r'C:\Users\16476\Downloads\SPL-Open-Data-main\feature_importance\feature_importance_over_time.csv')
df = df[df.iloc[:, 1:].sum(axis=1) != 0]

# Group by every 5 shots
df['shot_group'] = df['shot_number'] // 5
grouped_df = df.groupby('shot_group').mean()

# Calculate correlation matrix
corr_matrix = grouped_df.iloc[:, :-1].corr()

# Cluster features
n_clusters = 3
cluster = AgglomerativeClustering(n_clusters=n_clusters)
labels = cluster.fit_predict(corr_matrix)

# Group features by cluster
feature_clusters = {}
for i, feature in enumerate(corr_matrix.columns):
   if labels[i] not in feature_clusters:
       feature_clusters[labels[i]] = []
   feature_clusters[labels[i]].append(feature)

# Plot clusters
fig, axes = plt.subplots(n_clusters, 1, figsize=(15, 5*n_clusters))

for i, (cluster_id, features) in enumerate(feature_clusters.items()):
   ax = axes[i]
   for feature in features:
       ax.plot(grouped_df.index * 5, grouped_df[feature], 
               label=feature, alpha=0.7, linewidth=2)
   
   ax.set_xlabel('Shot Number (Grouped by 5)')
   ax.set_ylabel('Feature Importance')
   ax.set_title(f'Cluster {cluster_id+1} Features')
   ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
   ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('feature_clusters.png', bbox_inches='tight')

# Print clusters and correlations
print("\nFeature Clusters and Strong Correlations (|r| >= 0.5):")
for cluster_id, features in feature_clusters.items():
   print(f"\nCluster {cluster_id+1}:")
   print(features)
   for i, feat1 in enumerate(features):
       for feat2 in features[i+1:]:
           corr = corr_matrix.loc[feat1, feat2]
           if abs(corr) >= 0.5:
               print(f"{feat1} - {feat2}: {corr:.3f}")