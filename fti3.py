import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np

def create_cluster_plot(df, group_size, features):
   # Group shots
   df['shot_group'] = df['shot_number'] // group_size
   grouped_df = df.groupby('shot_group').mean()
   
   # Calculate correlation matrix and find optimal clusters
   corr_matrix = grouped_df[features].corr()
   silhouette_scores = []
   n_clusters_range = range(2, 6)
   
   for n_clusters in n_clusters_range:
       clusterer = AgglomerativeClustering(n_clusters=n_clusters)
       cluster_labels = clusterer.fit_predict(corr_matrix)
       silhouette_avg = silhouette_score(corr_matrix, cluster_labels)
       silhouette_scores.append(silhouette_avg)

   optimal_clusters = n_clusters_range[np.argmax(silhouette_scores)]
   
   # Cluster features
   cluster = AgglomerativeClustering(n_clusters=optimal_clusters)
   labels = cluster.fit_predict(corr_matrix)
   
   feature_clusters = {}
   for i, feature in enumerate(features):
       if labels[i] not in feature_clusters:
           feature_clusters[labels[i]] = []
       feature_clusters[labels[i]].append(feature)

   # Plot
   fig, axes = plt.subplots(optimal_clusters, 1, figsize=(15, 5*optimal_clusters))
   if optimal_clusters == 1:
       axes = [axes]

   for i, (cluster_id, cluster_features) in enumerate(feature_clusters.items()):
       ax = axes[i]
       for feature in cluster_features:
           ax.plot(grouped_df.index * group_size, grouped_df[feature], 
                   label=feature, alpha=0.7, linewidth=2)
       
       ax.set_xlabel(f'Shot Number (Grouped by {group_size})')
       ax.set_ylabel('Feature Importance')
       ax.set_title(f'Feature Cluster {cluster_id+1}')
       ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
       ax.grid(True, alpha=0.3)

   plt.tight_layout()
   plt.savefig(f'feature_clusters_{group_size}shots.png', bbox_inches='tight')
   plt.close()

   # Print correlations
   print(f"\nGroup Size: {group_size} shots")
   print(f"Optimal clusters: {optimal_clusters}")
   for cluster_id, cluster_features in feature_clusters.items():
       print(f"\nCluster {cluster_id+1}:")
       print(cluster_features)
       for i, feat1 in enumerate(cluster_features):
           for feat2 in cluster_features[i+1:]:
               corr = corr_matrix.loc[feat1, feat2]
               if abs(corr) >= 0.5:
                   print(f"{feat1} - {feat2}: {corr:.3f}")

# Read data
df = pd.read_csv(r'C:\Users\16476\Downloads\SPL-Open-Data-main\feature_importance\feature_importance_over_time.csv')
df = df[df.iloc[:, 1:].sum(axis=1) != 0]
features = [col for col in df.columns if col not in ['shot_number', 'shot_group']]

# Create plots for different group sizes
for group_size in [1, 3, 5, 10]:
   create_cluster_plot(df.copy(), group_size, features)