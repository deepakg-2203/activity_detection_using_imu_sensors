import pandas as pd
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load your dataset
dataset_path = 'E:/GEETHA/simple_final_combined_data.csv'
data = pd.read_csv(dataset_path)

# Specify the number of clusters
num_clusters = 3  # Adjust the number of clusters as needed

# Create a K-Means model
kmeans = KMeans(n_clusters=num_clusters, random_state=42,n_init=10)

# Fit the K-Means model to your data
kmeans.fit(data.drop('Label', axis=1))  # Assuming 'Label' contains the activity labels

# Save (dump) the trained K-Means model to a file using joblib
model_filename = 'kmeans_model.pkl'
joblib.dump(kmeans, model_filename)

print(f"K-Means model saved to {model_filename}")

# Predict cluster labels for the data
data['Cluster'] = kmeans.predict(data.drop(['Label'], axis=1))

# Calculate the most common activity label in each cluster
cluster_activities = data.groupby('Cluster')['Label'].value_counts().unstack(fill_value=0)
print("Cluster Activities:")
print(cluster_activities)

# Visualize clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
data_2d = pca.fit_transform(data.drop(['Label', 'Cluster'], axis=1))

# Plot the clusters
plt.figure(figsize=(8, 6))
for cluster in range(num_clusters):
    cluster_data = data_2d[data['Cluster'] == cluster]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f'Cluster {cluster}')

plt.title('Cluster Visualization')
plt.xlabel('PCA Dimension 1')
plt.ylabel('PCA Dimension 2')
plt.legend()
plt.show()