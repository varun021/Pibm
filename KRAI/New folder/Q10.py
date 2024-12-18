import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Define the data points
data_points = np.array([[8, 12], [12, 17], [20, 20], [25, 10], [22, 35],
                        [81, 65], [70, 75], [55, 65], [51, 60], [85, 93]])

# Step 2: Apply K-Means Clustering
# We assume k = 2 (as there appear to be two distinct groups in the data)
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(data_points)

# Step 3: Get the cluster centers and the labels
cluster_centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Step 4: Visualize the clusters
plt.figure(figsize=(8, 6))

# Plot data points with different colors based on their assigned cluster
plt.scatter(data_points[:, 0], data_points[:, 1], c=labels, s=100, cmap='viridis', marker='o', edgecolor='black')

# Plot the cluster centers
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', s=200, marker='X', label="Centroids")

plt.title("K-Means Clustering (k=2)", fontsize=16)
plt.xlabel("X-axis", fontsize=14)
plt.ylabel("Y-axis", fontsize=14)
plt.legend()
plt.grid(True)
plt.show()
