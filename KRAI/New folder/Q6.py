import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
dataset = pd.read_csv("Mall_Customers_Dataset.csv")

dataset = dataset.drop(columns=['CustomerID'])

dataset = pd.get_dummies(dataset, columns=['Gender'], drop_first=True)

features = dataset[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(features['Age'], features['Annual Income (k$)'], features['Spending Score (1-100)'], color='black', marker='*')
ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")
plt.show()

# Determine the optimal number of clusters using silhouette score
silhouette_scores = []
k_values = range(2, 11)  # Check for k clusters from 2 to 10
for k in k_values:
    # Disable multi-threading by using n_init=1 and n_jobs=None
    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=1, random_state=42)
    kmeans.fit(features)
    score = silhouette_score(features, kmeans.labels_)
    silhouette_scores.append(score)

# Identify the optimal k value
optimal_k = k_values[silhouette_scores.index(max(silhouette_scores))]
print(f"The optimal number of clusters is: {optimal_k}")

# Run KMeans with the optimal number of clusters
km = KMeans(n_clusters=optimal_k, init="k-means++", n_init=1, random_state=42)
clusters = km.fit_predict(features)

# Assign cluster labels to the dataset
dataset["label"] = clusters

# 3D plot of the clusters
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

colors = ['black', 'red', 'green', 'brown', 'purple', 'orange']
for i in range(optimal_k):
    ax.scatter(features['Age'][dataset.label == i],
               features['Annual Income (k$)'][dataset.label == i],
               features['Spending Score (1-100)'][dataset.label == i],
               color=colors[i % len(colors)],
               s=60, label=f'Cluster {i}')

ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")
plt.legend()
plt.grid()
plt.show()

