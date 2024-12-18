import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the iris dataset
iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(iris.data)

# Use PCA for 2D visualization (since there are 4 features)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(iris.data)
data['pca1'] = pca_result[:, 0]
data['pca2'] = pca_result[:, 1]

# Plot the clusters with PCA-reduced dimensions
plt.figure(figsize=(10, 6))
sns.scatterplot(x='pca1', y='pca2', hue='cluster', data=data, palette='Set2', s=100)
plt.title('K-Means Clustering of Iris Dataset (PCA-reduced dimensions)')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.legend(title='Cluster')
plt.show()
