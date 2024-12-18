import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load the Iris dataset (you can replace it with your own dataset)
iris = load_iris()
X = iris.data  # Features
y = iris.target  # Target classes

# Step 1: Standardizing the data (important for PCA)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA
# We'll reduce it to 2 dimensions for visualization, but you can change it to other values
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 3: Explained variance ratio (how much variance is retained)
print(f"Explained variance ratio for each component: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.2f}")

# Step 4: Visualizing the reduced data in 2D
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=80)
plt.title("PCA of Iris Dataset (2D Projection)", fontsize=16)
plt.xlabel("Principal Component 1", fontsize=14)
plt.ylabel("Principal Component 2", fontsize=14)
plt.colorbar(label='Target Class')
plt.show()
