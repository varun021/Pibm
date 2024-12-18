import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
dataset = pd.read_csv("Mall_Customers.csv")

# Drop unnecessary CustomerID column
dataset = dataset.drop(columns=['CustomerID'])

# Handle categorical data (Genre)
dataset = pd.get_dummies(dataset, columns=['Genre'], drop_first=True)

# Extract the features we want to use for clustering
features = dataset[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the features (scaling them)
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply K-Means clustering
# Fix FutureWarning by explicitly setting n_init to 10
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)
dataset['Cluster'] = kmeans.fit_predict(features_scaled)

# Visualizing the clusters based on Age and Annual Income
plt.figure(figsize=(8, 6))
plt.scatter(dataset['Age'], dataset['Annual Income (k$)'], c=dataset['Cluster'], cmap='viridis', s=100, edgecolor='black')
plt.title("Customer Segments Based on Age and Annual Income")
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
plt.colorbar(label='Cluster')
plt.show()

# Visualizing the clusters based on Age and Spending Score
plt.figure(figsize=(8, 6))
plt.scatter(dataset['Age'], dataset['Spending Score (1-100)'], c=dataset['Cluster'], cmap='viridis', s=100, edgecolor='black')
plt.title("Customer Segments Based on Age and Spending Score")
plt.xlabel("Age")
plt.ylabel("Spending Score (1-100)")
plt.colorbar(label='Cluster')
plt.show()
