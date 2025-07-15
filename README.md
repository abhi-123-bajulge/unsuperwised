# unsuperwised
Programs Unsupervised Learning • Clustering • Dimensionality Reduction
# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
# Load the Iris dataset
iris = load_iris()
X = iris.data
# Clustering: K-Means
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X)
# Adding the cluster data to the original dataset for visualization
iris_with_clusters = pd.DataFrame(X, columns=iris.feature_names)
iris_with_clusters['Cluster'] = clusters
print(f'Cluster Centers:\n{kmeans.cluster_centers_}')
print(f'Inertia: {kmeans.inertia_}')
# Dimensionality Reduction: PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
# Visualizing the clusters in the reduced dimension
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', edgecolor='k',
s=150)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('K-Means Clustering with PCA Reduction on Iris Dataset')
plt.show()
