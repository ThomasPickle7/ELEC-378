import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import scipy.io as sc
from scipy import signal, linalg
import matplotlib
import matplotlib.image as im
import time
from sklearn.decomposition import PCA
import pandas as pd
from scipy.spatial import distance
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

cancer = loadmat('ELEC-378\ELEC-378\HW3\cancer.mat')

X = cancer['X']
Y = cancer['Y']

X_c = X - X.mean()

X_u, X_s, X_vh = np.linalg.svd(X)

X_va = X_vh[:2, :]

X_pca = np.dot(X_c, X_va.T)

plt.scatter(X_pca[:, 0], X_pca[:, 1])

for i, tag in enumerate(Y):
    plt.annotate(tag[0][0], (X_pca[i, 0], X_pca[i, 1]))
plt.title("Types of Cancer")
plt.show()

# (c) Sorting genes by magnitude of coefficients in the most informative principal direction
pca = PCA(n_components=2)
pca.fit(X_c)
most_informative_direction = np.argmax(np.abs(pca.components_), axis=0)
sorted_genes_indices = np.argsort(np.abs(pca.components_[most_informative_direction[0]]))[::-1]

# Display heatmap of the data matrix after sorting the columns
plt.figure(figsize=(10, 6))
plt.imshow(X[:, sorted_genes_indices], aspect='auto', cmap='viridis')
plt.xlabel('Genes')
plt.ylabel('Patients')
plt.title('Heatmap of Gene Expression Data Sorted by Magnitude of Coefficients')
plt.colorbar(label='Expression Level')
plt.show()

pca = PCA(n_components=10)  # Retain more than two principal components
principal_components = pca.fit_transform(X_c)

# Perform K-means clustering
kmeans = KMeans(n_clusters=14)  # 14 different cancer types
kmeans.fit(principal_components)

# Visualize clusters
plt.scatter(principal_components[:, 0], principal_components[:, 1], c=kmeans.labels_, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering of Patients (More Than 2 Components)')
plt.show()