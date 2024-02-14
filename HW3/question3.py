import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the data from the 'cancer.mat' file
cancer = loadmat('ELEC-378\ELEC-378\HW3\cancer.mat')

# Extract features (X) and labels (Y) from the data
X = cancer['X']
Y = cancer['Y']

# Center the feature matrix
X_c = X - X.mean()

# Perform Singular Value Decomposition (SVD) on the centered feature matrix
X_u, X_s, X_vh = np.linalg.svd(X)

# Select the first two principal components
X_va = X_vh[:2, :]

# Project the centered feature matrix onto the first two principal components
X_pca = np.dot(X_c, X_va.T)

# Plot the PCA scatter plot with labels
plt.scatter(X_pca[:, 0], X_pca[:, 1])
for i, tag in enumerate(Y):
    plt.annotate(tag[0][0], (X_pca[i, 0], X_pca[i, 1]))
plt.title("Types of Cancer")
plt.show()

# Plot the PCA scatter plot with color-coded labels based on cancer type
for i, tag in enumerate(Y):
    lower_tag = str(tag[0][0]).lower().replace(" ", "")
    color = 'red' if lower_tag == 'melanoma' else 'blue'
    plt.annotate(tag[0][0], (X_pca[i, 0], X_pca[i, 1]))
    plt.scatter(X_pca[i, 0], X_pca[i, 1], color=color)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering of Patients (More Than 2 Components)')
plt.show()

# Sort the columns of X_vh to visualize the most relevant gene expressions
sorter = np.argsort(X_vh[0])
num_columns = min(100, X_vh.shape[1])

# Plot the heatmap of X after sorting columns
plt.imshow(X[:, sorter[:num_columns]], aspect='auto', cmap='viridis')
plt.colorbar()  # Add color bar for reference
plt.xlabel('Gene Expressions')
plt.ylabel('Cancer Diagnoses')
plt.title('Heatmap of X after Sorting Columns')
plt.show()

# (d) Demonstrate K-means clustering with more than two principal components retained
pca = PCA(n_components=32)  # Retaining more components
X_pca = pca.fit_transform(X)

# Perform K-means clustering
kmeans = KMeans(n_clusters=14, random_state=0)
labels = kmeans.fit_predict(X_pca)

# Visualize clustering results
for i, tag in enumerate(Y):
    plt.annotate(tag[0][0], (X_pca[i, 0], X_pca[i, 1]))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering of Cancer Patients (More Than 2 Components)')
plt.show()
