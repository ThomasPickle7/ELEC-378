# Import necessary libraries
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
import numpy as np
from sklearn.cluster import KMeans

# Load the data from the .mat file
eaves_file = loadmat('ELEC-378\ELEC-378\HW3\eavesdropping.mat')
eaves_mat = eaves_file["Y"]

# Center the data by subtracting the mean
eaves_center = eaves_mat - eaves_mat.mean()

# Singular Value Decomposition (SVD) to perform PCA
(eaves_u, eaves_s, eaves_vh) = np.linalg.svd(eaves_center)

# Print the shape of the right singular vectors matrix
print(eaves_vh.shape)

# Select the first two right singular vectors for PCA
eaves_v_approx = eaves_vh[:2, :]

# Project the centered data onto the first two principal components
eaves_pca = np.dot(eaves_center, eaves_v_approx.T)

# part a) Plot the PCA scatter plot without labels
plt.scatter(eaves_pca[:, 0], eaves_pca[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Unclabeled Scatter Plot")
plt.show()

# Clustering using KMeans with 2 clusters
kmeans = KMeans(n_clusters=2, random_state=0)
labels = kmeans.fit_predict(eaves_pca)

# Assign colors to clusters
colors = np.choose(labels, ['#00ff00', '#ff0000'])

# part b) Plot the PCA scatter plot with cluster labels
plt.scatter(eaves_pca[:, 0], eaves_pca[:, 1], c=colors)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

# Convert cluster labels to binary choices
choices = np.choose(labels, [1, 0])

# Reshape the binary choices to match the structure of the binary message
reshaped = np.reshape(choices, (27, 7))

# Decode the binary message
message = ""
for i in reshaped:
    temp = ""
    for num in i:
        temp = temp + str(num)
    char = int(temp, 2)
    message = message + chr(char)

# part c) Print the decoded message
print(message)