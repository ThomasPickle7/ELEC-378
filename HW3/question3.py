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

for i, tag in enumerate(Y):
    lower_tag = str(tag[0][0]).lower().replace(" ", "")
    color = 'red' if lower_tag == 'melanoma' else 'blue'
    plt.annotate(tag[0][0], (X_pca[i, 0], X_pca[i, 1]))
    plt.scatter(X_pca[i, 0], X_pca[i, 1], color=color)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('K-means Clustering of Patients (More Than 2 Components)')
plt.show()

sorter = np.argsort(X_vh[0])

num_columns = min(100, X_vh.shape[1])
plt.imshow(X[:, sorter[:num_columns]], aspect='auto', cmap='viridis')
plt.colorbar()  # Add color bar for reference
plt.xlabel('Gene Expressions')
plt.ylabel('Cancer Diagnoses')
plt.title('Heatmap of X after Sorting Columns')
plt.show()