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
from sklearn.cluster import KMeans

eaves_file = loadmat('ELEC-378\ELEC-378\HW3\eavesdropping.mat')
eaves_mat = eaves_file["Y"]

eaves_center = eaves_mat - eaves_mat.mean()

(eaves_u, eaves_s, eaves_vh) = np.linalg.svd(eaves_center)

print(eaves_vh.shape)

eaves_v_approx = eaves_vh[:2, :]

eaves_pca = np.dot(eaves_center, eaves_v_approx.T)

# part a)
# plt.scatter(eaves_pca[:, 0], eaves_pca[:, 1])
# plt.xlabel("Principal Component 1")
# plt.ylabel("Principal Component 2")
# plt.title("PCA Unclabeled Scatter Plot")
# plt.show()

#clustering
kmeans = KMeans(n_clusters=2, random_state=0)
labels = kmeans.fit_predict(eaves_pca)

#graph stuff
colors = np.choose(labels, ['#00ff00', '#ff0000'])
# part b)
plt.scatter(eaves_pca[:, 0], eaves_pca[:, 1], c=colors)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()

choices = np.choose(labels, [1,0])

reshaped = np.reshape(choices, (27,7))

message = ""
for i in reshaped:
    temp = ""
    for num in i:
        temp = temp + str(num)
    char = int(temp, 2)
    message = message + chr(char)

# part c)
 # print(message)
