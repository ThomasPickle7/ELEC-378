import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.image as im

# Load the image
image = im.imread('ELEC-378\ELEC-378\HW3\objection.png')

# Display the original image
plt.imshow(image)
plt.title('Original Image')
plt.show()

# Flatten the image into an array of RGB values
X = image.flatten().reshape(-1, 3)

# Run PCA
pca = PCA(n_components=2)

# Project pixels into 2D space
pixels_transformed = pca.fit_transform(X)

# Plot pixels in 2D space, with each pixel having its original color
plt.scatter(pixels_transformed[:, 0], pixels_transformed[:, 1], c=X)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

def kmeans(X, k=3, max_iterations=100):
    '''
    K-means clustering algorithm.

    Args:
    X: multidimensional data (ndarray)
    k: number of clusters (int)
    max_iterations: number of repetitions before clusters are established (int)

    Returns:
    P: an np.array containing class of each data point
    centroids: an np.array containing the centroid of each class
    '''

    # Convert data to numpy array if necessary
    X = np.array(X)

    # Initialize centroids by picking k random points from the data
    np.random.seed(0)  # For reproducibility
    centroids_indices = np.random.choice(X.shape[0], size=k, replace=False)
    centroids = X[centroids_indices]

    for _ in range(max_iterations):
        # Step 3: Find class (P) of each data point using Euclidean distance
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        P = np.argmin(distances, axis=0)

        # Step 4: Stop when max_iteration is reached or P matrix doesn't change
        new_centroids = np.array([X[P == i].mean(axis=0) for i in range(k)])
        if np.array_equal(centroids, new_centroids):
            break
        centroids = new_centroids

    return P, centroids

# Number of clusters for k-means
K = 2 ** 6

# Perform k-means clustering
labels, centroids = kmeans(X, k=K, max_iterations=100)

# Create a color-quantized image using the centroids found by k-means
color_quantized_data_matrix = np.vstack(centroids[labels])

# Display the color-quantized image
plt.imshow(color_quantized_data_matrix.reshape(image.shape))
plt.savefig('ELEC-378\ELEC-378\HW3\color_quantized_image.png')
plt.show()

# Perform k-means clustering using sklearn's KMeans
kmeans = KMeans(n_clusters=K, random_state=0).fit(X)

# Get the labels and cluster centers from sklearn's KMeans
labels = kmeans.predict(X)
kmeans_flat = kmeans.cluster_centers_[labels]

# Display the color-quantized image obtained from sklearn's KMeans
plt.imshow(kmeans_flat.reshape(image.shape))
plt.show()

# Plot the pixels in 2D space, using the color quantized data matrix
plt.scatter(pixels_transformed[:, 0], pixels_transformed[:, 1], c=color_quantized_data_matrix)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
