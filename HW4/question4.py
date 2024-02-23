import scipy.io as sc
import numpy as np
from scipy import signal, linalg
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as im
import matplotlib.pyplot as plt

file = sc.loadmat('ELEC-378/HW4/CS.mat')
y = file['y']
phi = file['Phi']
psi = file['Psi']
#Compute the inverse of phi*psi


# Calculate s using the equation y = Phi * Psi * s
s = np.linalg.inv(phi @ psi) @ y

# Calculate the image x using x = Psi * s
x = psi @ s

# Reshape x into a 64x64 image
x_image = np.reshape(x, (64, 64))

# Plot the image
plt.imshow(x_image)
plt.title('Ground Truth Image')
plt.show()

# select the same 2048 random entries from y and phi

# Randomly select 2048 indices
rng = np.random.default_rng()

indices = np.sort(rng.choice(4096, 2048, replace=False))
# Select the corresponding rows from y and phi
y_c = y[indices]
phi_c = phi[indices]

#solve y_c = phi_c*psi*s  for s using ridge regression

#solve for s_ridge with a variable parameter lmabda
from sklearn.linear_model import Ridge
# Create a ridge regression model
ridge = Ridge(alpha=.0004)

# Fit the model to the data
ridge.fit(phi_c @ psi, y_c)

# Calculate s using the model
s_ridge = ridge.coef_[0]

# Calculate the image x using x = Psi * s
x_ridge = psi @ s_ridge

# Reshape x into a 64x64 image
x_image_ridge = np.reshape(x_ridge, (64, 64))

# Plot the image
plt.imshow(x_image_ridge)
plt.title('Ridge Regression Reconstructed Image')
plt.show()

#calculate the recovery using LASSO with a variable parameter lambda
from sklearn.linear_model import Lasso
# Create a Lasso model
lasso = Lasso(alpha=.0004)

# Fit the model to the data
lasso.fit(phi_c @ psi, y_c)

# Calculate s using the model
s_lasso = lasso.coef_

# Calculate the image x using x = Psi * s
x_lasso = psi @ s_lasso

# Reshape x into a 64x64 image
x_image_lasso = np.reshape(x_lasso, (64, 64))

# Plot the image
plt.imshow(x_image_lasso)
plt.title('LASSO Reconstructed Image')
plt.show()


#construct a k-sparse vector s using the ground truth image, where only entries with magnitude greater than 15 are kept
s_sparse = s
s_sparse[np.abs(s_sparse) < 15] = 0

# Calculate the image x using x = Psi * s
x_sparse = psi @ s_sparse

# Reshape x into a 64x64 image
x_image_sparse = np.reshape(x_sparse, (64, 64))

# Plot the image
plt.imshow(x_image_sparse)
plt.title('Sparse Reconstructed Image')
plt.show()

#keep 2*K of the indices from earlier, without replacement
K = np.count_nonzero(s_sparse)
print(K)
sparse_indices = indices[np.sort(rng.choice(len(indices), 2*K, replace=False))]

y_c_sparse = y[sparse_indices]
phi_c_sparse = phi[sparse_indices]
#solve y_c = phi_c*psi*s  for s using ridge regression

#solve for s_ridge with a variable parameter lmabda
from sklearn.linear_model import Ridge
# Create a ridge regression model
ridge_sparse = Ridge(alpha=.0004)

# Fit the model to the data
ridge_sparse.fit(phi_c_sparse @ psi, y_c_sparse)

# Calculate s using the model
s_ridge_sparse = ridge_sparse.coef_[0]

# Calculate the image x using x = Psi * s
x_ridge_sparse = psi @ s_ridge_sparse

# Reshape x into a 64x64 image
x_image_ridge_sparse = np.reshape(x_ridge_sparse, (64, 64))

# Plot the image
plt.imshow(x_image_ridge_sparse)
plt.title('Sparse Ridge Regression Reconstructed Image')
plt.show()

#calculate the recovery using LASSO with a variable parameter lambda

# Create a Lasso model
lasso_sparse = Lasso(alpha=.0004)

# Fit the model to the data

lasso_sparse.fit(phi_c_sparse @ psi, y_c_sparse)

# Calculate s using the model
s_lasso_sparse = lasso_sparse.coef_

# Calculate the image x using x = Psi * s
x_lasso_sparse = psi @ s_lasso_sparse

# Reshape x into a 64x64 image

x_image_lasso_sparse = np.reshape(x_lasso_sparse, (64, 64))

# Plot the image
plt.imshow(x_image_lasso_sparse)
plt.title('Sparse LASSO Reconstructed Image')
plt.show()



#save the images
im.imsave('ground_truth.png', x_image)
im.imsave('ridge_reconstruction.png', x_image_ridge)
im.imsave('lasso_reconstruction.png', x_image_lasso)
im.imsave('sparse_reconstruction.png', x_image_sparse)
im.imsave('sparse_ridge_reconstruction.png', x_image_ridge_sparse)
im.imsave('sparse_lasso_reconstruction.png', x_image_lasso_sparse)



