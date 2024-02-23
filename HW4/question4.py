import scipy.io as sc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
from sklearn.linear_model import Lasso, Ridge


alpha = .99

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

# Randomly select 2048 indices
rng = np.random.default_rng()

indices = np.sort(rng.choice(4096, 2048, replace=False))
# Select the corresponding rows from y and phi
y_c = y[indices]
phi_c = phi[indices]


# Create a ridge regression model
ridge = Ridge(alpha=alpha)

# Fit the model to the data
ridge.fit(phi_c @ psi, y_c)
s_ridge = ridge.coef_[0]
x_ridge = psi @ s_ridge

# Reshape x into a 64x64 image
x_image_ridge = np.reshape(x_ridge, (64, 64))
plt.imshow(x_image_ridge)

# Create a Lasso model
lasso = Lasso(alpha=alpha)

# Fit the model to the data
lasso.fit(phi_c @ psi, y_c)
s_lasso = lasso.coef_
x_lasso = psi @ s_lasso

x_image_lasso = np.reshape(x_lasso, (64, 64))
plt.imshow(x_image_lasso)

#construct a k-sparse vector s using the ground truth image, where only entries with magnitude greater than 15 are kept
s_sparse = s
s_sparse[np.abs(s_sparse) < 15] = 0

# Calculate the image x using x = Psi * s
x_sparse = psi @ s_sparse

# Reshape x into a 64x64 image
x_image_sparse = np.reshape(x_sparse, (64, 64))
plt.imshow(x_image_sparse)

#keep 2*K of the indices from earlier, without replacement
K = np.count_nonzero(s_sparse)
sparse_indices = indices[np.sort(rng.choice(len(indices), 2*K, replace=False))]

y_c_sparse = y[sparse_indices]
phi_c_sparse = phi[sparse_indices]

#solve for s_ridge with a variable parameter lmabda

# Create a ridge regression model
ridge_sparse = Ridge(alpha=alpha)

# Fit the model to the data
ridge_sparse.fit(phi_c_sparse @ psi, y_c_sparse)
s_ridge_sparse = ridge_sparse.coef_[0]
x_ridge_sparse = psi @ s_ridge_sparse

# Reshape x into a 64x64 image
x_image_ridge_sparse = np.reshape(x_ridge_sparse, (64, 64))
plt.imshow(x_image_ridge_sparse)

#calculate the recovery using LASSO with a variable parameter lambda

# Create a Lasso model
lasso_sparse = Lasso(alpha=alpha)

lasso_sparse.fit(phi_c_sparse @ psi, y_c_sparse)
s_lasso_sparse = lasso_sparse.coef_
x_lasso_sparse = psi @ s_lasso_sparse

x_image_lasso_sparse = np.reshape(x_lasso_sparse, (64, 64))
plt.imshow(x_image_lasso_sparse)

#save the images to the current directory
im.imsave('ELEC-378/HW4/ground_truth.png', x_image)
im.imsave('ELEC-378/HW4/ridge_reconstruction.png', x_image_ridge)
im.imsave('ELEC-378/HW4/lasso_reconstruction.png', x_image_lasso)
im.imsave('ELEC-378/HW4/sparse_reconstruction.png', x_image_sparse)
im.imsave('ELEC-378/HW4/sparse_ridge_reconstruction.png', x_image_ridge_sparse)
im.imsave('ELEC-378/HW4/sparse_lasso_reconstruction.png', x_image_lasso_sparse)



