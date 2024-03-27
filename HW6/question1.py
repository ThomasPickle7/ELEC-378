import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# Load the data
train1 = loadmat('Train1.mat')
train2 = loadmat('Train2.mat')

X = train2["X"]
Y = train2["y"]

# X : data matrix of (n,p), Y are the labels of the data
# T : number of iterations
# ws[i], bs[i] : weight and bias at iteration i

# Initialize w and b to random alues and plot the resulting (random) hyperplane (a straight line since p = 2) on top of the scatter plot of the train1.mat data.

#initialize w0 to be a random vector of length p
w0 = np.random.rand(X.shape[1])

b0 = np.random.rand()


# Number of iterations
T = 1000


# ws and bs are the weight and bias and iteration
ws = np.zeros((T, X.shape[1]))

bs = np.zeros(T)

#ws and bs are the length of the number of iterations, or T.

ws[0] = w0

bs[0] = b0


# parameters n, p, lamb (da), and mu
n = X.shape[0]
p = X.shape[1]
lamb = 0.1
mu = 0.1


def plot_hyperplane(x_axis, w, b, c='k', label="Hyperplane"):
    slope = -w[0] / w[1]
    intercept = -b / w[1]
    plt.plot(x_axis, slope * x_axis + intercept, c=c, label=label)

# create the x-axis for the plot
x_axis = np.linspace(np.min(X[:,0]), np.max(X[:,0]), n)


# Plot the data
plt.figure()
plt.scatter(X[:,0], X[:,1], c=Y)
plot_hyperplane(x_axis, ws[0], bs[0], c='r', label="Initial Hyperplane")
plt.show()

# set up your gradient descent code such that it redraws the decision hyperplane each iteration of training. Run your code to convergence on Train1.mat, showing the projression of hyperplanes ffor lambda = 1. discuss the convergence in particulat to the ultimate support vectors. Try with a few different initializations for w and b.

for i in range(T - 1):
    # initialize the gradient as zeros
    sw = np.zeros(p)
    sb = 0;
    # iter


