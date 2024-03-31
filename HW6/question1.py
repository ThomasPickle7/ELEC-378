import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt


# Load in the data
train1 = loadmat('train1.mat')
train2 = loadmat('train2.mat')

X = train1['X']
Y = train1['y']

T = 100
# Initialize random weights and data
w0 = np.random.uniform(-1, 1, 2)
b0  = np.random.rand()
ws = np.zeros((T, 2))
bs = np.zeros(T)

ws[0] = w0
bs[0] = b0


# Plot the initial hyperplane
def plot_hyperplane(x_axis, w, b, c = 'k'):
    m = - w[0] / w[1]
    b = - b / w[1]
    plt.plot(x_axis, m * x_axis + b, c)


# Setting up the x-axis
x_axis = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
# a) Plot the initial hyperplane
plt.figure()
plt.ylim(-3, 5)
plt.style.use('fivethirtyeight')  
plt.title("initial hyperplane")  
plt.scatter(X[:, 0], X[:, 1], c = Y)
plot_hyperplane(x_axis, ws[0], bs[0])
plt.savefig("initial_hyperplane.png")
plt.show()



# Gradient descent alg
def gradient_descent(T, p, n, lamb, mu, ws, bs, X, Y): 
    for t in range(1,T):
        # Initialize ws and bs
        sw = np.zeros(p)
        sb = 0
        for i in range(n):
            # initialize x and y to a point
            x = X[i]
            y = Y[i]
            # Check if the point is misclassified
            if y * (np.dot(ws[t - 1], x) + bs[t - 1]) <= 1:
                # If it is, update the weights and bias
                siw = - y * x
                sib = - y
            else:
                # If it's not, set the weights and bias to 0
                siw = 0
                sib = 0
            # Update the weights and bias
            sw += siw
            sb += sib
        ws[t] = ws[t - 1] - mu * (sw + 2 * lamb * ws[t - 1])
        bs[t] = bs[t - 1] - mu * sb * bs[t - 1]

# used for plotting descended hyperplanes
def plot_hyperplane(x, w, b, color):
    y = - (w[0] * x + b) / w[1]
    plt.plot(x, y, color)

# Initializing params
# T: number of iterations
T = 100
# p: number of features
p = 2
# n: number of data points
n = 100
# lamb: lambda
lamb = 1
# mu: learning rate
mu = 0.01
# b) setting 
# Re-initialize ws and bs
ws = np.random.randn(T, p)
bs = np.random.randn(T)
gradient_descent(T, p, n, lamb, mu, ws, bs, X, Y)

# Plot the descended hyperplanes
plt.figure()
plt.title("Gradient-descended, random initialization")
plt.scatter(X[:, 0], X[:, 1], c = Y)
for t in range(T):
    plot_hyperplane(x_axis, ws[t], bs[t], 'cornflowerblue')
plt.savefig("gradient_descent_random.png")
plt.show()





# Re-initialize ws and bs differently
ws = np.zeros((T, 2))
bs = np.zeros(T)
ws[0] = w0
bs[0] = b0

plt.figure()
plt.title("Gradient-descended, zero initializations")
gradient_descent(T, p, n, lamb, mu, ws, bs, X, Y)
plt.scatter(X[:, 0], X[:, 1], c = Y)
colors = plt.cm.RdYlBu(np.linspace(0, 1, T + 1))
for t in range(T):
    plot_hyperplane(x_axis, ws[t], bs[t], colors[t])

# save an image of the plot
plt.savefig("gradient_descent.png")
plt.show()








# c) running the gradient descent algorithm with different values of lambda

lamb = 10
# Re-initialize ws and bs differently
ws = np.zeros((T, 2))
bs = np.zeros(T)
ws[0] = w0
bs[0] = b0

plt.figure()
plt.title("Gradient-descended, lambda = 10")
gradient_descent(T, p, n, lamb, mu, ws, bs, X, Y)
plt.scatter(X[:, 0], X[:, 1], c = Y)
colors = plt.cm.RdYlBu(np.linspace(0, 1, T + 1))
for t in range(T):
    plot_hyperplane(x_axis, ws[t], bs[t], colors[t])
plt.savefig("gradient_descent_lambda_10.png")
plt.show()



lamb = .1
# Re-initialize ws and bs
ws = np.zeros((T, 2))
bs = np.zeros(T)
ws[0] = w0
bs[0] = b0

plt.figure()
plt.title("Gradient-descended, lambda = .1")
gradient_descent(T, p, n, lamb, mu, ws, bs, X, Y)
plt.scatter(X[:, 0], X[:, 1], c = Y)
colors = plt.cm.RdYlBu(np.linspace(0, 1, T + 1))
for t in range(T):
    plot_hyperplane(x_axis, ws[t], bs[t], colors[t])
plt.savefig("gradient_descent_lambda_0.1.png")
plt.show()


lamb = 1
# Re-initialize ws and bs
ws = np.zeros((T, 2))
bs = np.zeros(T)
ws[0] = w0
bs[0] = b0

plt.figure()
plt.title("Gradient-descended, lambda = 1")
gradient_descent(T, p, n, lamb, mu, ws, bs, X, Y)
plt.scatter(X[:, 0], X[:, 1], c = Y)
colors = plt.cm.RdYlBu(np.linspace(0, 1, T + 1))
for t in range(T):
    plot_hyperplane(x_axis, ws[t], bs[t], colors[t])
plt.savefig("gradient_descent_lambda_1.png")
plt.show()


# d) implementing stochastic gradient descent
 
def stochastic_gradient_descent(T, p, n, lamb, mu, ws, bs, X, Y):
    for t in range(1,T):
        sw = np.zeros(p)
        sb = 0
        i = np.random.randint(n)
        x = X[i]
        y = Y[i]
        if y * (np.dot(ws[t], x) + bs[t]) <= 1:
            siw = - y * x
            sib = - y
        else:
            siw = 0
            sib = 0
        sw += siw
        sb += sib
        ws[t] = ws[t - 1] - mu * ( sw + 2 * lamb * ws[t - 1])
        bs[t] = bs[t - 1] - mu * sb * bs[t - 1]
    


# Stochastic gradient descent
ws = np.zeros((T, 2))
bs = np.zeros(T)
ws[0] = w0
bs[0] = b0
stochastic_gradient_descent(T, 2, 100, 1, 0.01, ws, bs, X, Y)
plt.figure()
plt.title("Stochastic gradient-descended hyperplanes")
plt.scatter(X[:, 0], X[:, 1], c = Y)
colors = plt.cm.RdYlBu(np.linspace(0, 1, T + 1))
for t in range(T):
    plot_hyperplane(x_axis, ws[t], bs[t], colors[t])
plt.savefig("stochastic_gradient_descent.png")
plt.show()






# Import training data 2
X = train2['X']
Y = train2['y']
w0 = np.random.uniform(-1, 1, 2)
b0  = np.random.rand()

# Initialize params
ws = np.zeros((T, 2))
bs = np.zeros(T)
ws[0] = w0
bs[0] = b0
gradient_descent(T, p, n, lamb, mu, ws, bs, X, Y)

# Plot the hyperplanes
x_axis = np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), 100)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c = Y)
plt.title("Gradient-descended on train2, random initialization")
colors = plt.cm.RdYlBu(np.linspace(0, 1, T + 1))
for t in range(T):
    plot_hyperplane(x_axis, ws[t], bs[t], colors[t])
plt.savefig("gradient_descent_train2_random.png")
plt.show()




# Plots the final hyperplane
plt.figure()
plt.title("Final hyperplane on train2")
plt.scatter(X[:, 0], X[:, 1], c = Y)
colors = plt.cm.RdYlBu(np.linspace(0, 1, T + 1))
t = T - 1
plot_hyperplane(x_axis, ws[t], bs[t], colors[t])
plt.savefig("final_hyperplane_train2.png")
plt.show()




# e) finding the misclassified points and calculating the error

# calculates misclassified points
misclassified = [1 for i in range(X.shape[0]) if np.sign(np.dot(ws[t], X[i]) + bs[t]) != Y[i]]
num_misclassified = sum(misclassified)
error = (num_misclassified / X.shape[0]) * 100

print("number of misclassified points for p = 2: ", num_misclassified)
print("Error: ", error, "%")


# f) Remapping the data to p = 3 via a non-linear function
# Define the nonlinear function
def phi(x):
    return np.array([x[0], x[1], x[0] ** 2 + x[1] ** 2])

# Remap the data
X_remapped = np.array([phi(x) for x in X])

# g) Repeat the gradient descent algorithm on the remapped data. Plot the final hyperplane
# in the original feature space.
ws = np.zeros((T, 3))
bs = np.zeros(T)
ws[0] = np.random.uniform(-1, 1, 3)
bs[0] = np.random.rand()
gradient_descent(T, 3, 100, 1, 0.01, ws, bs, X_remapped, Y)

# plot the final hyperplane
fig = plt.figure()  
ax = fig.add_subplot(projection='3d')
ax.scatter(X_remapped[:, 0], X_remapped[:, 1], X_remapped[:, 2], c = Y)
colors = plt.cm.RdYlBu(np.linspace(0, 1, T + 1))
ax.set_title("Remapped data to p = 3")
t = T - 1
# AAAAAAAAH 3 HOURS TO PLOT THIS STUPID HYPERPLANE PLEASE GIVE ME CREDIT
x = np.linspace(np.min(X_remapped[:, 0]), np.max(X_remapped[:, 0]), 100)
y = np.linspace(np.min(X_remapped[:, 1]), np.max(X_remapped[:, 1]), 100)
x, y = np.meshgrid(x, y)
z = (- ws[t][0] * x - ws[t][1] * y - bs[t]) / ws[t][2]
ax.plot_surface(x, y, z, alpha = 0.5)
plt.savefig("final_hyperplane_p3.png")
plt.show()




# calculating the misclassification rate for the remapped data
misclassified = [1 for i in range(X.shape[0]) if np.sign(np.dot(ws[T - 1], X_remapped[i]) + bs[T - 1]) != Y[i]]
num_misclassified = sum(misclassified)
error = (num_misclassified / X.shape[0]) * 100

print("number of misclassified points for p = 3: ", num_misclassified)
print("Error: ", error, "%")
