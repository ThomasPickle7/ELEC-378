import numpy as np
import time
# Define the gradient of the loss functions
def grad_L1(w):
    return np.array([4*w[0] + w[1], w[0] - 8*w[1]])

def grad_L2(w):
    return np.array([6*w[0] + 4*w[1], 4*w[0] + 10*w[1]])

def grad_L3(w):
    return np.array([-2*w[0] - 4*w[1], -4*w[0] + 6*w[1]])

# Define gradient descent algorithm
def gradient_descent(gradient_funcs, w_init, alpha, num_iterations):
    w = w_init
    for i in range(num_iterations):
        gradient = np.zeros_like(w)
        for gradient_func in gradient_funcs:
            gradient += gradient_func(w)
        w = w - alpha * gradient
    return w
#print the runtime of the gradient_descent function



# Define stochastic gradient descent algorithm
def stochastic_gradient_descent(gradient_funcs, w_init, alpha, num_iterations):
    w = w_init
    for i in range(num_iterations):
        j = np.random.randint(0, len(gradient_funcs))
        gradient_func = gradient_funcs[j]
        gradient = gradient_func(w)
        w = w - alpha * gradient
    return w




# Define the gradient functions
gradient_funcs = [grad_L1, grad_L2, grad_L3]

# Define the initial weights
w_init = np.array([394, 400])

# Define the learning rate
alpha = .0000002

import matplotlib.pyplot as plt

# Define the number of iterations
num_iterations = 1000

# Initialize lists to store the weights for each iteration
weights_gradient = []
weights_stochastic = []

# Perform gradient descent
w = w_init
start_time = time.time()
for i in range(num_iterations):
    w = gradient_descent(gradient_funcs, w, alpha, 1)
    weights_gradient.append(w)
end_time = time.time()

# Calculate the runtime of gradient descent
gradient_descent_runtime = end_time - start_time

# Perform stochastic gradient descent
w_stochastic = w_init
start_time = time.time()
for i in range(num_iterations):
    w_stochastic = stochastic_gradient_descent(gradient_funcs, w_stochastic, alpha, 1)
    weights_stochastic.append(w_stochastic)
end_time = time.time()

# Calculate the runtime of stochastic gradient descent
stochastic_gradient_descent_runtime = end_time - start_time

# Convert the weights lists to numpy arrays
weights_gradient = np.array(weights_gradient)
weights_stochastic = np.array(weights_stochastic)

# Plot the results
plt.plot(range(num_iterations), weights_gradient[:, 0], label='Gradient Descent')
plt.plot(range(num_iterations), weights_stochastic[:, 0], label='Stochastic Gradient Descent')
plt.xlabel('Number of Iterations')
plt.ylabel('Weight')
plt.title('Stochastic Gradient Descent vs Gradient Descent, Alpha = ' + str(alpha))
plt.legend()
plt.show()

# Print the runtime of each algorithm
print("Gradient Descent Runtime:", gradient_descent_runtime)
print("Stochastic Gradient Descent Runtime:", stochastic_gradient_descent_runtime)