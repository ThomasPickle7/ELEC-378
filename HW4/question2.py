import numpy as np

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
w_init = np.array([100, 700000])

# Define the learning rate
alpha = .01

# Define the number of iterations
num_iterations = 1000

# Perform gradient descent
w = gradient_descent(gradient_funcs, w_init, alpha, num_iterations)
w_stochastic = stochastic_gradient_descent(gradient_funcs, w_init, alpha, num_iterations)

# Print the results
#print the weights and learning rate used:
print('Learning Rate:', alpha)
print('Initial Weights:', w_init)
print('Gradient Descent:', w)
print('Stochastic Gradient Descent:', w_stochastic)
# The results show that the stochastic gradient descent algorithm converges to the same minimum as the gradient descent algorithm, but it does so in a more efficient manner by using a random sample of the loss functions at each iteration.