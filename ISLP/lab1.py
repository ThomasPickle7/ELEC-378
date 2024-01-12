import numpy as np

x = np.array([[[3,4,5],[2,3,4]], 
              [[1,9,3],[3,4,5]]])
y = np.array([4,9,7])
# print(np.sum(x))

x = np.array([1,2,3,4,5,6])
# print('beginning:\n', x)
x_reshape = x.reshape((3,2))
# print('reshaped:\n', x_reshape)

# print(x_reshape.shape, x_reshape.ndim, x_reshape.T)

print(x ** 2)
print(np.sqrt(x))
print(x ** 0.5)

x = np.random.normal(size=50)
print(x)