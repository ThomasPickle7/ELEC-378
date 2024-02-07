import numpy as np
import math
dimension = 2
samples = 5000000
vectors = np.random.rand(samples, dimension)
count = 0        

for vector in vectors:
    
    if (np.linalg.norm(vector, ord = 2) <= 1):
        count += 1

print((count/samples))