import sklearn
from keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from scipy.spatial.distance import cdist
from scipy.stats import mode
from sklearn.model_selection import train_test_split

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, shuffle=False)

def knearest(k, X_train, y_train, X_test, metric):
    if X_test.ndim < 2:
        X_test = [X_test]
    distances = cdist(X_test, X_train, metric)
    nearest = np.argsort(distances, axis=1)[:, :k]
    y_nearest = y_train[nearest]
        # return the 1d array of the most common element in each row
    return mode(y_nearest, axis=1)[0].ravel()

misclassification_rates = {'euclidean': [], 'cityblock': [], 'chebyshev': []}
for k in range(1, 21):
    for metric in misclassification_rates.keys():  
        y_pred = knearest(k, X_train, y_train, X_test, metric)
        misclassification_rate = np.mean(y_pred != y_test)
        misclassification_rates[metric].append(misclassification_rate)
        print(f'k={k}, misclassification rate={misclassification_rate}')
# Plot the misclassification rates
plt.plot(range(1, 21), misclassification_rates['euclidean'], label='euclidean')
plt.plot(range(1, 21), misclassification_rates['cityblock'], label='cityblock')
plt.plot(range(1, 21), misclassification_rates['chebyshev'], label='chebyshev')
plt.ylabel('Misclassification rate')
plt.legend()
plt.show()
