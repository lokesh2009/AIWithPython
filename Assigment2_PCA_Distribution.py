import matplotlib.pyplot as plt
import numpy as np

input = np.array(
    [
        [2.5, 2.4],
        [0.5, 0.4],
        [2.2, 2.9],
        [1.9, 2.2],
        [3.1, 3.0],
        [2.3, 2.7],
        [2, 1.6],
        [1, 1.1],
        [1.5, 1.6],
        [1.1, 0.9],
    ]
)
print("Original Input", input)
plt.title("Actual Dataset")
plt.scatter(input[:, 0], input[:, 1])

# Find out Mean and cov
adjusted_Mean = input - np.mean(input, axis=0)
print("Mean Adjusted input", adjusted_Mean)

plt.scatter(adjusted_Mean[:, 0], adjusted_Mean[:, 1])


cov = np.cov(adjusted_Mean[:, 0], adjusted_Mean[:, 1])
print(cov)

# Eigen values and Eigen vectors
from numpy import linalg as LA

e, w = LA.eig(cov)

print("Eigen values", e)
print("Eigen vector", w)

# eigen_input=np.dot(w.transpose(),adjusted_Mean.transpose()).transpose()
eigen_input = np.dot(adjusted_Mean, np.array(e.T[1]))
print("Eigen Inputs :", eigen_input.reshape(20, 1))


# Analysis PCA principal component analysis
# Alternative options directly
from sklearn.decomposition import PCA

pca = PCA(n_components=1)
pca.fit(input)
print("PCA Transform", pca.transform(input))
