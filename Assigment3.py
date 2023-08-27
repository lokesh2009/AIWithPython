import numpy as np
import matplotlib.pyplot as plt

# Define the data
input = np.array(
    [
        [7, 4, 3],
        [4, 1, 8],
        [6, 3, 5],
        [8, 6, 1],
        [8, 5, 7],
        [7, 2, 9],
        [5, 3, 3],
        [9, 5, 8],
        [7, 4, 5],
        [8, 2, 2],
    ]
)

# Standardize the data
mean = np.mean(input, axis=0)
std_dev = np.std(input, axis=0)
data_std = (input - mean) / std_dev

# Calculate the covariance matrix
cov_matrix = np.cov(data_std, rowvar=False)
# Calculate eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort eigenvalues and eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Choose the top k eigenvectors where k is the desired number of dimensions
k = 2  # You can choose the number of dimensions you want to reduce to
top_eigenvectors = eigenvectors[:, :k]

# Project the standardized data onto the new feature space
data_pca = np.dot(data_std, top_eigenvectors)

# Visualize the reduced-dimensional data using Matplotlib
plt.scatter(data_pca[:, 0], data_pca[:, 1])
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA Result")
plt.show()
