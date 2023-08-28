from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt

import numpy as np

x = np.array([[1, 2], [3, 4], [4, 5], [8, 7], [7, 8], [25, 80]])
print(x)
plt.scatter(x[:, 0], x[:, 1])


# Clustering
clustering = DBSCAN(eps=3, min_samples=2).fit(x)

#
clustering.labels_
