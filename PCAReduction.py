import numpy as np
import matplotlib.pyplot as plt


input1=np.array([2.5, 2.4], [0.5, 0.7], [2.2, 2.9], [1.9, 2.2], [3.1, 3.0], [2.3, 2.7], [2, 1.6],
                [1, 1.1], [1.5, 1.6], [1.1, 0.9])
print("original input :", input1)
plt.title("Actual Dataset")
plt.scatter(input1[:, 0], input1[:, 1])

adjusted_input= input1 - np.mean(input1, axis=0)
print("mean adjested input",adjusted_input)
plt.scatter(adjusted_input[:,0],adjusted_input[:,1])

# Calculate covarinace Matrix
cov=np.cov(adjusted_input[:,0],adjusted_input[:,1])
print(cov)

# Calculate EigenValue and Eigen vectors

from numpy import linalg as LA
w,v=LA.eig(cov)
print("Eigen Values")
print(w)
print("Eigen vectors")
print(v)


eigen_input = np.dot(adjusted_input,np.array(v.T[1]))
print(eigen_input)

from sklearn.decomposition import PCA
pca=PCA(n_components=1)
pca.fit(input1)
print(pca.transform(input1))

# try PCA for Wine data set
