import matplotlib.pyplot as plt
import numpy as np

input=np.array([[7,4,3],
                [4,1,8],
                [6,3,5],
                [8,6,1],
                [8,5,7],
                [7,2,9],
                [5,3,3],
                [9,5,8],
                [7,4,5],
                [8,2,2]])
print("Original Input",input)

plt.title("Actual Data set")
plt.scatter(input[:,0],input[:,1],input[:,2])

# Calculate the Mean
adjusted_input=np.mean(input,axis=0)
print("Mean adjusted input",adjusted_input)
plt.scatter(adjusted_input[:,0])

#Calculate standard deviation
Adjusted_std_dev=np.std(input,axis=0)
print("Adjusted standard input",Adjusted_std_dev)


data_std = (input - adjusted_input) / Adjusted_std_dev

# Calculate the Covarinace
cov=np.cov(data_std,rowvar=False)
print(cov)

#EigenValue and #Eigen Vectors
from numpy import linalg as LA
w,v=LA.eig(cov)
print("Eigen values")
print(w)
print("Eigen vectors")
print(v)

# Value

sorted_indices = np.argsort(w)[::-1]
eigenvalues = v[sorted_indices]
eigenvectors = v[:, sorted_indices]

#Project the Data points
eigen_input=np.dot(adjusted_input,np.array(v.T[1]))
print(eigen_input.reshape(10,1))