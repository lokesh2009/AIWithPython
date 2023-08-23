from sklearn.datasets import l
import matplotlib as plt
import numpy as np
import  skfuzzy as fuzz
from skfuzzy import control as ctrl


np.random.seed(0)
data=np.random.rand(100,2)

#develop clustering
n_clusters=3

cntr,u,u0,d,jm,p,fpc=fuzz.cluster.cmeans(data.T,n_clusters,2,error=0.005,maxiter=1000,init=None)
cluster_membership=np.argmax(u,axis=0)
print('Cluster centers',cntr)

print('Cluster Membership')

print('cluster membership',cluster_membership)