from sklearn.datasets import  make_blobs

# Generate sample data
centers =[[1,1],[-1,-1],[1,-1]]
X,label_true=make_blobs(n_samples=750,centers=centers,cluster_std=0.4,random_state=0)


