from sklearn.datasets import load_wine
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


wine = load_wine()
x = wine.data
y = wine.target
print("Winedataste size :", x.shape)
print("Wine data sheet size:", y.shape)

lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(x, y)

X_lda.shape


import matplotlib.pyplot as plt

plt.figure(figsize=[7, 5])
plt.scatter(X_lda[:0], X_lda[:1], c=y, s=25, cmap='plasma')
plt.title('LDA for Wine data with 2 component')
plt.xlabel('component1')
plt.ylabel('component2')
plt.savefig
