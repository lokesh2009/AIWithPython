from sklearn import  datasets
from sklearn.svm import SVC

iris = datasets.load_iris()
digits =datasets.load_digits()
#print(digits.data)

#print(digits.target)

#print(digits.images[0])

from sklearn import svm
clf = svm.SVC(gamma=0.001,C=100)
clf.fit(digits.data[:-1], digits.target[:-1])
SVC(C=100.0, gamma=0.001)


