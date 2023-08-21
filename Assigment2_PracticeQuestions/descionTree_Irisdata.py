from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from  sklearn.tree import export_text

from SclearnDatasetIris import iris

iris_dataset=load_iris()

descion_tree=DecisionTreeClassifier(random_state=0,max_depth=2)
descion_tree=descion_tree.fit(iris.data,iris.target)
r=export_text(descion_tree,feature_names=iris["feature_names"])
print(r)

X = [[0, 0], [2, 2]]
y = [0.5, 2.5]
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X, y)
clf.predict([[1, 1]])
