from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

dataset=load_wine
print(dataset.shape)

#Train the data set
X_train,Xtest,Y_train,y_test=train_test_split(dataset.data,dataset.target,test_size=0.30,random_state=45)


from sklearn import tree
classifier =tree.DecisionTreeClassifier()
classifier=classifier.fit(X_train,Y_train)


prediction=classifier.predict(Xtest)
print("Accuracy :",accuracy_score(prediction))