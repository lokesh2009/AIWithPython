
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


brestCancer_dataset=load_breast_cancer()


# Extracting Attribute
X=brestCancer_dataset.data

# Extracting Target / Class Labels
Y=brestCancer_dataset.target

# Train data
X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state = 50, test_size = 0.25)

# Creating Decision Tree Classifier

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)



# Predict Accuracy Score
y_pred = clf.predict(X_test)
print("Train data accuracy:",accuracy_score(y_true = y_train, y_pred=clf.predict(X_train)))
print("Test data accuracy:",accuracy_score(y_true = y_test, y_pred=y_pred))
