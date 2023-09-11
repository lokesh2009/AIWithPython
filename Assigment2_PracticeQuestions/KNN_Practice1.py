import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#
# # Load the data
# wine = load_wine()
# print(wine.data)
# print("Wine Data Shapes", wine.data.shape)
#
# # Print the target
# print(wine.target)
# print("Wine Target names", wine.target_names)
# print("Wine Feature names", wine.feature_names)
#
# X = wine.data
# y = wine.target
# Xtrain, Xtest, Ytrain, ytest = train_test_split(X, y, test_size=0.2)
# print("Xtrain Shape :", Xtrain.shape)
# print("Xtest Shape :", Xtest.shape)
#
# # KNN algorith apply for error and fix
#
# k = 3
# knn = KNeighborsClassifier(n_neighbors=k)
# knn.fit(Xtrain, Ytrain)
# yprediction = knn.predict(Xtest)
# print("Accuracy=", accuracy_score(ytest, yprediction))

#
# x_users = [[3, 9, 8, 5, 6, 100, 4, 3, 13,200,98,34,76,45]]
# y_users = knn.predict(x_users)
#
# print("Class belongs to Xusers", wine.target_names[y_users])


# Breast cancer related data

df = pd.read_csv(
    "/Users/lokeshsharma/Desktop/pythonProject/AIWithPython/TestData/sih.csv"
)
print(df.shape)

X_train, X_test, y_train, y_test = train_test_split(
    df.iloc[:, 1], df.iloc[:, 0], test_size=0.2, random_state=2
)
X_train.head
X_train.shape
print("Xtrain shape", X_test.shape)


# Apply KNN algo
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict
y_prid = knn.predict(X_test)
accuracy_score(y_test, y_pred=2)
