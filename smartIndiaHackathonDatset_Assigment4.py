import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

sih_data=pd.read_csv('/Users/lokeshsharma/Desktop/pythonProject/AIWithPython/TestData/sih.csv')

encoder=LabelEncoder()

for col in sih_data.columns:
  sih_data[col]=encoder.fit_transform(sih_data[col])

X_feature=sih_data.iloc[:,1:23]
y_label=sih_data.iloc[:,0]

scaler=StandardScaler()
X_feature=scaler.fit_transform(X_feature)

pca=PCA()
pca.fit_transform(X_feature)
pca_variance=pca.explained_variance_

plt.figure(figsize=(8, 6))
plt.bar(range(22), pca_variance, alpha=0.5, align='center', label='individual variance')
plt.legend()
plt.ylabel('Variance ratio')
plt.xlabel('Principal components')
plt.show()

