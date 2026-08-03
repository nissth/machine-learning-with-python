import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
datas = pd.read_csv('dimension-reduction/Wine.csv')
x = datas.iloc[:,0:13].values
y = datas.iloc[:,13].values

# splitting dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

# standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# pca
from sklearn.decomposition import PCA
pca = PCA(n_components=2) # number of components to keep

x_train2 = pca.fit_transform(x_train)
x_test2 = pca.transform(x_test)

from sklearn.linear_model import LogisticRegression

# lr before pca transformation
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

# lr after pca transformation
classifier2 = LogisticRegression(random_state=0)
classifier2.fit(x_train2,y_train)

y_pred = classifier.predict(x_test)
y_pred2 = classifier2.predict(x_test2)

from sklearn.metrics import confusion_matrix
print("actual / without pca")
cm = confusion_matrix(y_test,y_pred)
print(cm)

print("actual / with pca")
cm2 = confusion_matrix(y_test,y_pred2)
print(cm2)

print("without pca / with pca")
cm3 = confusion_matrix(y_pred,y_pred2)
print(cm3)

# lda
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)

x_train_lda = lda.fit_transform(x_train,y_train)
x_test_lda = lda.transform(x_test)

# lr after lda transformation
classifier_lda = LogisticRegression(random_state=0)
classifier_lda.fit(x_train_lda,y_train)

y_pred_lda = classifier_lda.predict(x_test_lda)

print("actual / with lda")
cm4 = confusion_matrix(y_pred,y_pred_lda)
print(cm4)