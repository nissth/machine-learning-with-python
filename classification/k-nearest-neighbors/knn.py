import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
datas = pd.read_csv('classification/veriler.csv')

x = datas.iloc[:,1:4].values # independent values
y = datas.iloc[:,4:].values # dependent value

# splitting dataset into train and test
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

# standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# k nearest neighbors
from sklearn.neighbors import KNeighborsClassifier
#knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski')
knn = KNeighborsClassifier(n_neighbors=1,metric='minkowski') # reducing the neighbors which are going to be looked at can impove the accuracy of the model
knn.fit(x_train,y_train.ravel())

y_pred = knn.predict(x_test)
print(y_pred)
print(y_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)