import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
datas = pd.read_csv('model-evaluation/Social_Network_Ads.csv')
x = datas.iloc[:,[2,3]].values
y = datas.iloc[:,4].values

# splitting dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

# standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# k-fold cross validation
from sklearn.model_selection import cross_val_score
cvs = cross_val_score(estimator= classifier, X=x_train, y=y_train, cv=4) 
# cv = cross-validation splitting strategy, in tis example dataset splits into 4 parts
print(cvs.mean())
print(cvs.std())