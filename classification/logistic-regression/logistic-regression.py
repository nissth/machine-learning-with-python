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

print(x_train)
print(' ')
print(x_test)

# logistic regression
from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression(random_state=0)
lgr.fit(x_train,y_train.ravel())

y_pred = lgr.predict(x_test)
print(y_pred)
print(y_test)

# confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)