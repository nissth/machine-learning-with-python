import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
datas = pd.read_csv('prediction/support-vector-regression/maaslar.csv')

x = datas.iloc[:,1:2].values
y = datas.iloc[:,2:].values # returns a dataframe which is 2d, even though there is only one colums

# standardization
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
x_scaled = sc1.fit_transform(x)
sc2 = StandardScaler()
y_scaled = sc2.fit_transform(y)

# support vector regression
from sklearn.svm import SVR

svr = SVR(kernel = 'rbf') # radial basis function
# kernel{‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’} or callable, default=’rbf’
svr.fit(x_scaled, y_scaled.ravel()) # ravel turns the 2d array into 1d shape

plt.scatter(x_scaled, y_scaled)
plt.plot(x_scaled, svr.predict(x_scaled))
plt.show()

print(svr.predict([[11]]))
print(svr.predict([[6.6]]))