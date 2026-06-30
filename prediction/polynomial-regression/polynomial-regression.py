import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
datas = pd.read_csv('prediction/polynomial-regression/maaslar.csv')

x = datas.iloc[:,1:2].values
y = datas.iloc[:,2:].values
#print(x)
#print(y)

# linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x,y) # learning function

plt.scatter(x,y)
plt.plot(x,lr.predict(x))
# plt.show()

# polynomial regression
from sklearn.preprocessing import PolynomialFeatures
pr = PolynomialFeatures(degree = 2) # increase of the degree, increases the reliability
x_poly = pr.fit_transform(x)
print(x_poly)

# transform values to polynomial first then use linear regression with the polynomial values
lr2 = LinearRegression()
lr2.fit(x_poly,y)

plt.scatter(x,y)
plt.plot(x,lr2.predict(pr.fit_transform(x)))
plt.show()

# testing the result of the regressions
# linear result
print(lr.predict([[11]]))
print(lr.predict([[6.6]]))

# polynomial result
print(lr2.predict(pr.fit_transform([[11]])))
print(lr2.predict(pr.fit_transform([[6.6]])))

from sklearn.metrics import r2_score
print(r2_score(y,lr2.predict(pr.fit_transform(x))))


print(r2_score(y,lr.predict(x)))

# print(lr.predict(pd.DataFrame([[11]], columns=x.columns)))
# print(lr.predict(pd.DataFrame([[6.6]], columns=x.columns)))
# print(lr2.predict(pr.fit_transform(pd.DataFrame([[11]], columns=x.columns))))
# print(lr2.predict(pr.fit_transform(pd.DataFrame([[6.6]], columns=x.columns))))


# from sklearn.preprocessing import PolynomialFeatures
# pr = PolynomialFeatures(degree = 4)
# x_poly = pr.fit_transform(x)
# print(x_poly)

# lr2 = LinearRegression()
# lr2.fit(x_poly,y)

# plt.scatter(x,y)
# plt.plot(x,lr2.predict(pr.fit_transform(x)))
# plt.show()