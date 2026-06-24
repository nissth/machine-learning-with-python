import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
datas = pd.read_csv('prediction/insanlar.csv')

ulke = datas.iloc[:,0:1].values.copy() #integer location
yas = datas.iloc[:,1:4].values

# categorical datas
from sklearn import preprocessing

le = preprocessing.LabelEncoder() #turns categorical values into numerical
ulke[:,0] = le.fit_transform(datas.iloc[:,0])

ohe = preprocessing.OneHotEncoder() #used when there is no order between the transformed numbers
ulke = ohe.fit_transform(ulke).toarray()

c = datas.iloc[:,-1].values.copy()

le = preprocessing.LabelEncoder() 
# c[:,-1] = le.fit_transform(datas.iloc[:,-1])
c = le.fit_transform(c).reshape(-1,1)

ohe = preprocessing.OneHotEncoder() 
c = ohe.fit_transform(c).toarray()

# data frames
result1 = pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
result2 = pd.DataFrame(data=yas, index=range(22), columns=['boy','kilo','yas'])
cinsiyet = datas.iloc[:,-1].values
result3 = pd.DataFrame(data=c[:,:1], index=range(22), columns=['cinsiyet'])
r1 = pd.concat([result1,result2], axis=1)
r2 = pd.concat([r1,result3], axis=1)
# print(c)
# print(r1)
# print(r2)


# splitting dataset into train and test
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(r1,result3,test_size=0.33,random_state=0)

# linear regresssion
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train) # learn y_train with respect to x_train
y_pred = regressor.predict(x_test)
# print(y_test)
# print(y_pred)

# multiple linear regression
boy = r2.iloc[:,3:4].values
# print(boy)

left = r2.iloc[:,:3]
right = r2.iloc[:,4:]

data = pd.concat([left,right], axis=1)
x_train,x_test,y_train,y_test = train_test_split(data,boy,test_size=0.33,random_state=0)
# print(x_train)
# print(y_train)

regressor2 = LinearRegression()
regressor2.fit(x_train,y_train)
y_pred = regressor2.predict(x_test)
# print(y_pred)
# print(y_test)

# backward elimination
import statsmodels.api as sm

X = np.append(arr=np.ones((22,1)).astype(int), values=data, axis=1) # to make beta0 = 1, created an all 1 array
print(X)

X_l = data.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(boy,X_l).fit() # Ordinary Least Squares
print(model.summary())

X_l = data.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(boy,X_l).fit() 
print(model.summary())

X_l = data.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(boy,X_l).fit() 
print(model.summary())

'''
R-squared: Shows how much of the dependent variable is explained by the model.
Adj. R-squared: Shows the model performance while considering the number of independent variables.
F-statistic: Tests whether the overall regression model is significant.
Prob (F-statistic): Shows the p-value of the overall model, where values below 0.05 indicate a significant model.
coef: Represents how much the dependent variable changes when the independent variable increases by one unit.
t: Measures how strongly an independent variable affects the dependent variable.
P>|t|: Shows whether an independent variable is statistically significant, where values below 0.05 indicate significance.
std err: Measures the uncertainty of the coefficient estimate.
[0.025 - 0.975]: Represents the 95% confidence interval of the coefficient.
'''

'''
if we want to use forward elimination we use ols then we keep the variables which has the minimum p-values
in this dataset 
x1            0.000  
x2            0.000  
x3            0.000     
x4            0.000 
x5            0.717   
x6            0.052  
we keep 0,1,2,3
'''
  