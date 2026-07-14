import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
datas = pd.read_csv('prediction/tennis-example/tennis.csv')

outlook = datas.iloc[:,0:1].values.copy() #integer location
temphum = datas.iloc[:,1:3].values.copy()
windy = datas.iloc[:,3:4].values.ravel()

# categorical datas
from sklearn import preprocessing

le = preprocessing.LabelEncoder() #turns categorical values into numerical
outlook[:,0] = le.fit_transform(datas.iloc[:,0])

ohe = preprocessing.OneHotEncoder() #used when there is no order between the transformed numbers
outlook = ohe.fit_transform(outlook).toarray()

play = datas.iloc[:,-1].values
play = le.fit_transform(play).reshape(-1,1)

ohe = preprocessing.OneHotEncoder() 
play = ohe.fit_transform(play).toarray()

# data frames
result1 = pd.DataFrame(data=outlook, index=range(14), columns=['overcast','rainy','sunny']) # alphabetical order
result12 = pd.DataFrame(data=temphum, index=range(14), columns=['temperature','humidity'])
result2 = pd.DataFrame(data=windy, index=range(14), columns=['windy'])
result3 = pd.DataFrame(data=play[:,:1], index=range(14), columns=['play'])
r1 = pd.concat([result1,result2], axis=1)
r12 = pd.concat([r1,result12], axis=1)
r2 = pd.concat([r12,result3], axis=1)

# splitting dataset into train and test
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(r12,result3,test_size=0.33,random_state=0)

# print(x_train)
# print(y_train)

# linear regresssion
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train) # learn y_train with respect to x_train
y_pred = regressor.predict(x_test)
print(y_test)
print(y_pred)

# multiple linear regression
play_target = r2.iloc[:,6:7].values

data = r2.iloc[:,:6]
x_train,x_test,y_train,y_test = train_test_split(data,play_target,test_size=0.33,random_state=0)
# print(x_train)
# print(y_train)

regressor2 = LinearRegression()
regressor2.fit(x_train,y_train)
y_pred = regressor2.predict(x_test)
print(y_pred)
print(y_test)

# plt.scatter(y_test, y_pred)
# plt.plot(y_test, y_test)
# plt.show()

# backward elimination
import statsmodels.api as sm

X = np.append(arr=np.ones((14,1)).astype(int), values=data, axis=1) # to make beta0 = 1, created an all 1 array
print(X)

X_l = data.iloc[:,[0,1,2,3,4,5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(play_target,X_l).fit() # Ordinary Least Squares
print(model.summary())

X_l = data.iloc[:,[0,1,2,3,5]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(play_target,X_l).fit() 
print(model.summary())

X_l = data.iloc[:,[0,1,2,3]].values
X_l = np.array(X_l, dtype=float)
model = sm.OLS(play_target,X_l).fit() 
print(model.summary())

