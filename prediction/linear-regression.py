import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
datas = pd.read_csv('prediction/satislar.csv')

months = datas[['Aylar']]
print(months)

sales = datas[['Satislar']]  # sales2 = datas.iloc[:,:1].values
print(sales)

# splitting dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(months,sales,test_size=0.33,random_state=0)

'''
# standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)
'''

# linear regression
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)

prediction = lr.predict(x_test)
print(y_test)
print(prediction)

# visualize
x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))

plt.title("Sales based on the Months")
plt.xlabel("Months")
plt.ylabel("Sales")

plt.show()
