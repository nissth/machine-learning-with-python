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

# standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

print(x_train)
print(' ')
print(x_test)