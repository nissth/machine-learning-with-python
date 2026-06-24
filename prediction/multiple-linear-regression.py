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
print(c)
print(r1)
print(r2)


# splitting dataset into train and test
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(r1,result3,test_size=0.33,random_state=0)

# x = dependent - ulke,boy,kilo,yas
# y = independent - cinsiyet
# finding y by using x

# standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

print(x_train)
print(' ')
print(x_test)