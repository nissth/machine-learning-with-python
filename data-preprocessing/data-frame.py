import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
datas = pd.read_csv('eksikveriler.csv')

# missing datas
ulke = datas.iloc[:,0:1].values.copy()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #puts the mean into the missing datas

yas = datas.iloc[:,1:4].values
imputer= imputer.fit(yas[:,1:4]) #learning
yas[:,1:4] = imputer.transform(yas[:,1:4]) #exec

# categorical datas
from sklearn import preprocessing

le = preprocessing.LabelEncoder() #turns categorical values into numerical
ulke[:,0] = le.fit_transform(datas.iloc[:,0])

ohe = preprocessing.OneHotEncoder() #used when there is no order between the transformed numbers
ulke = ohe.fit_transform(ulke).toarray()

# data frames
print(list(range(22)))
result1 = pd.DataFrame(data=ulke, index=range(22), columns=['fr','tr','us'])
print(result1) 

result2 = pd.DataFrame(data=yas, index=range(22), columns=['boy','kilo','yas'])
print(result2) 

cinsiyet = datas.iloc[:,-1].values
print(cinsiyet)

result3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=['cinsiyet'])
print(result3)

r1 = pd.concat([result1,result2], axis=1)
print(r1)

r2 = pd.concat([r1,result3], axis=1)
print(r2)
