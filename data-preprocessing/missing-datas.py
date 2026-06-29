import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing data
datas = pd.read_csv('eksikveriler.csv')

#print(datas)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #puts the mean into the missing datas

yas = datas.iloc[:,1:4].values
print(yas)
imputer= imputer.fit(yas[:,1:4]) #learning
yas[:,1:4] = imputer.transform(yas[:,1:4]) #exec
print(yas)