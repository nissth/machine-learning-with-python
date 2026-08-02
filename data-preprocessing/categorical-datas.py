import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing data
datas = pd.read_csv('data-preprocessing/eksikveriler.csv')

ulke = datas.iloc[:,0:1].values.copy()
print(ulke)

from sklearn import preprocessing

le = preprocessing.LabelEncoder() #turns categorical values into numerical
ulke[:,0] = le.fit_transform(datas.iloc[:,0])
print(ulke)

ohe = preprocessing.OneHotEncoder() #used when there is no order between the transformed numbers
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

#[0. 1. 0.]
#[0. 0. 1.]
#[1. 0. 0.]