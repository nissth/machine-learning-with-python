import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

# importing data
datas = pd.read_csv('deep-learning/Churn_Modelling.csv')

x = datas.iloc[:,3:13].values # input layer
y = datas.iloc[:,13].values # output layer

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
 
le = preprocessing.LabelEncoder() #turns categorical values into numerical
x[:,1] = le.fit_transform(x[:,1])
le2 = preprocessing.LabelEncoder()
x[:,2] = le2.fit_transform(x[:,2])

ohe = ColumnTransformer([("ohe", OneHotEncoder(dtype=float),[1])], remainder="passthrough")
x = ohe.fit_transform(x)
x = x[:,1:]

# splitting dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

# standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# artificial neural networks
classifier = Sequential() # initializing a neuron
# input layer
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=11))
# hidden layer
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
# output layer
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(x_train,y_train, epochs=50)
y_pred = classifier.predict(x_test)

y_pred= (y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)
