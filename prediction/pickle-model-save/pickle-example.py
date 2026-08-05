import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
url = "https://bilkav.com/satislar.csv"
datas = pd.read_csv(url)

datas = datas.values
x = datas[:,0:1]
y = datas[:,1]

split = 0.33

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=split,random_state=0)

from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train,y_train)
print(lr.predict(x_test))

# saving the model
import pickle 

myfile = "model.save"
pickle.dump(lr,open(file=myfile,mode='wb'))
save = pickle.load(open(file=myfile,mode='rb'))
print(save.predict(x_test))

'''
# to use the saved model
import pickle 

save = pickle.load(open(file=myfile,mode='rb'))
print(save.predict(x_test))

'''