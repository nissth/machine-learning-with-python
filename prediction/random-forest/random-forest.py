import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
datas = pd.read_csv('prediction/random-forest/maaslar.csv')

x = datas.iloc[:,1:2].values
y = datas.iloc[:,2:].values 

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=10, random_state=0) # how many decision trees will be drawn
rf.fit(x,y.ravel())

plt.scatter(x,y)
plt.plot(x,rf.predict(x))
plt.show()

# test
print(rf.predict([[11]]))
print(rf.predict([[6.6]]))