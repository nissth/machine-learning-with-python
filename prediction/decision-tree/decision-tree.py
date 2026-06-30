import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
datas = pd.read_csv('prediction/decision-tree/maaslar.csv')

x = datas.iloc[:,1:2].values
y = datas.iloc[:,2:].values 

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor(random_state=0)
dt.fit(x,y)

plt.scatter(x,y)
plt.plot(x,dt.predict(x))
plt.show()

# test
print(dt.predict([[11]]))
print(dt.predict([[6.6]]))