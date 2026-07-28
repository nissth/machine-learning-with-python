import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
datas = pd.read_csv('clustering/customers.csv')
# print(datas)  

x = datas.iloc[:,3:].values

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3,linkage='ward',metric='euclidean')
y_pred = ac.fit_predict(x)

print(y_pred)

plt.scatter(x[y_pred==0,0], x[y_pred==0,1], s=100, c='red')
plt.scatter(x[y_pred==1,0], x[y_pred==1,1], s=100, c='blue')
plt.scatter(x[y_pred==2,0], x[y_pred==2,1], s=100, c='green')
plt.show()

# dendrogram visualize
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))
plt.show()
