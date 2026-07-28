import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
datas = pd.read_csv('clustering/customers.csv')
# print(datas)  

x = datas.iloc[:,3:].values

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, init='k-means++')
kmeans.fit(x)

print(kmeans.cluster_centers_)

results = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=123) # start from the same state
    kmeans.fit(x)
    results.append(kmeans.inertia_) # summing wcss values

plt.plot(range(1,11), results)
plt.show()

kmeans = KMeans(n_clusters=3, init='k-means++')
y_pred = kmeans.fit_predict(x)
print(y_pred)

plt.scatter(x[y_pred==0,0], x[y_pred==0,1], s=100, c='red')
plt.scatter(x[y_pred==1,0], x[y_pred==1,1], s=100, c='blue')
plt.scatter(x[y_pred==2,0], x[y_pred==2,1], s=100, c='green')
plt.show()