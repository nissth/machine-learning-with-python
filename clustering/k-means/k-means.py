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