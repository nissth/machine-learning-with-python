import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
datas = pd.read_csv('reinforced-learning/Ads_CTR_Optimisation.csv')
# print(datas)  

import random

N = 10000
d = 10
sum = 0
chosen = []
for n in range(0,N):
    ad = random.randrange(d)
    chosen.append(ad)
    reward = datas.values[n,ad] # n id datas, if row=1 then reward = 1
    sum = sum + reward

print(sum)

plt.hist(chosen)
plt.show()