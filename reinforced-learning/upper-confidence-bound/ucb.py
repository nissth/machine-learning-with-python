import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
datas = pd.read_csv('reinforced-learning/Ads_CTR_Optimisation.csv')
# print(datas)  

import math 

N = 10000
d = 10

#Ri(n)
reward = [0] * d
#Ni(n)
clicks = [0] * d
sum = 0
chosen = []

for n in range(0,N):
    ad = 0
    max_ucb = 0
    for i in range(0,d):
        if(clicks[i] > 0):
            avg = reward[i] / clicks[i]
            delta = math.sqrt((3/2) * math.log(n+1)/ clicks[i])
            ucb = avg + delta
        else: 
            ucb = 1e400

        if max_ucb < ucb:
            max_ucb = ucb
            ad = i

    chosen.append(ad)
    clicks[ad] = clicks[ad] + 1

    r = datas.values[n,ad]
    reward[ad] = reward[ad] + r
    sum = sum + r

print(sum)

plt.hist(chosen)
plt.show()





