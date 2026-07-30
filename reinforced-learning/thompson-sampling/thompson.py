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

ones = [0] * d
zeros = [0] * d

for n in range(0,N):
    ad = 0
    max_th = 0
    for i in range(0,d):
        randbeta = random.betavariate(ones[i]+1, zeros[i]+1)
        if randbeta > max_th:
            max_th = randbeta
            ad = i 

    chosen.append(ad)
    r = datas.values[n,ad]

    if r==1: 
        ones[ad] += 1
    else: 
        zeros[ad] += 1
    sum = sum + r

print(sum)

plt.hist(chosen)
plt.show()





