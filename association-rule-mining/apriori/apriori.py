import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
datas = pd.read_csv('association-rule-mining/sepet.csv', header=None)
# print(datas)  

t = []
for i in range(0,7501):
    t.append([str(datas.values[i,j]) for j in range(0,20)]) 

# optimized version
# t = [[str(item) for item in row if pd.notna(item)] for row in datas.values]

from apyori import apriori
rules = apriori(t, min_support =0.01, min_confidence=0.2, min_lift=3, min_length=2)
print(list(rules))







