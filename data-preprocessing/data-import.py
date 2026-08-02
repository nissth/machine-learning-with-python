import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing data
datas = pd.read_csv('data-preprocessing/veriler.csv')

print(datas)

ulke = datas[['ulke']]
print(ulke)

boykilo = datas[['boy', 'kilo']]
print(boykilo)