import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# parsing every line
with open('natural-language-processing/Restaurant_Reviews.csv', encoding='utf-8') as f:
    lines = f.read().splitlines()

header = lines[0].split(',')
rows = []
for line in lines[1:]:
    line = line.strip()
    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1].replace('""', '"')
    text, label = line.rsplit(',', 1)
    rows.append([text, label])

datas = pd.DataFrame(rows, columns=header)
datas['Liked'] = datas['Liked'].astype(int)

import re 

review = re.sub('[^a-zA-Z]',' ', datas['Review'][0])
print(review)
# review1 = re.sub('[^a-zA-Z]',' ', datas['Review'][6])
# print(review1)

# Check how many rows successfully loaded
print("Total rows loaded:", len(datas))