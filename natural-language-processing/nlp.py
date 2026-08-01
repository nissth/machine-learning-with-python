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

# Check how many rows successfully loaded
print("Total rows loaded:", len(datas))

import re 
import nltk

stop = nltk.download('stopwords')
# print(stop)

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

from nltk.corpus import stopwords 

collection = []
for i in range(1000):
    review = re.sub('[^a-zA-Z]',' ', datas['Review'][i])

    review = review.lower() # turning every letter to lower case
    # print(review)
    review = review.split() # splitting every word into an array
    # print(review)

    # removing stopwords then add the rest into an array
    review = [ps.stem(word_) for word_ in review if not word_ in set(stopwords.words('english'))] 
    review = ' '.join(review) # reunite the words in the array as a string, add a space between the word 
    # print(review)

    collection.append(review)
print(collection)
