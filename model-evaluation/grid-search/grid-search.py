import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data
datas = pd.read_csv('model-evaluation/Social_Network_Ads.csv')
x = datas.iloc[:,[2,3]].values
y = datas.iloc[:,4].values

# splitting dataset into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=0)

# standardization
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

# grid search, parameter optimization and model selection
# parameters
p = [{'C':[1,2,3,4,5], 'kernel': ['linear']},
     {'C': [1,10,100,1000], 'kernel':['rbf'], 'gamma':[1,0.5,0.1,0.01,0.001]}]

from sklearn.model_selection import GridSearchCV
gs = GridSearchCV(estimator=classifier, param_grid=p, scoring='accuracy', cv=10, n_jobs=-1)

grid_search = gs.fit(x_train,y_train)
bestresult = grid_search.best_score_
bestparameters = grid_search.best_params_

print(bestresult)
print(bestparameters)