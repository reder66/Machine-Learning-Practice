# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ensemble.RandomForest import RandomForest
from metrics.classification_report import confusion_matrix, binary_classification_report

data = pd.read_csv('breast_cancer.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)

rf = RandomForest(n_estimate=10)
rf.fit(x_train, y_train)
yhat = rf.predict(x_test)
score = (yhat==y_test).sum()/len(y_test)
print('Testing score: %.4f'%score)
print('OOB score: %.4f'%rf.oob_score)

#%%
cm = confusion_matrix(yhat, y_test)
pr,re,f1,ka = binary_classification_report(cm)

