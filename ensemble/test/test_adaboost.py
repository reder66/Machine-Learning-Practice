# -*- coding: utf-8 -*-

import pandas as pd
from ensemble.AdaBoost import AdaBoost
from sklearn.model_selection import train_test_split
from metrics.classification_report import confusion_matrix, binary_classification_report

data = pd.read_csv('breast_cancer.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)

ada = AdaBoost(n_iter=50)
ada.fit(x_train, y_train)
print(ada.score(x_train, y_train))
#%%
yhat = ada.predict(x_test)
cm = confusion_matrix(yhat, y_test)
pr,re,f1,ka = binary_classification_report(cm)
