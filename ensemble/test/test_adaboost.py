# -*- coding: utf-8 -*-

import pandas as pd
from ensemble.AdaBoost import AdaBoost
from sklearn.model_selection import train_test_split

data = pd.read_csv('breast_cancer.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)

ada = AdaBoost(n_iter=100)
ada.fit(x_train, y_train)
print(ada.score(x_test, y_test))