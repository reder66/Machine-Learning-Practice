# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from ensemble.GradientBoostingTree import GradientBoostingTree

data = pd.read_csv('house_data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
#%%
gbdt = GradientBoostingTree(n_tree=30, max_depth=7)
gbdt.fit(x_train, y_train)
#%%
yhat = gbdt.predict(x_test)
Y = pd.DataFrame(y_test)
Y['yhat'] = yhat
Y = Y.reset_index(drop=True)
Y.plot()
print('MSE Score:', mean_squared_error(Y['MEDV'], Y['yhat']))
print('R2 Score:', r2_score(Y['MEDV'], Y['yhat']))
