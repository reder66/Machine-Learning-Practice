# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 12:06:55 2018

@author: Administrator
"""

from linear_model.LinearRegression import LinearRegression
from linear_model.ridge import ridgeRegress
from linear_model.LassoRegress import LassoRegress
from linear_model.Logit import LogisticRegress
from linear_model.lwlr import lwlr
import pandas as pd

df = pd.read_csv('data.csv')
X = df.iloc[:,:2].values
y = df.iloc[:,2].values
m, n = X.shape

lr = LinearRegression()
lr.fit(X, y)
print(lr.score(X, y))

ridge = ridgeRegress()
ridge.fit(X, y)
print(ridge.score(X, y))

lasso = LassoRegress()
lasso.fit(X, y)
print(lasso.coef_)

yhat = [(X[i, :]*lwlr(X[i, :], X, y))[0, 0] for i in range(m)]
rmse = np.sqrt(((y-yhat)**2).sum())/m
print(rmse)
    
