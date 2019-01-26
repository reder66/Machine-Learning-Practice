# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 12:11:28 2018

@author: Administrator
"""
from neighbors.knn import KNN
import pandas as pd


df = pd.read_csv('data.csv')
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
m, n = X.shape
knn = KNN()
knn.fit(X, y)

yhat = [knn.predict(X[i, :]) for i in range(m)]
print('Error classified:',(y-yhat).sum())
#%%
import numpy as np
from neighbors.kd_tree import kdTree
data = np.array([[3,1,4],[2,3,7],[4,3,4],[2,1,3],[2,4,5]])
tree = kdTree(data) 
