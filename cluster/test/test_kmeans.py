# -*- coding: utf-8 -*-
import pandas as pd
from cluster.kmeans import *

data = pd.read_csv('data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1].values

km = kmeans(k = 4)
km.fit(X)
km_yhat = km.clusterAssment[:, 0]

bikm = bikmeans(k = 4)
bikm.fit(X)
bikm_yhat = bikm.clusterAssment[:, 0]

m = X.shape[0]
print('kmeans score: %.3f'%((km_yhat==y).sum()/m))
print('bi-kmeans score: %.3f'%((bikm_yhat==y).sum()/m))
