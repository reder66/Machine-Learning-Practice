# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 12:10:43 2018

@author: Administrator
"""

import numpy as np
from tree.CART import CART
from tree.DecisionTreeClassifier import DecisionTree
import pandas as pd

df = pd.read_csv('data.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
dtc = DecisionTree()
dtc.fit(X, y)
tree = dtc.tree
print(dtc.getNumLeafs(tree))
print(dtc.getTreeDepth(tree))
dtc_yhat = dtc.predict(X)

cart = CART()
cart.fit(X, y)
cart_yhat = cart.predict(X.values)