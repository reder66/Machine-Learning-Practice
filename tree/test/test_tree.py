# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 12:10:43 2018

@author: Administrator
"""

import numpy as np
from tree.CART import CART
from tree.DecisionTreeClassifier import DecisionTree
import pandas as pd
from sklearn.model_selection import train_test_split

'''
Because our data is continuous, so we only use all dataset to fit DecisionTree Classifier,
only to test if there has bugs.
Using gini criterion, we can build cart to test prediction.
'''
df = pd.read_csv('data.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

dtc = DecisionTree()
dtc.fit(X, y)
tree = dtc.tree
print(dtc.getNumLeafs(tree))
print(dtc.getTreeDepth(tree))
dtc_yhat = dtc.predict(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.7)
cart = CART(tree_type='classify')
cart.fit(x_train, y_train)
cart_yhat = cart.predict(x_test.values)[:, 0]

print('DecisionTreeClassifier training score: %.3f'%((dtc_yhat==y).sum()/len(y)))
print('CART using gini testing score: %.3f'%((cart_yhat==y_test).sum()/len(y_test)))
print(cart_yhat[0])