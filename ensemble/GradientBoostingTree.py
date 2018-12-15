# -*- coding: utf-8 -*-

import numpy as np
from tree.CART import CART

class GradientBoostingTree:
    
    def __init__(self, n_tree = 30, max_depth = 7):
        self.n_tree = n_tree
        self.max_depth = max_depth
        
    def fit(self, X, y):
        #Based on square loss function
        xMat = np.array(X)
        yMat = np.array(y)
        
        root = CART(max_depth=1)
        root.fit(xMat, yMat)
        y0 = root.predict(xMat)
        error = yMat-y0
        treeList = [root]
        for t in range(self.n_tree):
            tree = CART(tree_type='reg', max_depth=self.max_depth, prune_size=0)
            tree.fit(xMat, error)
            errhat = tree.predict(xMat)
            y0 += errhat
            error = yMat-y0
            treeList.append(tree)
            if t%10==0:
                print('Building %d/%d......'%(t, self.n_tree))
            
        self.model = treeList
        
    def predict(self, X):
        X = np.array(X)
        m = X.shape[0]
        yhat = np.zeros(m)
        for i in range(self.n_tree):
            yhat += self.model[i].predict(X)
        return yhat