# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter
from multiprocessing import Process
from tree.CART import CART

cart = CART(tree_type='classify')

class RandomForest:
    
    def __init__(self, n_estimate = 10, min_samples_split = 1, min_tol_split = 1e-07, \
                 max_feature_size = 'auto', bagging_sample_size = 0.63):
        self.n_estimate = n_estimate
        self.min_samples_split = min_samples_split
        self.min_tol_split = min_tol_split
        self.max_feature_size = max_feature_size
        self.bagging_sample_size = bagging_sample_size
        self.max_feature_size = max_feature_size
        
    def fit(self, X, y):
        xMat = np.array(X)
        yMat = np.array(y)
        m, n = xMat.shape
        if self.max_feature_size == 'auto':
            feature_size = int(np.log2(n)) + 1
        elif isinstance(self.max_feature_size, float):
            feature_size = int(n*self.max_feature_size)
        else:
            print('Feature_size must be set between 0 to 1.')
            return
        
        treeList = []
        featureList = []
        oob_hat = [[] for i in range(m)]
        for k in range(self.n_estimate):
            row_index = np.random.choice(np.arange(m), size=int(m*self.bagging_sample_size), replace=False)
            col_index = np.random.choice(np.arange(n), size=feature_size, replace=False)
            featureList.append(col_index)
            x_train = xMat[row_index][:, col_index]
            y_train = yMat[row_index]
            #创建袋外样本
            oob_index = list(set(np.arange(m))-set(row_index))
            x_test = xMat[oob_index][:, col_index]
            cart = CART(tree_type='classify', min_samples_split=self.min_samples_split,\
                     min_tol_split=self.min_tol_split)
            cart.fit(x_train, y_train)
            treeList.append(cart)
            y_test = cart.predict(x_test)
            for j in range(len(y_test)):
                oob_hat[oob_index[j]].append(int(y_test[j]))
        
        oob_score = 0
        num = 0
        for i in range(m):
            if oob_hat[i] != []:
                c = sorted(Counter(oob_hat[i]).items(), key=lambda x:x[1], reverse=True)[0][0]
                if c==yMat[i]:
                    oob_score += 1
                    num += 1
        oob_score /= num
        self.feature = featureList
        self.model = treeList
        self.oob_score = oob_score
        
    def predict(self, inx):
        X = np.array(inx)
        m = X.shape[0]
        yhat = np.zeros((m, self.n_estimate))
        for i in range(self.n_estimate):
            x_test = X[:, self.feature[i]]
            yhat[:, i] = self.model[i].predict(x_test)
        result = []
        for i in range(m):
            c = sorted(Counter(yhat[i, :]).items(), key=lambda x:x[1], reverse=True)[0][0]
            result.append(c)
        return result
        
        
        
            
            
            
            
            
            
            
            
            
            
            