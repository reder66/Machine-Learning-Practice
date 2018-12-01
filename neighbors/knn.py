# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 00:16:26 2018

@author: Administrator
"""
import numpy as np
class KNN:
    def __init__(self,k=5):
        self.k = k

    def fit(self, X, y):
        #不知道sklearn怎么fit...
        self.X = X
        self.y = y

    def predict(self, inx):
        ##计算与其他的点的距离
        n = self.X.shape[0]
        diffMat = np.tile(inx, (n, 1)) - self.X
        dist = (diffMat**2).sum(axis=1)**0.5 #欧氏距离
        sortedDist = dist.argsort()
        
        classCount = {}
        for i in range(self.k):
            lab = self.y[sortedDist[i]]
            classCount[lab] = classCount.get(lab, 0) + 1
        pred = sorted(classCount.items(), key = lambda x:x[1], reverse = True)
        return pred[0][0]
 
