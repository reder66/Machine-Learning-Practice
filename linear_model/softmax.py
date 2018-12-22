# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 22:48:51 2018

@author: Administrator
"""
import numpy as np

def stable_softmax(xMat, W):
    #z=\w_j.T*x
    z = xMat*W
    z -= -np.max(z, axis=1)
    return np.exp(z)/np.exp(z).sum(axis=1) 


class SoftMax:
    
    def __init__(self, n_iter=100, alpha=0.1):
        self.n_iter = n_iter
        self.alpha = alpha
    
    def fit(self, X, y):
        xMat = np.matrix(X)
        yMat = np.array(y)
        m, n = xMat.shape
        k = len(np.unique(yMat))
        W = np.matrix(np.ones((n, k)))
        
        for i in range(self.n_iter):
            P = stable_softmax(xMat, W)
            for j in range(m):
                P[j, yMat[j]] -= 1
            W -= self.alpha/m*xMat.T*P
        self.weights = W
    
    def predict(self, X):
        xMat = np.matrix(X)
        y = xMat*self.weights
        yhat = np.argmax(y, axis=1)
        return np.array(yhat).ravel()

        