# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 11:27:57 2018

@author: Administrator
"""
import numpy as np

def normal(x, mu, sigma):
    return np.exp(-0.5*(x-mu)*np.linalg.pinv(sigma)*(x-mu).T)

class GDA:
    
    def fit(self, X, y):
        xMat = np.matrix(X)
        yMat = np.array(y)
        m, n = xMat.shape
        k = len(np.unique(yMat)) #类别数
        phi = []
        mu = []
        sigma = np.matrix(np.zeros((n, n)))
        #参数估计
        for i in range(k):
            xi = xMat[np.where(yMat==i)[0],:]
            mu_i = xi.mean(axis=0)
            sigma += (xi-mu_i).T*(xi-mu_i)
            mu.append(mu_i)
            phi.append(xi.shape[0]/m)
        
        sigma /= m
        self.phi = phi
        self.mu = mu
        self.sigma = sigma
        self.class_num = k
        
    def predict(self, X):
        xMat = np.matrix(X)
        m, n = xMat.shape
        yhat = []
        for i in range(m):
            x = xMat[i, :]
            P = []
            for j in range(self.class_num):
                mu = self.mu[j]
                phi = self.phi[j]
                pdf = normal(x, mu, self.sigma)
                P.append(pdf*phi)
            yhat.append(np.argmax(P))
        return np.array(yhat)