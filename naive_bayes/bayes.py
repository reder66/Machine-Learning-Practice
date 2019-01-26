# -*- coding: utf-8 -*-
import numpy as np
def normal(x, mu, sigma):
    return np.exp(-0.5*(x-mu)*np.linalg.pinv(sigma)*(x-mu).T)

def bernoulli(x, p):
    return np.multiply(np.power(p, x), np.power(1-p, 1-x))
    
class naive_bayes(object):
    def __init__(self, distribution='normal'):
        self.distribution = distribution
        
    def fit(self, X, y):
# =============================================================================
# 即估计P(x1...xn|yi)的条件分布，利用条件独立假设:
# P(x1...xn|yi)=P(x1|yi)*...*P(xn|yi)
# 根据假设的条件分布可估计参数，通常用高斯分布，而高斯分布可不依照条件独立假设，直接
# 求多元正态的参数，对于伯努利分布则利用条件独立假设。
# =============================================================================
        xMat = np.matrix(X)
        yMat = np.array(y)
        m, n = xMat.shape
        k = len(np.unique(yMat))
    
        if self.distribution=='normal':
            p_y = []
            mu = []
            sigma = np.matrix(np.zeros((n, n)))
            for i in range(k):
                index = yMat==i
                p_yi = yMat[index]/m
                sub_xMat = xMat[index, :]
                mu_i = sub_xMat.mean(axis=0)
                sigma += (sub_xMat-mu_i).T*(sub_xMat-mu_i)
                mu.append(mu_i)
                p_y.append(sub_xMat.shape[0]/m)
            sigma /= m
            self.p_y = p_y
            self.mu = mu
            self.sigma = sigma
            self.class_num = k
            
        elif self.distribution=='bernoulli':
            p = np.zeros((k, n)) #伯努利分布参数矩阵
            p_y = []
            for i in range(k):
                index = yMat==i
                for j in range(n):
                    sub_xMat = xMat[index, j]
                    p[i, j] = (sub_xMat.sum()+1)/(sub_xMat.shape[0]+len(np.unique(sub_xMat)))
                p_y.append((np.sum(index)+1)/(m+k))
                
            self.p = p
            self.p_y = p_y
            self.class_num = k
        
    
    def predict(self, X):
        xMat = np.matrix(X)
        m, n = xMat.shape
        if self.distribution=='normal':
            P = np.zeros((m, self.class_num))
            for i in range(m):
                for j in range(self.class_num):
                    pdf = normal(xMat[i, :], self.mu[j], self.sigma)
                    P[i, j] = pdf*self.p_y[j]
            
        elif self.distribution=='bernoulli':
            P = np.matrix(np.zeros((m, self.class_num)))
            for j in range(self.class_num):
                prob = self.p_y[j]
                for k in range(n):
                    prob = np.multiply(prob, bernoulli(xMat[:, k], self.p[j, k]))
                P[:, j] = prob
 
        yhat = np.argmax(np.array(P), axis=1)
        return yhat

            
                    
                
                
        
        