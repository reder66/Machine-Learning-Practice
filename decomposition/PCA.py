# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, k = 3):
        self.k = k

    
    def fit(self, X, cov = 'corr'):
        if cov not in ['corr', 'cov']:
            print("cov must be 'corr' or 'cov', which depends on your input X matrix")
            return
        
        xMat = np.matrix((X-np.mean(X, axis=0))/np.std(X, axis=0))
        if cov == 'corr':
            eigVal, eigVec = np.linalg.eig(np.corrcoef(xMat, rowvar=0))
        else:
            eigVal, eigVec = np.linalg.eig(np.cov(xMat, rowvar=0))
        self.eigVal = eigVal
        self.eigVec = eigVec
        self.xMat = xMat
        
    def fit_transform(self, X, cov = 'corr'):
        self.fit(X, cov)
        eigInd = np.argsort(-self.eigVal)[:self.k] #从大到小排序
        compenents = self.xMat*self.eigVec[:, eigInd]
        s_percent = self.eigVal[eigInd].sum()/self.eigVal.sum()
        self.compenents_ = compenents
        self.variance_explained_ratio_ = s_percent
        return compenents
        
    def scree_plot(self):
        s_all = np.sum(self.eigVal)
        s_com = self.eigVal[np.argsort(-self.eigVal)]
        s_percent = s_com/s_all
        plt.plot(s_percent)
        plt.show()
        
    
        