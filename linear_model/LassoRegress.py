import numpy as np
class LassoRegress:
    
    def __init__(self, lam=1.0, n_iter = 1000, tol = 0.0001):
        self.lam = lam
        self.n_iter = n_iter
        self.tol = tol
        
    def fit(self, X, y):
        #stepwise method
        xMat = np.matrix(X)
        yMat = np.matrix(y).T
        self.X = xMat
        self.y = yMat
        m, n = xMat.shape
        eps = 0.01
        ws = np.zeros((n, 1))
        for i in range(self.n_iter):
            bestres = np.inf
            wsMax = ws.copy()
            for j in range(n):
                #遍历每一个特征
                for sign in [-1, 1]:
                    wsTest = ws.copy()
                    wsTest[j] += eps*sign
                    yTest = xMat*wsTest
                    res = (yMat-yTest).T*(yMat-yTest)
                    if res<bestres:
                        wsMax = wsTest
                        bestres = res
                ws = wsMax.copy()
        self.coef_ = ws
