import numpy as np
class LassoRegress:
    
    def __init__(self, lam=1.0, n_iter = 1000, tol = 1e-16):
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
        
class nonNegativeLasso:
    
    def __init__(self, lam=1.0, n_iter = 1000, tol = 1e-16):
        self.lam = lam
        self.n_iter = n_iter
        self.tol = tol
        
    def fit(self, X, y):
        xMat = np.matrix(X)
        yMat = np.matrix(y).T
        m, n = xMat.shape
        beta = np.matrix(np.ones((n, 1)))
        #prepare work
        A = xMat.T * xMat
        A1 = A.copy()
        A2 = A.copy()
        A1[A1 <= 0] = 0
        A2[A2 >= 0] = 0
        A2 = np.abs(A2)
        b = self.lam*np.matrix(np.ones((n, 1)))-2*xMat.T*yMat
        #now start iteration
        for i in range(self.n_iter):
            a = A1 * beta
            if (a==0).sum() >= 1:
                #a为非0
                self.coef_ = beta
                return 
            c = A2 * beta
            beta0 = np.multiply(beta,-b+np.sqrt(np.power(b, 2)+4*np.multiply(a, c)))/a
            if np.abs(beta-beta0).sum() <= self.tol:
                self.coef_ = beta0
                return
            beta = beta0
        self.coef_ = beta
        
        
        
