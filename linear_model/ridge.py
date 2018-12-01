import numpy as np
class ridgeRegress:
    def __init__(self, lam=1.0):
        self.lam = lam
        
    def fit(self, X, y):
        xMat = np.matrix(X)
        yMat = np.matrix(y).T
        m = xMat.shape[1]
        xTx = xMat.T*xMat + self.lam*np.eye(m)
        w = xTx.I*xMat.T*yMat
        self.coef = w
    
    def predict(self, X):
        xMat = np.matrix(X)
        return xMat*self.coef
    
    def score(self, X, y):
        xMat = np.matrix(X)
        yMat = np.matrix(y).T
        yhat = xMat*self.coef
        n = yhat.shape[0]
        rmse = np.sqrt(np.power(yhat-yMat, 2).sum(axis=0))/n
        return rmse
