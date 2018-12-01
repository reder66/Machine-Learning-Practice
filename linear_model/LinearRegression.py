import numpy as np
class LinearRegression:

    def fit(self, X, y):
        xMat = np.matrix(X)
        yMat = np.matrix(y).T
        self.X = xMat
        self.y = yMat
        xTx = xMat.T*xMat
        w = xTx.I*(xMat.T*yMat)
        self.coef = w
    
    def predict(self, X):
        xMat = np.matrix(X)
        return xMat*self.coef
    
    def f_test(self):
        yhat = self.X*self.coef
        SSE = ((self.y-yhat).T*(self.y-yhat)).sum()
        SSR = ((yhat-self.y.mean()).T*(yhat-self.y.mean())).sum()
        f = SSR/SSE
        return f
        
    def score(self, X, y):
        xMat = np.matrix(X)
        yMat = np.matrix(y).T
        yhat = xMat*self.coef
        n = yhat.shape[0]
        rmse = np.sqrt(np.power(yhat-yMat, 2).sum(axis=0))/n
        return rmse
