import numpy as np
def sigmoid(z):
    return 1/(1+np.exp(-z))
class LogisticRegress:
    
    def __init__(self, learning_rate = 0.5, n_iter = 100):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        
    def fit(self, X, y):
        xMat = np.matrix(X)
        yMat = np.matrix(y).T
        m, n = xMat.shape
        theta = np.matrix(np.random.normal(size=(n, 1)))
        for i in range(self.n_iter):
            #随机梯度下降
            for j in range(m):
                alpha = self.learning_rate/(1+j+i) + 0.01
                rand = np.random.randint(m)
                xi = xMat[rand]
                z = xi*theta
                yhat = sigmoid(z)
                gradient = xi.T*(y[rand]-yhat)
                theta += alpha*gradient
        self.coef_ = theta
    
    def predict(self, X):
        xMat = np.matrix(X)
        z = xMat*self.coef_
        yhat = sigmoid(z)
        yhat[yhat>=0.5] = 1
        yhat[yhat<0.5] = 0
        return yhat
    
    def score(self, X, y):
        yMat = np.matrix(y).T
        yhat = self.predict(X)
        acc = np.abs(yMat-yhat).sum()
        return 1-acc
