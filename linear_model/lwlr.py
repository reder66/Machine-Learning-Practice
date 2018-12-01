import numpy as np

def lwlr(inx, X, y, k=1.0):
    xMat = np.matrix(X)
    yMat = np.matrix(y).T
    n = xMat.shape[0]
    diffMat = np.tile(inx, (n, 1))-X
    dist = (diffMat**2).sum(axis=1)
    w = np.diag(np.exp(dist/(-2*k**2))) #计算权重矩阵
    w = np.matrix(w)
    xTx = xMat.T*w*xMat
    return (xTx.I)*xMat.T*w*yMat
