import numpy as np

def stumpClassify(subdata, threshVal, threshIneq):
    result = np.zeros(len(subdata))
    if threshIneq == 'lt':
        result[subdata<=threshVal] = -1
    else:
        result[subdata>threshVal] = -1
    return result

def buildStump(data, label, D):
    m, n = data.shape
    minError = np.inf
    bestStump = {}
    for i in range(n):
        #遍历所有特征
        subdata = data[:, i]
        uniqueVal = set(subdata)
        for threshVal in uniqueVal:
            #遍历特征每个值
            for threshIneq in ['lt','rt']:
                #遍历每个不等号
                result = stumpClassify(subdata, threshVal, threshIneq)
                error = 0.5*np.abs(result - label)
                weightedErr = np.dot(D, error)
                if weightedErr < minError:
                    bestStump['feature'] = i
                    bestStump['threshVal'] = threshVal
                    bestStump['threshIneq'] = threshIneq
                    labelEst = result
                    minError = weightedErr
    return bestStump, labelEst, minError

class AdaBoost:
    def __init__(self, n_iter = 50, tol = 0.0):
        self.n_iter = n_iter
        self.tol = tol
    def fit(self, X, y, verbose = True):
        xMat = np.array(X)
        yMat = np.array(y)
        m, n = xMat.shape
        D = np.ones(m)/m
        model = []
        cur = 0
        trainError = 0
        for i in range(self.n_iter+1):
            submodel, labelEst, error = buildStump(xMat, yMat, D)
            alpha = 0.5*np.log((1-error)/max(error, 1e-16))
            submodel['alpha'] = alpha
            model.append(submodel)
            D[np.abs(labelEst - yMat) == 2] *= np.exp(alpha)
            D[np.abs(labelEst - yMat) == 0] *=  np.exp(-alpha)
            D = D/D.sum()
            cur += alpha * labelEst
            cur = np.sign(cur)
            trainError = 0.5*np.abs(cur-yMat).sum()/m
            if i%10==0 and verbose:
                print('%d/%d training, error:%.4f'%(i, self.n_iter, trainError))
            if trainError <= self.tol:
                break
        self.model = model
        
    def predict(self, inx):
        xMat = np.array(inx)
        n = len(xMat)
        i = 0
        final = np.zeros(n)
        for submodel in self.model:
            result = stumpClassify(xMat[:, submodel['feature']], submodel['threshVal'], submodel['threshIneq'])
            final += result * submodel['alpha']
        final = np.sign(final)
        return final
    
    def score(self, inx, y):
        yhat = self.predict(inx)
        error = 0.5*np.abs(yhat - y).sum()/inx.shape[0]
        return 1 - error
             
             
                
                
        