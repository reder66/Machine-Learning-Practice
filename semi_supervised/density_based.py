import numpy as np

def distance_matrix(A, B):
    #Use eucludean distance
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A,BT)
    # print(vecProd)
    SqA =  A**2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
 
    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))    
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0   
    ED = np.sqrt(SqED)
    return np.array(ED)
    
def find_center_bound(xMat, h, t):
    m, n = xMat.shape
    ED = distance_matrix(xMat, xMat)
    M = m*(m-1)/2
    #首先确定dc
    edVec = ED.flatten()
    edVec = np.sort(edVec[edVec!=0])
    dc = edVec[int(t*M)]
    #根据dc计算rou
    edJudge = ED.copy()
    edJudge[edJudge<=dc] = 1
    edJudge[edJudge>dc] = 0
    rou = edJudge.sum(axis=1)-1
    rou0 = rou[int(h*len(rou))]
    center = np.where(rou>=rou0)[0]
    boundary = np.where(rou<rou0)[0]
    bound = []
    for i in boundary:
        neighbor = np.where(edJudge[i, :]==1)[0]
        for j in neighbor:
            if j in center:
                bound.append(i)
                break
    return center, np.array(bound)
                
class density_method:
    def __init__(self, trainer, h=0.5, t=0.5):
        self.trainer = trainer
        self.h = h
        self.t = t
        
    def fit(self, X, y):
        xMat = np.array(X)
        k = len(y)
        L = xMat[:k, :]
        center, bound = find_center_bound(xMat, self.h, self.t)
        U_center = center[center>=k]
        U_bound = bound[bound>=k]
        ###核心点迭代
        for i in U_center:    
            self.trainer.fit(L, y)
            x = xMat[i, :]
            U_pred = self.trainer.predict(x)
            L = np.vstack((L, x))
            y = np.hstack((y, U_pred))
        ###边界点迭代
        for i in U_bound:
            self.trainer.fit(L, y)
            x = xMat[i, :]
            U_pred = self.trainer.predict(x)
            L = np.vstack((L, x))
            y = np.hstack((y, U_pred))
        
    def predict(self, X):
        return self.trainer.predict(X)