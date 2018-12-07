import numpy as np

def ecludDist(A, B):
    return np.sqrt(np.sum(np.power(A - B,2)))

def randCent(data, k):
    m, n = data.shape
    index = np.random.choice(range(m), k)
    cent = data[index, :]
    return cent

class kmeans:
    def __init__(self, k = 3, distFunc = ecludDist):
        self.k = k
        self.distFunc = distFunc
        
    def fit(self, data, verbose = True):
        X = np.matrix(data)
        m, n = X.shape
        clusterAssment = np.zeros((m, 2))
        cent = randCent(X, self.k)
        clusterChanged = True
        num = 0
        while clusterChanged:
            #当任意一个样本点的簇分类发生变化
            clusterChanged = False
            for i in range(m):
                minDist = np.inf
                for j in range(self.k):
                    dist = self.distFunc(X[i, :], cent[j, :])
                    if dist < minDist :
                        minDist = dist
                        minCluster = j
                if clusterAssment[i, 0] != minCluster:
                    clusterChanged = True
                    num += 1
                clusterAssment[i, 0] = minCluster
                clusterAssment[i, 1] = minDist
            for i in range(self.k):
                #更新所有质心
                cluster_i = X[np.nonzero(clusterAssment[:, 0] == i)[0]]
                cent[i, :] = np.mean(cluster_i, axis = 0)
        if verbose:
            print('Converge after %d iteration.'%num)
        self.centroids = cent
        self.clusterAssment = clusterAssment

#%%
class bikmeans:
    def __init__(self, k, distFunc = ecludDist):
        self.k = k
        self.distFunc = distFunc
        
    def fit(self, data):
        X = np.matrix(data)
        m, n = X.shape
        cent0 = np.mean(X, axis = 0)
        clusterAssment = np.zeros((m , 2))
        centlist = [cent0]
        for i in range(m):
            clusterAssment[i, 1] = self.distFunc(X[i, :], cent0)
        km = kmeans(2)
        while len(centlist) < self.k:
            minSSE = np.inf
            for i in range(len(centlist)):
                sub_x = X[np.nonzero(clusterAssment[:, 0] == i)[0]]
                km.fit(sub_x, verbose=False)
                subCluster = km.clusterAssment
                subSSE = subCluster[:, 1].sum()
                otherSSE = clusterAssment[clusterAssment[:, 0] != i][:, 1].sum()
                curSSE = subSSE + otherSSE
                if curSSE < minSSE:
                    minSSE = curSSE
                    bestCent = km.centroids
                    bestToSplit = i
                    bestClustAssment = subCluster
            #赋值顺序必须先是1再是0！
            bestClustAssment[bestClustAssment[:, 0] == 1, 0] = len(centlist)
            bestClustAssment[bestClustAssment[:, 0] == 0, 0] = bestToSplit
            centlist[bestToSplit] = bestCent[0, :]
            centlist.append(bestCent[1, :])
            clusterAssment[clusterAssment[:, 0] == bestToSplit] = bestClustAssment
        self.centroids = centlist
        self.clusterAssment = clusterAssment
        
        
    