# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:05:46 2018

@author: Administrator
"""
import numpy as np
def binSplitData(data, feature, value):
    mat0 = data[data[:, feature] < value]
    mat1 = data[data[:, feature] >= value]
    return mat0, mat1

def meanLeaf(data):
    #叶节点为均值
    return np.mean(data[:, -1])

def calErr(data):
    #以总方差为混乱度
    return np.var(data[:, -1])*data.shape[0]

def meanLeafEval(model, inx):
    return float(model)

#%%
def calGini(data):
    y = data[:, -1]
    n = len(y)
    uniquey = set(y)
    gini = 1
    for i in uniquey:
        num = len(y[y==i])
        gini -= (num/n)**2
    return gini

def classifyLeaf(data):
    y = data[:, -1]
    classCnt = {}
    for i in y:
        classCnt[i] = classCnt.get(i, 0) + 1
    return sorted(classCnt.items(), key = lambda x:x[1], reverse = True)[0][0]

def classifyLeafEval(model, inx):
    return model
    
#%%
def linearSolve(data):
    #最小二乘估计
    #包含截距项
    m, n = data.shape
    X = np.matrix(np.ones((m, n)))
    X[:, 1:n] = data[:, :-1]
    y = np.matrix(data[:, -1]).T
    xTx = X.T*X
    ws = xTx.I*X.T*y
    return ws, X, y
    
def modelLeaf(data):
    ws, X, y = linearSolve(data)
    return ws

def modelErr(data):
    ws, X, y = linearSolve(data)
    yhat = X*ws
    return np.sum(np.power(y-yhat, 2))

def modelLeafEval(model, inx):
    n = inx.shape[0]
    X = np.matrix(np.ones((1, n+1)))
    X[:,1:] = inx
    return X*model

#%%
def chooseBestSplit(data, leafType, errType, min_samples_split, min_tol_split):
    '''
    leafType：叶节点模型
    errType：混乱度计算
    min_tol_split：容许的误差下降值
    min_samples_split：叶节点切分的最少样本数
    '''
    m, n = data.shape
    if len(set(data[:, -1]))==1:
        return None, leafType(data)
    baseS = errType(data)
    bestS = np.inf
    bestFeature = 0
    bestVal = 0
    for i in range(n-1):
        uniqueVal = set(data[:, i])
        for val in uniqueVal:
            mat0, mat1 = binSplitData(data, i, val)
            if (mat0.shape[0] < min_samples_split) or (mat1.shape[0] < min_samples_split):
                continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestFeature = i
                bestS = newS
                bestVal = val
                
    if (baseS - bestS) < min_tol_split:
        return None, leafType(data)
    mat0, mat1 = binSplitData(data, bestFeature, bestVal)
    if (mat0.shape[0] < min_samples_split) or (mat1.shape[0] < min_samples_split):
        return None, leafType(data)
    return bestFeature, bestVal
    
def createTree(data, leafType, errType, min_samples_split, min_tol_split, max_depth, depth = 1):
    bestFeature, bestVal = chooseBestSplit(data, leafType, errType, min_samples_split, min_tol_split)
    if bestFeature==None:
        return bestVal
    mat0, mat1 = binSplitData(data, bestFeature, bestVal)
    retTree = {}
    retTree['splitFeature'] = bestFeature
    retTree['splitVal'] = bestVal
    if max_depth != 'None' and depth >= max_depth:
        retTree['left'] = leafType(mat0)
        retTree['right'] = leafType(mat1)
        return retTree
    retTree['left'] = createTree(mat0, leafType, errType, min_samples_split, min_tol_split, max_depth, depth+1)
    retTree['right'] = createTree(mat1, leafType, errType, min_samples_split, min_tol_split, max_depth, depth+1)
    return retTree

#%%
#Prediction
def istree(tree):
    return isinstance(tree, dict)

def prune(tree, test):
    #后剪枝
    if istree(tree['left']) or istree(tree['right']):
        mat0, mat1 = binSplitData(test, tree['splitFeature'], tree['splitVal'])
    if istree(tree['left']):
        tree['left'] = prune(tree['left'], mat0)
    if istree(tree['right']):
        tree['right'] = prune(tree['right'], mat1)
    if not istree(tree['left']) and not istree(tree['right']):
        #叶子结点处
        mat0, mat1 = binSplitData(test, tree['splitFeature'], tree['splitVal'])
        errNoMerge = np.power(mat0[:, -1] - tree['left'], 2).sum() + np.power(mat1[:, -1] - tree['right'], 2).sum()
        mergeMean = (tree['left'] + tree['right'])/2
        errMerge = np.power(mat0[:, -1] - mergeMean, 2).sum() + np.power(mat1[:, -1] - mergeMean, 2).sum()
        if errNoMerge > errMerge:
            #合并
            return mergeMean
        return tree
    return tree
        
def treeForecast(tree, inx, leafEval = modelLeafEval):
    if not istree(tree):
        return leafEval(tree, inx)
    if inx[tree['splitFeature']] < tree['splitVal']:
        if istree(tree['left']):
            return treeForecast(tree['left'], inx, leafEval)
        else:
            return leafEval(tree['left'], inx)
    else:
        if istree(tree['right']):
            return treeForecast(tree['right'], inx, leafEval)
        else:
            return leafEval(tree['right'], inx)
        
#%%   
class CART:
    def __init__(self, tree_type = 'reg', prune_size = 0.2,\
                 min_samples_split = 2, max_depth = 'None', min_tol_split = 1e-16):
        self.tree_type = tree_type
        self.prune_size = prune_size
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_tol_split = min_tol_split
        self.eval = {'reg':meanLeafEval, 'model':modelLeafEval, 'classify':classifyLeafEval}
        self.leafType = {'reg':meanLeaf, 'model':modelLeaf, 'classify':classifyLeaf}
        self.errType = {'reg':calErr, 'model':modelErr, 'classify':calGini}
        
    def fit(self, X, y): 
        data = np.column_stack((X, y))
        if self.prune_size and self.tree_type=='reg':
            np.random.shuffle(data)
            n_train = int(data.shape[0]*self.prune_size)
            train =  data[:n_train, :]
            test = data[n_train:, :]
            rawTree = createTree(train, self.leafType[self.tree_type], self.errType[self.tree_type], \
                               self.min_samples_split, self.min_tol_split, self.max_depth)
            retTree = prune(rawTree, test)
        else:
            retTree = createTree(data, self.leafType[self.tree_type], self.errType[self.tree_type], \
                               self.min_samples_split, self.min_tol_split, self.max_depth)
        self.tree = retTree
    
    def predict(self, X):
        xMat = np.array(X)
        n = len(xMat)
        yhat = np.zeros(n)
        for i in range(n):
            yhat[i] = treeForecast(self.tree, xMat[i, :], self.eval[self.tree_type])
        return yhat
    
    def getTreeDepth(self, tree):
        maxdepth = 0
        for i in ['left', 'right']:
            if isinstance(tree.get(i, 0), dict):
                depth = self.getTreeDepth(tree[i]) + 1
            else:
                depth = 1
            if depth>maxdepth:
                maxdepth = depth
        return maxdepth
