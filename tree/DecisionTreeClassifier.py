# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 16:08:42 2018

@author: Administrator
"""
import numpy as np
import pandas as pd
def calEnt(data):
    #最后一列为label
    label = data[:, -1]
    n = data.shape[0]
    labelcounts = {}
    entropy = 0
    for i in label:
        labelcounts[i] = labelcounts.get(i, 0) + 1
    for key in labelcounts:
        prob = labelcounts[key]/n
        entropy -= prob*np.log2(prob)
    return entropy

def splitData(data, i, val):
    #切分数据，只用于全离散变量
    sub = data[data[:,i]==val]
    sub = np.delete(sub, i, axis=1)
    return sub

def chooseBestFeatureToSplit(data):
    #从一个数据集中选择最好特征去切分
    #利用信息增益
    n, m = data.shape
    baseEnt = calEnt(data)
    bestGain = 0
    bestFeature = -1
    for i in range(m-1):
        uniqueVal = set(data[:,i])
        newEnt = 0
        for val in uniqueVal:
            subdata = splitData(data, i, val)
            subEnt = calEnt(subdata)
            prob = len(subdata)/n
            newEnt += prob*subEnt
        infoGain = baseEnt - newEnt
        if infoGain>bestGain:
            bestGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCnt = {}
    for i in classList:
        classCnt = classCnt.get(i, 0) + 1
    return sorted(classCnt.items(), key = lambda x:x[1], reverse = True)[0][0]

def createTree(data, colnames):
    classList = data[:, -1]
    if len(set(classList))==1: #只有1个类别
        return classList[0]
    if data.shape[1]==1: #用完了所有特征还没划分好
        return majorityCnt(classList)
    bestFeature = chooseBestFeatureToSplit(data)
    featureName = colnames[bestFeature]
    del colnames[bestFeature]
    mytree = {featureName:{}}
    uniqueVal = set(data[:,bestFeature])
    for val in uniqueVal:
        subdata = splitData(data, bestFeature, val)
        subnames = colnames.copy()
        mytree[featureName][val] = createTree(subdata, subnames)
    return mytree

#%%
class DecisionTree:
    
    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            colnames = list(X.columns)
        else:
            try:
                n = X.shape[1]
                colnames = ['x%d'%i for i in range(n)]   
            except:
                print('Please input dataframe.')
                return
        data = np.column_stack((X, y))            
        tree = createTree(data, colnames)
        self.tree = tree
    
    def getNumLeafs(self, tree):
        #有多少个叶结点，即多少个key
        numLeafs = 0
        for key, values in tree.items():
            if isinstance(tree[key], dict):
                numLeafs += self.getNumLeafs(tree[key])
            else:
                numLeafs += 1
        return numLeafs
    
    def getTreeDepth(self, tree):
        maxdepth = 0
        for key, values in tree.items():
            if isinstance(tree[key], dict):
                depth = self.getTreeDepth(tree[key]) + 1
            else:
                depth = 1
            if depth>maxdepth:
                maxdepth = depth
        return maxdepth
    
    def predict(self, X, verbose = False):
        n = len(X)
        yhat = np.array([0]*n)
        for i in range(n):
            yhat[i] = self.predict_path(self.tree, X.iloc[i, :], verbose)
        return yhat
        
    def predict_path(self, tree, inx, verbose):
        for key, values in tree.items():
            pred = inx[key]
            if verbose:
                print(key, pred)
            if isinstance(values[pred], dict):
                return self.predict_path(values[pred], inx, verbose)
            else:
                return values[pred]
