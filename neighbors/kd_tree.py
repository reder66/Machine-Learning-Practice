# -*- coding: utf-8 -*-
import numpy as np

class treeNode:
    def __init__(self, thresholdFea = None, thresholdVal = None):
        self.left = None
        self.right = None
        self.thresholdFea = thresholdFea
        self.thresholdVal = thresholdVal
        
def chooseBestToSplit(data):
    m, n = data.shape
    bestFeature = 0
    bestVal = 0
    maxVar = 0
    for fea in range(n):
        var = np.var(data[:, fea])
        if var > maxVar:
            bestFeature = fea
            bestVal = np.median(data[:, fea])
            maxVar = var
    if maxVar==0 or m==1:
        return None, bestVal
    return bestFeature, bestVal
    
def binSplitData(data, feature, pivot):
    mat0 = data[data[:, feature]<pivot]
    mat1 = data[data[:, feature]>=pivot]
    return mat0, mat1

        
def createTree(data):
    bestFeature, bestVal = chooseBestToSplit(data)
    if bestFeature==None:
        return data
    retTree = treeNode(thresholdFea=bestFeature, thresholdVal=bestVal)
    mat0, mat1 = binSplitData(data, bestFeature, bestVal)
    retTree.left = createTree(mat0)
    retTree.right = createTree(mat1)
    return retTree

#####################
class kdTree:
    def __init__(self, data):
        data = np.array(data)
        self.tree = createTree(data)
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        