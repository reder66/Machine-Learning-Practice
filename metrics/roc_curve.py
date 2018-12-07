# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 23:46:04 2018

@author: Administrator
"""
import numpy as np
import matplotlib.pyplot as plt
def roc_curve(y_pred, y_label):
    sortedIndex = np.argsort(y_pred)
    n = len(y_label)
    numPositive = np.sum(y_label==1)
    numNegative = n - numPositive
    yStep = 1/float(numPositive)
    xStep = 1/float(numNegative)
    #从FN=0,TN=0，即(1,1)开始作图
    cur = (1.0, 1.0)
    ySum = 0
    tpr = np.ones(n)
    fpr = np.ones(n)
    threshold = y_pred[sortedIndex][::-1] #从大到小
    i = n
    fig, ax = plt.subplots()
    for index in sortedIndex:
        if y_label[index] == 1:
            #FN+1,y坐标减少一个单位
            xDel = 0
            yDel = yStep 
        else:
            #TN+1,x坐标减少一个单位
            xDel = xStep
            yDel = 0
            ySum += cur[1]
        cur = (cur[0] - xDel, cur[1] - yDel)
        ax.plot(cur[0], cur[1])
        tpr[i] = cur[0]
        fpr[i] = cur[1]
        i -= 1
    plt.title('ROC Curve')
    plt.show()
    auc = ySum * xStep
    return tpr, fpr, threshold, auc
    
