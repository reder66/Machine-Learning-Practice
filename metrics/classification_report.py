# -*- coding: utf-8 -*-
'''
In this part we only care about binary-class label
'''

import numpy as np

def confusion_matrix(y_pred, y_true):
    y_pred = np.array(y_pred).ravel()
    y_true = np.array(y_true).ravel()
    N = len(y_pred)
    k = len(np.unique(y_true))
    loc_dict = {}
    for i in range(k):
        loc_dict[np.unique(y_true)[i]] = i 
    cm = np.zeros((k, k))
    for i in range(N):
        cm[loc_dict[y_true[i]], loc_dict[y_pred[i]]] += 1
    return cm
    
def precision(cm, i=0):
    return cm[i, i]/cm[:, i].sum()

def recall(cm, i=0):
    return cm[i, i]/cm[i, :].sum()

def F1_score(cm):
    pr = precision(cm)
    re = recall(cm)
    return 2*pr*re/(pr+re)

def kappa(cm):
    N = cm.sum()
    m = cm.shape[0]
    p0 = np.trace(cm)/N
    pe = 0
    for i in range(m):
        for j in range(m):
            pe += cm[i, j]*cm[j, i]
    pe /= N**2
    return (p0-pe)/(1-pe)

'''
In this part we will care about multi-class label
'''
def macro_average(cm):
    m = cm.shape[0]
    pr = []
    re = []
    for i in range(m):
        pr.append(precision(cm, i))
        re.append(recall(cm, i))
    pr = np.mean(pr)
    re = np.mean(re)
    return pr, re

def micro_average(cm):
    m = cm.shape[0]
    tp = 0
    fp = 0
    fn = 0
    for i in range(m):
        tp += cm[i, i]
        fp += cm[:, i].sum()
        fn += cm[i, :].sum()
    return tp/fp, tp/fn

'''
Then we create report
'''
def binary_classification_report(cm):
    pr = precision(cm)
    re = recall(cm)
    f1 = F1_score(cm)
    ka = kappa(cm)
    print('''
          Precision score: %.3f\n
          Recall score: %.3f\n
          F1 score: %.3f\n
          kappa coefficients: %.3f'''%(pr, re, f1, ka))
    return pr,re,f1,ka

def multiple_classification_report(cm):
    macro_pr, macro_re = macro_average(cm)
    micro_pr, micro_re = micro_average(cm)
    print('''
          Macro average precision score: %.3f\n
          Macro average recall score: %.3f\n
          Micro average precision score: %.3f\n
          Micro average recall score: %.3f'''%(macro_pr, macro_re, micro_pr, micro_re))
    return macro_pr, macro_re, micro_pr, micro_re

    