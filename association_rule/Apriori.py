# -*- coding: utf-8 -*-

# =============================================================================
#   These part belongs to apriori.fit, which is going to find frequent items.In 
# the next part we will define some functions to find association rule. 
# =============================================================================
def createC1(data):
    C1 = []
    for tid in data:
        for can in tid:
            if [can] not in C1:
                C1.append([can])
    return list(map(frozenset, C1))

def scanD(data, Ck, minSupport):
    ssCnt = {}
    for tid in data:
        for can in Ck:
            if can.issubset(tid):
                ssCnt[can] = ssCnt.get(can, 0) + 1 
    
    Lk = []
    supportData = {}
    N = len(data)
    for key in ssCnt:
        support = ssCnt[key]/N
        if support >= minSupport:
            Lk.append(key)
        supportData[can] = support
    return Lk, ssCnt

def aprioriGen(Lk, k):
    # create Ck from lk-1
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range(i+1, lenLk):
            l1 = list(Lk[i])[:k-2].sort()
            l2 = list(Lk[j])[:k-2].sort()
            if l1==l2:
                retList.append(Lk[i]|Lk[j])
    return retList

# =============================================================================
# Some functions to find association rules.
# =============================================================================
def calConf(freqSet, H, supportDict, brl, minConf):
    prunedH = []
    for conseq in H:
        conf = supportDict[freqSet]/supportDict[freqSet-conseq]
        if conf >= minConf:
            print(freqSet-conseq, '--->', conseq, 'conf: ', conf)
            brl.append((freqSet-conseq, conseq, conf))
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportDict, brl, minConf):
    m = len(H[0])
    if len(freqSet) > m+1 :
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calConf(freqSet, Hmp1, supportDict, brl, minConf)
        if len(Hmp1) > 1:
            rulesFromConseq(freqSet, Hmp1, supportDict, brl, minConf)
# =============================================================================
# Now we can build our class
# =============================================================================
class apriori:
    def __init__(self, minSupport=0.5, minConf=0.5):
        self.minSupport = minSupport
        self.minConf = minConf
        
    def fit(self, data):
         #L is all the frequent items, and supportDict save the support data
        C1 = createC1(data)
        D = list(map(set, data))
        L1, supportDict = scanD(D, C1, self.minSupport)
        L = [L1]
        k = 2
        while len(L[k-2])>0:
            Ck = aprioriGen(L[k-2], k)
            Lk, ssCnt = scanD(D, Ck, self.minSupport)
            L.append(Lk)
            supportDict.update(ssCnt)
            k += 1
        self.L = L
        self.supportDict = supportDict
    
    def generateRules(self):
        bigRuleList = []
        for i in range(1, len(self.L)):
            for freqSet in self.L[i]:
                H1 = [frozenset([item]) for item in freqSet]
                if i > 1:
                    rulesFromConseq(freqSet, H1, self.supportDict, bigRuleList, self.minConf)
                else:
                    calConf(freqSet, H1, self.supportDict, bigRuleList, self.minConf)
        self.rules = bigRuleList
        
        