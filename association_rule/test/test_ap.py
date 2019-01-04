# -*- coding: utf-8 -*-

from association_rule.Apriori import apriori

df = [[1,3,4], [2,3,5], [1,2,3,5], [2,5]]
ap = apriori(0.6)
ap.fit(df)
print(ap.L)
print(ap.supportDict)
print(ap.generateRules())