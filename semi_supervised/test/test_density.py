# -*- coding: utf-8 -*-
###########test#############
import numpy as np
import pandas as pd
from metrics.classification_report import binary_classification_report, confusion_matrix
from naive_bayes.bayes import naive_bayes
from semi_supervised.density_based import density_method


data = pd.read_csv('data.csv')
data = data.sample(frac=1)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

model = density_method(naive_bayes('normal'))
k = int(len(y)*0.25)
y_test = y[k:]

#####测试
model.fit(X, y[:k])
yhat = model.predict(X[k:, :])
accuracy = np.sum(yhat==y_test)/y_test.shape[0]
cm = confusion_matrix(yhat, y_test)
pr,re,f1,ka = binary_classification_report(cm)
print('Accuracy: {}'.format(accuracy))

