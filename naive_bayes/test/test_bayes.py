# -*- coding: utf-8 -*-

import pandas as pd
from Naive_Bayes.bayes import naive_bayes
from sklearn.model_selection import train_test_split
from metrics.classification_report import confusion_matrix, binary_classification_report


data = pd.read_csv('data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75)
bay = naive_bayes(distribution='normal')
bay.fit(x_train, y_train)

#%%
yhat = bay.predict(x_test)
accuracy = np.sum(yhat==y_test)/x_test.shape[0]
cm = confusion_matrix(yhat, y_test)
pr,re,f1,ka = binary_classification_report(cm)
print('Accuracy: {}'.format(accuracy))
