# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from linear_model.GDA import GDA
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

gda = GDA()
gda.fit(x_train, y_train)
yhat = gda.predict(x_test)


accuracy = np.sum(yhat==y_test)/x_test.shape[0]

print('GDA parameters:\n')
print(gda.phi, '\n', gda.mu, '\n', gda.sigma)
print('GDA testing accuracy: %.3f'%accuracy)