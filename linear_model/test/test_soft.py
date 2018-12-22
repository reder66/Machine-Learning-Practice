# -*- coding: utf-8 -*-

import pandas as pd
from linear_model.softmax import SoftMax
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.75)

soft = SoftMax(300, 0.05)
soft.fit(x_train, y_train)
yhat = soft.predict(x_test)


accuracy = np.sum(yhat==y_test)/x_test.shape[0]


print('SoftMax weights:\n',soft.weights)
print('SoftMax tesint accuracy: %.3f'%accuracy)
