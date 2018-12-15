# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from decomposition.PCA import PCA

data = pd.read_csv('breast_cancer.csv')
X = data.iloc[:,:-1]

pca = PCA(5)
x = pca.fit_transform(X)
pca.scree_plot()
print(pca.variance_explained_ratio_)