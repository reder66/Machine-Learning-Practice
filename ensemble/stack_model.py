# -*- coding: utf-8 -*-

from sklearn.model_selection import KFold

def get_oof(clf, x_train, y_train, x_test, kf):
    oof_train = np.zeros(x_train.shape[0])
    oof_test = []
    for i, (train_id, test_id) in enumerate(kf.split(x_train)):
        x_train_oof = x_train[train_id, :]
        y_train_oof = y_train[train_id]
        x_test_oof = x_train[test_id, :]
        clf.fit(x_train_oof, y_train_oof)
        oof_train[test_id] = clf.predict(x_test_oof)
        oof_test.apend(clf.predict(x_test))
        
    oof_test = np.array(oof_test).mean(axis=0)
    return oof_train, oof_test
        
def stacking(model1, model2, k=4):
    kf = KFold(k)
    train_sets = []
    test_sets = []
    for clf in model1:
        oof_train, oof_test = get_oof(clf, x_train, y_train, x_test, kf)
        
        
class stackingModel:
    def __init__(self, first_layer_model, second_layer_model, k=4):
        kf = KFold(k)
        self.first_layer_model = first_layer_model
        self.second_layer_model = second_layer_model
        self.kf = kf
        
    def fit(self, x_train, y_train, x_test):
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        train_sets = []
        test_sets = []
        for clf in self.first_layer_model:
            oof_train, oof_test = get_oof(clf, x_train, y_train, x_test, self.kf)
            train_sets.append(oof_train)
            test_sets.append(oof_test)
        train_sets = np.array(train_sets).T
        test_sets = np.array(test_sets).T
        self.second_layer_model.fit(train_sets, y_train)
        self.y_pred = self.second_layer_model.predict(test_sets)
    
        
    