# Author：MaXiao
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from sklearn.model_selection import KFold


def StackingModels(models, meta_model, X_train, y_train, X_test, task='clf', use_probas=True, cv=5, random_state=2018):
    ntrain, ntest = X_train.shape[0], X_test.shape[0]
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    def cross_validator(model):
        valid_pred = np.zeros((ntrain))
        test_pred = np.zeros((ntest, cv))
        for i, (train_index, valid_index) in enumerate(kf.split(X_train)):
            # 将初始训练集进行K折交叉检验，其中K-1折作为新的训练集，剩余1折作为验证集
            X_train_kfold, y_train_kfold = X_train[train_index], y_train[train_index]
            X_valid_kfold, y_valid_kfold = X_train[valid_index], y_train[valid_index]
            # 训练模型，并对验证集进行预测
            model.fit(X_train_kfold, y_train_kfold)
            valid_pred[valid_index] = model.predict(X_valid_kfold)
            # 对测试集进行预测
            test_pred[:, i] = model.predict(X_test)
            
        if task == 'clf':
            test_pred_final = np.array([1 if i>0.5 else 0 for i in test_pred.mean(axis=1)])
        elif task=='reg':
            test_pred_final = test_pred.mean(axis=1)
        return valid_pred, test_pred_final
    
    # 生成第二级的训练集和测试集
    train_second = np.zeros((ntrain, len(models)))
    test_second = np.zeros((ntest, len(models)))
    for i, j in enumerate(map(cross_validator, models)):
        train_second[:, i] = j[0]
        test_second[:, i] = j[1]
    assert train_second.shape == (ntrain, len(models))
    assert test_second.shape == (ntest, len(models))

    meta_model.fit(train_second, y_train)
    test_pred = meta_model.predict(test_second)
    
    if use_probas:
        test_prob = meta_model.predict_proba(test_second)[:, -1]
        return test_pred, test_prob
    else:
        return test_pred