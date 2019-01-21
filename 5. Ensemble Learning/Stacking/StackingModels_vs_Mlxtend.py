# Author：MaXiao
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  
from sklearn.linear_model import LogisticRegression, LinearRegression 
from mlxtend.classifier import StackingCVClassifier
from mlxtend.regressor import StackingCVRegressor
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from StackingModels import StackingModels


X, y  = make_classification(n_samples=10000, n_features=20, n_informative=18, n_clusters_per_class=3, hypercube=1, 
                            class_sep=0.85, random_state=2018)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2018)

scaler = StandardScaler()
X_train, X_test = map(scaler.fit_transform, [X_train, X_test])

rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=2018, n_jobs=8)
xgb = XGBClassifier(n_estimators=50, learning_rate=0.75, random_state=2018, n_jobs=8)
lgb = LGBMClassifier(n_estimators=50, learning_rate=0.75, random_state=2018, n_jobs=8)
svc = SVC(kernel='rbf', random_state=2018, probability=True, gamma='auto')
lr = LogisticRegression(max_iter=1000, solver='lbfgs', penalty='l2', n_jobs=8)
models = [rf, xgb, lgb, svc]
y_pred_self, y_prob_self = StackingModels(models=models, meta_model=lr, X_train=X_train, X_test=X_test, y_train=y_train)
acc = accuracy_score(y_test, y_pred_self)
auc = roc_auc_score(y_test, y_prob_self)
print('MyModel:  ACC = {:.6f}, AUC = {:.6f}'.format(acc, auc))
stack_clf = StackingCVClassifier(classifiers=models, meta_classifier=lr, cv=5).fit(X_train, y_train)
y_pred_mxltend, y_prob_mxltend = stack_clf.predict(X_test), stack_clf.predict_proba(X_test)[:, -1]
acc = accuracy_score(y_test, y_pred_mxltend)
auc = roc_auc_score(y_test, y_prob_mxltend)
print('Mlxtend:  ACC = {:.6f}, AUC = {:.6f}'.format(acc, auc))


X, y  = make_regression(n_samples=5000, n_features=20, n_informative=18, random_state=2018)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2018)
X_train, X_test = map(scaler.fit_transform, [X_train, X_test])

rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=2018, n_jobs=8)
xgb = XGBRegressor(n_estimators=50, learning_rate=0.75, random_state=2018, n_jobs=8)
lgb = LGBMRegressor(n_estimators=50, learning_rate=0.75, random_state=2018, n_jobs=8)
svr = SVR(kernel='rbf', gamma='auto')
lr = LinearRegression(n_jobs=8)
models = [rf, xgb, lgb, svr]

y_pred_self = StackingModels(models=models, meta_model=lr, X_train=X_train, 
                             X_test=X_test, y_train=y_train, use_probas=False, task_mode='reg')
mse = mean_squared_error(y_test, y_pred_self)
print('MyModel:  MSE = {:.6f}'.format(mse))

stack_reg = StackingCVRegressor(regressors=models, meta_regressor=lr, cv=5).fit(X_train, y_train)
y_pred_mxltend = stack_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred_mxltend)
print('Mlxtend:  MSE = {:.6f}'.format(mse))