# Author：MaXiao
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


cancer = load_breast_cancer()
X, y = load_breast_cancer(return_X_y=True)
X_scaled = StandardScaler().fit_transform(X)

# 基于L1正则化的特征选择
linear_svc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_scaled, y)
sfm_linear_svc = SelectFromModel(linear_svc, prefit=False)
sfm_linear_svc.fit(X_scaled, y)
X_selected = sfm_linear_svc.transform(X_scaled)
print('shape: {:} ---> shape:{:}\n'.format(X.shape, X_selected.shape))

# get_support属性返回布尔型列表，若特征被保留，则显示True
get_support = {'Support' : sfm_linear_svc.get_support()}
sfm_result = pd.DataFrame(get_support, index=cancer.feature_names)
print(sfm_result[sfm_result['Support']==True])


#基于树模型进行模型选择
rf = RandomForestClassifier(n_estimators=100, random_state=10)
rf.fit(X, y)

# 选择特征重要性为1.2倍均值的特征
sfm_rf = SelectFromModel(rf, threshold='1.2*mean',prefit=True)

#返回所选的特征
X_selected_rf = sfm_rf.transform(X)
print('\nshape:{:}--->shape:{:}'.format(X.shape, X_selected_rf.shape))

mask = sfm_rf.get_support()
plt.matshow(mask.reshape(1, -1), cmap=plt.cm.Reds)#, aspect='auto')
plt.xlabel('Sample index')
plt.ylim(-0.5, 0.5)
plt.yticks([-0.5, 0.5])
plt.show()
