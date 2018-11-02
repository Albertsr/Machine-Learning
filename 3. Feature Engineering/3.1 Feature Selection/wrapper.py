# Author：MaXiao
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.feature_selection import RFE, RFECV  
from xgboost import XGBRegressor

'''
API说明： 
sklearn.feature_selection模块提供了两个API可用于wrapper，分别为：
1. RFE(estimator, n_features_to_select=None, step=1, verbose=0)
2. RFECV(estimator, step=1, min_features_to_select=1, cv='warn', scoring=None, verbose=0, n_jobs=None)

- 两者的区别在于RFECV可以通过交叉验证的方式返回最佳的特征数，而RFE需要通过参数n_features_to_select预先指定；
- estimator：模型必须具备coef_或feature_importances_属性用于评估特征重要性。
             一般来说线性模型以及线性核SVM具备coef_属性、决策树类算法具备feature_importances_属性 
- step：整数或小数形式，表示每次迭代剔除的特征数或特征占比；

属性说明：
1. RFECV_XGB.support_ ：布尔值列表，若特征被保留则相应索引处为True，否则为False
2. RFECV_XGB.ranking_ ：数值型列表，若特征被保留则相应索引处为1，否则大于1，且ranking值越大，特征越不重要
3. RFECV_XGB.grid_scores_ ：数值型列表，表示特征子集的交叉验证分数，与特征是否被选择没有太大关系

'''

X, y = load_boston(return_X_y=True)
xgb = XGBRegressor(learning_rate=0.2, n_estimators=150, random_state=2017)
RFECV_XGB = RFECV(xgb, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
RFECV_XGB.fit(X, y)
print('Original features: {:}'.format(X.shape[1]))
print('RFECV_XGB features: {:}'.format(RFECV_XGB.n_features_))

# 将RFECV_XGB的训练结果用pandas.dataframe进行直观展示
feature_name = load_boston().feature_names
rfecv_dict = {'Support':RFECV_XGB.support_, 'Ranking':RFECV_XGB.ranking_, 'Grid_scores':RFECV_XGB.grid_scores_}
rfecv_result = pd.DataFrame(rfecv_dict, index=feature_name)
# 根据Ranking对rfecv_result升序排列
rfecv_result.sort_values('Ranking', inplace=True)

# 将保留特征对应的support_与ranking_属性标红
def highlight(s):
    if isinstance(s[0], np.bool_):
        cond = s == s.max()
    else:
        cond = s == s.min()
    return ['color: red' if v else '' for v in cond]
print(rfecv_result.style.apply(highlight, subset=['Support', 'Ranking']))
