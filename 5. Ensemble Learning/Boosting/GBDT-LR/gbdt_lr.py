# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from scipy.sparse import hstack
from sklearn.datasets import make_classification
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import OneHotEncoder


# 生成实验数据集
X, y  = make_classification(n_samples=10000, n_features=20, n_informative=18, n_redundant=2,
                            n_classes=2, n_clusters_per_class=3, random_state=2017)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


# 不生成新的特征，直接训练
clf = GradientBoostingClassifier(n_estimators=50)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
print("Original featrues")
print("GBDT_ACC: {:.6f}".format(acc))
print("GBDT_AUC: {:.6f}".format(auc))


# 生成的新特征, apply方法返回每个样本在每颗树叶节点的索引矩阵
X_train_leaves = clf.apply(X_train)[:, :, 0]
X_test_leaves = clf.apply(X_test)[:, :, 0]

# 将X_train_leaves, X_test_leaves在axis=0方向上合并，再进行OneHotEncoder操作
All_leaves = np.r_[X_train_leaves, X_test_leaves]

# 索引矩阵每列不是0/1二值型离散特征，因此需要OneHotEncoder操作
enc = OneHotEncoder(categories='auto')
new_features = enc.fit_transform(All_leaves)

# 根据原训练集、测试集的索引对新特征予以拆分
train_samples = X_train.shape[0]
X_train_new = new_features[:train_samples, :]
X_test_new = new_features[train_samples: , :]

# 将初始训练集与GBDT新生成的特征联合后再训练LR
X_train_hstack = hstack([X_train_new, X_train])
X_test_hstack = hstack([X_test_new, X_test])

lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X_train_hstack, y_train)
y_pred = lr.predict(X_test_hstack)
y_prob = lr.predict_proba(X_test_hstack)[:, 1]

# 进行预测
GBDT_LR_ACC = accuracy_score(y_test, y_pred)
GBDT_LR_AUC = roc_auc_score(y_test, y_prob)
print("\nNew featrues: ")
print('GBDT_LR_ACC: {:.6f}'.format(GBDT_LR_ACC))
print('GBDT_LR_AUC: {:.6f}'.format(GBDT_LR_AUC))