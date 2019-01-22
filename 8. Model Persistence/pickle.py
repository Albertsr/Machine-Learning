# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import pickle
from sklearn.datasets import load_breast_cancer 
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2018)
clf = LR().fit(X_train, y_train)

# 运用pickle序列化机器学习模型, 保存为字符串形式
s = pickle.dumps(clf)

# 反序列化
clf_load = pickle.loads(s)

# 输出模型预测精度
print(clf_load.score(X_test, y_test))

# 用dump(object, file) 将模型保存至磁盘
with open('clf_pickle', 'wb') as model:
    pickle.dump(clf, model)

# 运用pickle调用模型，并输出模型结果
with open('clf_pickle', 'rb') as model:
    loaded_clf = pickle.load(model)
    result = loaded_clf.score(X_test,y_test)
    print('算法评估结果：%.2f%%' % (result*100))