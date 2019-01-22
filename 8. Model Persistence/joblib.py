# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

from sklearn.datasets import load_breast_cancer 
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression as LR
from sklearn.model_selection import train_test_split


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2018)
clf = LR().fit(X_train, y_train)

运用joblib序列化和反序列化机器学习模型
with open('cancer_joblib', 'wb') as model:
    joblib.dump(clf, model)


with open('cancer_joblib','rb') as model:
    clf = joblib.load(model)
    result = clf.score(X_test, y_test)
    print('算法评估结果：{:.2%}'.format(result))