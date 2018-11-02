# Author：MaXiao
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr


from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.feature_selection import GenericUnivariateSelect, mutual_info_classif, mutual_info_regression

'''
API : GenericUnivariateSelect(score_func=mutual_info_classif, mode='percentile', param=85)
   
   1) 参数score_func为评分函数。
   - 对于分类问题，可以取值：chi2(卡方检验)、mutual_info_classif(互信息)、f_classif(F检验)
   - 对于回归问题，可以取值：mutual_info_regression、f_regression 
   
   注意事项：
   - 卡方检验可用于检测非负特征与分类标签列的独立性，卡方统计量越大，两者越可能相互独立
   - 互信息既能捕捉到线性关系，也能捕捉到非线性关系，因此多采用mutual_info_classif或mutual_info_regression
   
   2）参数mode为选择模式
   - 可以取值：{'percentile', 'k_best', 'fpr', 'fdr', 'fwe'}
   - 'fpr' : Select features based on a false positive rate test.只能用于分类问题.
   - 'fdr' : Select features based on an estimated false discovery rate.只能用于分类问题.
   - 'fwe' : Select features based on family-wise error rate.
   
   3) 参数param的取值范围由参数mode的取值决定，例如mode='percentile',param=80表示取分数位于前80%的特征   
'''

# 分类问题：乳腺癌数据集
X_cancer, y_cancer = load_breast_cancer(return_X_y=True)
transformer = GenericUnivariateSelect(score_func=mutual_info_classif, mode='percentile', param=85)
X_cancer_selected = transformer.fit_transform(X_cancer, y_cancer)
print("Cancer's shape: {} ---> {}".format(X_cancer.shape, X_cancer_selected.shape))

# 回归问题：波士顿房价数据集
X_boston, y_boston = load_boston(return_X_y=True)
transformer = GenericUnivariateSelect(score_func=mutual_info_regression, mode='percentile', param=85)
X_boston_selected = transformer.fit_transform(X_boston, y_boston)
print("Boston's shape: {} ---> {}".format(X_boston.shape, X_boston_selected.shape))
