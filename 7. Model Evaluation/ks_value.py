# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr


import numpy as np
from sklearn.metrics import confusion_matrix, make_scorer


def ks(y_true, y_prob, thresholds_num=1000):
    # 生成一系列阈值
    thresholds = np.linspace(np.min(y_prob), np.max(y_prob), thresholds_num) 
    
    def tpr_fpr_delta(threshold):
        y_pred = np.array([int(i>threshold) for i in y_prob])
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp+tn)
        tpr = tp / (tp+fn)
        delta = tpr - fpr
        return delta

    max_delta = np.max(list(map(tpr_fpr_delta, thresholds)))
    return max_delta

# ks_score可直接用于网格搜索scoring参数的取值
ks_score = make_scorer(ks, greater_is_better=True, needs_proba=True) 