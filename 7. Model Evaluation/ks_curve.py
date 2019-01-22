# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


def plot_ks(y_true, y_prob, thresholds_num=1000):
    
    thresholds = np.linspace(np.min(y_prob), np.max(y_prob), thresholds_num)
    def tpr_fpr_delta(threshold):
        y_pred = np.array([int(i>threshold) for i in y_prob])
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        tpr = tp / (tp+fn)
        fpr = fp / (fp+tn)
        delta = tpr - fpr
        return tpr, fpr, delta

    tprs, fprs, deltas = np.vectorize(tpr_fpr_delta)(thresholds)
    target_tpr = tprs[np.argmax(deltas)]
    target_fpr = fprs[np.argmax(deltas)]
    target_threshold = thresholds[np.argmax(deltas)]
    ks_value = np.max(deltas)
 
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, tprs, label='TPR', color='r', linestyle='-', linewidth=1.5)    
    plt.legend(loc='upper right')
    plt.plot(thresholds, fprs, label='FPR', color='k', linestyle='-', linewidth=1.5)
    plt.legend(loc='upper right')
    plt.xlabel('Threshold', fontsize=10)
    plt.ylabel('TPR, FPR', fontsize=10)
    plt.annotate('KS Value : {:.6f}'.format(ks_value), xy=(target_threshold+0.01, 0.1+0.5*ks_value))
    plt.xticks()


    # 要连接的两个点的坐标
    x = [[target_threshold, target_threshold]] 
    y = [[target_fpr, target_tpr]]

    for i in range(len(x)):
        plt.plot(x[i], y[i], 'b--', lw=1.5)
        plt.scatter(x[i], y[i], c='b', s=15) # s控制点的大小
        plt.annotate('TPR : {:.6f}'.format(target_tpr), xy=([target_threshold, target_tpr]), xytext=(0.3, target_tpr),
                 arrowprops=dict(arrowstyle="<-", color='r')) 
        plt.annotate('FPR : {:.6f}'.format(target_fpr), xy=([target_threshold, target_fpr]), xytext=(0.3, target_fpr),
                 arrowprops=dict(arrowstyle="<-", color='k')) 
        plt.show()
