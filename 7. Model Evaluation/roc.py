# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr


from sklearn.metrics import roc_curve
from matplotlib import pyplot as plt


def plot_roc(y_true, y_prob):
    fprs, tprs, thresholds = roc_curve(y_true, y_prob, pos_label=1) 
    plt.plot(fprs, tprs, 'r-', label='ROC', lw=1.5)    
    plt.legend(loc='lower right')
    plt.xlabel('FPR',fontsize=10)
    plt.ylabel('Recall', fontsize=10)
    plt.title('ROC')
    plt.show()