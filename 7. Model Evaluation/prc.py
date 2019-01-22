# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr


from sklearn.metrics import precision_recall_curve
from matplotlib import pyplot as plt


def plot_prc(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    plt.plot(recall, precision, color='red', linestyle='-', linewidth=1.5)   
    plt.xlabel('TPR', fontsize=10)
    plt.ylabel('Precison', fontsize=10)
    plt.title('Precison-Recall Curve')
    plt.show()