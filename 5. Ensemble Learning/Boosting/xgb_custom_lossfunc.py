# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr



import numpy as np
from xgboost import XGBClassifier, XGBRegressor


# 1. 以ln(cosh(x))为损失函数
def log_cosh_obj(y_true, y_pred):
    delta = y_pred - y_true 
    grad = np.tanh(delta)
    hess = (1.0 - grad*grad)
    return grad, hess

# 回归问题
model = XGBRegressor(objective=log_cosh_obj)
# 分类问题
model = XGBClassifier(objective=log_cosh_obj)



# 2. Pseudo-Huber loss function，可以近似替代MAE
def huber_approx_obj(y_true, y_pred, h=1):
    # h为Pseudo-Huber loss function中的参数，用于调节坡度，其值越大，图像越陡峭
    d = y_pred - y_true 
    scale = 1 + np.square(d / h)
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess

# 回归问题
model = XGBRegressor(objective=huber_approx_obj)
# 分类问题
model = XGBClassifier(objective=huber_approx_obj)


# 3. 以log(exp(-x) + exp(x))为损失函数：更适合处理分类问题
def log_exp(y_true, y_pred):
    d = y_pred - y_true
    t1 = np.exp(d) - np.exp(-d) 
    t2 = np.exp(d) + np.exp(-d) 
    grad = t1 / t2
    hess = 1.0 - grad**2 
    return grad, hess

# 分类问题
model = XGBClassifier(objective=log_exp)